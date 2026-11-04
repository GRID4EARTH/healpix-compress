"""Two-layer coding of a nested chunk of 16-bit integer data: a lossless parent
level and a predicted, quantised child level.

A chunk of :math:`4^\\delta` values at level :math:`L` is stored as

1. the **base**: the mean of the four children of every level-:math:`L-1`
   parent, rounded to the 16-bit digital-number lattice
   (:func:`hcc.compression.uint16_jxl.quantize`) and coded with lossless
   JPEG XL, so the parent level is exact up to that rounding;
2. a **prediction** of the children from the base: one least-squares filter
   per child phase over the :math:`k \\times k` parent neighbourhood (finite
   differences up to order :math:`k - 1`), its coefficients stored with the
   chunk;
3. the **detail**: the difference between the children and their prediction,
   quantised with a step :math:`s` (in digital numbers) and coded with
   lossless JPEG XL.

The step is set per chunk from the detail itself,
:math:`s = \\max(s_{\\min}, r \\, \\sigma)`, with :math:`\\sigma` the standard
deviation of the detail in digital numbers and :math:`r` the
``relative_step``. Rough chunks get a coarse step and smooth chunks a fine
one, which is what an error budget proportional to the local transfer error
asks for: for a 10 m reflectance band resampled to HEALPix with a
point-spread-function kernel, the error the transfer itself leaves on the native pixels
is close to :math:`0.058 \\sigma`, and the quantisation adds about
:math:`0.16 s` there, so :math:`r \\approx 0.36` puts the added error at the
transfer error, for a size of about 1.5 times the native tile under lossless
JPEG 2000. The default :math:`r = 0.57` aims at about 1.3 times that size,
for an added error of about 1.5 times the transfer error.
``relative_step = 0`` uses :math:`s = s_{\\min}`; with the default
:math:`s_{\\min} = 1` the children then sit on the DN lattice, as in
:mod:`hcc.compression.uint16_jxl`.

Cells that are ``NaN`` round-trip as ``NaN``: a parent without any valid
child is the base's no-data number, and the validity of the children of the
partially valid parents is stored as a bit mask.

The JPEG XL and zlib streams go through imagecodecs and zlib. The rest of
:func:`encode` and :func:`decode` runs in the optional native kernels when the
package ``healpix-compress-native`` is installed (see
:mod:`hcc.compression._native`), without holding the GIL; this module's
Python implementation stays the reference, and the kernels reproduce it bit
for bit.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import numpy.typing as npt

from hcc.compression import _native, uint16_jxl
from hcc.compression.zorder import expand_tile, flatten_tile

_MAGIC = b"HS2P"
_VERSION = 1
# magic, version, delta, k, flags, step, offset, then the byte length of the
# coefficients, the base codestream, the detail codestream and the mask
_HEADER = struct.Struct("<4sBBBBfiIIII")
_FLAG_MASK = 1
# the four children of a parent, as (row, column) offsets in the child raster
_PHASES = ((0, 0), (0, 1), (1, 0), (1, 1))

#: ``relative_step`` at which the added error matches the transfer error of
#: a point-spread-function resampling, measured on one 10 m reflectance band.
TRANSFER_FLOOR_RELATIVE_STEP = 0.36

#: Default ``relative_step``: about 1.3 times the native tile under lossless
#: JPEG 2000 on the same measurement, for an added error of about 1.5 times
#: the transfer error.
DEFAULT_RELATIVE_STEP = 0.57

#: Default JPEG XL ``decoding_speed``: on 10 m reflectance chunks at effort 7,
#: about 3 % larger than ``0`` for half the JPEG XL encoding and decoding times.
#: The decoded cells do not depend on it.
DEFAULT_DECODING_SPEED = 2


def _design(base: np.ndarray, k: int) -> np.ndarray:
    """``(h * h, k * k + 1)``: the edge-padded parent neighbourhoods, then ones."""
    h = base.shape[0]
    p = np.pad(base, k // 2, mode="edge")
    design = np.empty((h, h, k * k + 1))
    # one copy of the k x k windows into the columns, no stacked temporaries
    columns = design[..., : k * k].reshape(h, h, k, k)
    columns[...] = np.lib.stride_tricks.sliding_window_view(p, (k, k))
    design[..., k * k] = 1.0
    return design.reshape(h * h, k * k + 1)


def _replicate_coefficients(k: int) -> np.ndarray:
    coef = np.zeros((4, k * k + 1), dtype=np.float32)
    coef[:, (k // 2) * k + k // 2] = 1.0
    return coef


def _fit(
    base: np.ndarray,
    raster: np.ndarray,
    full: np.ndarray,
    k: int,
    design: np.ndarray | None = None,
) -> np.ndarray:
    """One least-squares filter per child phase, fitted on the fully valid parents.

    The four phases share the design matrix: one solve with four right-hand sides.
    """
    if design is None:
        design = _design(base, k)
    rows = full.ravel()
    if rows.sum() < design.shape[1]:
        return _replicate_coefficients(k)
    targets = np.stack(
        [raster[di::2, dj::2].ravel()[rows] for di, dj in _PHASES], axis=1
    )
    return _solve(design[rows], targets)


def _solve(design: np.ndarray, targets: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(design, targets, rcond=None)
    return np.ascontiguousarray(coef.T).astype(np.float32)


def _predict(
    base: np.ndarray, coef: np.ndarray, k: int, design: np.ndarray | None = None
) -> np.ndarray:
    if design is None:
        design = _design(base, k)
    h = base.shape[0]
    pred = np.empty((2 * h, 2 * h), dtype=np.float64)
    # one matrix-vector product per phase, not one matrix product for the four:
    # the summation order then matches the decoder the existing streams were
    # written for, bit for bit
    for p, (di, dj) in enumerate(_PHASES):
        pred[di::2, dj::2] = (design @ coef[p].astype(np.float64)).reshape(h, h)
    return pred


def _decoded_base(base_dn: np.ndarray, q: float, o: float) -> np.ndarray:
    """The base as the decoder sees it, with no-data parents filled by the mean."""
    base = uint16_jxl.dequantize(
        base_dn, quantification_value=q, add_offset=o, dtype=np.float64
    )
    finite = np.isfinite(base)
    return np.where(finite, base, base[finite].mean() if finite.any() else 0.0)


def _check_parent_range(
    cells: np.ndarray, q: float, o: float, out_of_range: str
) -> None:
    """Apply ``out_of_range`` to the parent means, which the base clips to 16 bits."""
    if out_of_range == "clip":
        return
    finite = np.isfinite(cells)
    if not finite.any():
        return
    # a mean lies between its children: no parent can leave the range unless a cell does
    lo, hi = np.rint(np.array([np.fmin.reduce(cells), np.fmax.reduce(cells)]) * q - o)
    if lo >= uint16_jxl.NODATA_DN + 1 and hi <= uint16_jxl.MAX_DN:
        return
    # the four children of a parent are consecutive cells of a nested chunk
    valid = finite.reshape(-1, 4)
    count = valid.sum(axis=1)
    sums = np.where(valid, cells.reshape(-1, 4), 0.0).sum(axis=1)
    mean = sums[count > 0] / count[count > 0]
    scaled = np.rint(mean * q - o)
    outside = (scaled < uint16_jxl.NODATA_DN + 1) | (scaled > uint16_jxl.MAX_DN)
    uint16_jxl.report_out_of_range(
        int(np.count_nonzero(outside)),
        int(mean.size),
        quantification_value=q,
        add_offset=o,
        out_of_range=out_of_range,
        what="parent means",
    )


def _encode_layers(
    cells: np.ndarray,
    k: int,
    q: float,
    o: float,
    relative_step: float,
    min_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, bytes | None]:
    """The base, the filters, the quantised detail with its step and offset, and
    the packed validity bits of the children when some parents are partially valid."""
    raster = expand_tile(cells)
    h = raster.shape[0] // 2
    valid = np.isfinite(raster)
    vb = valid.reshape(h, 2, h, 2)
    count = vb.sum(axis=(1, 3))
    sums = np.where(valid, raster, 0.0).reshape(h, 2, h, 2).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, sums / count, np.nan)
    # the range was checked by encode, on the same means
    base_dn = uint16_jxl.quantize(
        mean, quantification_value=q, add_offset=o, out_of_range="clip"
    )
    base = _decoded_base(base_dn, q, o)

    design = _design(base, k)
    coef = _fit(base, raster, count == 4, k, design)
    pred = _predict(base, coef, k, design)

    detail = raster - pred
    detail *= q
    sigma = float(detail[valid].std()) if valid.any() else 0.0
    step = float(np.float32(max(min_step, relative_step * sigma)))
    detail /= step
    r = np.where(valid, np.rint(detail), 0).astype(np.int64)
    offset = int(r.min())
    r -= offset
    r16 = np.clip(r, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    partial = (count > 0) & (count < 4)
    mask_bits = np.packbits(valid.ravel()).tobytes() if partial.any() else None
    return base_dn, coef, r16, step, offset, mask_bits


def encode(
    cells: npt.NDArray[np.floating],
    *,
    relative_step: float = DEFAULT_RELATIVE_STEP,
    min_step: float = 1.0,
    predictor_size: int = 5,
    quantification_value: float = uint16_jxl.DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = uint16_jxl.DEFAULT_ADD_OFFSET,
    effort: int = 7,
    decoding_speed: int = DEFAULT_DECODING_SPEED,
    out_of_range: str = uint16_jxl.DEFAULT_OUT_OF_RANGE,
) -> bytes:
    """Compress one nested chunk into a lossless base and a quantised detail.

    Parameters
    ----------
    cells : numpy.ndarray of floating dtype
        A 1-D nested chunk of :math:`4^\\delta` cells, :math:`\\delta \\ge 1`.
    relative_step : float, default 0.57
        The detail step as a fraction of the detail's standard deviation, in
        digital numbers; ``0`` uses ``min_step``.
    min_step : float, default 1.0
        The smallest step, in digital numbers.
    predictor_size : int, default 5
        Odd side of the parent neighbourhood the children are predicted from.
    quantification_value : float, default 10000.0
        The scale :math:`Q`, in steps per unit of the values.
    add_offset : float, default -1000.0
        The offset :math:`O`, in digital numbers.
    effort : int, default 7
        JPEG XL encoder effort in :math:`[1, 9]`.
    decoding_speed : int, default 2
        JPEG XL decoding speed tier in :math:`[0, 4]`: higher restricts the
        encoder to tools that decode faster, for a larger codestream. The
        JPEG XL layers are lossless, so the decoded cells do not depend on it
        (nor on ``effort``). On 10 m reflectance chunks at effort 7, ``0`` (the
        only setting before) is the smallest, ``1`` keeps that size and cuts
        both the JPEG XL encoding and decoding times by about a third, ``2``
        adds about 3 % to the size and halves them.
    out_of_range : {'warn', 'raise', 'clip'}, default 'warn'
        What to do with parent means outside the range the scale and offset
        can hold (see :func:`hcc.compression.uint16_jxl.quantize`); the
        children are coded as their difference from the prediction.

    Returns
    -------
    bytes
        The chunk's codestream.

    See Also
    --------
    decode : The inverse operation.
    stream_info : The step and predictor a codestream was written with.

    Examples
    --------
    >>> import numpy as np
    >>> cells = np.linspace(0.1, 0.3, 4**4)
    >>> stream = encode(cells, relative_step=0)
    >>> float(np.abs(decode(stream, dtype="float64") - cells).max()) <= 0.5e-4
    True
    """
    cells = np.asarray(cells, dtype=np.float64)
    delta = uint16_jxl.check_chunk_shape(cells.shape)
    if delta < 1:
        raise ValueError("a chunk needs at least one level of children (4 cells)")
    if relative_step < 0 or min_step <= 0:
        raise ValueError(
            f"need relative_step >= 0 and min_step > 0, got {relative_step!r}, {min_step!r}"
        )
    if predictor_size < 1 or predictor_size % 2 == 0:
        raise ValueError(
            f"predictor_size must be a positive odd integer, got {predictor_size!r}"
        )
    if not 1 <= effort <= 9:
        raise ValueError(f"effort must be in [1, 9], got {effort!r}")
    if not 0 <= decoding_speed <= 4:
        raise ValueError(f"decoding_speed must be in [0, 4], got {decoding_speed!r}")
    uint16_jxl.check_out_of_range(out_of_range)
    q, o = float(quantification_value), float(add_offset)
    k = predictor_size
    _check_parent_range(cells, q, o, out_of_range)

    native = _native.module
    if native is None:
        base_dn, coef, r16, step, offset, mask_bits = _encode_layers(
            cells, k, q, o, relative_step, min_step
        )
    else:
        enc = native.PyramidEncoder(np.ascontiguousarray(cells), k, q, o)
        system = enc.fit_system()
        coef = _replicate_coefficients(k) if system is None else _solve(*system)
        r16, step, offset, mask_bits = enc.finish(coef, relative_step, min_step)
        base_dn = enc.base_dn()

    flags = _FLAG_MASK if mask_bits is not None else 0
    mask = zlib.compress(mask_bits, 9) if flags else b""

    jxl = uint16_jxl._jpegxl()
    options = dict(lossless=True, effort=effort)
    if decoding_speed:
        options["decodingspeed"] = decoding_speed
    coef_bytes = coef.tobytes()
    base_bytes = bytes(jxl.jpegxl_encode(np.ascontiguousarray(base_dn), **options))
    detail_bytes = bytes(jxl.jpegxl_encode(np.ascontiguousarray(r16), **options))
    header = _HEADER.pack(
        _MAGIC,
        _VERSION,
        delta,
        k,
        flags,
        float(step),
        offset,
        len(coef_bytes),
        len(base_bytes),
        len(detail_bytes),
        len(mask),
    )
    return header + coef_bytes + base_bytes + detail_bytes + mask


def stream_info(stream: bytes) -> dict:
    """The parameters a codestream was written with.

    Parameters
    ----------
    stream : bytes
        A codestream produced by :func:`encode`.

    Returns
    -------
    dict
        ``delta``, ``predictor_size``, ``step`` (digital numbers), ``offset``,
        ``has_mask`` and the byte length of each part.

    Examples
    --------
    >>> import numpy as np
    >>> stream_info(encode(np.linspace(0.1, 0.3, 4**3), relative_step=0))["step"]
    1.0
    """
    magic, version, delta, k, flags, step, offset, n_coef, n_base, n_detail, n_mask = (
        _HEADER.unpack_from(stream)
    )
    if magic != _MAGIC or version != _VERSION:
        raise ValueError(f"not a {_MAGIC.decode()} v{_VERSION} codestream")
    return dict(
        delta=delta,
        predictor_size=k,
        step=float(step),
        offset=offset,
        has_mask=bool(flags & _FLAG_MASK),
        coefficient_bytes=n_coef,
        base_bytes=n_base,
        detail_bytes=n_detail,
        mask_bytes=n_mask,
    )


def decode(
    stream: bytes,
    *,
    quantification_value: float = uint16_jxl.DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = uint16_jxl.DEFAULT_ADD_OFFSET,
    dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray[np.floating]:
    """Decompress a codestream produced by :func:`encode` to a nested chunk.

    Parameters
    ----------
    stream : bytes
        The chunk's codestream.
    quantification_value : float, default 10000.0
        The :math:`Q` used when encoding.
    add_offset : float, default -1000.0
        The :math:`O` used when encoding.
    dtype : dtype-like, default ``numpy.float32``
        Floating dtype of the result.

    Returns
    -------
    numpy.ndarray
        The 1-D nested chunk; cells that were ``NaN`` are ``NaN``.

    See Also
    --------
    encode : The forward operation.

    Examples
    --------
    >>> import numpy as np
    >>> cells = np.full(4**3, 0.25)
    >>> cells[:4] = np.nan
    >>> out = decode(encode(cells))
    >>> bool(np.isnan(out[:4]).all()), float(out[4])
    (True, 0.25)
    """
    info = stream_info(stream)
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"dtype must be a floating dtype, got {dtype}")
    q, o = float(quantification_value), float(add_offset)
    k, h = info["predictor_size"], 2 ** (info["delta"] - 1)

    pos = _HEADER.size
    coef = np.frombuffer(
        stream, dtype=np.float32, count=info["coefficient_bytes"] // 4, offset=pos
    ).reshape(4, k * k + 1)
    pos += info["coefficient_bytes"]
    jxl = uint16_jxl._jpegxl()
    base_dn = np.asarray(
        jxl.jpegxl_decode(stream[pos : pos + info["base_bytes"]])
    ).reshape(h, h)
    pos += info["base_bytes"]
    r16 = np.asarray(
        jxl.jpegxl_decode(stream[pos : pos + info["detail_bytes"]])
    ).reshape(2 * h, 2 * h)
    pos += info["detail_bytes"]
    mask_bits = (
        zlib.decompress(stream[pos : pos + info["mask_bytes"]])
        if info["has_mask"]
        else None
    )

    native = _native.module
    if native is not None and dtype in (np.float32, np.float64):
        return native.pyramid_decode(
            np.ascontiguousarray(base_dn, dtype=np.uint16),
            np.ascontiguousarray(r16, dtype=np.uint16),
            coef.copy(),
            k,
            info["offset"],
            info["step"],
            q,
            o,
            mask_bits,
            dtype == np.float32,
        )

    raster = _predict(_decoded_base(base_dn, q, o), coef, k)
    # pred + (r16 + offset) * step / q, in place, rounded as numpy does it
    residual = r16.astype(np.float64)
    residual += info["offset"]
    residual *= info["step"]
    residual /= q
    raster += residual

    valid = np.broadcast_to(
        (base_dn != uint16_jxl.NODATA_DN)[:, None, :, None], (h, 2, h, 2)
    ).reshape(2 * h, 2 * h)
    if mask_bits is not None:
        bits = np.frombuffer(mask_bits, dtype=np.uint8)
        valid = valid & np.unpackbits(bits, count=valid.size).reshape(valid.shape).view(
            bool
        )
    raster[~valid] = np.nan
    return flatten_tile(raster).astype(dtype, copy=False)
