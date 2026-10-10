"""Two-layer Sentinel-2 coding of a nested chunk: a lossless parent level and
a predicted, quantised child level.

A chunk of :math:`4^\\delta` reflectances at level :math:`L` is stored as

1. the **base**: the mean of the four children of every level-:math:`L-1`
   parent, rounded to the Sentinel-2 digital-number lattice
   (:func:`hcc.compression.sentinel2.quantize`) and coded with lossless
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
asks for: for Sentinel-2 B04 resampled with the PSF settings of
legacy-converters, the error the transfer itself leaves on the native pixels
is close to :math:`0.058 \\sigma`, and the quantisation adds about
:math:`0.16 s` there, so :math:`r \\approx 0.36` puts the added error at the
transfer error. ``relative_step = 0`` uses :math:`s = s_{\\min}`; with the
default :math:`s_{\\min} = 1` the children then sit on the DN lattice, as in
:mod:`hcc.compression.sentinel2`.

Cells that are ``NaN`` round-trip as ``NaN``: a parent without any valid
child is the base's no-data number, and the validity of the children of the
partially valid parents is stored as a bit mask.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import numpy.typing as npt

from hcc.compression import sentinel2
from hcc.compression.zorder import expand_tile, flatten_tile

_MAGIC = b"HS2P"
_VERSION = 1
# magic, version, delta, k, flags, step, offset, then the byte length of the
# coefficients, the base codestream, the detail codestream and the mask
_HEADER = struct.Struct("<4sBBBBfiIIII")
_FLAG_MASK = 1

#: ``relative_step`` at which the added error matches the transfer error of
#: the legacy-converters Sentinel-2 PSF preset, measured on B04 of tile T31UDQ.
TRANSFER_FLOOR_RELATIVE_STEP = 0.36


def _neighbourhood(base: np.ndarray, k: int) -> np.ndarray:
    """``(h * h, k * k)`` parent neighbourhoods, edge-padded."""
    r = k // 2
    p = np.pad(base, r, mode="edge")
    h = base.shape[0]
    feats = np.stack(
        [p[i : i + h, j : j + h] for i in range(k) for j in range(k)], axis=-1
    )
    return feats.reshape(h * h, k * k)


def _design(base: np.ndarray, k: int) -> np.ndarray:
    feats = _neighbourhood(base, k)
    return np.concatenate([feats, np.ones((feats.shape[0], 1))], axis=1)


def _replicate_coefficients(k: int) -> np.ndarray:
    coef = np.zeros((4, k * k + 1), dtype=np.float32)
    coef[:, (k // 2) * k + k // 2] = 1.0
    return coef


def _fit(base: np.ndarray, raster: np.ndarray, full: np.ndarray, k: int) -> np.ndarray:
    """One least-squares filter per child phase, fitted on the fully valid parents."""
    design = _design(base, k)
    rows = full.ravel()
    if rows.sum() < design.shape[1]:
        return _replicate_coefficients(k)
    coef = np.empty((4, design.shape[1]), dtype=np.float64)
    for p, (di, dj) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        target = raster[di::2, dj::2].ravel()[rows]
        coef[p], *_ = np.linalg.lstsq(design[rows], target, rcond=None)
    return coef.astype(np.float32)


def _predict(base: np.ndarray, coef: np.ndarray, k: int) -> np.ndarray:
    design = _design(base, k)
    h = base.shape[0]
    pred = np.empty((2 * h, 2 * h), dtype=np.float64)
    for p, (di, dj) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
        pred[di::2, dj::2] = (design @ coef[p].astype(np.float64)).reshape(h, h)
    return pred


def _decoded_base(base_dn: np.ndarray, q: float, o: float) -> np.ndarray:
    """The base as the decoder sees it, with no-data parents filled by the mean."""
    base = sentinel2.dequantize(
        base_dn, quantification_value=q, add_offset=o, dtype=np.float64
    )
    finite = np.isfinite(base)
    return np.where(finite, base, base[finite].mean() if finite.any() else 0.0)


def encode(
    cells: npt.NDArray[np.floating],
    *,
    relative_step: float = TRANSFER_FLOOR_RELATIVE_STEP,
    min_step: float = 1.0,
    predictor_size: int = 5,
    quantification_value: float = sentinel2.DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = sentinel2.DEFAULT_ADD_OFFSET,
    effort: int = 7,
) -> bytes:
    """Compress one nested chunk of reflectances into a base and a quantised detail.

    Parameters
    ----------
    cells : numpy.ndarray of floating dtype
        A 1-D nested chunk of :math:`4^\\delta` cells, :math:`\\delta \\ge 1`.
    relative_step : float, default 0.36
        The detail step as a fraction of the detail's standard deviation, in
        digital numbers; ``0`` uses ``min_step``.
    min_step : float, default 1.0
        The smallest step, in digital numbers.
    predictor_size : int, default 5
        Odd side of the parent neighbourhood the children are predicted from.
    quantification_value : float, default 10000.0
        The :math:`Q` of the product metadata (``QUANTIFICATION_VALUE``).
    add_offset : float, default -1000.0
        The :math:`O` of the product metadata (``RADIO_ADD_OFFSET`` or
        ``BOA_ADD_OFFSET``).
    effort : int, default 7
        JPEG XL encoder effort in :math:`[1, 9]`.

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
    delta = sentinel2.check_chunk_shape(cells.shape)
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
    q, o = float(quantification_value), float(add_offset)

    raster = expand_tile(cells)
    side, h = raster.shape[0], raster.shape[0] // 2
    valid = np.isfinite(raster)
    vb = valid.reshape(h, 2, h, 2)
    count = vb.sum(axis=(1, 3))
    sums = np.where(valid, raster, 0.0).reshape(h, 2, h, 2).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(count > 0, sums / count, np.nan)
    base_dn = sentinel2.quantize(mean, quantification_value=q, add_offset=o)
    base = _decoded_base(base_dn, q, o)

    k = predictor_size
    coef = _fit(base, raster, count == 4, k)
    pred = _predict(base, coef, k)

    detail = (raster - pred) * q
    sigma = float(detail[valid].std()) if valid.any() else 0.0
    step = np.float32(max(min_step, relative_step * sigma))
    r = np.where(valid, np.rint(detail / float(step)), 0).astype(np.int64)
    offset = int(r.min())
    r16 = np.clip(r - offset, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    partial = (count > 0) & (count < 4)
    flags = _FLAG_MASK if partial.any() else 0
    mask = zlib.compress(np.packbits(valid.ravel()).tobytes(), 9) if flags else b""

    jxl = sentinel2._jpegxl()
    coef_bytes = coef.tobytes()
    base_bytes = bytes(
        jxl.jpegxl_encode(np.ascontiguousarray(base_dn), lossless=True, effort=effort)
    )
    detail_bytes = bytes(
        jxl.jpegxl_encode(np.ascontiguousarray(r16), lossless=True, effort=effort)
    )
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
    assert side == 2**delta
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
    quantification_value: float = sentinel2.DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = sentinel2.DEFAULT_ADD_OFFSET,
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
    jxl = sentinel2._jpegxl()
    base_dn = np.asarray(
        jxl.jpegxl_decode(stream[pos : pos + info["base_bytes"]])
    ).reshape(h, h)
    pos += info["base_bytes"]
    r16 = np.asarray(
        jxl.jpegxl_decode(stream[pos : pos + info["detail_bytes"]])
    ).reshape(2 * h, 2 * h)
    pos += info["detail_bytes"]

    pred = _predict(_decoded_base(base_dn, q, o), coef, k)
    raster = pred + (r16.astype(np.int64) + info["offset"]) * info["step"] / q

    valid = np.repeat(np.repeat(base_dn != sentinel2.NODATA_DN, 2, axis=0), 2, axis=1)
    if info["has_mask"]:
        bits = np.frombuffer(
            zlib.decompress(stream[pos : pos + info["mask_bytes"]]), dtype=np.uint8
        )
        valid &= np.unpackbits(bits)[: valid.size].reshape(valid.shape).astype(bool)
    raster[~valid] = np.nan
    return flatten_tile(raster).astype(dtype)
