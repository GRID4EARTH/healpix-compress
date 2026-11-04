"""Coding of chunks stored as 16-bit integers: a scale, an offset and
lossless JPEG XL.

Many products store a variable as unsigned or signed 16-bit integers, the
digital numbers (DN), with a scale and an offset per variable:

.. math::

    x = \\frac{DN + O}{Q}

with ``DN = 0`` reserved for no-data. :math:`Q` is the number of steps per
unit of :math:`x` and :math:`O` shifts the values into the range
:math:`[1, 65535]`; a signed variable uses a negative offset, e.g.
:math:`O = -32768`. The integers are compressed losslessly, so decoding
restores the quantised values exactly, and no error is added beyond the
initial rounding: data already on such a lattice round-trip unchanged.

This module applies that chain to a HEALPix chunk: the cells of one nested
chunk are quantised to DN, scattered into a 2-D raster by Morton (Z-order)
de-interleaving (:func:`hcc.compression.zorder.expand_tile`), and compressed
with lossless JPEG XL. Quantisation is the only lossy step: a cell value that
does not sit on the DN lattice is rounded to the nearest step :math:`1 / Q`, a
uniform error of RMS :math:`1 / (Q \\sqrt{12})`. Finite values outside the
range the scale and offset can hold, :math:`[(1 + O) / Q, (65535 + O) / Q]`,
are clipped to it; ``out_of_range`` says whether that warns, raises or is
silent.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from hcc.compression.zorder import expand_tile, flatten_tile

#: Digital number reserved for no-data (NaN cells).
NODATA_DN = 0

#: Largest digital number representable in the 16-bit raster.
MAX_DN = np.iinfo(np.uint16).max

#: Default :math:`Q`: steps of ``1e-4``, for values such as reflectances.
DEFAULT_QUANTIFICATION_VALUE = 10000.0

#: Default :math:`O`: values down to about ``-0.1`` with the default :math:`Q`.
DEFAULT_ADD_OFFSET = -1000.0

#: What happens to finite values outside the range the scale and offset can
#: hold: ``"warn"`` clips them with an :class:`OutOfRangeWarning`, ``"raise"``
#: raises a :class:`ValueError`, ``"clip"`` clips them silently.
OUT_OF_RANGE_POLICIES = ("warn", "raise", "clip")

#: Default ``out_of_range`` policy.
DEFAULT_OUT_OF_RANGE = "warn"


class OutOfRangeWarning(UserWarning):
    """Finite values were clipped to the range the scale and offset can hold."""


def value_range(
    *,
    quantification_value: float = DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = DEFAULT_ADD_OFFSET,
) -> tuple[float, float]:
    """The smallest and largest values a scale and an offset can hold.

    Parameters
    ----------
    quantification_value : float, default 10000.0
        The scale :math:`Q`.
    add_offset : float, default -1000.0
        The offset :math:`O`.

    Returns
    -------
    tuple of float
        :math:`((1 + O) / Q, (65535 + O) / Q)`.

    Examples
    --------
    >>> value_range(quantification_value=100.0, add_offset=-32768.0)
    (-327.67, 327.67)
    """
    q, o = float(quantification_value), float(add_offset)
    return (NODATA_DN + 1 + o) / q, (MAX_DN + o) / q


def check_out_of_range(policy: str) -> None:
    """Reject an unknown ``out_of_range`` policy.

    Parameters
    ----------
    policy : str
        One of :data:`OUT_OF_RANGE_POLICIES`.
    """
    if policy not in OUT_OF_RANGE_POLICIES:
        raise ValueError(
            f"out_of_range must be one of {OUT_OF_RANGE_POLICIES}, got {policy!r}"
        )


def report_out_of_range(
    n_outside: int,
    n_valid: int,
    *,
    quantification_value: float,
    add_offset: float,
    out_of_range: str,
    what: str = "values",
) -> None:
    """Warn or raise about clipped values, following ``out_of_range``.

    Parameters
    ----------
    n_outside : int
        Number of finite values outside the range.
    n_valid : int
        Number of finite values.
    quantification_value, add_offset : float
        The scale and the offset.
    out_of_range : {'warn', 'raise', 'clip'}
        The policy.
    what : str, default "values"
        What was clipped, for the message.
    """
    if n_outside == 0 or out_of_range == "clip":
        return
    lo, hi = value_range(
        quantification_value=quantification_value, add_offset=add_offset
    )
    message = (
        f"{n_outside} of {n_valid} finite {what} are outside [{lo:g}, {hi:g}], the range "
        f"quantification_value={quantification_value:g} and add_offset={add_offset:g} "
        "can hold in 16 bits"
    )
    if out_of_range == "raise":
        raise ValueError(message)
    warnings.warn(f"{message}; they are clipped to it", OutOfRangeWarning, stacklevel=3)


def _jpegxl():
    try:
        import imagecodecs
    except ImportError as e:  # pragma: no cover - exercised only without the dependency
        raise ImportError(
            "hcc.compression.uint16_jxl needs the optional dependency 'imagecodecs' "
            "(pip install imagecodecs / conda install -c conda-forge imagecodecs)"
        ) from e

    return imagecodecs


def check_chunk_shape(shape: tuple[int, ...]) -> int:
    """Check that a chunk is one nested parent cell and return its depth.

    Parameters
    ----------
    shape : tuple of int
        Chunk shape; must be 1-D with a power-of-four size.

    Returns
    -------
    int
        :math:`\\delta`, such that the chunk holds :math:`4^\\delta` cells.

    Examples
    --------
    >>> check_chunk_shape((1024 * 1024,))
    10
    """
    if len(shape) != 1:
        raise ValueError(f"expected a 1-D nested chunk, got a chunk of shape {shape}")
    (size,) = shape
    delta = round(math.log(size, 4)) if size > 0 else -1
    if size == 0 or 4**delta != size:
        raise ValueError(
            f"the chunk size must be a power of four (one nested parent cell), got {size}"
        )
    return delta


def check_rows_shape(shape: tuple[int, ...]) -> int:
    """Check that a chunk is a stack of nested parent cells along its last axis.

    The leading axes (time, band, ...) are free; the last one must hold the
    :math:`4^\\delta` cells of one nested parent cell.

    Parameters
    ----------
    shape : tuple of int
        Chunk shape, ``(*lead, 4**delta)``.

    Returns
    -------
    int
        :math:`\\delta`, such that each row holds :math:`4^\\delta` cells.

    Examples
    --------
    >>> check_rows_shape((1, 3, 4**8))
    8
    """
    if len(shape) == 0:
        raise ValueError("expected a chunk with at least one axis, got a scalar chunk")
    return check_chunk_shape((shape[-1],))


def encode_rows(
    cells: npt.NDArray[np.floating], encode_row: Callable[[np.ndarray], bytes]
) -> bytes:
    """Encode a chunk row by row along its last axis.

    A 1-D chunk is the codestream of its single row, unchanged. A chunk with
    leading axes is the little-endian ``uint64`` byte length of each row's
    codestream, rows in C order, followed by the codestreams.

    Parameters
    ----------
    cells : numpy.ndarray
        Chunk of shape ``(*lead, 4**delta)``.
    encode_row : callable
        Encodes one 1-D row of ``4**delta`` cells.

    Returns
    -------
    bytes

    See Also
    --------
    decode_rows : The inverse operation.
    """
    cells = np.asarray(cells)
    if cells.ndim == 1:
        return encode_row(cells)
    streams = [encode_row(row) for row in cells.reshape(-1, cells.shape[-1])]
    lengths = np.array([len(s) for s in streams], dtype="<u8")
    return lengths.tobytes() + b"".join(streams)


def decode_rows(
    data: bytes,
    shape: tuple[int, ...],
    decode_row: Callable[[bytes], np.ndarray],
) -> np.ndarray:
    """Decode a chunk written by :func:`encode_rows`.

    Parameters
    ----------
    data : bytes
        The encoded chunk.
    shape : tuple of int
        Chunk shape, ``(*lead, 4**delta)``.
    decode_row : callable
        Decodes one row's codestream to a 1-D array.

    Returns
    -------
    numpy.ndarray
        The chunk, of shape *shape*.
    """
    if len(shape) == 1:
        return decode_row(data)
    n_rows = math.prod(shape[:-1])
    header = 8 * n_rows
    if len(data) < header:
        raise ValueError(f"truncated chunk: {len(data)} bytes for {n_rows} rows")
    lengths = np.frombuffer(data[:header], dtype="<u8").astype(np.int64)
    if header + int(lengths.sum()) != len(data):
        raise ValueError(
            f"corrupt chunk: {int(lengths.sum())} bytes of rows after a {header}-byte "
            f"header, got {len(data)} bytes"
        )
    ends = header + np.cumsum(lengths)
    rows = [decode_row(data[end - size : end]) for end, size in zip(ends, lengths)]
    return np.stack(rows).reshape(shape)


def quantize(
    values: npt.NDArray[np.floating],
    *,
    quantification_value: float = DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = DEFAULT_ADD_OFFSET,
    out_of_range: str = DEFAULT_OUT_OF_RANGE,
) -> npt.NDArray[np.uint16]:
    """Convert values to 16-bit digital numbers.

    Applies :math:`DN = \\mathrm{round}(Q x - O)`, clipped to
    :math:`[1, 65535]`; non-finite values become the no-data number ``0``.

    Parameters
    ----------
    values : numpy.ndarray of floating dtype
        The values, any shape.
    quantification_value : float, default 10000.0
        The scale :math:`Q`, in steps per unit of the values.
    add_offset : float, default -1000.0
        The offset :math:`O`, in digital numbers.

    out_of_range : {'warn', 'raise', 'clip'}, default 'warn'
        What to do with finite values outside :func:`value_range`: clip them
        with an :class:`OutOfRangeWarning`, raise a :class:`ValueError`, or
        clip them silently.

    Returns
    -------
    numpy.ndarray of dtype uint16
        Digital numbers, same shape as *values*.

    See Also
    --------
    dequantize : The inverse mapping.

    Examples
    --------
    >>> import numpy as np
    >>> quantize(np.array([0.0, 0.12345, np.nan, 2.0]))
    array([ 1000,  2234,     0, 21000], dtype=uint16)

    A signed variable with steps of ``0.01``:

    >>> quantize(np.array([-12.5, 0.0, 40.25]), quantification_value=100, add_offset=-32768)
    array([31518, 32768, 36793], dtype=uint16)
    """
    values = np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"expected a floating dtype, got {values.dtype}")
    if quantification_value <= 0:
        raise ValueError(
            f"quantification_value must be positive, got {quantification_value!r}"
        )
    check_out_of_range(out_of_range)

    valid = np.isfinite(values)
    scaled = np.rint(np.where(valid, values, 0.0) * quantification_value - add_offset)
    if out_of_range != "clip":
        outside = valid & ((scaled < NODATA_DN + 1) | (scaled > MAX_DN))
        report_out_of_range(
            int(np.count_nonzero(outside)),
            int(np.count_nonzero(valid)),
            quantification_value=quantification_value,
            add_offset=add_offset,
            out_of_range=out_of_range,
        )
    dn = np.clip(scaled, NODATA_DN + 1, MAX_DN).astype(np.uint16)
    dn[~valid] = NODATA_DN
    return dn


def dequantize(
    dn: npt.NDArray[np.integer],
    *,
    quantification_value: float = DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = DEFAULT_ADD_OFFSET,
    dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray[np.floating]:
    """Convert 16-bit digital numbers back to values.

    Applies :math:`x = (DN + O) / Q`; the no-data number ``0`` becomes
    ``NaN``.

    Parameters
    ----------
    dn : numpy.ndarray of integer dtype
        Digital numbers, any shape.
    quantification_value : float, default 10000.0
        The scale :math:`Q`, in steps per unit of the values.
    add_offset : float, default -1000.0
        The offset :math:`O`, in digital numbers.
    dtype : dtype-like, default ``numpy.float32``
        Floating dtype of the result.

    Returns
    -------
    numpy.ndarray
        The values, same shape as *dn*.

    See Also
    --------
    quantize : The forward mapping.

    Examples
    --------
    >>> import numpy as np
    >>> dequantize(np.array([1000, 2234, 0], dtype="uint16"))
    array([0.    , 0.1234,    nan], dtype=float32)
    """
    dn = np.asarray(dn)
    if not np.issubdtype(dn.dtype, np.integer):
        raise TypeError(f"expected an integer dtype, got {dn.dtype}")
    dtype = np.dtype(dtype)
    if not np.issubdtype(dtype, np.floating):
        raise TypeError(f"dtype must be a floating dtype, got {dtype}")

    values = ((dn.astype(np.float64) + add_offset) / quantification_value).astype(dtype)
    values[dn == NODATA_DN] = np.nan
    return values


def encode(
    cells: npt.NDArray[np.floating],
    *,
    quantification_value: float = DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = DEFAULT_ADD_OFFSET,
    effort: int = 7,
    out_of_range: str = DEFAULT_OUT_OF_RANGE,
) -> bytes:
    """Compress one nested HEALPix chunk to a JPEG XL codestream.

    The chunk is quantised with :func:`quantize`, de-interleaved into a
    :math:`2^\\delta \\times 2^\\delta` raster and encoded losslessly.

    Parameters
    ----------
    cells : numpy.ndarray of floating dtype
        A 1-D nested chunk of :math:`4^\\delta` cells (all descendants, at
        :math:`\\delta` levels below, of one parent cell).
    quantification_value : float, default 10000.0
        The scale :math:`Q`, in steps per unit of the values.
    add_offset : float, default -1000.0
        The offset :math:`O`, in digital numbers.
    effort : int, default 7
        JPEG XL encoder effort in :math:`[1, 9]`; higher is slower and
        slightly smaller.
    out_of_range : {'warn', 'raise', 'clip'}, default 'warn'
        What to do with finite values outside :func:`value_range`: clip them
        with an :class:`OutOfRangeWarning`, raise a :class:`ValueError`, or
        clip them silently.

    Returns
    -------
    bytes
        The JPEG XL codestream.

    See Also
    --------
    decode : The inverse operation.

    Examples
    --------
    >>> import numpy as np
    >>> cells = np.linspace(0, 1, 4**4, dtype="float32")
    >>> stream = encode(cells)
    >>> np.array_equal(decode(stream), dequantize(quantize(cells)), equal_nan=True)
    True
    """
    cells = np.asarray(cells)
    check_chunk_shape(cells.shape)
    if not 1 <= effort <= 9:
        raise ValueError(f"effort must be in [1, 9], got {effort!r}")

    dn = quantize(
        cells,
        quantification_value=quantification_value,
        add_offset=add_offset,
        out_of_range=out_of_range,
    )
    raster = np.ascontiguousarray(expand_tile(dn))
    return bytes(_jpegxl().jpegxl_encode(raster, lossless=True, effort=effort))


def decode(
    stream: bytes,
    *,
    quantification_value: float = DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = DEFAULT_ADD_OFFSET,
    dtype: npt.DTypeLike = np.float32,
) -> npt.NDArray[np.floating]:
    """Decompress a JPEG XL codestream produced by :func:`encode` to a nested chunk.

    Parameters
    ----------
    stream : bytes
        The JPEG XL codestream.
    quantification_value : float, default 10000.0
        The :math:`Q` used when encoding.
    add_offset : float, default -1000.0
        The :math:`O` used when encoding.
    dtype : dtype-like, default ``numpy.float32``
        Floating dtype of the result.

    Returns
    -------
    numpy.ndarray
        The 1-D nested chunk of quantised values; no-data cells are ``NaN``.

    See Also
    --------
    encode : The forward operation.

    Examples
    --------
    >>> import numpy as np
    >>> cells = np.array([0.1, 0.2, np.nan, 0.4], dtype="float32")
    >>> decode(encode(cells))
    array([0.1, 0.2, nan, 0.4], dtype=float32)
    """
    raster = np.asarray(_jpegxl().jpegxl_decode(stream))
    if (
        raster.ndim != 2
        or raster.shape[0] != raster.shape[1]
        or raster.dtype != np.uint16
    ):
        raise ValueError(
            f"expected a square uint16 raster, got shape {raster.shape} and dtype {raster.dtype}"
        )
    dn = flatten_tile(raster)
    return dequantize(
        dn,
        quantification_value=quantification_value,
        add_offset=add_offset,
        dtype=dtype,
    )
