"""Sentinel-2 style coding of reflectance chunks: integer digital numbers
plus lossless JPEG XL.

Sentinel-2 L1C and L2A products store a reflectance band as 16-bit digital
numbers (DN) obtained from the reflectance :math:`\\rho` through the affine
relation

.. math::

    \\rho = \\frac{DN + O}{Q}

where :math:`Q` is ``QUANTIFICATION_VALUE`` and :math:`O` is
``RADIO_ADD_OFFSET`` (L1C) or ``BOA_ADD_OFFSET`` (L2A) in the product
metadata, with ``DN = 0`` reserved for no-data. The integers are then
compressed losslessly, so decoding restores the quantised reflectances
exactly, and no error is added beyond the initial rounding.

This module applies the same chain to a HEALPix chunk: the cells of one
nested chunk are quantised to DN, scattered into a 2-D raster by Morton
(Z-order) de-interleaving (:func:`hcc.compression.zorder.expand_tile`), and
compressed with lossless JPEG XL. Quantisation is the only lossy step: a
cell value that does not sit on the DN lattice is rounded to the nearest
step :math:`1 / Q`, a uniform error of RMS :math:`1 / (Q \\sqrt{12})`.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from hcc.compression.zorder import expand_tile, flatten_tile

#: Digital number reserved for no-data (NaN cells), as in Sentinel-2 products.
NODATA_DN = 0

#: Largest digital number representable in the 16-bit raster.
MAX_DN = np.iinfo(np.uint16).max

#: ``QUANTIFICATION_VALUE`` of Sentinel-2 L1C and L2A products.
DEFAULT_QUANTIFICATION_VALUE = 10000.0

#: ``RADIO_ADD_OFFSET`` / ``BOA_ADD_OFFSET`` of Sentinel-2 products since
#: processing baseline 04.00.
DEFAULT_ADD_OFFSET = -1000.0


def _jpegxl():
    try:
        import imagecodecs
    except ImportError as e:  # pragma: no cover - exercised only without the dependency
        raise ImportError(
            "hcc.compression.sentinel2 needs the optional dependency 'imagecodecs' "
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


def quantize(
    reflectance: npt.NDArray[np.floating],
    *,
    quantification_value: float = DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = DEFAULT_ADD_OFFSET,
) -> npt.NDArray[np.uint16]:
    """Convert reflectances to Sentinel-2 digital numbers.

    Applies :math:`DN = \\mathrm{round}(Q \\rho - O)`, clipped to
    :math:`[1, 65535]`; non-finite values become the no-data number ``0``.

    Parameters
    ----------
    reflectance : numpy.ndarray of floating dtype
        Reflectance values, any shape.
    quantification_value : float, default 10000.0
        The :math:`Q` of the product metadata (``QUANTIFICATION_VALUE``).
    add_offset : float, default -1000.0
        The :math:`O` of the product metadata (``RADIO_ADD_OFFSET`` or
        ``BOA_ADD_OFFSET``).

    Returns
    -------
    numpy.ndarray of dtype uint16
        Digital numbers, same shape as *reflectance*.

    See Also
    --------
    dequantize : The inverse mapping.

    Examples
    --------
    >>> import numpy as np
    >>> quantize(np.array([0.0, 0.12345, np.nan, 2.0]))
    array([ 1000,  2234,     0, 21000], dtype=uint16)
    """
    reflectance = np.asarray(reflectance)
    if not np.issubdtype(reflectance.dtype, np.floating):
        raise TypeError(f"expected a floating dtype, got {reflectance.dtype}")
    if quantification_value <= 0:
        raise ValueError(
            f"quantification_value must be positive, got {quantification_value!r}"
        )

    valid = np.isfinite(reflectance)
    scaled = np.rint(
        np.where(valid, reflectance, 0.0) * quantification_value - add_offset
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
    """Convert Sentinel-2 digital numbers back to reflectances.

    Applies :math:`\\rho = (DN + O) / Q`; the no-data number ``0`` becomes
    ``NaN``.

    Parameters
    ----------
    dn : numpy.ndarray of integer dtype
        Digital numbers, any shape.
    quantification_value : float, default 10000.0
        The :math:`Q` of the product metadata (``QUANTIFICATION_VALUE``).
    add_offset : float, default -1000.0
        The :math:`O` of the product metadata (``RADIO_ADD_OFFSET`` or
        ``BOA_ADD_OFFSET``).
    dtype : dtype-like, default ``numpy.float32``
        Floating dtype of the result.

    Returns
    -------
    numpy.ndarray
        Reflectances, same shape as *dn*.

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

    reflectance = ((dn.astype(np.float64) + add_offset) / quantification_value).astype(
        dtype
    )
    reflectance[dn == NODATA_DN] = np.nan
    return reflectance


def encode(
    cells: npt.NDArray[np.floating],
    *,
    quantification_value: float = DEFAULT_QUANTIFICATION_VALUE,
    add_offset: float = DEFAULT_ADD_OFFSET,
    effort: int = 7,
) -> bytes:
    """Compress one nested HEALPix chunk of reflectances to a JPEG XL codestream.

    The chunk is quantised with :func:`quantize`, de-interleaved into a
    :math:`2^\\delta \\times 2^\\delta` raster and encoded losslessly.

    Parameters
    ----------
    cells : numpy.ndarray of floating dtype
        A 1-D nested chunk of :math:`4^\\delta` cells (all descendants, at
        :math:`\\delta` levels below, of one parent cell).
    quantification_value : float, default 10000.0
        The :math:`Q` of the product metadata (``QUANTIFICATION_VALUE``).
    add_offset : float, default -1000.0
        The :math:`O` of the product metadata (``RADIO_ADD_OFFSET`` or
        ``BOA_ADD_OFFSET``).
    effort : int, default 7
        JPEG XL encoder effort in :math:`[1, 9]`; higher is slower and
        slightly smaller.

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
        cells, quantification_value=quantification_value, add_offset=add_offset
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
        The 1-D nested chunk of quantised reflectances; no-data cells are ``NaN``.

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
