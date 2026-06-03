"""
Block-DCT lossy compression for HEALPix tile residuals.

This module implements a hierarchical, adaptive compressor for square float32
arrays stored in Z-order (Morton) encoding.  The pipeline follows three stages:

1. **Low-frequency layer** — the input is spatially downsampled and stored
   losslessly as float32.  The downsampling factor is controlled by
   ``downsample_factor``.
2. **DCT residual coding** — the difference between the original data and a
   bilinear prediction (obtained by upsampling the low-frequency layer) is
   divided into non-overlapping ``window × window`` blocks.  Each block is
   transformed with a 2-D DCT-II and the resulting coefficients are uniformly
   quantised to int8 or int16 with a single global scale factor ``q_scale``.
3. **Entropy compression** — the quantised coefficient array is compressed
   with zlib.

This approach mirrors JPEG-style block coding applied to the prediction
residual: the DCT concentrates residual energy into a small number of
low-frequency coefficients that quantise efficiently, while high-frequency
coefficients in smooth residual regions remain near zero and are cheap to
entropy-code.

Public API
----------
compress : Compress a Z-order HEALPix chunk into a tuple of arrays.
decompress : Reconstruct a chunk from its compressed representation.
lookup_preset : Return the parameter dict for a named quality preset.
compression_info : Report size and compression-ratio statistics.

Lower-level helpers
-------------------
dct_blocks : Apply 2-D DCT-II to non-overlapping blocks of a 2-D array.
idct_blocks : Apply 2-D IDCT-II to non-overlapping blocks and reassemble.
compress_residuals : Quantise a residual array via block DCT.
decompress_residuals : Reconstruct a residual array from quantised DCT coefficients.
downsample : Spatially reduce a square float32 array.
upsample : Spatially magnify a square float32 array.

Design notes
------------
Algorithm parameters (``window``, ``quantize_bits``, ``downsample_factor``)
are **not** embedded in the compressed bytes.  They must be passed identically
to both :func:`compress` and :func:`decompress`, and should be stored in the
array's metadata document (e.g. ``zarr.json``) by the caller.

Only two data-dependent values are stored per chunk:

* ``n`` — the tile side length (uint32).
* ``q_scale`` — the uniform quantisation step size (float32).

These are written into the 8-byte binary header for the standalone
:func:`compress` / :func:`decompress` API, and stored as sidecar scalars
in the zarr codec path via :func:`compress_residuals` /
:func:`decompress_residuals`.

Use :func:`lookup_preset` to resolve a named preset into an explicit parameter
dict *before* writing; store that dict in the metadata, not the preset name.
This guarantees that changing or removing a preset in a future version of the
library cannot make existing data unreadable.
"""

import struct
import zlib
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.fft import dctn, idctn
from scipy.ndimage import zoom as sp_zoom

from hcc.compression.zorder import expand_tile, flatten_tile

__all__ = [
    "compress",
    "decompress",
    "lookup_preset",
    "compression_info",
    "dct_blocks",
    "idct_blocks",
    "compress_residuals",
    "decompress_residuals",
    "downsample",
    "upsample",
]

# ─────────────────────────────────────────────────────────────────
#  Binary format constants  (standalone compress / decompress API)
# ─────────────────────────────────────────────────────────────────

# Little-endian header: uint32 n  +  float32 q_scale
_HEADER_FMT = "<If"
_HEADER_BYTES = struct.calcsize(_HEADER_FMT)  # 8 bytes


# ─────────────────────────────────────────────────────────────────
#  Quality presets  (private — always access via lookup_preset)
# ─────────────────────────────────────────────────────────────────

_PRESETS: dict[str, dict[str, Any]] = {
    "low": dict(
        window=4,
        quantize_bits=8,
        clip_percentile=3.0,
        zlib_level=9,
        downsample_factor=4,
    ),
    "medium": dict(
        window=8,
        quantize_bits=8,
        clip_percentile=0.5,
        zlib_level=9,
        downsample_factor=2,
    ),
    "high": dict(
        window=8,
        quantize_bits=16,
        clip_percentile=0.1,
        zlib_level=6,
        downsample_factor=2,
    ),
    "ultra": dict(
        window=16,
        quantize_bits=16,
        clip_percentile=0.02,
        zlib_level=1,
        downsample_factor=1,
    ),
}


# ─────────────────────────────────────────────────────────────────
#  Signal processing primitives
# ─────────────────────────────────────────────────────────────────


def downsample(data: npt.NDArray[np.float32], factor: int) -> npt.NDArray[np.float32]:
    """Downsample a square array by an integer factor.

    Parameters
    ----------
    data : numpy.ndarray of shape (n, n), dtype float32
        Input array.  ``n`` must be divisible by *factor*.
    factor : {1, 2, 4}
        Spatial reduction factor.

        * ``1`` — returns a float32 copy without modification.
        * ``2`` — exact 2×2 block average.
        * ``4`` — bilinear downsampling via :func:`scipy.ndimage.zoom`.

    Returns
    -------
    numpy.ndarray of shape (n//factor, n//factor), dtype float32

    See Also
    --------
    upsample : Inverse operation.
    """
    if factor == 1:
        return data.astype(np.float32).copy()
    if factor == 2:
        n = data.shape[0]
        return (
            data.reshape(n // 2, 2, n // 2, 2)
            .mean(axis=(1, 3))
            .astype(np.float32)
        )
    return sp_zoom(data.astype(np.float64), 1.0 / factor, order=1).astype(np.float32)


def upsample(data: npt.NDArray[np.float32], factor: int) -> npt.NDArray[np.float32]:
    """Upsample a square array by an integer factor using bilinear interpolation.

    Parameters
    ----------
    data : numpy.ndarray of shape (m, m), dtype float32
        Input array.
    factor : {1, 2, 4}
        Spatial magnification factor.

        * ``1`` — returns a float32 copy without modification.
        * ``2`` or ``4`` — bilinear interpolation via :func:`scipy.ndimage.zoom`.

    Returns
    -------
    numpy.ndarray of shape (m*factor, m*factor), dtype float32

    See Also
    --------
    downsample : Inverse operation.
    """
    if factor == 1:
        return data.astype(np.float32).copy()
    return sp_zoom(data.astype(np.float64), factor, order=1).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
#  DCT block primitives
# ─────────────────────────────────────────────────────────────────


def dct_blocks(data: npt.NDArray[np.float32], block_size: int) -> npt.NDArray[np.float64]:
    """Apply a 2-D DCT-II to each non-overlapping square block of *data*.

    The array is partitioned into ``(n // block_size) ** 2`` non-overlapping
    blocks of shape ``(block_size, block_size)``.  A 2-D DCT-II with
    orthonormal (``norm='ortho'``) normalisation is applied independently to
    each block.  The coefficients are returned in the same spatial layout as
    the input: coefficient ``[r, c]`` within the block at grid position
    ``(i, j)`` is stored at output index
    ``[i * block_size + r, j * block_size + c]``.

    Parameters
    ----------
    data : numpy.ndarray of shape (n, n)
        Input array.  *n* must be a positive integer multiple of *block_size*.
    block_size : int
        Side length of each square DCT block.  Typical values: 4, 8, 16.

    Returns
    -------
    numpy.ndarray of shape (n, n), dtype float64
        DCT-II coefficients in block-spatial layout.

    Raises
    ------
    ValueError
        If *n* is not divisible by *block_size*.

    See Also
    --------
    idct_blocks : Exact inverse of this function.

    Notes
    -----
    The (0, 0) coefficient of each block is the block mean scaled by
    ``block_size`` (because of the orthonormal normalisation).  Smooth
    residual blocks therefore concentrate their energy in the DC coefficient
    and a few low-frequency AC terms.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.ones((8, 8), dtype=np.float32)
    >>> c = dct_blocks(x, block_size=4)
    >>> c.shape
    (8, 8)
    >>> # DC coefficient of each block equals block_size (ortho norm of constant)
    >>> float(round(c[0, 0], 6))
    2.0
    """
    n = data.shape[0]
    if n % block_size != 0:
        raise ValueError(
            f"Array size {n} is not divisible by block_size {block_size}."
        )
    nb = n // block_size  # number of blocks per spatial dimension

    # Reshape (n, n) → (nb, block_size, nb, block_size), then
    # transpose to (nb, nb, block_size, block_size) so that
    # blocks[i, j] is the (i, j)-th block.
    blocks = (
        data.reshape(nb, block_size, nb, block_size)
        .transpose(0, 2, 1, 3)
    )

    # Apply 2-D DCT on the last two axes (within each block).
    coeffs = dctn(blocks, axes=(-2, -1), norm="ortho")

    # Transpose back and reshape to (n, n) to preserve spatial layout.
    return coeffs.transpose(0, 2, 1, 3).reshape(n, n)


def idct_blocks(coeffs: npt.NDArray, block_size: int) -> npt.NDArray[np.float64]:
    """Apply a 2-D IDCT-II to each non-overlapping block and reassemble.

    This is the exact inverse of :func:`dct_blocks`: applying
    ``idct_blocks(dct_blocks(x, b), b)`` recovers *x* up to floating-point
    rounding.

    Parameters
    ----------
    coeffs : numpy.ndarray of shape (n, n)
        DCT-II coefficient array as produced by :func:`dct_blocks`.
        *n* must be divisible by *block_size*.
    block_size : int
        Side length of each square block — must match the value used
        when calling :func:`dct_blocks`.

    Returns
    -------
    numpy.ndarray of shape (n, n), dtype float64
        Reconstructed spatial-domain array.

    See Also
    --------
    dct_blocks : Forward transform.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal((8, 8)).astype(np.float32)
    >>> np.allclose(x, idct_blocks(dct_blocks(x, 4), 4), atol=1e-5)
    True
    """
    n = coeffs.shape[0]
    nb = n // block_size

    blocks = coeffs.reshape(nb, block_size, nb, block_size).transpose(0, 2, 1, 3)
    reconstructed = idctn(blocks, axes=(-2, -1), norm="ortho")
    return reconstructed.transpose(0, 2, 1, 3).reshape(n, n)


# ─────────────────────────────────────────────────────────────────
#  Residual codec  (used by the zarr ArrayArrayCodec layer)
# ─────────────────────────────────────────────────────────────────


def compress_residuals(
    residuals: npt.NDArray[np.float32],
    prediction: npt.NDArray[np.float32],
    window: int,
    quantize_bits: int,
    clip_percentile: float,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.float32]]:
    """Compress a residual array using block DCT and uniform quantisation.

    The residual is partitioned into non-overlapping ``window × window``
    blocks.  A 2-D DCT-II is applied to each block, and the resulting
    coefficients are uniformly quantised to int8 or int16 using a single
    global scale factor ``q_scale`` derived from the coefficient distribution.

    Parameters
    ----------
    residuals : numpy.ndarray of shape (n, n), dtype float32
        Residual signal ``original − bilinear_prediction``.
    prediction : numpy.ndarray of shape (n, n), dtype float32
        Bilinear prediction used to produce *residuals*.  Passed here for
        API symmetry with :func:`decompress_residuals` but not used by this
        function.
    window : int
        DCT block side length.  Must divide *n* evenly.  Typical values:
        4, 8, 16.
    quantize_bits : {8, 16}
        Quantisation depth.

        * ``8``  → ``int8``, 254 usable levels (±127), 1 byte per coefficient.
        * ``16`` → ``int16``, 65 534 usable levels (±32 767), 2 bytes per
          coefficient.
    clip_percentile : float
        Symmetric tail percentile used to determine ``q_scale``.
        ``q_scale`` is set so that the ``(100 − clip_percentile)``-th
        percentile of the absolute DCT coefficients maps to the maximum
        quantised level.  A larger value allows more saturation of large
        coefficients in exchange for finer resolution of the majority.
        Typical range: 0.02–5.0.

    Returns
    -------
    residuals_q : numpy.ndarray of shape (n, n), dtype int8 or int16
        Quantised DCT coefficients in block-spatial layout (same spatial
        ordering as *residuals*; coefficient ``[r, c]`` of block ``(i, j)``
        is at index ``[i*window + r, j*window + c]``).
    q_scale : numpy.ndarray of shape (), dtype float32
        Scalar quantisation step size.  Must be stored as metadata and
        passed to :func:`decompress_residuals` to reconstruct the residual.

    See Also
    --------
    decompress_residuals : Inverse of this function.
    dct_blocks : 2-D block DCT primitive.

    Notes
    -----
    Uniform quantisation is chosen because the residual DCT coefficients of
    a prediction error tend to be sparse and concentrated near zero, making
    the distribution relatively uniform within the quantisation range.
    A single ``q_scale`` scalar is sufficient to reconstruct the signal,
    which minimises per-chunk metadata overhead.
    """
    # Transform residuals to DCT domain, block by block.
    coeffs = dct_blocks(residuals.astype(np.float32), block_size=window)

    # Derive quantisation scale from the coefficient distribution.
    # max_abs is the (100 - clip_percentile)-th percentile of |coeffs|,
    # so clip_percentile % of coefficients will saturate (be clipped).
    max_level = (2 ** (quantize_bits - 1)) - 1  # 127 or 32 767
    abs_coeffs = np.abs(coeffs)
    max_abs = float(np.percentile(abs_coeffs, 100.0 - clip_percentile))

    # Guard against a near-zero residual (e.g. perfectly flat tile).
    if max_abs == 0.0:
        max_abs = 1.0

    q_scale = np.float32(max_abs / max_level)

    # Clip and round to integer levels in the symmetric range [−max_level, +max_level].
    coeffs_clipped = np.clip(coeffs, -max_abs, max_abs)
    coeffs_int = np.round(coeffs_clipped / q_scale).astype(
        np.int8 if quantize_bits == 8 else np.int16
    )

    return coeffs_int, np.array(q_scale, dtype=np.float32)


def decompress_residuals(
    compressed: npt.NDArray[np.integer],
    prediction: npt.NDArray[np.float32],
    q_scale: npt.NDArray[np.float32],
    *,
    window: int,
    quantize_bits: int,
) -> npt.NDArray[np.float32]:
    """Reconstruct a residual array from quantised DCT coefficients.

    This is the inverse of :func:`compress_residuals`.

    Parameters
    ----------
    compressed : numpy.ndarray of shape (n, n), dtype int8 or int16
        Quantised DCT coefficients in block-spatial layout, as returned by
        :func:`compress_residuals`.
    prediction : numpy.ndarray of shape (n, n), dtype float32
        Bilinear prediction.  Passed here for API symmetry but not used by
        this function; the spatial context is not needed for IDCT-based
        reconstruction.
    q_scale : numpy.ndarray of shape () or (1,), dtype float32
        Quantisation step size returned by :func:`compress_residuals`.
    window : int
        DCT block side length — must match the value used at compress time.
    quantize_bits : {8, 16}
        Quantisation depth — must match the value used at compress time.
        (Not used directly here; kept for API symmetry.)

    Returns
    -------
    numpy.ndarray of shape (n, n), dtype float32
        Reconstructed residual array.

    See Also
    --------
    compress_residuals : Inverse of this function.
    idct_blocks : 2-D block IDCT primitive.
    """
    # Dequantise: scale integer levels back to float coefficients.
    coeffs = compressed.astype(np.float32) * float(q_scale)

    # Apply inverse DCT block by block and cast back to float32.
    return idct_blocks(coeffs, block_size=window).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
#  Internal encoder / decoder  (2-D tile ↔ tuple of arrays)
# ─────────────────────────────────────────────────────────────────


def encode(
    data: npt.NDArray[np.float32],
    window: int,
    quantize_bits: int,
    clip_percentile: float,
    downsample_factor: int,
) -> tuple[npt.NDArray[np.integer], int, float, npt.NDArray[np.float32]]:
    """Encode a square 2-D float32 tile into a tuple of arrays.

    This is an internal function.  For external use prefer :func:`compress`.
    Entropy compression (zlib) is intentionally absent so that the zarr
    codec layer can handle serialisation independently.

    Parameters
    ----------
    data : numpy.ndarray of shape (n, n), dtype float32
        Square 2-D input tile.
    window : int
        DCT block side length.
    quantize_bits : {8, 16}
        Quantisation depth.
    clip_percentile : float
        Tail percentile for ``q_scale`` derivation (see
        :func:`compress_residuals`).
    downsample_factor : {1, 2, 4}
        Spatial reduction factor for the low-frequency layer.

    Returns
    -------
    R_q : numpy.ndarray of shape (n, n), dtype int8 or int16
        Quantised DCT coefficients.
    n : int
        Tile side length.
    q_scale : float
        Quantisation step size.
    low : numpy.ndarray of shape (n//downsample_factor, n//downsample_factor), dtype float32
        Lossless low-frequency layer.

    See Also
    --------
    decode : Inverse of this function.
    compress : Public API — adds Z-order conversion, zlib, and binary packing.
    """
    data = data.astype(np.float32)
    n = int(data.shape[0])

    # 1. Low-frequency layer (lossless).
    low = downsample(data, factor=downsample_factor)

    # 2. Bilinear prediction from the low-frequency layer.
    pred = upsample(low, factor=downsample_factor)

    # 3. Residual.
    R = data - pred

    # 4. Block-DCT quantisation of the residual.
    R_q, q_scale_arr = compress_residuals(
        R,
        prediction=pred,
        window=window,
        quantize_bits=quantize_bits,
        clip_percentile=clip_percentile,
    )

    return R_q, n, float(q_scale_arr), low


def decode(
    encoded: tuple,
    window: int,
    quantize_bits: int,
    downsample_factor: int,
) -> npt.NDArray[np.float32]:
    """Reconstruct a 2-D float32 tile from a tuple produced by :func:`encode`.

    This is an internal function.  For external use prefer :func:`decompress`.

    Parameters
    ----------
    encoded : tuple
        ``(R_q, n, q_scale, low)`` as returned by :func:`encode`.
    window : int
        DCT block side length — must match the value used at encode time.
    quantize_bits : {8, 16}
        Quantisation depth — must match encode time.
    downsample_factor : {1, 2, 4}
        Low-frequency reduction factor — must match encode time.

    Returns
    -------
    numpy.ndarray of shape (n, n), dtype float32
        Reconstructed 2-D tile.

    See Also
    --------
    encode : Forward transform.
    decompress : Public API — adds Z-order conversion and binary unpacking.
    """
    R_q, n, q_scale, low = encoded

    # Reconstruct prediction from the low-frequency layer.
    pred = upsample(np.reshape(low, (int(n) // downsample_factor, int(n) // downsample_factor)), factor=downsample_factor)

    # Dequantise and apply IDCT.
    R = decompress_residuals(
        np.reshape(R_q, (n, n)),
        prediction=pred,
        q_scale=np.float32(q_scale),
        window=window,
        quantize_bits=quantize_bits,
    )

    return pred + R


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────


def compress(
    data: npt.NDArray[np.float32],
    window: int,
    quantize_bits: int = 8,
    clip_percentile: float = 0.5,
    downsample_factor: int = 2,
    zlib_level: int = 9,
    force_compress_low: bool = True,
) -> tuple:
    """Compress a HEALPix chunk stored in Z-order into a tuple of arrays.

    The Z-order (Morton) input is first expanded to a 2-D square grid,
    compressed via :func:`encode` (low-frequency layer + block-DCT residual),
    then the quantised residual is zlib-compressed.

    Parameters are **not** embedded in the returned bytes.  The caller is
    responsible for storing them (e.g. in ``zarr.json``) and passing them
    unchanged to :func:`decompress`.  Use :func:`lookup_preset` to resolve
    a named preset into an explicit parameter dict before writing.

    Parameters
    ----------
    data : numpy.ndarray
        1-D array in Z-order (Morton) encoding, as stored in HEALPix NESTED
        scheme tiles, dtype float32.
    window : int
        DCT block side length.  Must divide the tile side length *n* evenly.
        Typical values: 4, 8, 16.
    quantize_bits : {8, 16}, default 8
        Quantisation depth.

        * ``8``  → int8, 254 usable levels, 1 byte per coefficient.
        * ``16`` → int16, 65 534 usable levels, 2 bytes per coefficient.
    clip_percentile : float, default 0.5
        Symmetric tail percentile for ``q_scale`` derivation.  A larger
        value allows more saturation in exchange for finer resolution of
        the majority of coefficients.  Typical range: 0.02–5.0.
    downsample_factor : {1, 2, 4}, default 2
        Spatial reduction factor for the lossless low-frequency layer.

        * ``1`` — full-resolution (lossless low frequencies).
        * ``2`` — half resolution; recommended default.
        * ``4`` — quarter resolution; smallest file, most loss.
    zlib_level : int, default 9
        zlib compression level for the quantised residual (0–9).
    force_compress_low : bool, default True
        If ``True``, the low-frequency layer is also zlib-compressed before
        being stored in the returned tuple.  Set to ``False`` to keep it as
        a raw float32 array (e.g. when the caller handles serialisation
        externally).  Must match the value passed to :func:`decompress`.

    Returns
    -------
    tuple
        ``(compressed_bytes, n, q_scale, low)`` where:

        * ``compressed_bytes`` — zlib-compressed int8/int16 residual bytes.
        * ``n`` — tile side length (int).
        * ``q_scale`` — quantisation step size (float).
        * ``low_bytes`` — zlib-compressed float32 low-frequency layer if
          *force_compress_low* is ``True``, otherwise a raw float32 array.

    Raises
    ------
    ValueError
        If *quantize_bits* or *downsample_factor* are not in their allowed
        sets, or if *clip_percentile* is out of range.

    See Also
    --------
    decompress : Inverse of this function.
    lookup_preset : Resolve a named preset into an explicit parameter dict.
    compression_info : Inspect size and ratio statistics after encoding.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chunk = rng.standard_normal(64 * 64).astype(np.float32)
    >>> params = lookup_preset("medium")
    >>> enc = compress(chunk, **params)
    >>> isinstance(enc, tuple)
    True
    """
    if quantize_bits not in (8, 16):
        raise ValueError("quantize_bits must be 8 or 16")
    if downsample_factor not in (1, 2, 4):
        raise ValueError("downsample_factor must be 1, 2 or 4")
    if not (0.0 < clip_percentile < 50.0):
        raise ValueError("clip_percentile must be in the open interval (0, 50)")

    tile_2d = expand_tile(data.astype(np.float32))
    R_q, n, q_scale, low = encode(
        tile_2d,
        window=window,
        quantize_bits=quantize_bits,
        clip_percentile=clip_percentile,
        downsample_factor=downsample_factor,
    )

    compressed_bytes = zlib.compress(R_q.tobytes(), level=zlib_level)
    low_out = (
        zlib.compress(low.ravel().tobytes(), level=zlib_level)
        if force_compress_low
        else low.ravel()
    )
    return compressed_bytes, n, q_scale, low_out


def decompress(
    compressed_data: tuple,
    window: int,
    quantize_bits: int = 8,
    downsample_factor: int = 2,
    force_compress_low: bool = True,
    **kwargs,
) -> npt.NDArray[np.float32]:
    """Decompress a tuple produced by :func:`compress`.

    The parameter signature mirrors :func:`compress` so that a codec can pass
    the same metadata dict to both functions.  Only *window*, *quantize_bits*,
    and *downsample_factor* are required for decoding; any extra keyword
    arguments (e.g. ``clip_percentile``, ``zlib_level``) are accepted but
    ignored for API symmetry.

    Parameters
    ----------
    compressed_data : tuple
        ``(compressed_bytes, n, q_scale, low)`` as returned by
        :func:`compress`.
    window : int
        DCT block side length — must match compress time.
    quantize_bits : {8, 16}, default 8
        Quantisation depth — must match compress time.
    downsample_factor : {1, 2, 4}, default 2
        Low-frequency layer reduction factor — must match compress time.
    force_compress_low : bool, default True
        Must match the value used at compress time.  When ``True``, the
        low-frequency layer is decompressed with zlib before use; when
        ``False``, it is read directly as a float32 array.
    **kwargs
        Accepted for API symmetry; ignored.

    Returns
    -------
    numpy.ndarray
        Reconstructed chunk in Z-order encoding, same shape and dtype
        (float32) as the original input to :func:`compress`.

    See Also
    --------
    compress : Inverse of this function.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chunk = rng.standard_normal(64 * 64).astype(np.float32)
    >>> params = lookup_preset("medium")
    >>> enc = compress(chunk, **params)
    >>> rec = decompress(enc, **params)
    >>> rec.shape == chunk.shape
    True
    """
    compressed_bytes, n, q_scale, low_data = compressed_data

    residual_dtype = np.int8 if quantize_bits == 8 else np.int16
    R_q = np.frombuffer(zlib.decompress(compressed_bytes), dtype=residual_dtype).reshape(n, n)

    low_m = n // downsample_factor
    if force_compress_low:
        low_2d = np.frombuffer(zlib.decompress(low_data), dtype=np.float32).reshape(low_m, low_m)
    else:
        low_2d = np.asarray(low_data, dtype=np.float32).reshape(low_m, low_m)

    tile_2d = decode(
        (R_q, n, q_scale, low_2d),
        window=window,
        quantize_bits=quantize_bits,
        downsample_factor=downsample_factor,
    )

    return flatten_tile(tile_2d)


def lookup_preset(name: Literal["low", "medium", "high", "ultra"]) -> dict[str, Any]:
    """Return a copy of the parameter dict for a named quality preset.

    Resolve the preset *before* writing and store the returned dict (not the
    preset name) in the array metadata.  This guarantees that future changes
    to preset values cannot affect the readability of existing data.

    Parameters
    ----------
    name : {'low', 'medium', 'high', 'ultra'}
        Name of the preset to look up.

        +----------+--------+---------------+-----------------+------------+-------------------+
        | name     | window | quantize_bits | clip_percentile | zlib_level | downsample_factor |
        +==========+========+===============+=================+============+===================+
        | low      | 4      | 8             | 3.0             | 9          | 4                 |
        +----------+--------+---------------+-----------------+------------+-------------------+
        | medium   | 8      | 8             | 0.5             | 9          | 2                 |
        +----------+--------+---------------+-----------------+------------+-------------------+
        | high     | 8      | 16            | 0.1             | 6          | 2                 |
        +----------+--------+---------------+-----------------+------------+-------------------+
        | ultra    | 16     | 16            | 0.02            | 1          | 1                 |
        +----------+--------+---------------+-----------------+------------+-------------------+

    Returns
    -------
    dict[str, Any]
        A fresh copy of the preset's parameter dict.  Keys match the
        keyword arguments of :func:`compress` and :func:`decompress`.

    Raises
    ------
    ValueError
        If *name* is not one of the supported preset names.

    See Also
    --------
    compress : Pass the returned dict as ``**params`` to compress data.
    decompress : Pass the same dict as ``**params`` to decompress data.

    Examples
    --------
    >>> params = lookup_preset("high")
    >>> sorted(params.keys())
    ['clip_percentile', 'downsample_factor', 'quantize_bits', 'window', 'zlib_level']
    >>> params["quantize_bits"]
    16
    """
    if name not in _PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. Available presets: {list(_PRESETS)}"
        )
    return _PRESETS[name].copy()


# ─────────────────────────────────────────────────────────────────
#  Utility: compression statistics
# ─────────────────────────────────────────────────────────────────


def compression_info(
    compressed_data: tuple,
    downsample_factor: int,
    original_n: int | None = None,
    force_compress_low: bool = True,
) -> dict:
    """Return size and compression-ratio statistics for a compressed chunk.

    Parameters
    ----------
    compressed_data : tuple
        ``(compressed_bytes, n, q_scale, low)`` as returned by
        :func:`compress`.
    downsample_factor : {1, 2, 4}
        The downsampling factor used at compress time — needed to compute
        the size of the low-frequency layer.
    original_n : int or None, optional
        Side length of the original square tile in pixels.  When ``None``,
        inferred from *n* in *compressed_data*.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``n`` : int
            Tile side length.
        ``downsample_factor`` : int
            Downsampling factor.
        ``q_scale`` : float
            Quantisation step size.
        ``low_grid`` : str
            Shape of the embedded low-frequency layer as ``"m×m"``.
        ``bytes_low_layer`` : int
            Size of the low-frequency layer in bytes (float32).
        ``bytes_residual_compressed`` : int
            Size of the zlib-compressed residual in bytes.
        ``total_bytes`` : int
            Total compressed size in bytes.
        ``original_bytes`` : int
            Uncompressed size of the original tile in bytes (float32).
        ``compression_ratio`` : float
            ``original_bytes / total_bytes``, rounded to 2 decimal places.
        ``size_reduction_pct`` : float
            ``(1 − total_bytes / original_bytes) × 100``.

    See Also
    --------
    compress : Produces the tuple consumed by this function.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chunk = rng.standard_normal(64 * 64).astype(np.float32)
    >>> params = lookup_preset("medium")
    >>> enc = compress(chunk, **params)
    >>> info = compression_info(enc, downsample_factor=params["downsample_factor"])
    >>> info["compression_ratio"] > 1.0
    True
    """
    compressed_bytes, n, q_scale, low_data = compressed_data

    bytes_low = len(low_data) if force_compress_low else np.asarray(low_data).nbytes
    bytes_residual = len(compressed_bytes)
    total_bytes = bytes_residual + bytes_low

    orig_n = original_n if original_n is not None else n
    original_bytes = orig_n * orig_n * 4
    ratio = original_bytes / total_bytes if total_bytes > 0 else float("inf")

    return {
        "n": int(n),
        "downsample_factor": downsample_factor,
        "q_scale": float(q_scale),
        "low_grid": f"{n // downsample_factor}x{n // downsample_factor}",
        "bytes_low_layer": bytes_low,
        "bytes_residual_compressed": bytes_residual,
        "total_bytes": total_bytes,
        "original_bytes": original_bytes,
        "compression_ratio": round(ratio, 2),
        "size_reduction_pct": round((1 - total_bytes / original_bytes) * 100, 1),
    }
