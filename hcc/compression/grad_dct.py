"""
Gradient-residual DCT-style lossy compression for HEALPix tiles.

This module implements a hierarchical, adaptive-quantisation compressor
for square float32 arrays stored in Z-order (Morton) encoding.  The
pipeline follows three broad stages:

1. **Low-frequency layer** – the input is downsampled and stored losslessly.
2. **Residual coding** – the difference between the original and a bilinear
   prediction is normalised by a local standard-deviation map, then
   quantised to int8 or int16.
3. **Entropy compression** – the quantised residual is compressed with zlib.

Public API
----------
compress : Compress a Z-order HEALPix chunk into a flat byte array.
decompress : Reconstruct a chunk from its compressed byte array.
lookup_preset : Return the parameter dict for a named quality preset.
compression_info : Report size and compression-ratio statistics.

Design notes
------------
Algorithm parameters (``window``, ``quantize_bits``, etc.) are **not**
embedded in the compressed bytes.  They must be passed identically to
both :func:`compress` and :func:`decompress`, and should be stored in
the array's metadata document (e.g. ``zarr.json``) by the caller.

Only the three data-dependent scalars ``q_lo``, ``q_hi``, and
``sigma_floor`` — which cannot be recomputed without the original data —
are stored in the binary header together with ``n`` (the tile side
length).

Use :func:`lookup_preset` to resolve a named preset into an explicit
parameter dict *before* writing; store that dict in the metadata, not
the preset name.  This guarantees that changing or removing a preset in
a future version of the library cannot make existing data unreadable.
"""

import struct
from typing import Any, Literal

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage import zoom as sp_zoom

__all__ = [
    "compress",
    "decompress",
    "lookup_preset",
    "compression_info",
    "block_mean_2x2",
    "downsample",
    "upsample",
    "local_std",
]

# ─────────────────────────────────────────────────────────────────
#  Binary format constants
# ─────────────────────────────────────────────────────────────────

# Little-endian header: uint32 n  +  float32 q_lo, q_hi, sigma_floor
_HEADER_FMT = "<Ifff"
_HEADER_BYTES = struct.calcsize(_HEADER_FMT)  # 16 bytes


# ─────────────────────────────────────────────────────────────────
#  Quality presets  (private — always access via lookup_preset)
# ─────────────────────────────────────────────────────────────────

_PRESETS: dict[str, dict[str, Any]] = {
    "low": dict(
        window=4,
        quantize_bits=8,
        clip_percentile=3.0,
        zlib_level=9,
        sigma_floor_ratio=0.15,
        downsample_factor=4,
    ),
    "medium": dict(
        window=8,
        quantize_bits=8,
        clip_percentile=0.5,
        zlib_level=9,
        sigma_floor_ratio=0.05,
        downsample_factor=2,
    ),
    "high": dict(
        window=8,
        quantize_bits=16,
        clip_percentile=0.1,
        zlib_level=6,
        sigma_floor_ratio=0.02,
        downsample_factor=2,
    ),
    "ultra": dict(
        window=16,
        quantize_bits=16,
        clip_percentile=0.02,
        zlib_level=1,
        sigma_floor_ratio=0.01,
        downsample_factor=1,
    ),
}


# ─────────────────────────────────────────────────────────────────
#  Internal helpers (Z-order / Morton encoding)
# ─────────────────────────────────────────────────────────────────


def _as_square_array(arr):
    edge_length = int(np.sqrt(arr.size))

    return np.reshape(arr, (edge_length, edge_length))


def _decode_zorder(delta: int):
    """Decode Z-order (Morton) curve indices into (x, y) coordinate arrays.

    Parameters
    ----------
    delta : int
        Recursion depth; the tile side length is ``2**delta``.

    Returns
    -------
    x : numpy.ndarray of shape (4**delta,), dtype uint64
        Column indices corresponding to each Morton code.
    y : numpy.ndarray of shape (4**delta,), dtype uint64
        Row indices corresponding to each Morton code.
    """
    size = 4**delta
    x = np.zeros((size,), dtype="uint64")
    y = np.zeros((size,), dtype="uint64")
    code = np.arange(size, dtype="uint64")
    for bit in range(delta):
        x |= ((code >> (bit << 1)) & 1) << bit
        y |= ((code >> ((bit << 1) + 1)) & 1) << bit
    return x, y


def _flatten_tile(data: np.ndarray) -> np.ndarray:
    """Convert a 2-D tile back to a 1-D Z-order encoded array.

    Parameters
    ----------
    data : numpy.ndarray
        2-D array of shape ``(..., 2**delta, 2**delta)``.

    Returns
    -------
    numpy.ndarray
        1-D array of shape ``(4**delta,)`` in Morton order.
    """
    delta = int(np.log(data.size) / np.log(4))
    x, y = _decode_zorder(delta)
    return np.squeeze(data[..., x, y])


def _expand_tile(data: np.ndarray) -> np.ndarray:
    """Convert a 1-D Z-order encoded array into a 2-D tile.

    Parameters
    ----------
    data : numpy.ndarray
        Array with last dimension of size ``4**delta`` in Morton order.

    Returns
    -------
    numpy.ndarray
        Array with last two dimensions replaced by ``(2**delta, 2**delta)``.
    """
    nside = int(np.sqrt(data.shape[-1]))
    delta = int(np.log2(nside))
    x, y = _decode_zorder(delta)
    shape = (*data.shape[:-1], 2**delta, 2**delta)
    out = np.empty(shape, dtype=data.dtype)
    out[..., x, y] = data
    return out


# ─────────────────────────────────────────────────────────────────
#  Signal processing primitives
# ─────────────────────────────────────────────────────────────────


def block_mean_2x2(data: np.ndarray) -> np.ndarray:
    """Average a square array over non-overlapping 2×2 blocks.

    Parameters
    ----------
    data : numpy.ndarray of shape (n, n), dtype float32
        Input array.  ``n`` must be even.

    Returns
    -------
    numpy.ndarray of shape (n//2, n//2), dtype float32
        Block-averaged output.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(16, dtype=np.float32).reshape(4, 4)
    >>> block_mean_2x2(x)
    array([[ 2.5,  4.5],
           [10.5, 12.5]], dtype=float32)
    """
    n = data.shape[0]
    return data.reshape(n // 2, 2, n // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a square array by an integer factor.

    Parameters
    ----------
    data : numpy.ndarray of shape (n, n), dtype float32
        Input array.  ``n`` must be divisible by *factor*.
    factor : {1, 2, 4}
        Spatial reduction factor.

        * ``1`` – returns a float32 copy without modification.
        * ``2`` – applies :func:`block_mean_2x2` (exact block average).
        * ``4`` – applies bilinear downsampling via ``scipy.ndimage.zoom``.

    Returns
    -------
    numpy.ndarray of shape (n//factor, n//factor), dtype float32
        Downsampled array.

    See Also
    --------
    upsample : Inverse operation (bilinear interpolation).
    block_mean_2x2 : Block-averaging primitive used when *factor* = 2.
    """
    if factor == 1:
        return data.astype(np.float32).copy()
    if factor == 2:
        return block_mean_2x2(data)
    return sp_zoom(data.astype(np.float64), 1.0 / factor, order=1).astype(np.float32)


def upsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a square array by an integer factor using bilinear interpolation.

    Parameters
    ----------
    data : numpy.ndarray of shape (m, m), dtype float32
        Input array.
    factor : {1, 2, 4}
        Spatial magnification factor.

        * ``1`` – returns a float32 copy without modification.
        * ``2`` or ``4`` – bilinear interpolation via ``scipy.ndimage.zoom``.

    Returns
    -------
    numpy.ndarray of shape (m*factor, m*factor), dtype float32
        Upsampled array.

    See Also
    --------
    downsample : Inverse operation.
    """
    if factor == 1:
        return data.astype(np.float32).copy()
    return sp_zoom(data.astype(np.float64), factor, order=1).astype(np.float32)


def local_std(data: np.ndarray, window: int = 8) -> np.ndarray:
    """Compute a local standard-deviation map using a uniform sliding window.

    Uses the identity ``Var(X) = E[X²] − E[X]²`` to compute the map in
    O(n) time via two calls to :func:`scipy.ndimage.uniform_filter`.

    Parameters
    ----------
    data : numpy.ndarray of shape (n, n)
        Input array (any floating-point dtype; internally cast to float64).
    window : int, optional
        Side length of the square averaging window, by default 8.
        Larger values produce a smoother sigma map.

    Returns
    -------
    numpy.ndarray of shape (n, n), dtype float32
        Per-pixel local standard deviation, clipped to zero from below to
        avoid numerical negatives from floating-point cancellation.

    Notes
    -----
    The uniform filter is applied with default boundary conditions
    (``mode='reflect'``), which may introduce mild edge artefacts for
    very large *window* values relative to the array size.
    """
    f = data.astype(np.float64)
    mu = uniform_filter(f, size=window)
    mu2 = uniform_filter(f**2, size=window)
    return np.sqrt(np.maximum(mu2 - mu**2, 0)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
#  Internal encoder / decoder  (2-D tile ↔ tuple of arrays)
# ─────────────────────────────────────────────────────────────────


def encode(
    data: np.ndarray,
    window: int,
    quantize_bits: int,
    clip_percentile: float,
    sigma_floor_ratio: float,
    downsample_factor: int,
) -> tuple:
    """Encode a square 2-D float32 tile into a tuple of arrays.

    This is an internal function.  Prefer :func:`compress` for external use.

    Entropy compression (zlib) is intentionally absent here so that the
    zarr codec layer can handle serialisation independently.

    Parameters
    ----------
    data : numpy.ndarray of shape (n, n), dtype float32
        Square 2-D input tile.
    window : int
        Uniform-filter window size for :func:`local_std`.
    quantize_bits : {8, 16}
        Residual quantisation depth.
    clip_percentile : float
        Symmetric percentile for residual clipping before quantisation.
    sigma_floor_ratio : float
        Minimum sigma expressed as a fraction of ``std(R)``.
    downsample_factor : {1, 2, 4}
        Spatial reduction factor for the low-frequency layer.

    Returns
    -------
    R_q : numpy.ndarray
        Quantised normalised residuals of shape (n, n), dtype int8 or int16.
    n : int
        Tile side length.
    q_lo : float
        Lower quantisation bound.
    q_hi : float
        Upper quantisation bound.
    sigma_floor : float
        Variance normalisation floor.
    low : numpy.ndarray
        Low-frequency layer of shape (n//downsample_factor, n//downsample_factor), dtype float32

    See Also
    --------
    decode : Inverse of this function.
    compress : Public API — adds Z-order conversion, zlib, and binary packing.
    """
    data = data.astype(np.float32)
    n = np.array(data.shape[0], dtype="uint64")

    # 1. Low-frequency layer
    low = downsample(data, factor=downsample_factor)

    # 2. Prediction via bilinear upsampling
    pred = upsample(low, factor=downsample_factor)

    # 3. Residual
    R = data - pred

    # 4. Local variance map
    #    sigma(x,y) ~ local activity in the prediction:
    #    smooth regions -> low sigma -> small expected residual
    #    active regions -> high sigma -> large expected residual
    sigma = local_std(pred, window=window)
    sigma_floor = np.std(R) * sigma_floor_ratio
    sigma = np.maximum(sigma, sigma_floor)

    # 5. Adaptive variance normalisation (whitens the signal)
    R_norm = (R / sigma).astype(np.float32)

    # 6. Quantisation
    n_levels = 2**quantize_bits  # 256 or 65 536
    half = n_levels // 2  # 128 or 32 768

    q_lo = np.percentile(R_norm, clip_percentile)
    q_hi = np.percentile(R_norm, 100.0 - clip_percentile)
    scale = (q_hi - q_lo) / (n_levels - 1)

    R_clipped = np.clip(R_norm, q_lo, q_hi)
    R_int = np.round((R_clipped - q_lo) / scale - half)

    R_q = R_int.astype(np.int8 if quantize_bits == 8 else np.int16)

    return (np.ravel(R_q), n, q_lo, q_hi, sigma_floor, np.ravel(low))


def decode(
    encoded: tuple,
    window: int,
    quantize_bits: int,
    downsample_factor: int,
) -> np.ndarray:
    """Reconstruct a 2-D float32 tile from a tuple produced by :func:`encode`.

    This is an internal function.  Prefer :func:`decompress` for external use.

    Parameters
    ----------
    encoded : tuple
        ``(R_q, n, q_lo, q_hi, sigma_floor, low)`` as returned by
        :func:`encode`.
    window : int
        Uniform-filter window size — must match the value used at encode time.
    quantize_bits : {8, 16}
        Residual quantisation depth — must match the value used at encode time.
    downsample_factor : {1, 2, 4}
        Low-frequency layer reduction factor — must match encode time.

    Returns
    -------
    numpy.ndarray of shape (n, n), dtype float32
        Reconstructed 2-D tile.

    Notes
    -----
    The sigma map is not stored in the tuple; it is recomputed
    deterministically from the low-frequency layer at decode time.

    See Also
    --------
    encode : Inverse of this function.
    decompress : Public API — adds Z-order conversion and binary unpacking.
    """
    R_q, n, q_lo, q_hi, sigma_floor, low = encoded

    R_q = _as_square_array(R_q)
    low = _as_square_array(low)

    # Reconstruct prediction and sigma map
    pred = upsample(low, factor=downsample_factor)
    sigma = local_std(pred, window=window)
    sigma = np.maximum(sigma, sigma_floor)

    n_levels = 2**quantize_bits
    half = n_levels // 2

    # Dequantisation + variance denormalisation
    scale = (q_hi - q_lo) / (n_levels - 1)
    R = ((R_q.astype(np.float32) + half) * scale + q_lo) * sigma

    return pred + R


# ─────────────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────────────


def compress(
    data: np.ndarray,
    window: int,
    quantize_bits: int = 8,
    clip_percentile: float = 0.5,
    sigma_floor_ratio: float = 0.05,
    downsample_factor: int = 2,
) -> tuple[np.ndarray, ...]:
    """Compress a HEALPix chunk stored in Z-order into a flat byte array.

    Parameters are **not** embedded in the returned bytes.  The caller is
    responsible for storing them (e.g. in ``zarr.json``) and passing them
    unchanged to :func:`decompress`.  Use :func:`lookup_preset` to resolve
    a named preset into an explicit parameter dict before writing.

    Parameters
    ----------
    data : numpy.ndarray
        1-D (or batched) array in Z-order (Morton) encoding, as stored in
        HEALPix NESTED scheme tiles, dtype float32.
    window : int
        Side length of the uniform window used by :func:`local_std` to
        build the local standard-deviation map.  Typical range: 4–32.
        Larger values produce smoother sigma maps.
    quantize_bits : {8, 16}, default: 8
        Bit depth of the quantised residual.

        * ``8``  → ``int8``,  256 levels, 1 byte per pixel.
        * ``16`` → ``int16``, 65 536 levels, 2 bytes per pixel.
    clip_percentile : float, default: 0.5
        Symmetric percentile used to define the quantisation range. The residual
        is clipped to ``[percentile(p), percentile(100−p)]`` before
        quantisation. Higher values improve entropy compression at the cost of
        larger reconstruction error in the tails. Typical range: 0.05–5.0.
    sigma_floor_ratio : float, default: 0.05
        Minimum sigma expressed as a fraction of ``std(R)``.
        Prevents division by near-zero sigma in flat regions.
        Typical range: 0.005–0.20.
    downsample_factor : {1, 2, 4}, default: 2
        Spatial reduction factor for the low-frequency layer.

        * ``1`` – full-resolution low layer (lossless low frequencies).
        * ``2`` – layer at half resolution; recommended default.
        * ``4`` – layer at quarter resolution; smallest file, most loss.

    Returns
    -------
    R_q : numpy.ndarray
        Quantised normalised residuals of shape (n, n), dtype int8 or int16.
    n : int
        Tile side length.
    q_lo : float
        Lower quantisation bound.
    q_hi : float
        Upper quantisation bound.
    sigma_floor : float
        Variance normalisation floor.
    low : numpy.ndarray
        Low-frequency layer of shape (n//downsample_factor, n//downsample_factor), dtype float32

    Raises
    ------
    ValueError
        If *quantize_bits* or *downsample_factor* are not in their allowed
        sets, or if *clip_percentile* or *sigma_floor_ratio* are out of range.

    See Also
    --------
    decompress : Inverse of this function.
    lookup_preset : Resolve a named preset into an explicit parameter dict.
    compression_info : Inspect size and ratio statistics after encoding.

    Examples
    --------
    Compress using parameters from a preset:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chunk = rng.standard_normal(64 * 64).astype(np.float32)
    >>> params = lookup_preset("high")
    >>> enc = compress(chunk, **params)
    >>> isinstance(enc, np.ndarray) and enc.dtype == np.uint8
    True

    Compress with explicit parameters:

    >>> enc = compress(chunk, window=8, quantize_bits=16, downsample_factor=2)
    """
    if quantize_bits not in (8, 16):
        raise ValueError("quantize_bits must be 8 or 16")
    if downsample_factor not in (1, 2, 4):
        raise ValueError("downsample_factor must be 1, 2 or 4")
    if not (0.005 <= sigma_floor_ratio <= 0.5):
        raise ValueError("sigma_floor_ratio out of range [0.005, 0.5]")
    if not (0.0 < clip_percentile < 50.0):
        raise ValueError("clip_percentile out of range (0, 50)")

    tile_2d = _expand_tile(data.astype(np.float32))
    return encode(
        tile_2d,
        window,
        quantize_bits,
        clip_percentile,
        sigma_floor_ratio,
        downsample_factor,
    )


def decompress(
    compressed_data: tuple[np.ndarray, ...],
    window: int,
    quantize_bits: int = 8,
    downsample_factor: int = 2,
) -> np.ndarray:
    """Decompress a byte array produced by :func:`compress`.

    The parameter signature mirrors :func:`compress` so that a codec can
    pass the same metadata dict to both functions.  Only *window*,
    *quantize_bits*, and *downsample_factor* are required for decoding;
    the remaining arguments are accepted for API symmetry but are not used.

    Parameters
    ----------
    compressed_data : tuple of numpy.ndarray
        Compressed component arrays as returned by :func:`compress`.
    window : int
        Uniform-filter window size — must match the value used at
        compress time.
    quantize_bits : {8, 16}, default: 8
        Residual quantisation depth — must match compress time, by default 8.
    downsample_factor : {1, 2, 4}, default: 2
        Low-frequency layer reduction factor — must match compress time,
        by default 2.

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
    tile_2d = decode(compressed_data, window, quantize_bits, downsample_factor)

    return _flatten_tile(tile_2d)


def lookup_preset(name: Literal["low", "medium", "high", "ultra"]) -> dict[str, Any]:
    """Return a copy of the parameter dict for a named quality preset.

    Resolve the preset *before* writing and store the returned dict (not
    the preset name) in the array metadata.  This guarantees that future
    changes to preset values cannot affect the readability of existing data.

    Parameters
    ----------
    name : {'low', 'medium', 'high', 'ultra'}
        Name of the preset to look up.

        +----------+--------+---------------+------------------+------------+--------------------+-------------------+
        | name     | window | quantize_bits | clip_percentile  | zlib_level | sigma_floor_ratio  | downsample_factor |
        +==========+========+===============+==================+============+====================+===================+
        | low      | 4      | 8             | 3.0              | 9          | 0.15               | 4                 |
        +----------+--------+---------------+------------------+------------+--------------------+-------------------+
        | medium   | 8      | 8             | 0.5              | 9          | 0.05               | 2                 |
        +----------+--------+---------------+------------------+------------+--------------------+-------------------+
        | high     | 8      | 16            | 0.1              | 6          | 0.02               | 2                 |
        +----------+--------+---------------+------------------+------------+--------------------+-------------------+
        | ultra    | 16     | 16            | 0.02             | 1          | 0.01               | 1                 |
        +----------+--------+---------------+------------------+------------+--------------------+-------------------+

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
    ['clip_percentile', 'downsample_factor', 'quantize_bits', 'sigma_floor_ratio', 'window', 'zlib_level']
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
    binary_data: np.ndarray,
    downsample_factor: int,
    original_n: int | None = None,
) -> dict:
    """Return size and compression-ratio statistics for a compressed chunk.

    Parameters
    ----------
    binary_data : numpy.ndarray of dtype uint8
        Compressed byte array as returned by :func:`compress`.
    downsample_factor : {1, 2, 4}
        The downsampling factor used when compressing — needed to compute
        the size of the low-frequency layer embedded in the binary.
    original_n : int or None, optional
        Side length of the original square tile in pixels.  When ``None``,
        inferred from the ``n`` field in the binary header.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``n`` : int
            Tile side length read from the binary header.
        ``downsample_factor`` : int
            Downsampling factor passed as argument.
        ``low_grid`` : str
            Shape of the embedded low-frequency layer as ``"m×m"``.
        ``bytes_header`` : int
            Fixed header size in bytes (always 16).
        ``bytes_low_layer`` : int
            Size of the embedded low-frequency layer in bytes (float32).
        ``bytes_residual_compressed`` : int
            Size of the zlib-compressed residual in bytes.
        ``total_bytes`` : int
            Total size of *binary_data* in bytes.
        ``original_bytes`` : int
            Uncompressed size of the original tile in bytes (float32).
        ``compression_ratio`` : float
            ``original_bytes / total_bytes``, rounded to 2 decimal places.
        ``size_reduction_pct`` : float
            ``(1 − total_bytes / original_bytes) × 100``.

    See Also
    --------
    compress : Produces the byte array consumed by this function.

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
    buf = bytes(binary_data)
    n, _q_lo, _q_hi, _sigma_floor = struct.unpack(_HEADER_FMT, buf[:_HEADER_BYTES])

    low_m = n // downsample_factor
    bytes_low = low_m * low_m * 4  # float32
    bytes_residual = len(buf) - _HEADER_BYTES - bytes_low
    total_bytes = len(buf)

    orig_n = original_n if original_n is not None else n
    original_bytes = orig_n * orig_n * 4
    ratio = original_bytes / total_bytes if total_bytes > 0 else float("inf")

    return {
        "n": int(n),
        "downsample_factor": downsample_factor,
        "low_grid": f"{low_m}x{low_m}",
        "bytes_header": _HEADER_BYTES,
        "bytes_low_layer": bytes_low,
        "bytes_residual_compressed": bytes_residual,
        "total_bytes": total_bytes,
        "original_bytes": original_bytes,
        "compression_ratio": round(ratio, 2),
        "size_reduction_pct": round((1 - total_bytes / original_bytes) * 100, 1),
    }
