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
compress : Compress a Z-order HEALPix chunk.
decompress : Reconstruct a chunk from its compressed form.
encode : Low-level encoder (operates on 2-D tiles).
decode : Low-level decoder (operates on 2-D tiles).
compression_info : Report size and compression-ratio statistics.

Compression presets
-------------------
Four named presets are available via the ``preset`` argument of
:func:`compress`.  See :data:`QUALITY_PRESETS` for their exact values.

Notes
-----
All parameters needed for decoding are stored inside the encoded list;
the decoder requires no extra arguments beyond the encoded data itself.
"""

import zlib

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage import zoom as sp_zoom

__all__ = [
    "QUALITY_PRESETS",
    "compress",
    "decompress",
    "encode",
    "decode",
    "compression_info",
    "block_mean_2x2",
    "downsample",
    "upsample",
    "local_std",
]


# ─────────────────────────────────────────────────────────────────
#  Quality presets
# ─────────────────────────────────────────────────────────────────

QUALITY_PRESETS: dict = {
    "low": dict(
        quantize_bits=8,
        clip_percentile=3.0,
        zlib_level=9,
        sigma_floor_ratio=0.15,
        downsample_factor=4,
        window=4,
    ),
    "medium": dict(
        quantize_bits=8,
        clip_percentile=0.5,
        zlib_level=9,
        sigma_floor_ratio=0.05,
        downsample_factor=2,
        window=8,
    ),
    "high": dict(
        quantize_bits=16,
        clip_percentile=0.1,
        zlib_level=6,
        sigma_floor_ratio=0.02,
        downsample_factor=2,
        window=8,
    ),
    "ultra": dict(
        quantize_bits=16,
        clip_percentile=0.02,
        zlib_level=1,
        sigma_floor_ratio=0.01,
        downsample_factor=1,
        window=16,
    ),
}
"""dict : Named compression presets.

Each key maps to a ``dict`` of keyword arguments accepted by
:func:`compress` / :func:`encode`.

+----------+----------------+------------------+------------+--------------------+-------------------+--------+
| Preset   | quantize_bits  | clip_percentile  | zlib_level | sigma_floor_ratio  | downsample_factor | window |
+==========+================+==================+============+====================+===================+========+
| low      | 8              | 3.0              | 9          | 0.15               | 4                 | 4      |
+----------+----------------+------------------+------------+--------------------+-------------------+--------+
| medium   | 8              | 0.5              | 9          | 0.05               | 2                 | 8      |
+----------+----------------+------------------+------------+--------------------+-------------------+--------+
| high     | 16             | 0.1              | 6          | 0.02               | 2                 | 8      |
+----------+----------------+------------------+------------+--------------------+-------------------+--------+
| ultra    | 16             | 0.02             | 1          | 0.01               | 1                 | 16     |
+----------+----------------+------------------+------------+--------------------+-------------------+--------+
"""


# ─────────────────────────────────────────────────────────────────
#  Internal helpers (Z-order / Morton encoding)
# ─────────────────────────────────────────────────────────────────


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
#  Encoder
# ─────────────────────────────────────────────────────────────────


def encode(
    data: np.ndarray,
    window: int = 8,
    quantize_bits: int = 8,
    clip_percentile: float = 0.5,
    zlib_level: int = 9,
    sigma_floor_ratio: float = 0.05,
    downsample_factor: int = 2,
) -> list:
    """Encode a square float32 array into a list of compressed numpy arrays.

    The encoder applies the following pipeline:

    1. Downsample the input by *downsample_factor* to obtain the
       low-frequency layer (stored without further lossy processing).
    2. Predict the full-resolution data by bilinear upsampling.
    3. Compute the residual ``R = data − prediction``.
    4. Build a local standard-deviation map *sigma* from the prediction
       and clamp it to a minimum of ``std(R) × sigma_floor_ratio``.
    5. Normalise ``R_norm = R / sigma`` (whitens the residual).
    6. Clip *R_norm* at the *clip_percentile* / (100 − *clip_percentile*)
       percentiles and quantise to int8 or int16.
    7. Compress the quantised array with zlib at *zlib_level*.

    Parameters
    ----------
    data : numpy.ndarray of shape (n, n), dtype float32
        Input data to compress.  ``n`` must be a power of 2 and greater
        than 1.
    window : int, optional
        Side length of the uniform window used by :func:`local_std` to
        build the local standard-deviation map, by default 8.
        Typical range: 4–32.  Larger values produce smoother sigma maps,
        which may increase residual magnitudes in high-detail zones.
    quantize_bits : {8, 16}, optional
        Bit depth of the quantised residual, by default 8.

        * ``8``  → ``int8``,  256 levels, 1 byte per pixel.
        * ``16`` → ``int16``, 65 536 levels, 2 bytes per pixel.
    clip_percentile : float, optional
        Symmetric percentile used to define the quantisation range, by
        default 0.5.  The residual is clipped to
        ``[percentile(R_norm, p), percentile(R_norm, 100−p)]`` before
        mapping onto the integer range.  Increasing this value widens the
        clipped tail and improves entropy compression at the cost of larger
        reconstruction error at the extremes.  Typical range: 0.05–5.0.
    zlib_level : int, optional
        Compression level passed to :func:`zlib.compress`, by default 9.
        ``0`` disables compression; ``9`` is maximum effort.  This
        parameter is lossless and does not affect reconstruction quality.
    sigma_floor_ratio : float, optional
        Ratio applied to ``std(R)`` to define the minimum allowable sigma,
        by default 0.05.  Prevents division by near-zero sigma in flat
        regions.  Smaller values preserve more dynamic range in smooth
        zones; larger values increase quantisation noise there while
        improving overall compressibility.  Typical range: 0.005–0.20.
    downsample_factor : {1, 2, 4}, optional
        Spatial reduction factor for the low-frequency layer, by default 2.

        * ``1`` – full-resolution low layer (lossless low frequencies,
          larger file).
        * ``2`` – layer is ``(n/2)×(n/2)``; recommended default.
        * ``4`` – layer is ``(n/4)×(n/4)``; smallest file, most loss.

    Returns
    -------
    list of numpy.ndarray
        Six-element list carrying all compressed data and metadata:

        ``enc[0]`` : numpy.ndarray, shape (4,), dtype uint64
            Metadata vector ``[n, downsample_factor, quantize_bits, window]``.
        ``enc[1]`` : numpy.ndarray, shape ((n//downsample_factor)**2,), dtype float32
            Flattened low-frequency layer.
        ``enc[2]`` : numpy.ndarray, shape (1,), dtype float32
            Lower quantisation bound ``q_lo``.
        ``enc[3]`` : numpy.ndarray, shape (1,), dtype float32
            Upper quantisation bound ``q_hi``.
        ``enc[4]`` : numpy.ndarray, shape (1,), dtype float32
            Sigma floor value used during encoding.
        ``enc[5]`` : numpy.ndarray, shape (k,), dtype uint8
            zlib-compressed residual bytes (length *k* varies with data).

    Raises
    ------
    AssertionError
        If *data* is not square, or if ``n`` is not a power of 2, or if
        any parameter is outside its valid range.

    See Also
    --------
    decode : Inverse of this function.
    compress : High-level wrapper that handles Z-order tiles and presets.
    QUALITY_PRESETS : Named parameter bundles for common use cases.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((64, 64)).astype(np.float32)
    >>> enc = encode(data)
    >>> len(enc)
    6
    >>> enc[0]  # metadata: n=64, downsample_factor=2, quantize_bits=8, window=8
    array([64,  2,  8,  8], dtype=uint64)
    """
    assert quantize_bits in (8, 16), "quantize_bits must be 8 or 16"
    assert downsample_factor in (1, 2, 4), "downsample_factor must be 1, 2 or 4"
    assert 0.005 <= sigma_floor_ratio <= 0.5, "sigma_floor_ratio out of range [0.005, 0.5]"
    assert 0.0 < clip_percentile < 50.0, "clip_percentile out of range (0, 50)"

    data = data.astype(np.float32)
    n = data.shape[0]
    assert data.shape == (n, n), "data must be square"
    assert (n & (n - 1)) == 0 and n > 1, "n must be a power of 2"

    # 1. Low-frequency layer
    low = downsample(data, factor=downsample_factor)

    # 2. Prediction via bilinear upsampling
    pred = upsample(low, factor=downsample_factor)

    # 3. Residual
    R = data - pred

    # 4. Local variance map
    #    sigma(x,y) ~ local activity in the prediction:
    #    smooth regions -> low sigma  -> small expected residual
    #    active regions -> high sigma -> large expected residual
    sigma = local_std(pred, window=window)
    sigma_floor = float(np.std(R)) * sigma_floor_ratio
    sigma = np.maximum(sigma, sigma_floor)

    # 5. Adaptive variance normalisation (whitens the signal)
    R_norm = (R / sigma).astype(np.float32)

    # 6. Quantisation
    n_levels = 2**quantize_bits   # 256 or 65 536
    half = n_levels // 2          # 128 or 32 768

    q_lo = float(np.percentile(R_norm, clip_percentile))
    q_hi = float(np.percentile(R_norm, 100.0 - clip_percentile))
    scale = (q_hi - q_lo) / (n_levels - 1)

    R_clipped = np.clip(R_norm, q_lo, q_hi)
    R_int = np.round((R_clipped - q_lo) / scale - half)

    if quantize_bits == 8:
        R_q = R_int.astype(np.int8)
    else:
        R_q = R_int.astype(np.int16)

    # 7. Entropy compression — variance whitening improves zlib ratio
    residual_bytes = zlib.compress(R_q.tobytes(), level=zlib_level)

    # 8. Pack everything; enc[0] carries all parameters for the decoder
    return [
        np.array([n, downsample_factor, quantize_bits, window], dtype=np.uint64),
        low.ravel(),
        np.array([q_lo], dtype=np.float32),
        np.array([q_hi], dtype=np.float32),
        np.array([sigma_floor], dtype=np.float32),
        np.frombuffer(residual_bytes, dtype=np.uint8),
    ]


# ─────────────────────────────────────────────────────────────────
#  Decoder
# ─────────────────────────────────────────────────────────────────


def decode(enc: list) -> np.ndarray:
    """Reconstruct a float32 array from a compressed encoding produced by :func:`encode`.

    All compression parameters are read from the metadata stored in
    ``enc[0]``; no extra arguments are required.

    Parameters
    ----------
    enc : list of numpy.ndarray
        Six-element list as returned by :func:`encode`.

    Returns
    -------
    numpy.ndarray of shape (n, n), dtype float32
        Reconstructed data array.

    Notes
    -----
    The decoder mirrors the encoder pipeline exactly:

    1. Reshape the low-frequency layer from ``enc[1]``.
    2. Upsample it to full resolution to recover the prediction.
    3. Recompute the sigma map from the prediction (not stored).
    4. Decompress the zlib residual from ``enc[5]``.
    5. Dequantise and denormalise: ``R = (R_q + half) * scale + q_lo) * sigma``.
    6. Return ``prediction + R``.

    The sigma map is not stored in the encoded list; it is recomputed
    deterministically from the prediction at decode time.

    See Also
    --------
    encode : Inverse of this function.
    decompress : High-level wrapper for Z-order tiles.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> data = rng.standard_normal((64, 64)).astype(np.float32)
    >>> enc = encode(data)
    >>> rec = decode(enc)
    >>> rec.shape
    (64, 64)
    >>> float(np.max(np.abs(rec - data))) < 1.0  # lossy but bounded
    True
    """
    meta = enc[0].astype(int)
    n               = meta[0]
    downsample_factor = meta[1]
    quantize_bits   = meta[2]
    window          = meta[3]

    low_size = n // downsample_factor
    low = enc[1].reshape(low_size, low_size)

    q_lo        = float(enc[2][0])
    q_hi        = float(enc[3][0])
    sigma_floor = float(enc[4][0])

    # Prediction (identical to the encoder)
    pred = upsample(low, factor=downsample_factor)
    sigma = local_std(pred, window=window)
    sigma = np.maximum(sigma, sigma_floor)

    # zlib decompression
    raw = zlib.decompress(enc[5].tobytes())

    n_levels = 2**quantize_bits
    half = n_levels // 2

    if quantize_bits == 8:
        R_q = np.frombuffer(raw, dtype=np.int8).reshape(n, n).astype(np.float32)
    else:
        R_q = np.frombuffer(raw, dtype=np.int16).reshape(n, n).astype(np.float32)

    # Dequantisation + variance denormalisation
    scale = (q_hi - q_lo) / (n_levels - 1)
    R = ((R_q + half) * scale + q_lo) * sigma

    return pred + R


# ─────────────────────────────────────────────────────────────────
#  Public API: compress / decompress
# ─────────────────────────────────────────────────────────────────


def compress(
    chunk: np.ndarray,
    window: int = 8,
    quantize_bits: int = 8,
    clip_percentile: float = 0.5,
    zlib_level: int = 9,
    sigma_floor_ratio: float = 0.05,
    downsample_factor: int = 2,
    preset: str = None,
) -> list:
    """Compress a HEALPix chunk stored in Z-order into a list of numpy arrays.

    This is the main entry point of the compression API.  Internally it
    converts the 1-D Z-order chunk to a 2-D tile, delegates to
    :func:`encode`, and returns the encoded list.

    Parameters
    ----------
    chunk : numpy.ndarray
        1-D (or batched) array in Z-order (Morton) encoding, as stored in
        HEALPix NESTED scheme tiles.
    window : int, optional
        Window size for local variance estimation, by default 8.
        See :func:`encode` for details.
    quantize_bits : {8, 16}, optional
        Quantisation bit depth, by default 8.
        See :func:`encode` for details.
    clip_percentile : float, optional
        Symmetric percentile for residual clipping, by default 0.5.
        See :func:`encode` for details.
    zlib_level : int, optional
        zlib compression level (0–9), by default 9.
        See :func:`encode` for details.
    sigma_floor_ratio : float, optional
        Minimum sigma as a fraction of ``std(R)``, by default 0.05.
        See :func:`encode` for details.
    downsample_factor : {1, 2, 4}, optional
        Low-frequency layer downsampling factor, by default 2.
        See :func:`encode` for details.
    preset : {'low', 'medium', 'high', 'ultra'} or None, optional
        Named quality preset, by default ``None``.  When provided, its
        values populate all parameters not explicitly set by the caller.
        Any parameter passed explicitly alongside a preset takes precedence
        over the preset value for that parameter.

    Returns
    -------
    list of numpy.ndarray
        Encoded list as returned by :func:`encode`.  Pass directly to
        :func:`decompress` to reconstruct the chunk.

    Raises
    ------
    ValueError
        If *preset* is not one of the keys in :data:`QUALITY_PRESETS`.

    See Also
    --------
    decompress : Inverse of this function.
    encode : Low-level encoder (operates on 2-D tiles).
    QUALITY_PRESETS : Full parameter tables for each named preset.
    compression_info : Inspect size and ratio statistics after encoding.

    Notes
    -----
    **Parameter priority.**  When *preset* is given, its values are used
    as defaults.  Any parameter whose value differs from the function
    default overrides the corresponding preset value.

    **Effect on output size** (rough ordering, most to least impactful):

    * ``downsample_factor`` 2 → 4 : low-res layer shrinks by 4×.
    * ``quantize_bits`` 16 → 8 : residual block halves in raw size.
    * ``clip_percentile`` increase : narrower quantisation range,
      better zlib ratio.
    * ``zlib_level`` increase : smaller output, no quality change.
    * ``sigma_floor_ratio`` increase : smoother residual, better
      zlib ratio.

    Examples
    --------
    Compress with the ``"high"`` preset:

    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> chunk = rng.standard_normal(64 * 64).astype(np.float32)
    >>> enc = compress(chunk, preset="high")

    Override one parameter from a preset:

    >>> enc = compress(chunk, preset="high", zlib_level=9)

    Fine-grained control without a preset:

    >>> enc = compress(
    ...     chunk,
    ...     quantize_bits=16,
    ...     clip_percentile=0.1,
    ...     zlib_level=6,
    ...     downsample_factor=2,
    ... )
    """
    kwargs = dict(
        window=window,
        quantize_bits=quantize_bits,
        clip_percentile=clip_percentile,
        zlib_level=zlib_level,
        sigma_floor_ratio=sigma_floor_ratio,
        downsample_factor=downsample_factor,
    )
    if preset is not None:
        if preset not in QUALITY_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. Choices: {list(QUALITY_PRESETS)}"
            )
        base = QUALITY_PRESETS[preset].copy()
        defaults = dict(
            window=8,
            quantize_bits=8,
            clip_percentile=0.5,
            zlib_level=9,
            sigma_floor_ratio=0.05,
            downsample_factor=2,
        )
        for k, v in kwargs.items():
            if v != defaults[k]:   # caller explicitly changed this parameter
                base[k] = v
        kwargs = base

    internal_data = _expand_tile(chunk)
    return encode(internal_data, **kwargs)


def decompress(compressed: list) -> np.ndarray:
    """Decompress a chunk produced by :func:`compress`.

    Parameters
    ----------
    compressed : list of numpy.ndarray
        Encoded list as returned by :func:`compress`.

    Returns
    -------
    numpy.ndarray
        Reconstructed chunk in Z-order encoding, same shape and dtype
        (float32) as the original input to :func:`compress`.

    See Also
    --------
    compress : Inverse of this function.
    decode : Low-level decoder (operates on 2-D tiles).

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chunk = rng.standard_normal(64 * 64).astype(np.float32)
    >>> enc = compress(chunk, preset="medium")
    >>> rec = decompress(enc)
    >>> rec.shape == chunk.shape
    True
    """
    reconstructed = decode(compressed)
    return _flatten_tile(reconstructed)


# ─────────────────────────────────────────────────────────────────
#  Utility: compression statistics
# ─────────────────────────────────────────────────────────────────


def compression_info(enc: list, original_n: int = None) -> dict:
    """Return size and compression-ratio statistics for an encoded chunk.

    Parameters
    ----------
    enc : list of numpy.ndarray
        Encoded list as returned by :func:`compress` or :func:`encode`.
    original_n : int or None, optional
        Side length of the original square data array (number of pixels
        along one axis).  When ``None``, inferred from the metadata stored
        in ``enc[0]``.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``n`` : int
            Side length of the original array.
        ``downsample_factor`` : int
            Downsampling factor used during encoding.
        ``quantize_bits`` : int
            Quantisation bit depth (8 or 16).
        ``window`` : int
            Local variance window size.
        ``low_grid`` : str
            Shape of the stored low-frequency layer as ``"m×m"``.
        ``bytes_low_layer`` : int
            Uncompressed size of the low-frequency layer in bytes
            (float32, 4 bytes per value).
        ``bytes_residual_compressed`` : int
            Size of the zlib-compressed residual block in bytes.
        ``total_bytes`` : int
            Total encoded size in bytes (all six arrays combined).
        ``original_bytes`` : int
            Original data size assuming float32 (4 bytes per pixel).
        ``compression_ratio`` : float
            ``original_bytes / total_bytes``, rounded to 2 decimal places.
        ``size_reduction_pct`` : float
            ``(1 − total_bytes / original_bytes) × 100``, i.e. the
            percentage of the original size saved.

    See Also
    --------
    compress : Produces the encoded list consumed by this function.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chunk = rng.standard_normal(64 * 64).astype(np.float32)
    >>> enc = compress(chunk, preset="medium")
    >>> info = compression_info(enc, original_n=64)
    >>> info["compression_ratio"] > 1.0
    True
    """
    meta = enc[0].astype(int)
    n               = meta[0]
    downsample_factor = meta[1]
    quantize_bits   = meta[2]
    window          = meta[3]

    low_size = n // downsample_factor
    bytes_low      = enc[1].nbytes * 4     # float32 = 4 bytes per value
    bytes_residual = enc[5].nbytes         # already zlib-compressed
    total_bytes    = sum(a.nbytes for a in enc)

    orig_n = original_n if original_n is not None else n
    original_bytes = orig_n * orig_n * 4   # float32
    ratio = original_bytes / total_bytes if total_bytes > 0 else float("inf")

    return {
        "n": n,
        "downsample_factor": downsample_factor,
        "quantize_bits": quantize_bits,
        "window": window,
        "low_grid": f"{low_size}x{low_size}",
        "bytes_low_layer": bytes_low,
        "bytes_residual_compressed": bytes_residual,
        "total_bytes": total_bytes,
        "original_bytes": original_bytes,
        "compression_ratio": round(ratio, 2),
        "size_reduction_pct": round((1 - total_bytes / original_bytes) * 100, 1),
    }
