import zlib

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.ndimage import zoom as sp_zoom


# ─────────────────────────────────────────────────────────────────
#  Quality presets
# ─────────────────────────────────────────────────────────────────

QUALITY_PRESETS: dict = {
    # Smallest output — high loss, heavily compressed residuals
    "low": dict(
        quantize_bits=8,
        clip_percentile=3.0,
        zlib_level=9,
        sigma_floor_ratio=0.15,
        downsample_factor=4,
        window=4,
    ),
    # Balanced size/quality — behaviour close to the original code
    "medium": dict(
        quantize_bits=8,
        clip_percentile=0.5,
        zlib_level=9,
        sigma_floor_ratio=0.05,
        downsample_factor=2,
        window=8,
    ),
    # Good quality — 16-bit residuals, less clipping of outliers
    "high": dict(
        quantize_bits=16,
        clip_percentile=0.1,
        zlib_level=6,
        sigma_floor_ratio=0.02,
        downsample_factor=2,
        window=8,
    ),
    # Maximum quality — low-frequency layer is not downsampled
    "ultra": dict(
        quantize_bits=16,
        clip_percentile=0.02,
        zlib_level=1,
        sigma_floor_ratio=0.01,
        downsample_factor=1,
        window=16,
    ),
}


# ─────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────


def _decode_zorder(delta: int):
    """Decode Z-order (Morton) curve indices into (x, y) coordinate arrays."""
    size = 4**delta
    x = np.zeros((size,), dtype="uint64")
    y = np.zeros((size,), dtype="uint64")

    code = np.arange(size, dtype="uint64")
    for bit in range(delta):
        x |= ((code >> (bit << 1)) & 1) << bit
        y |= ((code >> ((bit << 1) + 1)) & 1) << bit

    return x, y


def _flatten_tile(data: np.ndarray):
    """Convert a 2-D tile back to a 1-D Z-order encoded array."""
    delta = int(np.log(data.size) / np.log(4))
    x, y = _decode_zorder(delta)
    return np.squeeze(data[..., x, y])


def _expand_tile(data: np.ndarray):
    """Convert a 1-D Z-order encoded array into a 2-D tile."""
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
    """[n,n] -> [n//2, n//2]: average over non-overlapping 2x2 blocks."""
    n = data.shape[0]
    return data.reshape(n // 2, 2, n // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """
    Generic downsampling [n,n] -> [n//factor, n//factor].

    factor=1  : copy without modification (lossless low layer, larger file)
    factor=2  : original behaviour — good size/quality trade-off
    factor=4  : aggressive downsampling — very small low-frequency layer
    """
    if factor == 1:
        return data.astype(np.float32).copy()
    if factor == 2:
        return block_mean_2x2(data)
    # General case via scipy zoom
    return sp_zoom(data.astype(np.float64), 1.0 / factor, order=1).astype(np.float32)


def upsample(data: np.ndarray, factor: int) -> np.ndarray:
    """[m,m] -> [m*factor, m*factor]: bilinear interpolation."""
    if factor == 1:
        return data.astype(np.float32).copy()
    return sp_zoom(data.astype(np.float64), factor, order=1).astype(np.float32)


def local_std(data: np.ndarray, window: int = 8) -> np.ndarray:
    """
    Local standard-deviation map computed with a uniform filter.
    Uses the identity  Var(X) = E[X^2] - E[X]^2  to stay O(n).
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
    """
    Encode a [n,n] float32 array into a list of compressed numpy arrays.

    Parameters
    ----------
    data              : np.ndarray, shape (n,n), dtype float32
    window            : int  [4-32]
                        Window size for local variance estimation.
                        Larger  -> softer normalisation, larger residuals.
    quantize_bits     : {8, 16}
                        8  -> int8,  256 levels  (1 byte/value)  -- smallest size
                        16 -> int16, 65 536 levels (2 bytes/value) -- higher quality
    clip_percentile   : float [0.05 - 5.0]
                        Percentage clipped at both ends before quantisation.
                        Higher value -> narrower range -> better compression,
                        but more loss at the extremes.
    zlib_level        : int [0-9]
                        0 = no compression (fast, large file)
                        9 = maximum compression (slow, small file)
    sigma_floor_ratio : float [0.005 - 0.20]
                        Sigma floor = ratio * std(R).
                        Small -> preserves smooth regions, larger dynamic range.
                        Large -> smooths noise, less fine detail.
    downsample_factor : {1, 2, 4}
                        Downsampling factor for the low-frequency layer.
                        1 -> no loss on low frequencies (heavier file)
                        2 -> original behaviour (recommended)
                        4 -> low-frequency layer 16x smaller, more loss

    Returns
    -------
    list of 6 np.ndarray (metadata + compressed data)
    """
    assert quantize_bits in (8, 16), "quantize_bits must be 8 or 16"
    assert downsample_factor in (1, 2, 4), "downsample_factor must be 1, 2 or 4"
    assert 0.005 <= sigma_floor_ratio <= 0.5, "sigma_floor_ratio out of range [0.005, 0.5]"
    assert 0.0 < clip_percentile < 50.0, "clip_percentile out of range (0, 50)"

    data = data.astype(np.float32)
    n = data.shape[0]
    assert data.shape == (n, n), "data must be square"
    assert (n & (n - 1)) == 0 and n > 1, "n must be a power of 2"

    # -- 1. Low-frequency layer -----------------------------------------------
    low = downsample(data, factor=downsample_factor)

    # -- 2. Prediction via bilinear upsampling --------------------------------
    pred = upsample(low, factor=downsample_factor)

    # -- 3. Residual -----------------------------------------------------------
    R = data - pred

    # -- 4. Local variance map ------------------------------------------------
    #    sigma(x,y) proportional to local activity in the prediction:
    #    smooth regions -> low sigma  -> little residual expected
    #    active regions -> high sigma -> large residual expected
    sigma = local_std(pred, window=window)
    sigma_floor = float(np.std(R)) * sigma_floor_ratio
    sigma = np.maximum(sigma, sigma_floor)

    # -- 5. Adaptive variance normalisation -----------------------------------
    #    After normalisation R_norm is statistically more white/homogeneous
    #    -> better entropy compressibility
    R_norm = (R / sigma).astype(np.float32)

    # -- 6. Quantisation -------------------------------------------------------
    n_levels = 2**quantize_bits   # 256 or 65536
    half = n_levels // 2          # 128 or 32768

    q_lo = float(np.percentile(R_norm, clip_percentile))
    q_hi = float(np.percentile(R_norm, 100.0 - clip_percentile))
    scale = (q_hi - q_lo) / (n_levels - 1)

    R_clipped = np.clip(R_norm, q_lo, q_hi)
    R_int = np.round((R_clipped - q_lo) / scale - half)

    if quantize_bits == 8:
        R_q = R_int.astype(np.int8)
    else:
        R_q = R_int.astype(np.int16)

    # -- 7. Entropy compression with zlib -------------------------------------
    #    Variance whitening improves zlib compression ratio
    residual_bytes = zlib.compress(R_q.tobytes(), level=zlib_level)

    # -- 8. Metadata (stored so the decoder is self-contained) ----------------
    # enc[0] = [n, downsample_factor, quantize_bits, window]
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
    """
    Reconstruct a [n,n] float32 array from a list of compressed numpy arrays.
    All parameters are read from enc[0] (metadata) — no extra arguments needed.

    Note: the old signature decode(enc, window) is no longer required;
    window is now stored in the metadata.
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
    """
    Compress a HEALPix chunk (Z-order) into a list of numpy arrays.

    Quick usage with a preset
    --------------------------
    >>> enc = compress(chunk, preset="high")

    Fine-grained parameter control
    --------------------------------
    >>> enc = compress(chunk,
    ...                quantize_bits=16,
    ...                clip_percentile=0.1,
    ...                zlib_level=6,
    ...                downsample_factor=2)

    Explicit parameters always override the preset values.

    Available presets : "low", "medium", "high", "ultra"
    See QUALITY_PRESETS for the exact values used by each preset.

    Parameter effect on output size
    --------------------------------
    quantize_bits=8  vs 16  : x2 on residuals (8-bit is half the size)
    downsample_factor=2 vs 4: low-res layer is 4x smaller with factor=4
    clip_percentile higher  : better zlib compression of residuals
    zlib_level higher       : smaller file, slower encoding
    sigma_floor_ratio higher: smoother residuals, better compression
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
        # Start from preset values, then override with explicitly passed parameters
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
            if v != defaults[k]:   # user explicitly changed this parameter
                base[k] = v
        kwargs = base

    internal_data = _expand_tile(chunk)
    return encode(internal_data, **kwargs)


def decompress(compressed: list) -> np.ndarray:
    """
    Decompress a chunk produced by compress().
    No additional parameters required: everything is read from the metadata.
    """
    reconstructed = decode(compressed)
    return _flatten_tile(reconstructed)


# ─────────────────────────────────────────────────────────────────
#  Utility: compression statistics
# ─────────────────────────────────────────────────────────────────


def compression_info(enc: list, original_n: int = None) -> dict:
    """
    Return a dictionary of statistics about a compressed chunk.

    Parameters
    ----------
    enc        : list returned by compress()
    original_n : original chunk side length (number of pixels along one axis);
                 if None, inferred from n stored in the metadata

    Example
    -------
    >>> enc = compress(chunk, preset="medium")
    >>> info = compression_info(enc, original_n=chunk.shape[0])
    >>> print(info)
    {
        'n': 64,
        'downsample_factor': 2,
        'quantize_bits': 8,
        'window': 8,
        'low_grid': '32x32',
        'bytes_low_layer': 4096,
        'bytes_residual_compressed': 1234,
        'total_bytes': 5330,
        'original_bytes': 16384,
        'compression_ratio': 3.07,
        'size_reduction_pct': 67.5
    }
    """
    meta = enc[0].astype(int)
    n               = meta[0]
    downsample_factor = meta[1]
    quantize_bits   = meta[2]
    window          = meta[3]

    low_size = n // downsample_factor
    bytes_low      = enc[1].nbytes * 4     # float32 = 4 bytes
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
