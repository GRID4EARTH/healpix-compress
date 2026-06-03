"""
Wavelet-based lossy compression for HEALPix tiles.

This module implements a compressor based on a multi-level 2-D Discrete
Wavelet Transform (DWT), following the same principles as JPEG 2000.  The
pipeline has two stages:

1. **DWT** — the 2-D input tile is decomposed with ``pywt.wavedec2`` using
   the ``'periodization'`` boundary mode.  This mode guarantees that the
   coefficient array has *exactly* the same shape as the input, so no padding
   metadata needs to be stored.  The wavelet naturally separates the tile into
   multi-scale subbands: the low-frequency approximation (LL) concentrates
   most of the signal energy, while the detail subbands (LH, HL, HH) carry
   progressively finer structure.
2. **Uniform quantisation + entropy coding** — all coefficients are uniformly
   quantised to int8 or int16 with a single global scale ``q_scale``, then
   compressed with zlib.  Because the DWT concentrates energy and produces
   sparse detail subbands, zlib achieves much better ratios on the quantised
   coefficients than on the raw spatial data.

Compared to :mod:`hcc.compression.grad_dct`:

* **No separate low-frequency layer** — the DWT approximation subband already
  encodes the low-frequency content; there is no need to store a downsampled
  float32 copy alongside the residuals.
* **Better energy compaction** — wavelet bases have longer support than DCT
  blocks and avoid blocking artefacts at subband boundaries.
* **Simpler binary format** — the compressed tuple contains only the zlib
  blob, the tile size ``n``, and the scalar ``q_scale``.

Public API
----------
compress : Compress a Z-order HEALPix chunk.
decompress : Reconstruct a chunk from its compressed representation.
lookup_preset : Return the parameter dict for a named quality preset.
compression_info : Report size and compression-ratio statistics.

Lower-level helpers
-------------------
dwt_forward : Apply 2-D DWT and return a flat coefficient array.
dwt_inverse : Reconstruct from a flat coefficient array via 2-D IDWT.
compress_residuals : Quantise a residual array via DWT (zarr codec interface).
decompress_residuals : Reconstruct a residual from quantised DWT coefficients.

Design notes
------------
Algorithm parameters (``wavelet``, ``levels``, ``quantize_bits``) are
**not** stored in the compressed bytes.  They must be passed identically to
both :func:`compress` and :func:`decompress` and stored in the caller's
metadata (e.g. ``zarr.json``).

Only two data-dependent scalars are stored per chunk:

* ``n`` — tile side length (uint32, 4 bytes).
* ``q_scale`` — uniform quantisation step size (float32, 4 bytes).

The ``'periodization'`` DWT boundary mode ensures the coefficient array has
exactly shape ``(n, n)``, so the coefficient layout can be reconstructed from
``(n, wavelet, levels)`` alone — see :func:`_coeff_slices`.
"""

import zlib
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from hcc.compression.zorder import expand_tile, flatten_tile

__all__ = [
    "compress",
    "decompress",
    "lookup_preset",
    "compression_info",
    "dwt_forward",
    "dwt_inverse",
    "compress_residuals",
    "decompress_residuals",
]


# ─────────────────────────────────────────────────────────────────
#  Quality presets
# ─────────────────────────────────────────────────────────────────

_PRESETS: dict[str, dict[str, Any]] = {
    "low": dict(
        wavelet="db2",
        levels=4,
        quantize_bits=8,
        clip_percentile=3.0,
        zlib_level=9,
    ),
    "medium": dict(
        wavelet="db4",
        levels=5,
        quantize_bits=8,
        clip_percentile=0.5,
        zlib_level=9,
    ),
    "high": dict(
        wavelet="db4",
        levels=6,
        quantize_bits=16,
        clip_percentile=0.1,
        zlib_level=6,
    ),
    "ultra": dict(
        wavelet="db6",
        levels=6,
        quantize_bits=16,
        clip_percentile=0.02,
        zlib_level=1,
    ),
}

# Presets for the zarr codec path (HccDecomposeCodec still manages the
# low-frequency layer; levels here apply to the *residual* DWT).
_CODEC_PRESETS: dict[str, dict[str, Any]] = {
    "low": dict(
        wavelet="db2",
        levels=3,
        quantize_bits=8,
        clip_percentile=3.0,
        zlib_level=9,
        downsample_factor=4,
    ),
    "medium": dict(
        wavelet="db4",
        levels=4,
        quantize_bits=8,
        clip_percentile=0.5,
        zlib_level=9,
        downsample_factor=2,
    ),
    "high": dict(
        wavelet="db4",
        levels=5,
        quantize_bits=16,
        clip_percentile=0.1,
        zlib_level=6,
        downsample_factor=2,
    ),
    "ultra": dict(
        wavelet="db6",
        levels=6,
        quantize_bits=16,
        clip_percentile=0.02,
        zlib_level=1,
        downsample_factor=1,
    ),
}


# ─────────────────────────────────────────────────────────────────
#  Internal: DWT coefficient layout (cached)
# ─────────────────────────────────────────────────────────────────


@lru_cache(maxsize=64)
def _coeff_slices(shape: tuple[int, int], wavelet: str, levels: int):
    """Return the DWT coefficient slices for the given parameters.

    The slice layout produced by :func:`pywt.coeffs_to_array` depends only
    on *shape*, *wavelet*, and *levels*.  This function computes it once per
    unique parameter set and caches the result.

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of the square input array ``(n, n)``.
    wavelet : str
        PyWavelets wavelet name (e.g. ``'db4'``).
    levels : int
        Number of DWT decomposition levels.

    Returns
    -------
    slices : list
        Slice metadata as returned by :func:`pywt.coeffs_to_array`.  Pass
        directly to :func:`pywt.array_to_coeffs` during reconstruction.
    """
    import pywt

    dummy = np.zeros(shape, dtype=np.float32)
    coeffs = pywt.wavedec2(dummy, wavelet=wavelet, level=levels, mode="periodization")
    _, slices = pywt.coeffs_to_array(coeffs)
    return slices


def _max_levels(n: int, wavelet: str) -> int:
    """Return the maximum safe DWT decomposition level for an ``n×n`` tile.

    Ensures that the coarsest subband is at least as wide as the wavelet
    filter, preventing boundary artefacts in the ``'periodization'`` mode.

    Parameters
    ----------
    n : int
        Tile side length.
    wavelet : str
        PyWavelets wavelet name.

    Returns
    -------
    int
        Maximum safe level.
    """
    import pywt

    filter_len = pywt.Wavelet(wavelet).dec_len
    max_lv = 0
    size = n
    while size >= filter_len:
        size //= 2
        max_lv += 1
    return max_lv


# ─────────────────────────────────────────────────────────────────
#  DWT primitives
# ─────────────────────────────────────────────────────────────────


def dwt_forward(
    data: npt.NDArray[np.float32],
    wavelet: str,
    levels: int,
) -> tuple[npt.NDArray[np.float32], list]:
    """Apply a 2-D multi-level DWT to *data*.

    The ``'periodization'`` boundary mode is used so that the output
    coefficient array has *exactly* the same shape as the input.  The
    multi-level decomposition produces a pyramid of subbands:

    * **LL** (approximation) at the coarsest level — captures low-frequency
      energy; size ``n // 2**levels × n // 2**levels``.
    * **LH, HL, HH** (detail) at each level — capture oriented edges and
      texture at that scale.

    Parameters
    ----------
    data : numpy.ndarray of shape (n, n), dtype float32
        Square 2-D input tile.  *n* should be a power of two, and *levels*
        must not exceed :func:`_max_levels`.
    wavelet : str
        PyWavelets wavelet name.  Recommended choices:

        * ``'db2'`` — fast, moderate quality (Daubechies 2).
        * ``'db4'`` — good quality/speed trade-off; default.
        * ``'db6'`` — high quality, slightly slower.
        * ``'bior4.4'`` — biorthogonal, symmetric; used in JPEG 2000.
    levels : int
        Number of decomposition levels.  Typical range: 3–6.  More levels
        capture lower-frequency structure at the cost of slightly larger
        filter boundary effects.

    Returns
    -------
    coeffs_arr : numpy.ndarray of shape (n, n), dtype float32
        All DWT coefficients arranged in a single array.  The LL subband
        occupies the top-left corner; detail subbands fill the rest in a
        standard quad-tree layout.
    slices : list
        Slice metadata needed to reconstruct the coefficient list from the
        flat array.  Pass to :func:`dwt_inverse` or :func:`pywt.array_to_coeffs`.

    See Also
    --------
    dwt_inverse : Exact inverse of this function.
    _coeff_slices : Cached version that computes slices without the data.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal((64, 64)).astype(np.float32)
    >>> coeffs, slices = dwt_forward(x, wavelet='db4', levels=3)
    >>> coeffs.shape
    (64, 64)
    """
    import pywt

    coeffs = pywt.wavedec2(
        data.astype(np.float64), wavelet=wavelet, level=levels, mode="periodization"
    )
    arr, slices = pywt.coeffs_to_array(coeffs)
    return arr.astype(np.float32), slices


def dwt_inverse(
    coeffs_arr: npt.NDArray,
    slices: list,
    wavelet: str,
    levels: int,
) -> npt.NDArray[np.float32]:
    """Reconstruct a 2-D array from DWT coefficients.

    This is the exact inverse of :func:`dwt_forward`:
    ``dwt_inverse(dwt_forward(x, w, l)[0], dwt_forward(x, w, l)[1], w, l)``
    recovers *x* up to floating-point rounding.

    Parameters
    ----------
    coeffs_arr : numpy.ndarray of shape (n, n)
        Flat coefficient array as produced by :func:`dwt_forward`.
    slices : list
        Slice metadata as returned by :func:`dwt_forward` or
        :func:`_coeff_slices`.
    wavelet : str
        PyWavelets wavelet name — must match the value used at
        :func:`dwt_forward` time.
    levels : int
        Number of decomposition levels — must match :func:`dwt_forward`.

    Returns
    -------
    numpy.ndarray of shape (n, n), dtype float32
        Reconstructed spatial-domain tile.

    See Also
    --------
    dwt_forward : Forward transform.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal((64, 64)).astype(np.float32)
    >>> coeffs, slices = dwt_forward(x, 'db4', 3)
    >>> np.allclose(x, dwt_inverse(coeffs, slices, 'db4', 3), atol=1e-4)
    True
    """
    import pywt

    coeffs = pywt.array_to_coeffs(
        coeffs_arr.astype(np.float64), slices, output_format="wavedec2"
    )
    return pywt.waverec2(coeffs, wavelet=wavelet, mode="periodization").astype(
        np.float32
    )


# ─────────────────────────────────────────────────────────────────
#  Residual codec  (zarr ArrayArrayCodec interface)
# ─────────────────────────────────────────────────────────────────


def compress_residuals(
    residuals: npt.NDArray[np.float32],
    prediction: npt.NDArray[np.float32],
    wavelet: str,
    levels: int,
    quantize_bits: int,
    clip_percentile: float,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.float32]]:
    """Compress a residual array using DWT and uniform quantisation.

    Applies a multi-level 2-D DWT to *residuals*, then uniformly quantises
    all coefficients to int8 or int16.  A single global scale factor
    ``q_scale`` is derived from the coefficient distribution and stored as
    metadata.

    Parameters
    ----------
    residuals : numpy.ndarray of shape (n, n), dtype float32
        Residual array ``original − bilinear_prediction``.
    prediction : numpy.ndarray of shape (n, n), dtype float32
        Bilinear prediction used to produce *residuals*.  Accepted for API
        symmetry with :func:`decompress_residuals` but not used here.
    wavelet : str
        PyWavelets wavelet name (e.g. ``'db4'``).
    levels : int
        Number of DWT decomposition levels.
    quantize_bits : {8, 16}
        Quantisation depth.  ``8`` → int8 (±127 levels);
        ``16`` → int16 (±32 767 levels).
    clip_percentile : float
        Tail percentile for ``q_scale`` derivation.  The
        ``(100 − clip_percentile)``-th percentile of ``|coefficients|`` is
        mapped to the maximum quantised level.  Typical range: 0.02–5.0.

    Returns
    -------
    coeffs_q : numpy.ndarray of shape (n, n), dtype int8 or int16
        Quantised DWT coefficients in the standard quad-tree layout.
    q_scale : numpy.ndarray of shape (), dtype float32
        Quantisation step size.  Must be stored as sidecar metadata and
        passed unchanged to :func:`decompress_residuals`.

    See Also
    --------
    decompress_residuals : Inverse of this function.
    dwt_forward : Underlying DWT primitive.
    """
    coeffs_arr, _ = dwt_forward(residuals.astype(np.float32), wavelet, levels)

    max_level = (2 ** (quantize_bits - 1)) - 1  # 127 or 32 767
    max_abs = float(np.percentile(np.abs(coeffs_arr), 100.0 - clip_percentile))
    if max_abs == 0.0:
        max_abs = 1.0

    q_scale = np.float32(max_abs / max_level)
    coeffs_clipped = np.clip(coeffs_arr, -max_abs, max_abs)
    coeffs_q = np.round(coeffs_clipped / q_scale).astype(
        np.int8 if quantize_bits == 8 else np.int16
    )
    return coeffs_q, np.array(q_scale, dtype=np.float32)


def decompress_residuals(
    compressed: npt.NDArray[np.integer],
    prediction: npt.NDArray[np.float32],
    q_scale: npt.NDArray[np.float32],
    *,
    wavelet: str,
    levels: int,
    quantize_bits: int,
) -> npt.NDArray[np.float32]:
    """Reconstruct a residual from quantised DWT coefficients.

    This is the inverse of :func:`compress_residuals`.

    Parameters
    ----------
    compressed : numpy.ndarray of shape (n, n), dtype int8 or int16
        Quantised DWT coefficients as returned by :func:`compress_residuals`.
    prediction : numpy.ndarray of shape (n, n), dtype float32
        Bilinear prediction.  Accepted for API symmetry; not used here.
    q_scale : numpy.ndarray of shape () or (1,), dtype float32
        Quantisation step size from :func:`compress_residuals`.
    wavelet : str
        PyWavelets wavelet name — must match compress time.
    levels : int
        Number of decomposition levels — must match compress time.
    quantize_bits : {8, 16}
        Quantisation depth — accepted for API symmetry; not used here.

    Returns
    -------
    numpy.ndarray of shape (n, n), dtype float32
        Reconstructed residual array.

    See Also
    --------
    compress_residuals : Inverse of this function.
    """
    coeffs_float = compressed.astype(np.float32) * float(q_scale)
    slices = _coeff_slices(compressed.shape, wavelet, levels)
    return dwt_inverse(coeffs_float, slices, wavelet, levels)


# ─────────────────────────────────────────────────────────────────
#  Public API  (standalone compress / decompress — full tile DWT)
# ─────────────────────────────────────────────────────────────────


def compress(
    data: npt.NDArray[np.float32],
    wavelet: str = "db4",
    levels: int = 5,
    quantize_bits: int = 8,
    clip_percentile: float = 0.5,
    zlib_level: int = 9,
    **kwargs,
) -> tuple[bytes, int, float]:
    """Compress a HEALPix chunk stored in Z-order using a full-tile DWT.

    Unlike :func:`hcc.compression.grad_dct.compress`, this function applies
    the DWT to the *entire* 2-D tile rather than only to a prediction
    residual.  There is no separate low-frequency layer: the DWT approximation
    subband naturally encodes low-frequency content, and detail subbands are
    sparsified and compressed by zlib.

    Parameters are **not** embedded in the returned bytes.  Store them in the
    array metadata alongside the compressed data and pass them unchanged to
    :func:`decompress`.

    Parameters
    ----------
    data : numpy.ndarray
        1-D array in Z-order (Morton) encoding, dtype float32.
    wavelet : str, default ``'db4'``
        PyWavelets wavelet name.  Good choices:

        * ``'db2'`` — fastest, moderate quality.
        * ``'db4'`` — good default.
        * ``'db6'`` or ``'bior4.4'`` — highest quality, closest to JPEG 2000.
    levels : int, default 5
        Number of DWT decomposition levels.  Must satisfy
        ``2**levels ≤ n / filter_length``.  Typical range: 3–6.
    quantize_bits : {8, 16}, default 8
        Quantisation depth.  ``8`` → int8, 1 byte/coefficient;
        ``16`` → int16, 2 bytes/coefficient.
    clip_percentile : float, default 0.5
        Tail percentile for ``q_scale``.  The ``(100 − clip_percentile)``-th
        percentile of ``|coefficients|`` maps to the max quantised level.
        Typical range: 0.02–5.0.
    zlib_level : int, default 9
        zlib compression level (0–9).
    **kwargs
        Accepted for API symmetry with other codecs; ignored.

    Returns
    -------
    compressed_bytes : bytes
        zlib-compressed quantised DWT coefficients.
    n : int
        Tile side length.
    q_scale : float
        Quantisation step size — store in metadata for :func:`decompress`.

    Raises
    ------
    ValueError
        If *levels* exceeds the safe maximum for the tile size and wavelet.

    See Also
    --------
    decompress : Inverse of this function.
    lookup_preset : Resolve a named preset into an explicit parameter dict.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chunk = rng.standard_normal(64 * 64).astype(np.float32)
    >>> params = lookup_preset("medium")
    >>> enc = compress(chunk, **params)
    >>> isinstance(enc, tuple) and len(enc) == 3
    True
    """
    if quantize_bits not in (8, 16):
        raise ValueError("quantize_bits must be 8 or 16")
    if not (0.0 < clip_percentile < 50.0):
        raise ValueError("clip_percentile must be in the open interval (0, 50)")

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        # Already a spatial 2D tile — use as-is.
        tile_2d = data
    else:
        # 1-D Z-order (Morton) encoded array — expand to 2-D.
        tile_2d = expand_tile(data)
    n = tile_2d.shape[0]

    max_lv = _max_levels(n, wavelet)
    if levels > max_lv:
        raise ValueError(
            f"levels={levels} exceeds the safe maximum {max_lv} "
            f"for a {n}×{n} tile with wavelet '{wavelet}'."
        )

    coeffs_arr, _ = dwt_forward(tile_2d, wavelet, levels)

    max_level = (2 ** (quantize_bits - 1)) - 1
    max_abs = float(np.percentile(np.abs(coeffs_arr), 100.0 - clip_percentile))
    if max_abs == 0.0:
        max_abs = 1.0
    q_scale = float(np.float32(max_abs / max_level))

    coeffs_clipped = np.clip(coeffs_arr, -max_abs, max_abs)
    coeffs_q = np.round(coeffs_clipped / q_scale).astype(
        np.int8 if quantize_bits == 8 else np.int16
    )

    compressed_bytes = zlib.compress(coeffs_q.tobytes(), level=zlib_level)
    return compressed_bytes, n, q_scale


def decompress(
    compressed_data: tuple[bytes, int, float],
    wavelet: str = "db4",
    levels: int = 5,
    quantize_bits: int = 8,
    **kwargs,
) -> npt.NDArray[np.float32]:
    """Decompress a tuple produced by :func:`compress`.

    Parameters
    ----------
    compressed_data : tuple
        ``(compressed_bytes, n, q_scale)`` as returned by :func:`compress`.
    wavelet : str, default ``'db4'``
        Wavelet name — must match compress time.
    levels : int, default 5
        Number of DWT levels — must match compress time.
    quantize_bits : {8, 16}, default 8
        Quantisation depth — must match compress time.
    **kwargs
        Accepted for API symmetry; ignored.

    Returns
    -------
    numpy.ndarray, dtype float32
        Reconstructed chunk in Z-order (Morton) encoding, shape ``(n²,)``.
        This mirrors the expected input format of :func:`compress`: if you
        passed a 1-D Z-order array to :func:`compress`, you get a 1-D Z-order
        array back here.  Pass it to ``_regrid_2d`` (or
        :func:`~hcc.compression.zorder.expand_tile`) to recover the 2-D
        spatial tile.

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
    compressed_bytes, n, q_scale = compressed_data

    dtype = np.int8 if quantize_bits == 8 else np.int16
    coeffs_q = np.frombuffer(
        zlib.decompress(compressed_bytes), dtype=dtype
    ).reshape(n, n)

    coeffs_float = coeffs_q.astype(np.float32) * float(q_scale)
    slices = _coeff_slices((n, n), wavelet, levels)
    tile_2d = dwt_inverse(coeffs_float, slices, wavelet, levels)
    return flatten_tile(tile_2d)


def lookup_preset(
    name: Literal["low", "medium", "high", "ultra"],
    codec: bool = False,
) -> dict[str, Any]:
    """Return a copy of the parameter dict for a named quality preset.

    Parameters
    ----------
    name : {'low', 'medium', 'high', 'ultra'}
        Quality preset name.

        Standalone presets (``codec=False``):

        +----------+---------+--------+---------------+-----------------+------------+
        | name     | wavelet | levels | quantize_bits | clip_percentile | zlib_level |
        +==========+=========+========+===============+=================+============+
        | low      | db2     | 4      | 8             | 3.0             | 9          |
        +----------+---------+--------+---------------+-----------------+------------+
        | medium   | db4     | 5      | 8             | 0.5             | 9          |
        +----------+---------+--------+---------------+-----------------+------------+
        | high     | db4     | 6      | 16            | 0.1             | 6          |
        +----------+---------+--------+---------------+-----------------+------------+
        | ultra    | db6     | 6      | 16            | 0.02            | 1          |
        +----------+---------+--------+---------------+-----------------+------------+

        Codec presets (``codec=True``) additionally include
        ``downsample_factor`` for use with
        :class:`~hcc.codecs.decompose.HccDecomposeCodec`.

    codec : bool, default False
        If ``True``, return the codec-path preset that includes
        ``downsample_factor``.

    Returns
    -------
    dict[str, Any]
        A fresh copy of the preset's parameter dict.

    Raises
    ------
    ValueError
        If *name* is not a known preset.

    Examples
    --------
    >>> p = lookup_preset("high")
    >>> sorted(p.keys())
    ['clip_percentile', 'levels', 'quantize_bits', 'wavelet', 'zlib_level']
    >>> p["wavelet"]
    'db4'
    """
    table = _CODEC_PRESETS if codec else _PRESETS
    if name not in table:
        raise ValueError(
            f"Unknown preset {name!r}. Available: {list(table)}"
        )
    return table[name].copy()


# ─────────────────────────────────────────────────────────────────
#  Utility: compression statistics
# ─────────────────────────────────────────────────────────────────


def compression_info(
    compressed_data: tuple[bytes, int, float],
    quantize_bits: int = 8,
) -> dict:
    """Return size and compression-ratio statistics for a compressed chunk.

    Parameters
    ----------
    compressed_data : tuple
        ``(compressed_bytes, n, q_scale)`` as returned by :func:`compress`.
    quantize_bits : {8, 16}, default 8
        Quantisation depth used at compress time — needed to report the
        uncompressed coefficient size.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``n`` : int
            Tile side length.
        ``q_scale`` : float
            Quantisation step size.
        ``bytes_coefficients_raw`` : int
            Uncompressed coefficient array size (n² × bytes-per-coeff).
        ``bytes_compressed`` : int
            zlib-compressed size in bytes.
        ``total_bytes`` : int
            Same as ``bytes_compressed`` (no sidecar in standalone format).
        ``original_bytes`` : int
            Uncompressed float32 tile size (n² × 4).
        ``compression_ratio`` : float
            ``original_bytes / total_bytes``, rounded to 2 decimal places.
        ``size_reduction_pct`` : float
            ``(1 − total_bytes / original_bytes) × 100``.
        ``coefficient_sparsity`` : str
            Qualitative hint — ``'dense'`` for int16, ``'sparse'`` for int8.

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
    >>> info = compression_info(enc, quantize_bits=params["quantize_bits"])
    >>> info["compression_ratio"] > 1.0
    True
    """
    compressed_bytes, n, q_scale = compressed_data

    bytes_per_coeff = 1 if quantize_bits == 8 else 2
    bytes_raw = n * n * bytes_per_coeff
    bytes_compressed = len(compressed_bytes)
    original_bytes = n * n * 4  # float32

    ratio = original_bytes / bytes_compressed if bytes_compressed > 0 else float("inf")

    return {
        "n": int(n),
        "q_scale": float(q_scale),
        "bytes_coefficients_raw": bytes_raw,
        "bytes_compressed": bytes_compressed,
        "total_bytes": bytes_compressed,
        "original_bytes": original_bytes,
        "compression_ratio": round(ratio, 2),
        "size_reduction_pct": round((1 - bytes_compressed / original_bytes) * 100, 1),
        "coefficient_sparsity": "sparse" if quantize_bits == 8 else "dense",
    }
