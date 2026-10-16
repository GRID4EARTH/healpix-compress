"""
Wavelet-based lossy compression for HEALPix tiles.

This module implements a compressor based on a multi-level 2-D Discrete
Wavelet Transform (DWT), following the same principles as JPEG 2000.

Pipeline
--------
1. **DWT** — the 2-D input tile is decomposed with ``pywt.wavedec2`` using
   the ``'periodization'`` boundary mode.  The wavelet separates the tile
   into a pyramid of subbands: the LL approximation concentrates most of the
   signal energy; the detail subbands (LH, HL, HH) capture oriented structure
   at each scale.
2. **LL stored losslessly** — the LL approximation subband (of size
   ``n/2^levels × n/2^levels``, typically 32×32 for *levels* = 5) is stored
   as raw float32.  Because LL is small (≤4 KB) this has negligible impact on
   file size.
3. **Detail subbands uniformly quantised** — the remaining ~99 % of
   coefficients (all detail subbands) are uniformly quantised to int8 or
   int16 with a scale ``q_scale`` derived from the maximum absolute detail
   coefficient, then compressed with zlib.  For smooth signals the detail
   coefficients are sparse (near-zero), so zlib achieves high compression
   ratios — comparable to JPEG 2000.

Why two-tier storage?
~~~~~~~~~~~~~~~~~~~~~
A single global ``q_scale = max(|all coeffs|) / max_level`` is calibrated on
the LL subband (large values).  At this scale the detail coefficients, which
are orders of magnitude smaller, all round to zero, and the LL values are
quantised with only ~127 or ~32 767 levels — causing significant RMSE.
Storing LL in float32 removes this tension: ``q_scale`` is set by the detail
subband alone, giving fine quantisation where it matters.

Binary format (standalone API)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``compressed_bytes`` is structured as::

    ┌─────────────────────────────┬──────────────────┬────────────────────┬──────────────────────┐
    │ header (13 bytes)           │ LL float32       │ scales float32     │ zlib(detail int8 or  │
    │ uint32 n, uint32 ll_numel,  │ (ll_numel × 4 B) │ (one per detail    │ int16), quantised    │
    │ uint32 n_subbands, uint8 0  │ lossless         │ subband)           │ detail subbands      │
    └─────────────────────────────┴──────────────────┴────────────────────┴──────────────────────┘

Algorithm parameters (``wavelet``, ``levels``, ``quantize_bits``) are **not**
stored in the bytes — they must be passed identically to both :func:`compress`
and :func:`decompress`.

Compared to :mod:`hcc.compression.grad_dct`
--------------------------------------------
* No block-boundary artefacts (global DWT vs block DCT).
* LL stored losslessly → lower RMSE at equivalent compressed size.
* Comparable file size to JPEG 2000 lossy at the ``medium`` preset.

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
"""

import struct
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
    # LL is stored losslessly in all presets; quantize_bits applies to
    # detail subbands only.  For smooth astronomical data the detail
    # subbands are sparse → int8 already gives excellent RMSE.
    "low": dict(
        wavelet="db2",
        levels=4,
        quantize_bits=8,
        clip_percentile=0.5,   # kept for API compat; unused in compress()
        zlib_level=9,
    ),
    "medium": dict(
        # Comparable to JPEG 2000 lossy at ~0.2 MB for 1024×1024 tiles.
        wavelet="bior4.4",
        levels=5,
        quantize_bits=8,
        clip_percentile=0.5,
        zlib_level=9,
    ),
    "high": dict(
        # Better than JPEG 2000 lossy; ~0.3–0.6 MB for 1024×1024 tiles.
        wavelet="bior4.4",
        levels=6,
        quantize_bits=16,
        clip_percentile=0.5,
        zlib_level=6,
    ),
    "ultra": dict(
        # Near-lossless; LL exact, detail subbands finely quantised.
        wavelet="bior4.4",
        levels=6,
        quantize_bits=16,
        clip_percentile=0.5,
        zlib_level=1,
    ),
}

# Presets for the zarr codec path (HccDecomposeCodec manages the
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


def _subband_slices(slices: list) -> list:
    """Flatten a pywt ``coeffs_to_array`` slice tree into an ordered list of
    independent 2-D subband slices: ``[LL, *level-L details, ..., *level-1
    details]``. Iterates ``.values()`` rather than hardcoding orientation
    key names (``'da'``/``'ad'``/``'dd'``) since pywt's exact key spelling
    is an implementation detail we don't want to depend on; every per-level
    dict always has exactly the 3 detail orientations for a 2-D wavedec2,
    regardless of naming. The order is fully determined by *slices* itself
    (in turn fully determined by array shape/wavelet/levels, not by data),
    so :func:`compress_residuals` and :func:`decompress_residuals` always
    agree on it without needing to store it.
    """
    return [slices[0]] + [sl for level in slices[1:] for sl in level.values()]


def compress_residuals(
    residuals: npt.NDArray[np.float32],
    prediction: npt.NDArray[np.float32],
    wavelet: str,
    levels: int,
    quantize_bits: int,
    clip_percentile: float,
) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Compress a residual array using DWT and per-subband quantisation.

    Applies a multi-level 2-D DWT to *residuals*, then uniformly quantises
    every *detail* subband (all levels' LH/HL/HH orientations) to int8 or
    int16 using its *own* independent scale factor, rather than one scale
    shared across the whole coefficient array. Subbands in a wavelet
    decomposition have very different magnitude/statistical distributions
    (different detail orientations/levels are not comparable), so
    calibrating a single scale from the array-wide maximum (or array-wide
    percentile) either wastes precision on most subbands to accommodate the
    few with large coefficients, or clips a heavy-tailed subband's genuine
    structure to protect others.

    The **LL approximation subband is stored losslessly** (raw float32),
    exactly as in the standalone :func:`compress`/:func:`decompress` pair
    (see this module's "Why two-tier storage?" docstring section) rather
    than being quantised like the detail subbands: LL is small
    (``1/4**levels`` of the coefficient count) but concentrates most of the
    signal's energy, so quantising it inflates the error far more than
    storing it as float32 costs in bytes.

    Parameters
    ----------
    residuals : numpy.ndarray of shape (n, n), dtype float32
        The array to transform: a whole tile (the
        :class:`~hcc.codecs.grad_dwt.HccGradDwtCodec` use) or a prediction
        residual.
    prediction : numpy.ndarray of shape (n, n), dtype float32
        Accepted for API symmetry with :func:`decompress_residuals`; not
        used.
    wavelet : str
        PyWavelets wavelet name (e.g. ``'db4'``).
    levels : int
        Number of DWT decomposition levels.
    quantize_bits : {8, 16}
        Quantisation depth for detail subbands.  ``8`` → int8 (±127
        levels); ``16`` → int16 (±32 767 levels). Does not affect the LL
        subband, which is always stored losslessly.
    clip_percentile : float
        Tail percentile for each detail subband's scale derivation, applied
        *within* that subband only (not across the whole coefficient
        array).  The ``(100 − clip_percentile)``-th percentile of
        ``|subband coefficients|`` is mapped to the maximum quantised
        level.  ``0`` uses the subband's exact maximum (no clipping).
        Typical range: 0.02–5.0.

    Returns
    -------
    coeffs_q : numpy.ndarray of shape (n, n), dtype int8 or int16
        Quantised DWT *detail* coefficients in the standard quad-tree
        layout; the LL region is always zero here (LL is returned
        separately in *ll_values*, not duplicated in this array).
    q_scale : numpy.ndarray of shape (3 * levels,), dtype float32
        One quantisation step size per *detail* subband, in the fixed
        order defined by :func:`_subband_slices` with the LL entry
        dropped (each level's three detail orientations, coarsest level
        first). Must be stored as sidecar metadata and passed unchanged to
        :func:`decompress_residuals`.
    ll_values : numpy.ndarray of shape (n // 2**levels, n // 2**levels),
        dtype float32
        The LL approximation subband, losslessly. Must also be stored as
        sidecar metadata and passed unchanged to
        :func:`decompress_residuals`.

    See Also
    --------
    decompress_residuals : Inverse of this function.
    dwt_forward : Underlying DWT primitive.
    """
    if quantize_bits not in (8, 16):
        # the output dtype is int8 or int16: any other width would pay the
        # int16 storage for a coarser step
        raise ValueError(f"quantize_bits must be 8 or 16, got {quantize_bits!r}")

    coeffs_arr, slices = dwt_forward(residuals.astype(np.float32), wavelet, levels)
    max_level = (2 ** (quantize_bits - 1)) - 1  # 127 or 32 767
    dtype_q = np.int8 if quantize_bits == 8 else np.int16

    subbands = _subband_slices(slices)
    ll_slice = subbands[0]
    detail_subbands = subbands[1:]

    ll_values = np.ascontiguousarray(coeffs_arr[ll_slice], dtype=np.float32)

    coeffs_q = np.zeros_like(coeffs_arr, dtype=dtype_q)
    q_scale = np.empty(len(detail_subbands), dtype=np.float32)

    for i, sl in enumerate(detail_subbands):
        block = coeffs_arr[sl]
        abs_block = np.abs(block)
        # Per-subband clipping (see docstring): a single outlier in ONE
        # subband can no longer drag down the precision of every OTHER
        # subband, since each has its own scale. clip_percentile=0 uses the
        # subband's exact max (no clipping at all, most conservative).
        if clip_percentile > 0 and abs_block.size > 1:
            max_abs = float(np.percentile(abs_block, 100.0 - clip_percentile))
        else:
            max_abs = float(np.max(abs_block)) if abs_block.size else 0.0
        if max_abs == 0.0:
            max_abs = 1.0

        scale = np.float32(max_abs / max_level)
        coeffs_q[sl] = np.round(np.clip(block, -max_abs, max_abs) / scale).astype(dtype_q)
        q_scale[i] = scale

    return coeffs_q, q_scale, ll_values


def decompress_residuals(
    compressed: npt.NDArray[np.integer],
    prediction: npt.NDArray[np.float32],
    q_scale: npt.NDArray[np.float32],
    ll_values: npt.NDArray[np.float32],
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
        Quantised DWT *detail* coefficients as returned by
        :func:`compress_residuals` (LL region ignored -- *ll_values* is
        used instead).
    prediction : numpy.ndarray of shape (n, n), dtype float32
        Accepted for API symmetry; not used.
    q_scale : numpy.ndarray of shape (3*levels,), dtype float32
        Per-detail-subband quantisation step sizes from
        :func:`compress_residuals`, in the fixed order produced by
        :func:`_subband_slices` with the LL entry dropped (each level's
        three detail orientations, coarsest first).
    ll_values : numpy.ndarray of shape (n // 2**levels, n // 2**levels),
        dtype float32
        Losslessly-stored LL approximation subband from
        :func:`compress_residuals`.
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
    # q_scale/ll_values come back from the Zarr scalar sidecar
    # (hcc.codecs.scalars) as flat float32 arrays: restore their shapes.
    slices = _coeff_slices(compressed.shape, wavelet, levels)
    subbands = _subband_slices(slices)
    ll_slice = subbands[0]
    detail_subbands = subbands[1:]

    q_scale = np.ravel(np.asarray(q_scale, dtype=np.float32))
    if q_scale.size != len(detail_subbands):
        raise ValueError(
            f"q_scale has {q_scale.size} element(s) but {len(detail_subbands)} "
            f"detail subbands are expected for wavelet={wavelet!r}, "
            f"levels={levels} on a {compressed.shape} array. q_scale must "
            "come from the matching compress_residuals() call (same "
            "wavelet/levels/shape)."
        )

    ll_shape = tuple(s.stop - (s.start or 0) for s in ll_slice)
    ll_values = np.ravel(np.asarray(ll_values, dtype=np.float32)).reshape(ll_shape)

    coeffs_float = np.empty(compressed.shape, dtype=np.float32)
    coeffs_float[ll_slice] = ll_values
    for sl, scale in zip(detail_subbands, q_scale):
        coeffs_float[sl] = compressed[sl].astype(np.float32) * float(scale)

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
    subband naturally encodes low-frequency content, and the detail subbands
    are normalised per subband, quantised and compressed with zlib.

    Parameters are **not** embedded in the returned bytes.  Store them in the
    array metadata alongside the compressed data and pass them unchanged to
    :func:`decompress`.

    Parameters
    ----------
    data : numpy.ndarray
        1-D array in Z-order (Morton) encoding, or a 2-D tile, dtype float32.
    wavelet : str, default ``'db4'``
        PyWavelets wavelet name.  Good choices:

        * ``'db2'`` — fastest, moderate quality.
        * ``'db4'`` — good default.
        * ``'db6'`` or ``'bior4.4'`` — highest quality, closest to JPEG 2000.
    levels : int, default 5
        Number of DWT decomposition levels.  Must satisfy
        ``2**levels ≤ n / filter_length``.  Typical range: 3–6.
    quantize_bits : {8, 16}, default 8
        Quantisation depth.  ``8`` → int8, ``16`` → int16.
    clip_percentile : float, default 0.5
        Accepted for API symmetry with :func:`compress_residuals`; the
        standalone path scales each detail subband by its maximum.
    zlib_level : int, default 9
        zlib compression level (0–9).
    **kwargs
        Accepted for API symmetry with other codecs; ignored.

    Returns
    -------
    compressed_bytes : bytes
        Header, lossless LL subband, per-subband scales and the
        zlib-compressed quantised detail coefficients.
    n : int
        Tile side length.
    q_scale : float
        Always ``1.0``; the per-subband scales are embedded in the bytes.

    Raises
    ------
    ValueError
        If *quantize_bits* is not 8 or 16, or *levels* exceeds the safe
        maximum for the tile size and wavelet.

    See Also
    --------
    decompress : Inverse of this function.
    lookup_preset : Resolve a named preset into an explicit parameter dict.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chunk = rng.standard_normal(1024 * 1024).astype(np.float32)
    >>> params = lookup_preset("medium")
    >>> enc = compress(chunk, **params)
    >>> isinstance(enc, tuple) and len(enc) == 3
    True
    """
    if quantize_bits not in (8, 16):
        raise ValueError(f"quantize_bits must be 8 or 16; got {quantize_bits!r}.")

    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        tile_2d = data
    else:
        tile_2d = expand_tile(data)
    n = tile_2d.shape[0]

    max_lv = _max_levels(n, wavelet)
    if levels > max_lv:
        raise ValueError(
            f"levels={levels} exceeds the safe maximum {max_lv} "
            f"for a {n}×{n} tile with wavelet '{wavelet}'."
        )

    coeffs_arr, _ = dwt_forward(tile_2d, wavelet, levels)
    return _compress_zlib(coeffs_arr, n, wavelet, levels, quantize_bits, zlib_level)


def _compress_zlib(
    coeffs_arr: npt.NDArray[np.float32],
    n: int,
    wavelet: str,
    levels: int,
    quantize_bits: int,
    zlib_level: int,
) -> tuple[bytes, int, float]:
    """Per-subband normalisation, quantisation and zlib for :func:`compress`."""
    import pywt as _pywt

    slices = _coeff_slices((n, n), wavelet, levels)

    # Decompose the flat coefficient array back into the pywt subband list
    # so we can normalise each subband independently.
    coeffs_list = _pywt.array_to_coeffs(
        coeffs_arr.astype(np.float64), slices, output_format="wavedec2"
    )
    # coeffs_list[0]   = LL approximation  (shape n_ll × n_ll)
    # coeffs_list[i>0] = (cH, cV, cD) tuple at decomposition level i

    # ── LL: stored losslessly as float32 ─────────────────────────────
    ll_values = np.asarray(coeffs_list[0], dtype=np.float32)  # (n_ll, n_ll)

    # ── Detail subbands: per-subband normalisation ────────────────────
    # Each subband is independently scaled to fill ±max_int.  This mirrors
    # JPEG 2000's approach and prevents fine-scale subbands (small values)
    # from being rounded to zero because of a q_scale dominated by
    # coarse-scale subbands (large values).
    max_int = (2 ** (quantize_bits - 1)) - 1   # 127 or 32 767
    subband_scales: list[np.float32] = []
    norm_list: list = [np.zeros_like(ll_values, dtype=np.float64)]  # LL → 0

    for detail_tuple in coeffs_list[1:]:        # (cH, cV, cD) per level
        norm_details = []
        for subband in detail_tuple:
            scale = float(np.max(np.abs(subband)))
            if scale == 0.0:
                scale = 1.0
            subband_scales.append(np.float32(scale))
            norm_details.append(subband * (max_int / scale))
        norm_list.append(tuple(norm_details))

    # Re-pack normalised subbands into a single flat array, then quantise
    norm_flat, _ = _pywt.coeffs_to_array(norm_list)
    norm_flat = np.asarray(norm_flat, dtype=np.float32)
    dtype_q = np.int8 if quantize_bits == 8 else np.int16
    detail_q = np.round(np.clip(norm_flat, -max_int, max_int)).astype(dtype_q)
    # LL region in detail_q is 0 (norm_list[0] is all zeros)

    # ── Binary format ─────────────────────────────────────────────────
    # header (13 B) : uint32 n, uint32 ll_numel, uint32 n_subbands, uint8 backend=0
    # ll_float32    : ll_numel × 4 B  (lossless)
    # scales_float32: n_subbands × 4 B  (one per detail subband)
    # payload       : zlib(detail_q)
    ll_bytes     = ll_values.ravel().tobytes()
    scales_bytes = np.array(subband_scales, dtype=np.float32).tobytes()
    payload      = zlib.compress(detail_q.tobytes(), level=zlib_level)
    header       = struct.pack("<IIIB", n, ll_values.size, len(subband_scales), 0)
    compressed_bytes = header + ll_bytes + scales_bytes + payload

    # q_scale = 1.0 sentinel; all per-subband scales are embedded in the bytes
    return compressed_bytes, n, 1.0


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
        Pass it to :func:`~hcc.compression.zorder.expand_tile` to recover the
        2-D spatial tile.

    Raises
    ------
    ValueError
        If the payload was written by an unknown entropy back-end.

    See Also
    --------
    compress : Inverse of this function.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> chunk = rng.standard_normal(1024 * 1024).astype(np.float32)
    >>> params = lookup_preset("medium")
    >>> enc = compress(chunk, **params)
    >>> rec = decompress(enc, **params)
    >>> rec.shape == chunk.shape
    True
    """
    compressed_bytes, n, _ = compressed_data   # q_scale ignored (embedded)

    _HDR = struct.calcsize("<IIIB")
    n_stored, ll_numel, n_subbands, backend = struct.unpack("<IIIB", compressed_bytes[:_HDR])
    if backend != 0:
        raise ValueError(f"unknown entropy back-end tag {backend}; only zlib (0) is supported.")

    return _decompress_zlib(
        compressed_bytes, _HDR, n_stored, ll_numel, n_subbands, wavelet, levels, quantize_bits
    )


def _decompress_zlib(
    compressed_bytes: bytes,
    hdr_size: int,
    n: int,
    ll_numel: int,
    n_subbands: int,
    wavelet: str,
    levels: int,
    quantize_bits: int,
) -> npt.NDArray[np.float32]:
    """Inverse of :func:`_compress_zlib`."""
    import pywt as _pywt

    # ── Read LL float32 (lossless) ────────────────────────────────────
    ll_end = hdr_size + ll_numel * 4
    ll_values = np.frombuffer(compressed_bytes[hdr_size:ll_end], dtype=np.float32).copy()
    ll_side = int(round(ll_numel ** 0.5))

    # ── Read per-subband scales ────────────────────────────────────────
    scales_end = ll_end + n_subbands * 4
    subband_scales = np.frombuffer(
        compressed_bytes[ll_end:scales_end], dtype=np.float32
    ).copy()

    # ── Decompress quantised detail coefficients ──────────────────────
    dtype_q = np.int8 if quantize_bits == 8 else np.int16
    detail_q = np.frombuffer(
        zlib.decompress(compressed_bytes[scales_end:]), dtype=dtype_q
    ).reshape(n, n)

    # ── Denormalise per subband and restore LL ────────────────────────
    slices = _coeff_slices((n, n), wavelet, levels)
    max_int = (2 ** (quantize_bits - 1)) - 1

    # Split the quantised flat array back into subband list
    norm_list = _pywt.array_to_coeffs(
        detail_q.astype(np.float64), slices, output_format="wavedec2"
    )
    # norm_list[0] is the LL region (all 0s); replace with stored float32 LL
    restored_list: list = [ll_values.reshape(ll_side, ll_side).astype(np.float64)]

    scale_idx = 0
    for detail_tuple in norm_list[1:]:          # (cH, cV, cD) per level
        denorm = []
        for subband_norm in detail_tuple:
            scale = float(subband_scales[scale_idx])
            denorm.append(subband_norm * (scale / max_int))
            scale_idx += 1
        restored_list.append(tuple(denorm))

    # Inverse DWT directly from the subband list (skip the array round-trip)
    tile_2d = _pywt.waverec2(
        restored_list, wavelet=wavelet, mode="periodization"
    ).astype(np.float32)
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

        Standalone presets (``codec=False``).  LL is always stored
        losslessly; ``quantize_bits`` applies only to the detail subbands.

        +----------+----------+--------+---------------+------------+---------------------------+
        | name     | wavelet  | levels | quantize_bits | zlib_level | expected size (1024²)     |
        +==========+==========+========+===============+============+===========================+
        | low      | db2      | 4      | 8             | 9          | ~80–150 KB                |
        +----------+----------+--------+---------------+------------+---------------------------+
        | medium   | bior4.4  | 5      | 8             | 9          | ~150–300 KB (≈ JPEG 2000) |
        +----------+----------+--------+---------------+------------+---------------------------+
        | high     | bior4.4  | 6      | 16            | 6          | ~300–600 KB               |
        +----------+----------+--------+---------------+------------+---------------------------+
        | ultra    | bior4.4  | 6      | 16            | 1          | ~400–800 KB               |
        +----------+----------+--------+---------------+------------+---------------------------+

        Codec presets (``codec=True``) apply to the residual of
        :class:`~hcc.codecs.decompose.HccDecomposeCodec` and additionally
        include its ``downsample_factor``.

    codec : bool, default False
        If ``True``, return the codec-path preset.

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
    'bior4.4'
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
        ``ll_numel`` : int
            Number of losslessly stored LL coefficients.
        ``n_subbands`` : int
            Number of detail subbands.
        ``bytes_header_ll_scales`` : int
            Size of the header, LL subband and per-subband scales.
        ``bytes_detail_raw`` : int
            Uncompressed size of the quantised detail coefficients.
        ``bytes_detail_compressed`` : int
            zlib-compressed size of the detail coefficients.
        ``total_bytes`` : int
            Total compressed size in bytes.
        ``original_bytes`` : int
            Uncompressed float32 tile size (n² × 4).
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
    >>> chunk = rng.standard_normal(1024 * 1024).astype(np.float32)
    >>> params = lookup_preset("medium")
    >>> enc = compress(chunk, **params)
    >>> info = compression_info(enc, quantize_bits=params["quantize_bits"])
    >>> info["compression_ratio"] > 1.0
    True
    """
    compressed_bytes, n, _ = compressed_data

    _HDR = struct.calcsize("<IIIB")
    _, ll_numel, n_subbands, _ = struct.unpack("<IIIB", compressed_bytes[:_HDR])

    ll_end    = _HDR + ll_numel * 4
    scales_end = ll_end + n_subbands * 4
    detail_compressed_size = len(compressed_bytes) - scales_end

    bytes_per_coeff = 1 if quantize_bits == 8 else 2
    detail_raw_size  = (n * n - ll_numel) * bytes_per_coeff
    original_bytes   = n * n * 4          # float32
    total_bytes      = len(compressed_bytes)

    ratio = original_bytes / total_bytes if total_bytes > 0 else float("inf")

    return {
        "n": int(n),
        "ll_numel": int(ll_numel),
        "n_subbands": int(n_subbands),
        "bytes_header_ll_scales": int(scales_end),
        "bytes_detail_raw": int(detail_raw_size),
        "bytes_detail_compressed": int(detail_compressed_size),
        "total_bytes": int(total_bytes),
        "original_bytes": int(original_bytes),
        "compression_ratio": round(ratio, 2),
        "size_reduction_pct": round((1 - total_bytes / original_bytes) * 100, 1),
    }
