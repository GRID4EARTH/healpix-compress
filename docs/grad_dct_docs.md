# grad_dct — Compression Parameter Reference

This document describes every tunable parameter exposed by the `compress()` / `encode()` functions in `grad_dct.py`, explains the underlying mechanism each one controls, and gives practical guidance on how to use them to trade off **output size** against **reconstruction quality**.

---

## How the compression pipeline works

Understanding the pipeline helps interpret what each parameter does.

```
Original data [n×n float32]
        │
        ▼
 1. Downsample ──────────────────────────────► low-frequency layer (stored lossless)
        │
        ▼
 2. Upsample (bilinear) ────────────────────► prediction [n×n]
        │
        ▼
 3. Residual  =  original − prediction ─────► R [n×n]
        │
        ▼
 4. Local standard-deviation map ───────────► sigma [n×n]
        │
        ▼
 5. Variance normalisation  R_norm = R/sigma  (whitens the signal)
        │
        ▼
 6. Quantise R_norm → int8 or int16
        │
        ▼
 7. zlib entropy compression
        │
        ▼
 Encoded list  [metadata, low, q_lo, q_hi, sigma_floor, residual_bytes]
```

The decoder runs this pipeline in reverse. All parameters needed for decoding are stored inside the encoded list — no extra arguments are required at decode time.

---

## Parameters

### `window` — local variance window size

| | |
|---|---|
| **Type** | `int` |
| **Default** | `8` |
| **Recommended range** | `4` – `32` |

Controls the size of the uniform filter used to compute the local standard-deviation map `sigma`. This map is used in step 5 to normalise the residual so that it has a more uniform statistical distribution before quantisation and entropy coding.

**Effect on quality and size**

- A **larger window** produces a smoother sigma map. This is more accurate for large, slowly-varying structures but less adaptive to sharp local transitions. Residuals in high-detail areas may be slightly larger, but the overall signal is whiter and more compressible.
- A **smaller window** makes the sigma map more local and reactive. Sharp edges get a more accurate local scale, which can reduce quantisation artefacts in those areas.

**Practical guidance**

For most HEALPix scientific maps, `window=8` is a good default. Increase to `16` or `32` for maps with very smooth large-scale structure. Decrease to `4` for maps dominated by point sources or sharp transitions.

---

### `quantize_bits` — quantisation depth

| | |
|---|---|
| **Type** | `int` |
| **Allowed values** | `8`, `16` |
| **Default** | `8` |

Determines the number of discrete levels used to represent the normalised residual `R_norm`.

| Value | Integer type | Number of levels | Bytes per pixel | Notes |
|---|---|---|---|---|
| `8` | `int8` | 256 | 1 | Smallest output, original behaviour |
| `16` | `int16` | 65 536 | 2 | ~256× finer precision, ~2× larger residual block |

**Effect on quality and size**

- `quantize_bits=8` keeps the residual layer at **1 byte per pixel** (before zlib). Suitable for lossy scientific compression where a few percent error is acceptable.
- `quantize_bits=16` raises the precision to **65 536 levels**, reducing quantisation noise by a factor of ~256 in the normalised domain. This is the dominant cost driver when switching from `8` to `16`: the raw residual block doubles in size, though zlib partially recovers this.

**Practical guidance**

Use `8` for storage and transfer where bandwidth is the primary constraint. Use `16` when downstream analysis (e.g. power-spectrum estimation, map differencing) requires low quantisation noise.

---

### `clip_percentile` — outlier clipping percentage

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.5` |
| **Recommended range** | `0.05` – `5.0` |

Before quantisation, the normalised residual `R_norm` is clipped symmetrically: the bottom `clip_percentile` % and the top `clip_percentile` % of values are clamped to the corresponding percentile boundaries `q_lo` and `q_hi`.

This maps the clipped range `[q_lo, q_hi]` onto the full integer range (e.g. `[-128, 127]` for `int8`), which maximises the use of available quantisation levels for the bulk of the data.

**Effect on quality and size**

- A **higher** `clip_percentile` narrows the quantised range. More values fall in the same bin, the distribution is more peaked, and **zlib achieves a better ratio**. However, extreme values (the clipped tail) are hard-clipped and reconstructed with significant error.
- A **lower** `clip_percentile` preserves more of the dynamic range. Extreme values are reconstructed more accurately, but the broader distribution is less compressible.

**Practical guidance**

The default `0.5` clips the 0.5th and 99.5th percentiles. This is a good starting point for smooth cosmological or meteorological maps. For maps with heavy tails (e.g. point-source catalogues, unmasked CMB temperature maps), consider reducing to `0.1` or lower. If output size is the priority and the tails are scientifically unimportant, increase to `1.0`–`3.0`.

---

### `zlib_level` — entropy compression effort

| | |
|---|---|
| **Type** | `int` |
| **Allowed values** | `0` – `9` |
| **Default** | `9` |

Controls the compression level passed directly to Python's `zlib.compress()`. This affects only the entropy coding step (step 7) and has no impact on the quantisation noise or the low-frequency layer.

| Level | Speed | Output size |
|---|---|---|
| `0` | Fastest (no compression) | Largest |
| `1` | Very fast | Large |
| `6` | Moderate | Moderate |
| `9` | Slowest | Smallest |

**Effect on quality and size**

`zlib_level` has **no effect on reconstruction quality** — it is lossless. It only affects how much the already-quantised integer array is compressed. The gain from increasing from level `6` to `9` is typically small (a few percent) for residual arrays that are already well-whitened by variance normalisation.

**Practical guidance**

Keep `zlib_level=9` for archival compression. Drop to `6` or `1` to accelerate batch processing pipelines where CPU time is a bottleneck and slight size increases are acceptable.

---

### `sigma_floor_ratio` — variance normalisation noise floor

| | |
|---|---|
| **Type** | `float` |
| **Default** | `0.05` |
| **Recommended range** | `0.005` – `0.20` |

The local sigma map can be very close to zero in perfectly flat regions of the prediction. Dividing the residual by a near-zero sigma would produce arbitrarily large `R_norm` values. To prevent this, sigma is clamped from below:

```
sigma_floor = std(R) × sigma_floor_ratio
sigma       = max(sigma, sigma_floor)
```

**Effect on quality and size**

- A **smaller** `sigma_floor_ratio` allows very smooth regions to use a finer normalisation scale. This preserves subtle structure in flat areas but may produce large `R_norm` spikes if sigma is very small, hurting compression.
- A **larger** `sigma_floor_ratio` raises the floor, effectively smoothing out residuals in flat areas. This makes the normalised residual more uniform and improves zlib compression at the cost of slightly increased error in flat zones.

**Practical guidance**

The default `0.05` (5% of the global residual standard deviation) works well for most maps. Reduce to `0.01` for maps where flat regions carry scientifically important small-amplitude signals. Increase to `0.10`–`0.15` if compression ratio is the priority and flat-region accuracy is less critical.

---

### `downsample_factor` — low-frequency layer resolution

| | |
|---|---|
| **Type** | `int` |
| **Allowed values** | `1`, `2`, `4` |
| **Default** | `2` |

Determines how aggressively the low-frequency layer is downsampled before storage. This layer is stored **without additional lossy compression** (only zlib on the float32 values), so its size scales with the square of the inverse factor.

| Factor | Low-layer size | Low-layer area relative to original |
|---|---|---|
| `1` | `n × n` | 100 % (no downsampling) |
| `2` | `(n/2) × (n/2)` | 25 % |
| `4` | `(n/4) × (n/4)` | 6.25 % |

**Effect on quality and size**

- `downsample_factor=1`: The low-frequency layer is stored at full resolution. The prediction is exact (prediction = low), so the residual is zero everywhere, and the encoded size is dominated by the raw float32 layer. This mode is essentially lossless at the low-frequency level. It produces the largest files.
- `downsample_factor=2`: The default. The prediction is a bilinear upsample from an `(n/2)×(n/2)` grid. This is the main source of lossy compression. The low layer takes 25% of the original size.
- `downsample_factor=4`: The prediction is upsampled from an `(n/4)×(n/4)` grid. Significantly more structure falls into the residual, which must be captured by the quantised int8/int16 values. Both the low layer (6.25%) and the residual quality matter more.

**Practical guidance**

This is the **single most impactful parameter** for output size. Use `factor=2` as the default. Use `factor=4` only when extreme compression is needed and the map is smooth enough that a coarser prediction still captures most of the signal. Use `factor=1` when you need lossless low frequencies and only want to compress the high-frequency detail.

---

## Using presets

Four presets are provided as convenient starting points. They can be selected with `preset="name"` in `compress()`. Any parameter passed explicitly alongside a preset will override the preset value for that parameter.

| Preset | `quantize_bits` | `clip_percentile` | `zlib_level` | `sigma_floor_ratio` | `downsample_factor` | `window` |
|---|---|---|---|---|---|---|
| `"low"` | 8 | 3.0 | 9 | 0.15 | 4 | 4 |
| `"medium"` | 8 | 0.5 | 9 | 0.05 | 2 | 8 |
| `"high"` | 16 | 0.1 | 6 | 0.02 | 2 | 8 |
| `"ultra"` | 16 | 0.02 | 1 | 0.01 | 1 | 16 |

**Example — use the `"high"` preset but force maximum zlib compression:**

```python
enc = compress(chunk, preset="high", zlib_level=9)
```

---

## Inspecting compression results

Use `compression_info()` to get a detailed breakdown of sizes and ratios after encoding:

```python
enc  = compress(chunk, preset="medium")
info = compression_info(enc, original_n=chunk.shape[0])
print(info)
# {
#     'n': 64,
#     'downsample_factor': 2,
#     'quantize_bits': 8,
#     'window': 8,
#     'low_grid': '32x32',
#     'bytes_low_layer': 4096,
#     'bytes_residual_compressed': 1234,
#     'total_bytes': 5330,
#     'original_bytes': 16384,
#     'compression_ratio': 3.07,
#     'size_reduction_pct': 67.5
# }
```

---

## Quick decision guide

| Goal | Recommended change |
|---|---|
| Reduce output size as much as possible | `downsample_factor=4`, `clip_percentile=2.0`, `quantize_bits=8` |
| Improve reconstruction accuracy | `quantize_bits=16`, `clip_percentile=0.1`, `downsample_factor=2` |
| Faster encoding (same quality) | `zlib_level=1` or `zlib_level=6` |
| Preserve flat-region accuracy | Lower `sigma_floor_ratio` (e.g. `0.01`) |
| Handle maps with heavy tails | Lower `clip_percentile` (e.g. `0.05`–`0.1`) |
| Lossless low frequencies | `downsample_factor=1` |
