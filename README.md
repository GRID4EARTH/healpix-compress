# healpix-based chunk compression

This module contains compression algorithms and codecs to compress / uncompress chunks that are aligned with healpix cells.

The idea is that we can downscale the input chunk by a given amount of refinement levels using a simple aggregation (mean), then compute the residuals between input data and aggregation result, and apply a lossy compression algorithm to the residuals.

The result is multiple arrays (one array for the result of the aggregation, and at least one array of potentially different shape for the residuals), which for the zarr codec are converted to bytes and packed into a single binary blob.

## Parameters

The codec has two parameters:

- `delta`: how many refinement levels to average over. The averages will then have a refinement level of `original_level - delta`.
- `top_k`: how many parameters to keep. This is equivalent to a low-pass filter. The details depend on the actual compression algorithm.

## 16-bit integer data: `hcc-uint16-jxl`

Many products store a variable as unsigned or signed 16-bit integers with a scale and an offset. The `hcc-uint16-jxl` serializer stores a chunk the same way: the cells are rounded to 16-bit digital numbers, `DN = round(Q * x - O)` with `DN = 0` for no-data, de-interleaved into a square raster and compressed with lossless JPEG XL. Decoding restores the quantised values exactly, `x = (DN + O) / Q`; the rounding to the DN lattice is the only error, so data that were already stored as 16-bit integers with that scale and offset round-trip unchanged. The last axis of a chunk must have a power-of-four size (one nested parent cell); leading axes (time, band, ...) are allowed, each row being coded on its own.

`Q` is the number of steps per unit and `O` shifts the values into `[1, 65535]`: the codec holds values from `(1 + O) / Q` to `(65535 + O) / Q`. Finite values outside that range are clipped, with an `OutOfRangeWarning` by default (`out_of_range="raise"` raises instead, `out_of_range="clip"` stays silent). Data stored as floats, or packed with a scale that changes per field or per time step, would be rounded by this codec.

It needs the optional dependency `imagecodecs` (`pip install healpix-compress[jpegxl]`).

```python
import zarr
import hcc

z = root.create_array(
    "reflectance",
    data=data,
    fill_value=float("nan"),
    chunks=(4**10,),
    # steps of 1e-4 from -0.0999 to 6.4535: the codec's defaults, the "reflectance" preset
    serializer=hcc.codecs.uint16_jxl.HccUint16JxlCodec(
        quantification_value=10000.0, add_offset=-1000.0
    ),
    filters=None,
    compressors=None,
    dimension_names=["cells"],
)
```

A signed variable stored as `int16` with a scale of `0.01`, so from `-327.67` to `327.67`, takes `Q = 100` and `O = -32768`:

```python
serializer = hcc.codecs.uint16_jxl.HccUint16JxlCodec(
    quantification_value=100.0, add_offset=-32768.0
)
```

For packed integers `p` with a CF `scale_factor` and `add_offset`, `Q = 1 / scale_factor` and `O = add_offset / scale_factor`, plus `-32768` for a signed type; keep the packed no-data value out of the range, since `DN = 0` is the codec's own no-data.

### Two-layer variant: lossless parents, quantised children

`hcc-uint16-jxl-pyramid` keeps the parent level (the mean of each group of four children) lossless on the DN lattice, predicts the children from it with a least-squares filter over the 5 x 5 parent neighbourhood, and quantises the difference with a step proportional to that difference's standard deviation in the chunk (`relative_step`). Rough chunks get a coarse step and smooth chunks a fine one. For a 10 m reflectance band resampled with a point-spread-function kernel, `relative_step=0.36` (the `transfer-floor` preset) puts the error it adds on the native pixels at the error the transfer itself leaves there; `relative_step=0` keeps the children on the DN lattice. The same scale, offset and `out_of_range` apply to the parent means.

```python
serializer = hcc.codecs.uint16_jxl_pyramid.HccUint16JxlPyramidCodec(relative_step=0.36)
```

The JPEG XL layers are lossless, so `effort` and `decoding_speed` change the size and the speed of the chunks, not the decoded cells. On 10 m reflectance chunks at effort 7, the default `decoding_speed=2` makes the chunks about 3 % larger than `decoding_speed=0` (the only setting of earlier versions) for half the JPEG XL encoding and decoding times; `decoding_speed=1` keeps the size of `0` and saves about a third.

### Codec ids before 0.5.0

Up to 0.4.x the two serializers had other ids, `hcc-sentinel2` and `hcc-sentinel2-pyramid`. Those ids stay registered: stores written with them open unchanged, and their `zarr.json` keeps the old id. The old modules `hcc.codecs.sentinel2*` and `hcc.compression.sentinel2*` still import, with a `DeprecationWarning`; new code should use `hcc.codecs.uint16_jxl` and `hcc.codecs.uint16_jxl_pyramid`.

#### Native kernels

The optional package in [`rust/`](rust/) (`healpix-compress-native`, module `hcc_native`, Rust and pyo3) runs everything but the JPEG XL and zlib streams and the least-squares fit natively, without holding the GIL. It gives the same codestreams and the same decoded cells as the Python implementation, bit for bit; the codec uses it when it is installed, unless `HCC_NATIVE=0`. It is built with a Rust toolchain and maturin:

```bash
pip install ./rust
```

## Wavelet residuals: `hcc-grad-dwt`

`hcc-grad-dwt` codes the residual of `hcc-decompose` with a multi-level 2-D wavelet transform: the approximation subband is kept lossless (float32), each detail subband is quantised to int8 or int16 with its own scale. It needs the optional dependency PyWavelets (`pip install healpix-compress[wavelets]`).

```python
from hcc.codecs.presets import preset

[serializer] = preset("hcc-grad-dwt", "medium")
```

## Usage of the zarr codec

### Direct use with `zarr-python`

To write the data:

```python
import zarr
import hcc

data = ...
chunks = ...

store = ...
root = zarr.api.synchronous.create_group(store, zarr_format=3)

z = root.create_array(
    "array1",
    data=data,
    fill_value=float("nan"),
    chunks=chunks,
    serializer=hcc.codecs.dct.HccDctCodec(delta=delta, top_k=top_k),
    filters=None,
    compressors=None,
    dimension_names=["cells"],
)

print(z[:])
```

To read the data back into memory, all we have to do is import the package:

```python
import zarr
import hcc

store = ...
root = zarr.open_group(store, mode="r")
z = root["z"]
print(z[:])
```

### Through `xarray`

Writing:

```python
import xarray as xr
import hcc

ds = ...
store = ...
chunks = ...

ds.to_zarr(
    store,
    encoding={
        "array1": {
            "serializer": {
                "name": "hcc-dct",
                "configuration": {"delta": delta, "top_k": top_k},
            },
            "chunks": chunks,
        }
    },
)
```

Reading:

```python
import xarray as xr
import hcc

ds = xr.open_dataset(store, engine="zarr", chunks={})
ds
```
