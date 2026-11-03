# healpix-based chunk compression

This module contains compression algorithms and codecs to compress / uncompress chunks that are aligned with healpix cells.

The idea is that we can downscale the input chunk by a given amount of refinement levels using a simple aggregation (mean), then compute the residuals between input data and aggregation result, and apply a lossy compression algorithm to the residuals.

The result is multiple arrays (one array for the result of the aggregation, and at least one array of potentially different shape for the residuals), which for the zarr codec are converted to bytes and packed into a single binary blob.

## Parameters

The codec has two parameters:

- `delta`: how many refinement levels to average over. The averages will then have a refinement level of `original_level - delta`.
- `top_k`: how many parameters to keep. This is equivalent to a low-pass filter. The details depend on the actual compression algorithm.

## Sentinel-2 chain: digital numbers and lossless JPEG XL

The `hcc-sentinel2` serializer stores a chunk of reflectances the way a Sentinel-2 product stores a band: the cells are rounded to 16-bit digital numbers, `DN = round(Q * reflectance - O)` with `DN = 0` for no-data, de-interleaved into a square raster and compressed with lossless JPEG XL. Decoding restores the quantised reflectances exactly, `(DN + O) / Q`; the rounding to the DN lattice is the only error. The last axis of a chunk must have a power-of-four size (one nested parent cell); leading axes (time, band, ...) are allowed, each row being coded on its own.

It needs the optional dependency `imagecodecs` (`pip install healpix-compress[jpegxl]`).

```python
import zarr
import hcc

z = root.create_array(
    "reflectance",
    data=data,
    fill_value=float("nan"),
    chunks=(4**10,),
    serializer=hcc.codecs.sentinel2.HccSentinel2Codec(
        quantification_value=10000.0, add_offset=-1000.0
    ),
    filters=None,
    compressors=None,
    dimension_names=["cells"],
)
```

### Two-layer variant: lossless parents, quantised children

`hcc-sentinel2-pyramid` keeps the parent level (the mean of each group of four children) lossless on the DN lattice, predicts the children from it with a least-squares filter over the 5 x 5 parent neighbourhood, and quantises the difference with a step proportional to that difference's standard deviation in the chunk (`relative_step`). Rough chunks get a coarse step and smooth chunks a fine one. For a 10 m reflectance band resampled with a point-spread-function kernel, `relative_step=0.36` (the `transfer-floor` preset) puts the error it adds on the native pixels at the error the transfer itself leaves there; `relative_step=0` keeps the children on the DN lattice.

```python
serializer = hcc.codecs.sentinel2_pyramid.HccSentinel2PyramidCodec(relative_step=0.36)
```

The JPEG XL layers are lossless, so `effort` and `decoding_speed` change the size and the speed of the chunks, not the decoded cells. On 10 m reflectance chunks at effort 7, the default `decoding_speed=2` makes the chunks about 3 % larger than `decoding_speed=0` (the only setting of earlier versions) for half the JPEG XL encoding and decoding times; `decoding_speed=1` keeps the size of `0` and saves about a third.

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
