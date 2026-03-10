# healpix-based chunk compression

This module contains compression algorithms and codecs to compress / uncompress chunks that are aligned with healpix cells.

The idea is that we can downscale the input chunk by a given amount of refinement levels using a simple aggregation (mean), then compute the residuals between input data and aggregation result, and apply a lossy compression algorithm to the residuals.

The result is multiple arrays (one array for the result of the aggregation, and at least one array of potentially different shape for the residuals), which for the zarr codec are converted to bytes and packed into a single binary blob.

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
