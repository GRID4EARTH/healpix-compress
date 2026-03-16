import numpy as np
import pytest
import zarr

import hcc.codecs.dct  # noqa: F401
from hcc.tests import is_noise


@pytest.mark.parametrize("top_k", [16, 32])
@pytest.mark.parametrize("delta", [1, 2])
@pytest.mark.parametrize("chunksize", [16, 64])
def test_codec(top_k, delta, chunksize):
    data = np.concatenate([np.array([0.0], dtype="float64"), np.linspace(1, 0, 255)])
    print(f"chunk shape: {chunksize}")

    with zarr.storage.MemoryStore() as store:
        g = zarr.api.synchronous.create_group(store)
        z = g.create_array(
            "z",
            data=data,
            chunks=chunksize,
            serializer={
                "name": "hcc-dct",
                "configuration": {"delta": delta, "top_k": top_k},
            },
        )
        z[:] = data

        roundtripped = z[:]

    assert roundtripped.shape == data.shape
    assert roundtripped.dtype == data.dtype

    # properties of a lossy compression algorithm:
    # - if no signal in input data → no signal in output data
    # - if signal in the input: strong correlation between input and output data
    assert (is_noise(data) and is_noise(roundtripped)) or np.corrcoef(
        roundtripped, data
    )[0, 1] >= 0.5
