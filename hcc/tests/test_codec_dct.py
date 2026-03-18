import numpy as np
import pytest
import zarr

from hcc.codecs import dct
from hcc.tests import is_noise


@pytest.mark.parametrize("delta", [1, 2])
@pytest.mark.parametrize("top_k", [100, 1200, 14000])
def test_codec_serialization_roundtrip(delta, top_k):
    codec = dct.HccDctCodec(delta=2, top_k=100)

    serialized = codec.to_dict()
    reconstructed = dct.HccDctCodec.from_dict(serialized)

    assert codec == reconstructed


def test_evolve_from_array_spec():
    codec = dct.HccDctCodec(delta=2, top_k=100)
    value_dtype = zarr.dtype.Float32(endianness="little")
    array_spec = zarr.core.array_spec.ArraySpec(
        shape=(4**8,),
        dtype=value_dtype,
        fill_value=np.nan,
        config=None,
        prototype=zarr.core.buffer.numpy_buffer_prototype(),
    )

    evolved_codec = codec.evolve_from_array_spec(array_spec)

    expected_dtypes = [
        value_dtype,
        value_dtype,
        zarr.dtype.Int16(endianness="little"),
        zarr.dtype.UInt64(endianness="little"),
    ]
    assert evolved_codec.dtypes == expected_dtypes


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
