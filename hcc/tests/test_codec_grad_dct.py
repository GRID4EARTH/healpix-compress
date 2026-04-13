import numpy as np
import pytest
import zarr

from hcc.codecs import grad_dct
from hcc.tests import is_noise


@pytest.mark.parametrize("window", [8, 16, 32])
def test_codec_serialization_roundtrip(window):
    codec = grad_dct.HccGradDctCodec(window=window)

    serialized = codec.to_dict()
    reconstructed = grad_dct.HccGradDctCodec.from_dict(serialized)

    assert codec == reconstructed


@pytest.mark.parametrize("quantize_bits", [8, 16])
def test_evolve_from_array_spec(quantize_bits):
    codec = grad_dct.HccGradDctCodec(window=8, quantize_bits=quantize_bits)
    value_dtype = zarr.dtype.Float32(endianness="little")
    array_spec = zarr.core.array_spec.ArraySpec(
        shape=(4**8,),
        dtype=value_dtype,
        fill_value=np.nan,
        config=None,
        prototype=zarr.core.buffer.numpy_buffer_prototype(),
    )

    evolved_codec = codec.evolve_from_array_spec(array_spec)

    residuals_dtype = (
        zarr.dtype.Int8()
        if quantize_bits == 8
        else zarr.dtype.Int16(endianness="little")
    )
    expected_dtypes = [
        residuals_dtype,
        zarr.dtype.UInt64(endianness="little"),
        zarr.dtype.Float32(endianness="little"),
        zarr.dtype.Float32(endianness="little"),
        zarr.dtype.Float32(endianness="little"),
        zarr.dtype.Float32(endianness="little"),
    ]
    assert evolved_codec.dtypes == expected_dtypes


@pytest.mark.parametrize("chunksize", [32, 64])
@pytest.mark.parametrize("window", [8, 16, 32])
def test_codec(window, chunksize):
    data = np.concatenate([np.array([0.0], dtype="float64"), np.linspace(1, 0, 255)])
    print(f"chunk shape: {chunksize}")

    with zarr.storage.MemoryStore() as store:
        g = zarr.api.synchronous.create_group(store)
        z = g.create_array(
            "z",
            data=data,
            chunks=chunksize,
            serializer={
                "name": "hcc-grad-dct",
                "configuration": {"window": window},
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
