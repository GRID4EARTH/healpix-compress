import numpy as np
import pytest
import zarr

from hcc.codecs import decompose


def test_array_spec_from_array():
    array = np.ones(4**4, dtype=np.float32)
    prototype = zarr.buffer.cpu.numpy_buffer_prototype()
    actual = decompose.array_spec_from_array(array, prototype)

    assert actual.prototype == prototype
    assert actual.shape == array.shape
    assert np.isnan(actual.fill_value)
    assert actual.dtype.to_native_dtype() == array.dtype

    assert isinstance(actual.config.scalars, dict)


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", [1, 2, 4])
async def test_codec_serialization_roundtrip(delta):
    prototype = zarr.buffer.cpu.numpy_buffer_prototype()

    codec = decompose.HccDecompose(
        factor=delta,
        codecs={"downsampled": None, "residuals": None},
    )

    data = np.concatenate(
        [
            np.array([0], dtype=np.float32),
            np.linspace(1, 0, 255, dtype=np.float32),
        ],
        axis=0,
    )
    spec = decompose.array_spec_from_array(data, prototype)

    input_buffer = prototype.nd_buffer.from_ndarray_like(data)
    encoded = await codec._encode_single(input_buffer, spec)
    decoded = await codec._decode_single(encoded, spec)

    np.testing.assert_allclose(decoded.as_numpy_array(), data)
