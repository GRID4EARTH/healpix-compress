import numpy as np
import pytest
import zarr
from zarr.abc.codec import Codec
from zarr.buffer.cpu import numpy_buffer_prototype
from zarr.codecs import BytesCodec, GzipCodec, ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.dtype import Float32, Float64

from hcc.codecs import decompose


def test_array_spec_from_array():
    prototype = numpy_buffer_prototype()
    spec = ArraySpec(
        shape=16,
        dtype=Float64(endianness="little"),
        prototype=prototype,
        config=ArrayConfig.from_dict({}),
        fill_value=np.nan,
    )
    actual = decompose.array_spec_from_array_spec(
        spec,
        dtype=Float32(endianness="little"),
        shape=(4, 4),
    )

    assert actual.prototype == prototype
    assert actual.shape == (4, 4)
    assert np.isnan(actual.fill_value)
    assert actual.dtype.to_native_dtype() == np.float32

    assert isinstance(actual.config.scalars, dict)


def test_codec_from_dict():
    factor = 2

    codec = decompose.HccDecompose.from_dict(
        {
            "name": "hcc-decompose",
            "configuration": {
                "factor": factor,
                "codecs": {
                    "downsampled": [
                        {"name": "bytes"},
                        {"name": "zstd"},
                    ],
                    "residuals": [
                        {"name": "bytes"},
                        {"name": "gzip", "level": 5},
                    ],
                },
            },
        }
    )

    assert codec.factor == factor
    assert len(codec.codecs) == 2

    assert all(isinstance(subcodec, Codec) for subcodec in codec.codecs["downsampled"])
    assert all(isinstance(subcodec, Codec) for subcodec in codec.codecs["residuals"])


def test_codec_to_dict():
    factor = 4
    downsampled_codecs = [
        BytesCodec(endian="little"),
        ZstdCodec(),
    ]
    residuals_codecs = [
        BytesCodec(endian="little"),
        GzipCodec(level=5),
    ]

    codec = decompose.HccDecompose(
        factor=factor,
        codecs={"downsampled": downsampled_codecs, "residuals": residuals_codecs},
    )

    actual = codec.to_dict()
    assert actual["name"] == codec.name
    assert sorted(actual["configuration"].keys()) == ["codecs", "factor"]


@pytest.mark.asyncio
@pytest.mark.parametrize("delta", [1, 2, 4])
async def test_codec_serialization_roundtrip_default_subcodecs(delta):
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
    spec = ArraySpec(
        shape=data.shape,
        dtype=Float32(endianness="little"),
        prototype=prototype,
        fill_value=np.nan,
        config=ArrayConfig.from_dict({}),
    )

    input_buffer = prototype.nd_buffer.from_ndarray_like(data)
    encoded = await codec._encode_single(input_buffer, spec)
    decoded = await codec._decode_single(encoded, spec)

    np.testing.assert_allclose(decoded.as_numpy_array(), data)
