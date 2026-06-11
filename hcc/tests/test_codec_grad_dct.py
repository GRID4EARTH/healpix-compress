import numpy as np
import pytest
from zarr.buffer.cpu import numpy_buffer_prototype
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.dtype import Float32, parse_dtype

from hcc.codecs import grad_dct
from hcc.codecs.decompose import ArrayConfigWithScalars
from hcc.tests import is_noise


@pytest.mark.parametrize("window", [8, 16, 32])
@pytest.mark.parametrize("quantize_bits", [8, 16])
@pytest.mark.parametrize("sigma_floor_ratio", [0.05, 0.10])
@pytest.mark.parametrize("clip_percentile", [0.5, 0.1])
def test_codec_serialization_roundtrip(
    window, quantize_bits, sigma_floor_ratio, clip_percentile
):
    codec = grad_dct.HccGradDctCodec(
        window=window,
        quantize_bits=quantize_bits,
        sigma_floor_ratio=sigma_floor_ratio,
        clip_percentile=clip_percentile,
    )

    serialized = codec.to_dict()
    reconstructed = grad_dct.HccGradDctCodec.from_dict(serialized)

    assert codec == reconstructed


@pytest.mark.parametrize("quantize_bits", [8, 16])
def test_resolve_metadata(quantize_bits):
    prototype = numpy_buffer_prototype()
    chunk_spec = ArraySpec(
        shape=(16, 16),
        dtype=Float32(endianness="little"),
        prototype=prototype,
        fill_value=np.nan,
        config=ArrayConfig.from_dict({}),
    )
    codec = grad_dct.HccGradDctCodec(
        window=8,
        quantize_bits=quantize_bits,
        sigma_floor_ratio=0.05,
        clip_percentile=0.5,
    )

    actual = codec.resolve_metadata(chunk_spec)
    dtype = f"int{quantize_bits}"

    assert actual.prototype == prototype
    assert actual.shape == chunk_spec.shape
    assert actual.dtype == parse_dtype(dtype, zarr_format=3)
    assert actual.fill_value == np.iinfo(dtype).min
    assert actual.config == chunk_spec.config


@pytest.mark.asyncio
@pytest.mark.parametrize("chunksize", [64, 256])
@pytest.mark.parametrize("window", [8, 16, 32])
@pytest.mark.parametrize("quantize_bits", [8, 16])
@pytest.mark.parametrize("sigma_floor_ratio", [0.05, 0.10])
@pytest.mark.parametrize("clip_percentile", [0.5, 0.1])
async def test_codec(
    chunksize, window, quantize_bits, sigma_floor_ratio, clip_percentile
):
    print(f"chunk shape: {chunksize}")
    edge_length = int(np.sqrt(chunksize))
    data = np.reshape(
        np.concatenate(
            [np.array([0.0], dtype="float32"), np.linspace(1, 0, 255, dtype="float32")]
        )[:chunksize],
        (edge_length, edge_length),
    )

    block_shape = edge_length // 2
    downsampled = np.mean(
        np.reshape(data, (block_shape, 2, block_shape, 2)), axis=(1, 3)
    )
    broadcasted = np.broadcast_to(
        downsampled[:, None, :, None], (block_shape, 2, block_shape, 2)
    )
    prediction = np.reshape(broadcasted, data.shape)
    residuals = data - prediction

    codec = grad_dct.HccGradDctCodec(
        window=window,
        quantize_bits=quantize_bits,
        sigma_floor_ratio=sigma_floor_ratio,
        clip_percentile=clip_percentile,
    )

    prototype = numpy_buffer_prototype()
    array_spec = ArraySpec(
        shape=residuals.shape,
        dtype=Float32(endianness="little"),
        prototype=prototype,
        fill_value=np.nan,
        config=ArrayConfigWithScalars.from_dict({}),
    )
    array_spec.config.context["prediction"] = np.astype(prediction, np.float32)

    input_buffer = prototype.nd_buffer.from_ndarray_like(
        np.astype(residuals, np.float32)
    )
    encoded = await codec._encode_single(input_buffer, array_spec)
    decoded = await codec._decode_single(encoded, array_spec)
    roundtripped = decoded.as_numpy_array()

    assert roundtripped.shape == data.shape
    assert roundtripped.dtype == data.dtype

    # properties of a lossy compression algorithm:
    # - if no signal in input data → no signal in output data
    # - if signal in the input: strong correlation between input and output data
    ravelled_data = np.ravel(data)
    ravelled_roundtripped = np.ravel(roundtripped)
    assert (
        is_noise(np.ravel(ravelled_data)) and is_noise(ravelled_roundtripped)
    ) or np.corrcoef(ravelled_roundtripped, ravelled_data)[0, 1] >= 0.5


@pytest.mark.parametrize("name", ["low", "medium", "high", "ultra"])
def test_lookup_preset(name):
    from hcc.codecs.decompose import HccDecomposeCodec
    from hcc.compression.grad_dct import lookup_preset as lookup_raw_preset

    presets = lookup_raw_preset(name)
    codec_pipeline = grad_dct.lookup_preset(name)
    codec = codec_pipeline[0]

    assert isinstance(codec, HccDecomposeCodec)
    assert codec.factor == presets["downsample_factor"]
    assert codec.codecs["downsampled"][-1].level == presets["zlib_level"]

    grad_dct_codec = codec.codecs["residuals"][0]
    assert grad_dct_codec.window == presets["window"]
    assert grad_dct_codec.quantize_bits == presets["quantize_bits"]
    assert grad_dct_codec.sigma_floor_ratio == presets["sigma_floor_ratio"]
    assert grad_dct_codec.clip_percentile == presets["clip_percentile"]

    assert codec.codecs["residuals"][-1].level == presets["zlib_level"]
