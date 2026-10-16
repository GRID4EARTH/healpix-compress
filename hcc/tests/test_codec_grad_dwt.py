import numpy as np
import pytest
import zarr
from zarr.buffer.cpu import numpy_buffer_prototype
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.dtype import Float32, parse_dtype

from hcc.codecs import grad_dwt
from hcc.codecs.decompose import ArrayConfigWithScalars, HccDecomposeCodec
from hcc.compression import grad_dwt as compression
from hcc.compression.zorder import flatten_tile

pytest.importorskip("pywt")


def _make_tile(edge_length=64):
    x = np.linspace(0, 4, edge_length, dtype="float32")
    xx, yy = np.meshgrid(x, x)
    return (np.sin(xx) * np.cos(yy)).astype("float32")


def _array_spec(shape):
    return ArraySpec(
        shape=shape,
        dtype=Float32(endianness="little"),
        prototype=numpy_buffer_prototype(),
        fill_value=np.nan,
        config=ArrayConfigWithScalars.from_dict({}),
    )


class TestHccGradDwtCodec:
    @pytest.mark.parametrize("levels", [3, 4])
    @pytest.mark.parametrize("quantize_bits", [8, 16])
    def test_serialization_roundtrip(self, levels, quantize_bits):
        codec = grad_dwt.HccGradDwtCodec(
            wavelet="db4", levels=levels, quantize_bits=quantize_bits
        )

        assert grad_dwt.HccGradDwtCodec.from_dict(codec.to_dict()) == codec

    @pytest.mark.parametrize("quantize_bits", [8, 16])
    def test_resolve_metadata(self, quantize_bits):
        prototype = numpy_buffer_prototype()
        chunk_spec = ArraySpec(
            shape=(64, 64),
            dtype=Float32(endianness="little"),
            prototype=prototype,
            fill_value=np.nan,
            config=ArrayConfig.from_dict({}),
        )
        codec = grad_dwt.HccGradDwtCodec(quantize_bits=quantize_bits)

        actual = codec.resolve_metadata(chunk_spec)
        dtype = f"int{quantize_bits}"

        assert actual.shape == chunk_spec.shape
        assert actual.dtype == parse_dtype(dtype, zarr_format=3)
        assert actual.fill_value == np.iinfo(dtype).min

    def test_resolve_metadata_rejects_other_widths(self):
        codec = grad_dwt.HccGradDwtCodec(quantize_bits=12)

        with pytest.raises(ValueError, match="8 or 16"):
            codec.resolve_metadata(_array_spec((64, 64)))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ["quantize_bits", "max_relative_error"], [(8, 2e-2), (16, 1e-4)]
    )
    async def test_roundtrip_error(self, quantize_bits, max_relative_error):
        data = _make_tile(64)
        codec = grad_dwt.HccGradDwtCodec(
            wavelet="db4", levels=3, quantize_bits=quantize_bits, clip_percentile=0
        )
        spec = _array_spec(data.shape)
        prototype = spec.prototype

        encoded = await codec._encode_single(
            prototype.nd_buffer.from_ndarray_like(data), spec
        )
        decoded = (await codec._decode_single(encoded, spec)).as_numpy_array()

        assert set(spec.config.scalars) == {"q_scale", "ll_values"}
        assert decoded.shape == data.shape and decoded.dtype == np.float32
        rms = np.sqrt(np.mean((decoded - data) ** 2))
        assert rms <= max_relative_error * np.abs(data).max()


class TestLookupPreset:
    @pytest.mark.parametrize("name", ["low", "medium", "high", "ultra"])
    def test_pipeline(self, name):
        expected = compression.lookup_preset(name, codec=True)

        [pipeline] = grad_dwt.lookup_preset(name)

        assert isinstance(pipeline, HccDecomposeCodec)
        assert pipeline.factor == expected["downsample_factor"]
        codec = pipeline.codecs["residuals"][0]
        assert (codec.wavelet, codec.levels) == (expected["wavelet"], expected["levels"])
        assert codec.quantize_bits == expected["quantize_bits"]
        assert codec.clip_percentile == expected["clip_percentile"]
        assert pipeline.codecs["residuals"][-1].level == expected["zlib_level"]

    def test_unknown(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            grad_dwt.lookup_preset("extreme")


class TestZarrArray:
    @pytest.mark.parametrize("name", ["medium", "high"])
    def test_roundtrip(self, name):
        data = flatten_tile(_make_tile(256))

        store = zarr.storage.MemoryStore()
        array = zarr.create_array(
            store,
            shape=data.shape,
            chunks=data.shape,
            dtype="float32",
            serializer=grad_dwt.lookup_preset(name)[0],
            compressors=None,
            fill_value=np.nan,
        )
        array[:] = data

        reopened = zarr.open_array(store, mode="r")
        actual = reopened[:]

        assert reopened.metadata.codecs[0].codecs["residuals"][0].name == "hcc-grad-dwt"
        rms = np.sqrt(np.mean((actual - data) ** 2))
        assert rms <= 1e-2 * np.abs(data).max()


class TestStandaloneCompress:
    @pytest.mark.parametrize("name", ["low", "medium", "high", "ultra"])
    def test_roundtrip(self, name):
        params = compression.lookup_preset(name)
        data = flatten_tile(_make_tile(1024))

        encoded = compression.compress(data, **params)
        decoded = compression.decompress(encoded, **params)
        info = compression.compression_info(
            encoded, quantize_bits=params["quantize_bits"]
        )

        assert decoded.shape == data.shape
        assert np.sqrt(np.mean((decoded - data) ** 2)) <= 1e-2 * np.abs(data).max()
        assert info["total_bytes"] == len(encoded[0])
        assert info["bytes_detail_compressed"] > 0

    def test_rejects_other_widths(self):
        with pytest.raises(ValueError, match="8 or 16"):
            compression.compress(_make_tile(64), quantize_bits=12)

    def test_rejects_too_many_levels(self):
        with pytest.raises(ValueError, match="exceeds the safe maximum"):
            compression.compress(_make_tile(16), wavelet="db6", levels=6)
