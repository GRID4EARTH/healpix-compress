from contextlib import nullcontext

import numpy as np
import pytest
import zarr
from numpy.testing import assert_equal

from hcc.codecs import uint16_jxl as codec_module
from hcc.compression import uint16_jxl

pytest.importorskip("imagecodecs")


class TestHccUint16JxlCodec:
    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="defaults"),
            pytest.param(
                {"quantification_value": 1000.0, "add_offset": 0.0}, id="custom-scale"
            ),
            pytest.param({"effort": 3}, id="effort"),
            pytest.param({"out_of_range": "raise"}, id="out-of-range"),
        ],
    )
    def test_serialization_roundtrip(self, kwargs):
        codec = codec_module.HccUint16JxlCodec(**kwargs)

        actual = codec_module.HccUint16JxlCodec.from_dict(codec.to_dict())

        assert actual == codec

    def test_default_out_of_range_is_not_written(self):
        # keeps the metadata of the default readable by earlier versions
        configuration = codec_module.HccUint16JxlCodec().to_dict()["configuration"]

        assert "out_of_range" not in configuration

    def test_unknown_out_of_range(self):
        with pytest.raises(ValueError, match="out_of_range"):
            codec_module.HccUint16JxlCodec(out_of_range="ignore")

    def test_zarr_roundtrip_signed(self):
        packed = np.arange(-32000, 32000, 1000, dtype=np.int16)
        data = np.zeros(4**3)
        data[: packed.size] = packed / 100.0
        codec = codec_module.HccUint16JxlCodec(
            quantification_value=100.0, add_offset=-32768.0, effort=1
        )

        with zarr.storage.MemoryStore() as store:
            g = zarr.api.synchronous.create_group(store)
            z = g.create_array(
                "z",
                data=data,
                chunks=(4**3,),
                serializer=codec,
                filters=None,
                compressors=None,
            )

            actual = z[:]

        assert_equal(np.rint(actual * 100.0), np.rint(data * 100.0))

    def test_from_dict_without_configuration(self):
        actual = codec_module.HccUint16JxlCodec.from_dict({"name": "hcc-uint16-jxl"})

        assert actual == codec_module.HccUint16JxlCodec()

    @pytest.mark.parametrize("delta", [2, 3], ids=lambda d: f"delta-{d}")
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_zarr_roundtrip(self, delta, dtype):
        rng = np.random.default_rng(0)
        data = rng.uniform(0.0, 1.0, size=3 * 4**delta).astype(dtype)
        data[5] = np.nan

        with zarr.storage.MemoryStore() as store:
            g = zarr.api.synchronous.create_group(store)
            z = g.create_array(
                "z",
                data=data,
                chunks=(4**delta,),
                fill_value=np.nan,
                serializer={"name": "hcc-uint16-jxl", "configuration": {"effort": 3}},
                filters=None,
                compressors=None,
            )
            z[:] = data

            actual = z[:]

        expected = uint16_jxl.dequantize(uint16_jxl.quantize(data), dtype=dtype)
        assert actual.dtype == data.dtype
        assert_equal(actual, expected)

    def test_zarr_roundtrip_partial_last_chunk(self):
        data = np.linspace(0, 1, 40, dtype="float32")

        with zarr.storage.MemoryStore() as store:
            g = zarr.api.synchronous.create_group(store)
            z = g.create_array(
                "z",
                data=data,
                chunks=(16,),
                fill_value=np.nan,
                serializer=codec_module.HccUint16JxlCodec(effort=1),
                filters=None,
                compressors=None,
            )
            z[:] = data

            actual = z[:]

        expected = uint16_jxl.dequantize(uint16_jxl.quantize(data))
        assert_equal(actual, expected)

    @pytest.mark.parametrize(
        "chunks", [(1, 3, 4**3), (2, 1, 4**3)], ids=["per-time", "per-band"]
    )
    def test_zarr_roundtrip_leading_axes(self, chunks):
        rng = np.random.default_rng(0)
        data = rng.uniform(0.0, 0.5, size=(2, 3, 4**3)).astype("float32")
        data[1, 2, 7] = np.nan

        with zarr.storage.MemoryStore() as store:
            g = zarr.api.synchronous.create_group(store)
            z = g.create_array(
                "z",
                data=data,
                chunks=chunks,
                fill_value=np.nan,
                serializer=codec_module.HccUint16JxlCodec(effort=1),
                filters=None,
                compressors=None,
            )

            actual = z[:]

        expected = uint16_jxl.dequantize(uint16_jxl.quantize(data))
        assert_equal(actual, expected)

    def test_one_dimensional_chunk_bytes_unchanged(self):
        """1-D chunks keep the single-codestream format of existing stores."""
        data = np.linspace(0.1, 0.4, 4**3, dtype="float32")

        with zarr.storage.MemoryStore() as store:
            g = zarr.api.synchronous.create_group(store)
            g.create_array(
                "z",
                data=data,
                chunks=(4**3,),
                serializer=codec_module.HccUint16JxlCodec(effort=1),
                filters=None,
                compressors=None,
            )
            actual = bytes(store._store_dict["z/c/0"].to_bytes())

        assert actual == uint16_jxl.encode(data, effort=1)

    @pytest.mark.parametrize(
        ["dtype", "chunks", "expectation"],
        [
            pytest.param("float32", (16,), nullcontext(), id="valid"),
            pytest.param("int16", (16,), pytest.raises(ValueError), id="integer-dtype"),
            pytest.param(
                "float32",
                (12,),
                pytest.raises(ValueError),
                id="chunk-not-power-of-four",
            ),
            pytest.param("float32", (2, 16), nullcontext(), id="leading-axis"),
            pytest.param(
                "float32",
                (2, 12),
                pytest.raises(ValueError),
                id="leading-axis-not-power-of-four",
            ),
        ],
    )
    def test_validate(self, dtype, chunks, expectation):
        with zarr.storage.MemoryStore() as store, expectation:
            g = zarr.api.synchronous.create_group(store)
            g.create_array(
                "z",
                shape=(2, 32)[2 - len(chunks) :],
                dtype=dtype,
                chunks=chunks,
                serializer=codec_module.HccUint16JxlCodec(),
                filters=None,
                compressors=None,
            )
