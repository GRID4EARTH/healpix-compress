from contextlib import nullcontext

import numpy as np
import pytest
import zarr
from numpy.testing import assert_equal

from hcc.codecs import sentinel2 as codec_module
from hcc.compression import sentinel2

pytest.importorskip("imagecodecs")


class TestHccSentinel2Codec:
    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="defaults"),
            pytest.param(
                {"quantification_value": 1000.0, "add_offset": 0.0}, id="custom-scale"
            ),
            pytest.param({"effort": 3}, id="effort"),
        ],
    )
    def test_serialization_roundtrip(self, kwargs):
        codec = codec_module.HccSentinel2Codec(**kwargs)

        actual = codec_module.HccSentinel2Codec.from_dict(codec.to_dict())

        assert actual == codec

    def test_from_dict_without_configuration(self):
        actual = codec_module.HccSentinel2Codec.from_dict({"name": "hcc-sentinel2"})

        assert actual == codec_module.HccSentinel2Codec()

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
                serializer={"name": "hcc-sentinel2", "configuration": {"effort": 3}},
                filters=None,
                compressors=None,
            )
            z[:] = data

            actual = z[:]

        expected = sentinel2.dequantize(sentinel2.quantize(data), dtype=dtype)
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
                serializer=codec_module.HccSentinel2Codec(effort=1),
                filters=None,
                compressors=None,
            )
            z[:] = data

            actual = z[:]

        expected = sentinel2.dequantize(sentinel2.quantize(data))
        assert_equal(actual, expected)

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
        ],
    )
    def test_validate(self, dtype, chunks, expectation):
        with zarr.storage.MemoryStore() as store, expectation:
            g = zarr.api.synchronous.create_group(store)
            g.create_array(
                "z",
                shape=(32,),
                dtype=dtype,
                chunks=chunks,
                serializer=codec_module.HccSentinel2Codec(),
                filters=None,
                compressors=None,
            )
