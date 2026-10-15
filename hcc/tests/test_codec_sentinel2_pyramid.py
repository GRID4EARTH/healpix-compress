from contextlib import nullcontext

import numpy as np
import pytest
import zarr
from numpy.testing import assert_equal

from hcc.codecs import sentinel2_pyramid as codec_module
from hcc.compression import sentinel2_pyramid

pytest.importorskip("imagecodecs")


class TestHccSentinel2PyramidCodec:
    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({}, id="defaults"),
            pytest.param({"relative_step": 0.0, "min_step": 2.0}, id="fixed-step"),
            pytest.param({"predictor_size": 3, "effort": 3}, id="small-predictor"),
        ],
    )
    def test_serialization_roundtrip(self, kwargs):
        codec = codec_module.HccSentinel2PyramidCodec(**kwargs)

        actual = codec_module.HccSentinel2PyramidCodec.from_dict(codec.to_dict())

        assert actual == codec

    def test_default_relative_step(self):
        actual = codec_module.HccSentinel2PyramidCodec().relative_step

        assert actual == sentinel2_pyramid.DEFAULT_RELATIVE_STEP == 0.57

    @pytest.mark.parametrize(
        ["name", "expected"],
        [
            pytest.param("hcc-sentinel2", "hcc-sentinel2", id="lossless"),
            pytest.param(
                "hcc-sentinel2-pyramid", "hcc-sentinel2-pyramid", id="pyramid"
            ),
        ],
    )
    def test_write_and_reopen_by_name(self, name, expected):
        """The serializers resolve by name alone, as zarr does when it opens a store."""
        data = np.linspace(0.1, 0.4, 2 * 4**4, dtype="float32")

        with zarr.storage.MemoryStore() as store:
            g = zarr.create_group(store, zarr_format=3)
            g.create_array(
                "b04",
                data=data,
                chunks=(4**4,),
                fill_value=np.nan,
                serializer={"name": name},
                filters=None,
                compressors=None,
            )
            reopened = zarr.open_group(store, mode="r")["b04"]
            actual = reopened.metadata.to_dict()["codecs"][0]["name"]
            values = reopened[:]

        assert actual == expected
        # a quarter of a DN for the lattice, half a step at most for the pyramid detail
        assert np.abs(values - data).max() < 5e-3

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_zarr_roundtrip(self, dtype):
        rng = np.random.default_rng(0)
        data = (0.2 + 0.05 * rng.standard_normal(3 * 4**4)).astype(dtype)
        data[7] = np.nan
        codec = codec_module.HccSentinel2PyramidCodec(relative_step=0.5, effort=1)

        with zarr.storage.MemoryStore() as store:
            g = zarr.api.synchronous.create_group(store)
            z = g.create_array(
                "z",
                data=data,
                chunks=(4**4,),
                fill_value=np.nan,
                serializer=codec,
                filters=None,
                compressors=None,
            )
            z[:] = data

            actual = z[:]

        expected = np.concatenate(
            [
                sentinel2_pyramid.decode(
                    sentinel2_pyramid.encode(
                        data[i : i + 4**4], relative_step=0.5, effort=1
                    ),
                    dtype=dtype,
                )
                for i in range(0, data.size, 4**4)
            ]
        )
        assert actual.dtype == data.dtype
        assert_equal(actual, expected)

    @pytest.mark.parametrize(
        ["dtype", "chunks", "expectation"],
        [
            pytest.param("float32", (16,), nullcontext(), id="valid"),
            pytest.param(
                "uint16", (16,), pytest.raises(ValueError), id="integer-dtype"
            ),
            pytest.param(
                "float32", (1,), pytest.raises(ValueError), id="single-cell-chunks"
            ),
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
                serializer=codec_module.HccSentinel2PyramidCodec(),
                filters=None,
                compressors=None,
            )
