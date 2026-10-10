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
