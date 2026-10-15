from contextlib import nullcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from hcc.compression import sentinel2

pytest.importorskip("imagecodecs")


class TestQuantize:
    @pytest.mark.parametrize(
        ["reflectance", "expected"],
        [
            pytest.param([0.0], [1000], id="zero"),
            pytest.param([0.12345], [2234], id="rounding"),
            pytest.param([np.nan, np.inf], [0, 0], id="non-finite-is-nodata"),
            pytest.param([-0.5, 10.0], [1, 65535], id="clipping"),
        ],
    )
    def test_values(self, reflectance, expected):
        actual = sentinel2.quantize(np.array(reflectance, dtype="float32"))

        assert actual.dtype == np.uint16
        assert_equal(actual, np.array(expected, dtype="uint16"))

    @pytest.mark.parametrize(
        ["kwargs", "expectation"],
        [
            pytest.param({}, nullcontext(), id="defaults"),
            pytest.param(
                {"quantification_value": 1.0, "add_offset": 0.0},
                nullcontext(),
                id="unit",
            ),
            pytest.param(
                {"quantification_value": 0.0}, pytest.raises(ValueError), id="zero-q"
            ),
        ],
    )
    def test_parameters(self, kwargs, expectation):
        with expectation:
            sentinel2.quantize(np.array([0.5], dtype="float32"), **kwargs)

    def test_rejects_integers(self):
        with pytest.raises(TypeError):
            sentinel2.quantize(np.array([1, 2], dtype="uint16"))


class TestDequantize:
    @pytest.mark.parametrize(
        ["dn", "expected"],
        [
            pytest.param([1000], [0.0], id="zero"),
            pytest.param([2234], [0.1234], id="value"),
            pytest.param([0], [np.nan], id="nodata"),
        ],
    )
    def test_values(self, dn, expected):
        actual = sentinel2.dequantize(np.array(dn, dtype="uint16"))

        assert actual.dtype == np.float32
        assert_allclose(actual, np.array(expected, dtype="float32"))

    @pytest.mark.parametrize(
        ["dtype", "expectation"],
        [
            pytest.param("float32", nullcontext(), id="float32"),
            pytest.param("float64", nullcontext(), id="float64"),
            pytest.param("int32", pytest.raises(TypeError), id="integer-target"),
        ],
    )
    def test_dtype(self, dtype, expectation):
        with expectation:
            actual = sentinel2.dequantize(np.array([1000], dtype="uint16"), dtype=dtype)
            assert actual.dtype == np.dtype(dtype)

    def test_roundtrip_is_identity_on_the_lattice(self):
        dn = np.arange(0, 65536, dtype="uint16")

        actual = sentinel2.quantize(sentinel2.dequantize(dn, dtype="float64"))

        assert_equal(actual, dn)


class TestEncodeDecode:
    @pytest.mark.parametrize("delta", [1, 2, 4], ids=lambda d: f"delta-{d}")
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_roundtrip_is_exact_after_quantisation(self, delta, dtype):
        rng = np.random.default_rng(0)
        cells = rng.uniform(-0.05, 1.5, size=4**delta).astype(dtype)
        cells[::7] = np.nan

        actual = sentinel2.decode(sentinel2.encode(cells), dtype=dtype)
        expected = sentinel2.dequantize(sentinel2.quantize(cells), dtype=dtype)

        assert actual.dtype == np.dtype(dtype)
        assert_equal(actual, expected)

    def test_quantisation_error_is_bounded_by_half_a_step(self):
        rng = np.random.default_rng(1)
        cells = rng.uniform(0.0, 1.0, size=4**4).astype("float64")

        actual = sentinel2.decode(sentinel2.encode(cells), dtype="float64")

        # half of the DN step 1 / quantification_value, plus float rounding
        assert (
            np.max(np.abs(actual - cells))
            <= 0.5 / sentinel2.DEFAULT_QUANTIFICATION_VALUE + 1e-9
        )

    @pytest.mark.parametrize(
        ["kwargs", "expectation"],
        [
            pytest.param({}, nullcontext(), id="defaults"),
            pytest.param({"effort": 1}, nullcontext(), id="fast"),
            pytest.param({"effort": 0}, pytest.raises(ValueError), id="effort-too-low"),
            pytest.param(
                {"effort": 10}, pytest.raises(ValueError), id="effort-too-high"
            ),
        ],
    )
    def test_effort(self, kwargs, expectation):
        cells = np.linspace(0, 1, 4**3, dtype="float32")
        with expectation:
            sentinel2.encode(cells, **kwargs)

    @pytest.mark.parametrize(
        ["shape", "expectation"],
        [
            pytest.param((16,), nullcontext(), id="power-of-four"),
            pytest.param((12,), pytest.raises(ValueError), id="not-power-of-four"),
            pytest.param((0,), pytest.raises(ValueError), id="empty"),
            pytest.param((4, 4), pytest.raises(ValueError), id="two-dimensional"),
        ],
    )
    def test_chunk_shape(self, shape, expectation):
        cells = np.zeros(shape, dtype="float32")
        with expectation:
            sentinel2.encode(cells)

    def test_smooth_field_compresses(self):
        x = np.linspace(0, 1, 4**5, dtype="float32")

        stream = sentinel2.encode(x)

        assert len(stream) < x.size * 2  # smaller than the raw uint16 raster


class TestRows:
    @pytest.mark.parametrize(
        ["shape", "expectation"],
        [
            pytest.param((16,), nullcontext(), id="one-dimensional"),
            pytest.param((1, 3, 16), nullcontext(), id="leading-axes"),
            pytest.param((3, 12), pytest.raises(ValueError), id="not-power-of-four"),
            pytest.param((), pytest.raises(ValueError), id="scalar"),
        ],
    )
    def test_check_rows_shape(self, shape, expectation):
        with expectation:
            assert sentinel2.check_rows_shape(shape) == 2

    @pytest.mark.parametrize("shape", [(16,), (3, 16), (2, 3, 16)])
    def test_roundtrip(self, shape):
        rng = np.random.default_rng(0)
        cells = rng.uniform(0.0, 0.5, size=shape).astype("float32")

        data = sentinel2.encode_rows(cells, sentinel2.encode)
        actual = sentinel2.decode_rows(data, shape, sentinel2.decode)

        assert actual.shape == shape
        assert_equal(actual, sentinel2.dequantize(sentinel2.quantize(cells)))

    def test_one_dimensional_is_the_codestream(self):
        cells = np.linspace(0.0, 0.5, 16, dtype="float32")

        actual = sentinel2.encode_rows(cells, sentinel2.encode)

        assert actual == sentinel2.encode(cells)

    @pytest.mark.parametrize(
        "cut", [slice(0, 10), slice(0, -1)], ids=["truncated-header", "truncated-rows"]
    )
    def test_corrupt_chunk_is_rejected(self, cut):
        cells = np.linspace(0.0, 0.5, 2 * 16, dtype="float32").reshape(2, 16)
        data = sentinel2.encode_rows(cells, sentinel2.encode)

        with pytest.raises(ValueError):
            sentinel2.decode_rows(data[cut], (2, 16), sentinel2.decode)
