from contextlib import nullcontext

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from hcc.compression import uint16_jxl

pytest.importorskip("imagecodecs")


class TestQuantize:
    @pytest.mark.parametrize(
        ["values", "expected"],
        [
            pytest.param([0.0], [1000], id="zero"),
            pytest.param([0.12345], [2234], id="rounding"),
            pytest.param([np.nan, np.inf], [0, 0], id="non-finite-is-nodata"),
            pytest.param([-0.5, 10.0], [1, 65535], id="clipping"),
        ],
    )
    def test_values(self, values, expected):
        actual = uint16_jxl.quantize(
            np.array(values, dtype="float32"), out_of_range="clip"
        )

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
            uint16_jxl.quantize(np.array([2.5], dtype="float32"), **kwargs)

    def test_rejects_integers(self):
        with pytest.raises(TypeError):
            uint16_jxl.quantize(np.array([1, 2], dtype="uint16"))


class TestOutOfRange:
    @pytest.mark.parametrize(
        ["kwargs", "expected"],
        [
            pytest.param({}, (-0.0999, 6.4535), id="defaults"),
            pytest.param(
                {"quantification_value": 100.0, "add_offset": -32768.0},
                (-327.67, 327.67),
                id="signed",
            ),
            pytest.param(
                {"quantification_value": 1.0, "add_offset": 0.0},
                (1.0, 65535.0),
                id="unit",
            ),
        ],
    )
    def test_value_range(self, kwargs, expected):
        assert_allclose(uint16_jxl.value_range(**kwargs), expected)

    def test_values_in_range_do_not_warn(self, recwarn):
        lo, hi = uint16_jxl.value_range()
        uint16_jxl.quantize(np.array([lo, hi, np.nan, -np.inf]))

        assert not [w for w in recwarn if w.category is uint16_jxl.OutOfRangeWarning]

    def test_warn_is_the_default(self):
        values = np.array([0.1, -0.5, 7.0, np.nan])

        with pytest.warns(uint16_jxl.OutOfRangeWarning, match="2 of 3 finite values"):
            actual = uint16_jxl.quantize(values)

        assert_equal(actual, uint16_jxl.quantize(values, out_of_range="clip"))

    def test_raise(self):
        with pytest.raises(ValueError, match="outside"):
            uint16_jxl.quantize(np.array([0.1, 7.0]), out_of_range="raise")

    def test_unknown_policy(self):
        with pytest.raises(ValueError, match="out_of_range"):
            uint16_jxl.quantize(np.array([0.1]), out_of_range="ignore")

    def test_encode_passes_the_policy(self):
        cells = np.full(4**2, 0.1)
        cells[3] = 10.0

        with pytest.raises(ValueError, match="outside"):
            uint16_jxl.encode(cells, out_of_range="raise")


class TestSigned:
    def test_signed_16_bit_data_round_trip_exactly(self):
        # a signed variable with steps of 0.01, as packed integers would hold it
        rng = np.random.default_rng(0)
        packed = rng.integers(-32767, 32768, size=4**4, dtype=np.int16)
        values = packed.astype(np.float64) / 100.0
        kwargs = {"quantification_value": 100.0, "add_offset": -32768.0}

        stream = uint16_jxl.encode(values, effort=1, **kwargs)
        actual = uint16_jxl.decode(stream, dtype="float64", **kwargs)

        assert_equal(np.rint(actual * 100.0).astype(np.int16), packed)


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
        actual = uint16_jxl.dequantize(np.array(dn, dtype="uint16"))

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
            actual = uint16_jxl.dequantize(
                np.array([1000], dtype="uint16"), dtype=dtype
            )
            assert actual.dtype == np.dtype(dtype)

    def test_roundtrip_is_identity_on_the_lattice(self):
        dn = np.arange(0, 65536, dtype="uint16")

        actual = uint16_jxl.quantize(uint16_jxl.dequantize(dn, dtype="float64"))

        assert_equal(actual, dn)


class TestEncodeDecode:
    @pytest.mark.parametrize("delta", [1, 2, 4], ids=lambda d: f"delta-{d}")
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_roundtrip_is_exact_after_quantisation(self, delta, dtype):
        rng = np.random.default_rng(0)
        cells = rng.uniform(-0.05, 1.5, size=4**delta).astype(dtype)
        cells[::7] = np.nan

        actual = uint16_jxl.decode(uint16_jxl.encode(cells), dtype=dtype)
        expected = uint16_jxl.dequantize(uint16_jxl.quantize(cells), dtype=dtype)

        assert actual.dtype == np.dtype(dtype)
        assert_equal(actual, expected)

    def test_quantisation_error_is_bounded_by_half_a_step(self):
        rng = np.random.default_rng(1)
        cells = rng.uniform(0.0, 1.0, size=4**4).astype("float64")

        actual = uint16_jxl.decode(uint16_jxl.encode(cells), dtype="float64")

        # half of the DN step 1 / quantification_value, plus float rounding
        assert (
            np.max(np.abs(actual - cells))
            <= 0.5 / uint16_jxl.DEFAULT_QUANTIFICATION_VALUE + 1e-9
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
            uint16_jxl.encode(cells, **kwargs)

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
            uint16_jxl.encode(cells)

    def test_smooth_field_compresses(self):
        x = np.linspace(0, 1, 4**5, dtype="float32")

        stream = uint16_jxl.encode(x)

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
            assert uint16_jxl.check_rows_shape(shape) == 2

    @pytest.mark.parametrize("shape", [(16,), (3, 16), (2, 3, 16)])
    def test_roundtrip(self, shape):
        rng = np.random.default_rng(0)
        cells = rng.uniform(0.0, 0.5, size=shape).astype("float32")

        data = uint16_jxl.encode_rows(cells, uint16_jxl.encode)
        actual = uint16_jxl.decode_rows(data, shape, uint16_jxl.decode)

        assert actual.shape == shape
        assert_equal(actual, uint16_jxl.dequantize(uint16_jxl.quantize(cells)))

    def test_one_dimensional_is_the_codestream(self):
        cells = np.linspace(0.0, 0.5, 16, dtype="float32")

        actual = uint16_jxl.encode_rows(cells, uint16_jxl.encode)

        assert actual == uint16_jxl.encode(cells)

    @pytest.mark.parametrize(
        "cut", [slice(0, 10), slice(0, -1)], ids=["truncated-header", "truncated-rows"]
    )
    def test_corrupt_chunk_is_rejected(self, cut):
        cells = np.linspace(0.0, 0.5, 2 * 16, dtype="float32").reshape(2, 16)
        data = uint16_jxl.encode_rows(cells, uint16_jxl.encode)

        with pytest.raises(ValueError):
            uint16_jxl.decode_rows(data[cut], (2, 16), uint16_jxl.decode)
