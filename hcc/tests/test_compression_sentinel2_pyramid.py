from contextlib import nullcontext

import numpy as np
import pytest
from numpy.testing import assert_equal

from hcc.compression import sentinel2_pyramid as pyramid
from hcc.compression.zorder import flatten_tile

pytest.importorskip("imagecodecs")


def _field(delta, seed=0):
    """A smooth field with edges and noise, as a nested chunk of 4**delta cells."""
    side = 2**delta
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:side, 0:side] / side
    raster = (
        0.2
        + 0.1 * np.sin(6 * xx) * np.cos(4 * yy)
        + 0.15 * (xx > 0.6)
        + 0.004 * rng.standard_normal((side, side))
    )
    return flatten_tile(raster)


class TestEncodeDecode:
    @pytest.mark.parametrize("delta", [1, 2, 5], ids=lambda d: f"delta-{d}")
    def test_unit_step_keeps_children_on_the_dn_lattice(self, delta):
        cells = _field(delta)

        actual = pyramid.decode(pyramid.encode(cells, relative_step=0), dtype="float64")

        # half a digital number, plus float rounding
        assert np.abs(actual - cells).max() <= 0.5e-4 + 1e-12

    @pytest.mark.parametrize("relative_step", [0.1, 0.36, 1.0])
    def test_error_is_bounded_by_half_the_step(self, relative_step):
        cells = _field(6)
        stream = pyramid.encode(cells, relative_step=relative_step)
        step = pyramid.stream_info(stream)["step"]

        actual = pyramid.decode(stream, dtype="float64")

        assert step > 1
        # half the step in digital numbers, plus float rounding
        assert np.abs(actual - cells).max() <= 0.5 * step * 1e-4 + 1e-9

    def test_coarser_step_gives_smaller_stream(self):
        cells = _field(6)

        sizes = [len(pyramid.encode(cells, relative_step=r)) for r in (0.0, 0.36, 2.0)]

        assert sizes[0] > sizes[1] > sizes[2]

    def test_parent_means_are_kept_on_the_dn_lattice(self):
        cells = _field(5)
        means = cells.reshape(-1, 4).mean(axis=1)

        actual = pyramid.decode(
            pyramid.encode(cells, relative_step=2.0), dtype="float64"
        ).reshape(-1, 4)

        # the base is exact; the decoded children do not have to average to it,
        # but they stay within half a step of the cells, around exact parents
        expected = pyramid.sentinel2.dequantize(
            pyramid.sentinel2.quantize(means), dtype="float64"
        )
        assert (
            np.abs(actual.mean(axis=1) - expected).max()
            < 0.5
            * pyramid.stream_info(pyramid.encode(cells, relative_step=2.0))["step"]
            * 1e-4
            + 1e-9
        )

    @pytest.mark.parametrize(
        "nan_cells",
        [
            pytest.param(slice(0, 4), id="whole-parent"),
            pytest.param(slice(0, 3), id="partial-parent"),
            pytest.param(slice(0, 4**4), id="all"),
        ],
    )
    def test_nan_cells_round_trip(self, nan_cells):
        cells = _field(4)
        cells[nan_cells] = np.nan

        actual = pyramid.decode(pyramid.encode(cells), dtype="float64")

        assert_equal(np.isnan(actual), np.isnan(cells))

    @pytest.mark.parametrize(
        ["kwargs", "expectation"],
        [
            pytest.param({}, nullcontext(), id="defaults"),
            pytest.param({"predictor_size": 3}, nullcontext(), id="predictor-3"),
            pytest.param(
                {"predictor_size": 4}, pytest.raises(ValueError), id="even-predictor"
            ),
            pytest.param(
                {"relative_step": -1.0}, pytest.raises(ValueError), id="negative-step"
            ),
            pytest.param(
                {"min_step": 0.0}, pytest.raises(ValueError), id="zero-min-step"
            ),
            pytest.param({"effort": 0}, pytest.raises(ValueError), id="effort-too-low"),
        ],
    )
    def test_parameters(self, kwargs, expectation):
        with expectation:
            pyramid.encode(_field(3), **kwargs)

    @pytest.mark.parametrize(
        ["shape", "expectation"],
        [
            pytest.param((16,), nullcontext(), id="power-of-four"),
            pytest.param((1,), pytest.raises(ValueError), id="single-cell"),
            pytest.param((12,), pytest.raises(ValueError), id="not-power-of-four"),
        ],
    )
    def test_chunk_shape(self, shape, expectation):
        with expectation:
            pyramid.encode(np.full(shape, 0.2))


class TestStreamInfo:
    def test_header(self):
        actual = pyramid.stream_info(
            pyramid.encode(_field(3), relative_step=0, predictor_size=3)
        )

        assert actual["delta"] == 3
        assert actual["predictor_size"] == 3
        assert actual["step"] == 1.0

    def test_rejects_other_streams(self):
        with pytest.raises(ValueError):
            pyramid.stream_info(b"\x00" * 64)
