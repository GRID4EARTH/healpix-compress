"""The native kernels against the Python reference: same streams, same cells, bit for bit."""

import numpy as np
import pytest

from hcc.compression import _native
from hcc.compression import sentinel2_pyramid as pyramid
from hcc.compression.zorder import flatten_tile

pytest.importorskip("imagecodecs")
hcc_native = pytest.importorskip("hcc_native")


@pytest.fixture
def python_only(monkeypatch):
    monkeypatch.setattr(_native, "module", None)


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(_native, "module", hcc_native)


def _texture(delta, seed):
    """A Sentinel-2-like raster: power-law texture plus sensor noise."""
    side = 2**delta
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(side)
    k = np.hypot(*np.meshgrid(f, f, indexing="ij"))
    k[0, 0] = 1
    spec = rng.standard_normal((side, side)) + 1j * rng.standard_normal((side, side))
    tex = np.fft.ifft2(spec / k**1.4).real
    tex = (tex - tex.mean()) / (tex.std() or 1.0)
    return 0.12 + 0.05 * tex + 0.002 * rng.standard_normal((side, side))


def _cells(delta, kind, seed=0):
    raster = _texture(delta, seed)
    side = 2**delta
    yy, xx = np.mgrid[0:side, 0:side] / side
    rng = np.random.default_rng(seed + 100)
    match kind:
        case "full":
            pass
        case "swath-edge":
            raster[xx + 0.3 * yy > 0.85] = np.nan
        case "holes":
            raster[rng.random((side, side)) < 0.01] = np.nan
        case "edge-and-holes":
            raster[xx > 0.6] = np.nan
            raster[rng.random((side, side)) < 0.001] = np.nan
        case "one-valid":
            raster[...] = np.nan
            raster[0, 1] = 0.3
        case "all-nan":
            raster[...] = np.nan
        case "constant":
            raster[...] = 0.2
        case "noise":
            raster = rng.random((side, side))
    return flatten_tile(raster)


KINDS = [
    "full",
    "swath-edge",
    "holes",
    "edge-and-holes",
    "one-valid",
    "all-nan",
    "constant",
    "noise",
]


def _encode_both(monkeypatch, cells, **kwargs):
    monkeypatch.setattr(_native, "module", None)
    reference = pyramid.encode(cells, **kwargs)
    monkeypatch.setattr(_native, "module", hcc_native)
    return reference, pyramid.encode(cells, **kwargs)


def _decode_both(monkeypatch, stream, dtype):
    monkeypatch.setattr(_native, "module", None)
    reference = pyramid.decode(stream, dtype=dtype)
    monkeypatch.setattr(_native, "module", hcc_native)
    return reference, pyramid.decode(stream, dtype=dtype)


def _assert_bits_equal(actual, expected):
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    uint = np.dtype(f"u{expected.dtype.itemsize}")
    np.testing.assert_array_equal(actual.view(uint), expected.view(uint))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("delta", [1, 2, 3, 5, 8], ids=lambda d: f"delta-{d}")
@pytest.mark.parametrize("relative_step", [0.0, 0.57])
def test_encode_gives_the_same_stream(monkeypatch, delta, kind, relative_step):
    cells = _cells(delta, kind)

    expected, actual = _encode_both(monkeypatch, cells, relative_step=relative_step)

    assert actual == expected


@pytest.mark.parametrize("predictor_size", [1, 3, 7])
def test_encode_other_predictor_sizes(monkeypatch, predictor_size):
    cells = _cells(6, "swath-edge", seed=3)

    expected, actual = _encode_both(monkeypatch, cells, predictor_size=predictor_size)

    assert actual == expected


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("delta", [1, 2, 3, 5, 8], ids=lambda d: f"delta-{d}")
def test_decode_gives_the_same_cells(monkeypatch, delta, kind, dtype):
    monkeypatch.setattr(_native, "module", None)
    stream = pyramid.encode(_cells(delta, kind, seed=1))

    expected, actual = _decode_both(monkeypatch, stream, dtype)

    _assert_bits_equal(actual, expected)


@pytest.mark.parametrize("predictor_size", [1, 3, 7])
def test_decode_other_predictor_sizes(monkeypatch, predictor_size):
    monkeypatch.setattr(_native, "module", None)
    stream = pyramid.encode(
        _cells(6, "edge-and-holes", seed=4), predictor_size=predictor_size
    )

    expected, actual = _decode_both(monkeypatch, stream, "float64")

    _assert_bits_equal(actual, expected)


def test_other_dtypes_fall_back_to_python(monkeypatch, native):
    stream = pyramid.encode(_cells(4, "holes"))

    expected, actual = _decode_both(monkeypatch, stream, "float16")

    _assert_bits_equal(actual, expected)


def test_non_contiguous_cells(native):
    cells = np.repeat(_cells(4, "holes"), 2)[::2]

    assert pyramid.encode(cells) == pyramid.encode(np.ascontiguousarray(cells))


def test_decode_rejects_a_truncated_mask(native):
    base_dn = np.full((2, 2), 3000, dtype=np.uint16)
    r16 = np.zeros((4, 4), dtype=np.uint16)
    coef = np.zeros((4, 26), dtype=np.float32)

    with pytest.raises(ValueError, match="mask"):
        hcc_native.pyramid_decode(
            base_dn, r16, coef, 5, 0, 1.0, 1e4, -1e3, b"\x00", False
        )
