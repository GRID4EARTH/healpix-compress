import pytest
from zarr.abc.codec import Codec

from hcc.codecs import presets


@pytest.mark.parametrize(
    ["codec", "preset"],
    (
        ("hcc-grad-dct", "low"),
        ("hcc-grad-dct", "high"),
        ("hcc-sentinel2", "lossless"),
        ("hcc-sentinel2-pyramid", "transfer-floor"),
        ("hcc-sentinel2-pyramid", "lossless"),
    ),
)
def test_preset(codec, preset) -> None:
    actual = presets.preset(codec, preset)
    assert isinstance(actual, list) and len(actual) >= 1
    assert isinstance(actual[0], Codec)
