import pytest
from zarr.abc.codec import Codec

from hcc.codecs import presets


@pytest.mark.parametrize(
    ["codec", "preset"],
    (
        ("hcc-grad-dct", "low"),
        ("hcc-grad-dct", "high"),
    ),
)
def test_preset(codec, preset) -> None:
    actual = presets.preset(codec, preset)
    assert isinstance(actual, list) and len(actual) >= 1
    assert isinstance(actual[0], Codec)
