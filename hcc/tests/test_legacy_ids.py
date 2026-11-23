import importlib
import json
import sys

import numpy as np
import pytest
import zarr
from numpy.testing import assert_equal
from zarr.core.buffer import cpu

from hcc.codecs import _aliases, presets
from hcc.codecs.uint16_jxl import HccUint16JxlCodec
from hcc.codecs.uint16_jxl_pyramid import HccUint16JxlPyramidCodec
from hcc.compression import uint16_jxl
from hcc.tests import legacy_store_041

pytest.importorskip("imagecodecs")

OLD_IDS = {
    "hcc-sentinel2": HccUint16JxlCodec,
    "hcc-sentinel2-pyramid": HccUint16JxlPyramidCodec,
}


def _legacy_store() -> zarr.storage.MemoryStore:
    files = {k: cpu.Buffer.from_bytes(v) for k, v in legacy_store_041.FILES.items()}
    return zarr.storage.MemoryStore(files, read_only=True)


@pytest.mark.parametrize("name", ["single", "pyramid"])
def test_store_written_by_041_opens(name):
    store = _legacy_store()

    array = zarr.open_array(store, path=name, mode="r")
    actual = array[:]

    assert_equal(actual, legacy_store_041.VALUES[name])


@pytest.mark.parametrize(
    ["name", "old_id"],
    [("single", "hcc-sentinel2"), ("pyramid", "hcc-sentinel2-pyramid")],
)
def test_store_written_by_041_keeps_its_id(name, old_id):
    store = _legacy_store()

    array = zarr.open_array(store, path=name, mode="r")
    (serializer,) = [c for c in array.metadata.codecs if c.to_dict()["name"] == old_id]

    assert isinstance(serializer, OLD_IDS[old_id])


@pytest.mark.parametrize(
    ["name", "new_cls"],
    [("single", HccUint16JxlCodec), ("pyramid", HccUint16JxlPyramidCodec)],
)
def test_new_ids_write_the_same_bytes(name, new_cls):
    """The codec under its new id writes the chunks 0.4.1 wrote under the old one."""
    metadata = json.loads(legacy_store_041.FILES[f"{name}/zarr.json"])
    (serializer,) = metadata["codecs"]
    values = legacy_store_041.INPUTS[name]

    with zarr.storage.MemoryStore() as store:
        g = zarr.create_group(store, zarr_format=3)
        g.create_array(
            "z",
            data=values,
            chunks=(16,),
            fill_value=np.nan,
            serializer=new_cls(**serializer["configuration"]),
            filters=None,
            compressors=None,
        )
        for i in range(2):
            actual = bytes(store._store_dict[f"z/c/{i}"].to_bytes())
            assert actual == legacy_store_041.FILES[f"{name}/c/{i}"]


@pytest.mark.parametrize("old_id", OLD_IDS)
def test_old_id_round_trips(old_id):
    codec_cls = zarr.registry.get_codec_class(old_id)
    codec = codec_cls(effort=3)

    assert isinstance(codec, OLD_IDS[old_id])
    assert codec.to_dict()["name"] == old_id
    assert codec_cls.from_dict(codec.to_dict()) == codec


@pytest.mark.parametrize(
    ["codec", "preset", "old_cls"],
    [
        ("hcc-sentinel2", "lossless", _aliases.LegacyHccUint16JxlCodec),
        (
            "hcc-sentinel2-pyramid",
            "transfer-floor",
            _aliases.LegacyHccUint16JxlPyramidCodec,
        ),
    ],
)
def test_presets_under_old_ids(codec, preset, old_cls):
    (actual,) = presets.preset(codec, preset)

    assert type(actual) is old_cls
    assert actual.to_dict()["name"] == codec


@pytest.mark.parametrize(
    ["module", "attribute", "expected"],
    [
        ("hcc.compression.sentinel2", "quantize", uint16_jxl.quantize),
        ("hcc.compression.sentinel2_pyramid", "DEFAULT_RELATIVE_STEP", 0.57),
        ("hcc.codecs.sentinel2", "HccSentinel2Codec", _aliases.LegacyHccUint16JxlCodec),
        (
            "hcc.codecs.sentinel2_pyramid",
            "HccSentinel2PyramidCodec",
            _aliases.LegacyHccUint16JxlPyramidCodec,
        ),
    ],
)
def test_old_modules_warn_and_reexport(module, attribute, expected):
    sys.modules.pop(module, None)

    with pytest.warns(DeprecationWarning, match="0.5.0"):
        old = importlib.import_module(module)

    assert getattr(old, attribute) == expected
