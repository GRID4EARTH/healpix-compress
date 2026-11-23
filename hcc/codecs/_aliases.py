"""The codec ids used before 0.5.0, kept so that existing stores still open.

``hcc-uint16-jxl`` and ``hcc-uint16-jxl-pyramid`` were registered under other
ids up to 0.4.x, and ``zarr.json`` files written then name those ids. The
classes here are the current codecs under the old ids: same configuration,
same bytes, and ``to_dict`` keeps the old id, so rewriting the metadata of
such an array does not change it.
"""

from __future__ import annotations

from typing import ClassVar

from hcc.codecs import uint16_jxl, uint16_jxl_pyramid
from hcc.codecs.uint16_jxl import HccUint16JxlCodec
from hcc.codecs.uint16_jxl_pyramid import HccUint16JxlPyramidCodec


class LegacyHccUint16JxlCodec(HccUint16JxlCodec):
    """:class:`~hcc.codecs.uint16_jxl.HccUint16JxlCodec` under its id before 0.5.0."""

    name: ClassVar[str] = "hcc-sentinel2"


class LegacyHccUint16JxlPyramidCodec(HccUint16JxlPyramidCodec):
    """:class:`~hcc.codecs.uint16_jxl_pyramid.HccUint16JxlPyramidCodec` under its
    id before 0.5.0."""

    name: ClassVar[str] = "hcc-sentinel2-pyramid"


#: The codecs under the ids used before 0.5.0.
LEGACY_CODECS = (LegacyHccUint16JxlCodec, LegacyHccUint16JxlPyramidCodec)


def lookup_preset(name: str) -> list[LegacyHccUint16JxlCodec]:
    """:func:`hcc.codecs.uint16_jxl.lookup_preset` under the old id."""
    return uint16_jxl._lookup_preset(LegacyHccUint16JxlCodec, name)


def lookup_pyramid_preset(name: str) -> list[LegacyHccUint16JxlPyramidCodec]:
    """:func:`hcc.codecs.uint16_jxl_pyramid.lookup_preset` under the old id."""
    return uint16_jxl_pyramid._lookup_preset(LegacyHccUint16JxlPyramidCodec, name)
