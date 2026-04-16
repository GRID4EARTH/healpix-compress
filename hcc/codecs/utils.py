from __future__ import annotations

from typing import TYPE_CHECKING

from zarr.abc.codec import Codec

if TYPE_CHECKING:
    from zarr.core.common import JSON


def _codec_as_json(codec: Codec | dict[str, JSON]) -> dict[str, JSON]:
    if isinstance(codec, Codec):
        return codec.to_dict()

    return codec


def _codec_pipeline_as_json(
    codecs: list[Codec | dict[str, JSON]] | None,
) -> list[dict[str, JSON]] | None:
    if codecs is None:
        return codecs

    return [_codec_as_json(codec) for codec in codecs]
