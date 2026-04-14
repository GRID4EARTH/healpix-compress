# goal:
# - transform from chunk order to chunk tile order (using the z-order curve)
# - downsample
# - upsample
# - compute differences
# - apply subcodec pipelines
# - merge scalars
from dataclasses import dataclass, field
from typing import ClassVar, Self

from zarr.abc.codec import ArrayBytesCodec, Codec, CodecJSON_V3
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration

from hcc.codecs.utils import _codec_pipeline_as_json

CodecLike = Codec | CodecJSON_V3


@dataclass(frozen=True)
class HccDecompose(ArrayBytesCodec):
    name: ClassVar[str] = "hcc-decompose"

    delta: int

    codecs: dict[str, list[CodecLike] | None] = field(factory_method=list)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, cls.name, require_configuration=True
        )
        if not configuration_parsed:
            configuration_parsed = {}

        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        codec_json = {
            name: _codec_pipeline_as_json(pipeline)
            for name, pipeline in self.codecs.items()
        }
        return {
            "name": self.name,
            "configuration": {"delta": self.delta, "codecs": codec_json},
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        pass

    def _decode_sync(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        pass

    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        pass

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        pass

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer:
        pass
