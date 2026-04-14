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
from zarr.registry import get_pipeline_class

from hcc.codecs.utils import _codec_pipeline_as_json
from hcc.compression.decompose import downsample, upsample
from hcc.compression.zorder import expand_tile

CodecLike = Codec | CodecJSON_V3


@dataclass(frozen=True)
class HccDecompose(ArrayBytesCodec):
    name: ClassVar[str] = "hcc-decompose"

    factor: int

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
            "configuration": {"factor": self.factor, "codecs": codec_json},
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        # modifications:
        # - dtypes for downsampled / residuals
        # - scalar dtypes
        # - endianness for bytes codecs
        pass

    def _decode_sync(self, chunk_data: Buffer, chunk_spec: ArraySpec) -> NDBuffer:
        # steps:
        # - split into array buffers (lookup table at the end)
        # - for each array:
        #   - split into scalar and array buffer
        #   - construct array spec for each array (pass along scalars)
        #   - construct codec pipelines
        #   - call decode
        # - upsample downsampled
        # - add residuals to upsampled
        # - flatten tile
        # - construct NDBuffer
        pass

    def _encode_sync(self, chunk_array: NDBuffer, chunk_spec: ArraySpec) -> Buffer:
        pipeline_cls = get_pipeline_class()

        data = expand_tile(chunk_array.as_numpy_array())
        downsampled = downsample(data, self.factor)
        upsampled = upsample(downsampled, self.factor)
        residuals = data - upsampled  # noqa: F841

        # steps (to repeat for each array):
        # - construct array spec
        # - construct codec pipeline
        # - construct ndbuffer
        # - call encode
        # - merge result with scalar buffers
        downsampled_pipeline = pipeline_cls.from_codecs(  # noqa: F841
            self.codecs["downsampled"]
        )

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        return self._decode_sync(chunk_data, chunk_spec)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer:
        return self._encode_sync(chunk_array, chunk_spec)
