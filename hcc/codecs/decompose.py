# goal:
# - transform from chunk order to chunk tile order (using the z-order curve)
# - downsample
# - upsample
# - compute differences
# - apply subcodec pipelines
# - merge scalars
from dataclasses import asdict, dataclass, field, replace
from typing import ClassVar, Self

import numpy as np
from zarr.abc.codec import (
    ArrayArrayCodec,
    ArrayBytesCodec,
    Codec,
    CodecJSON_V3,
    CodecPipeline,
)
from zarr.codecs import BytesCodec, ZstdCodec
from zarr.core.array_spec import ArrayConfig, ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.dtype import Float32, ZDType
from zarr.registry import get_codec_class, get_pipeline_class

from hcc.codecs.packing import pack_buffers, unpack_buffers
from hcc.codecs.utils import _codec_pipeline_as_json
from hcc.compression.decompose import downsample, upsample
from hcc.compression.zorder import expand_tile, flatten_tile

CodecLike = Codec | CodecJSON_V3


def default_codec_pipeline(dtype: ZDType) -> list[Codec]:
    serializer = BytesCodec(endian=getattr(dtype, "endianness", None))
    compressor = ZstdCodec()

    return [serializer, compressor]


@dataclass(frozen=True)
class ArrayConfigWithScalars(ArrayConfig):
    scalars: dict[str, Buffer] = field(default_factory=dict)


def array_spec_from_array_spec(
    array_spec: ArraySpec, dtype: ZDType, shape: tuple[int, ...] | None = None
) -> ArraySpec:
    config = ArrayConfigWithScalars(**asdict(array_spec.config))

    to_replace = {"config": config, "dtype": dtype}
    if shape is not None:
        to_replace["shape"] = shape

    return replace(array_spec, **to_replace)


def parse_codec(codec: CodecLike) -> Codec:
    if isinstance(codec, str):
        return get_codec_class(codec)()
    elif isinstance(codec, dict):
        codec_cls = get_codec_class(codec["name"])

        codec.setdefault("configuration", {})

        return codec_cls.from_dict(codec)
    else:
        return codec


def extract_scalar_order(codecs: list[ArrayArrayCodec]) -> list[str]:
    order = []
    for codec in codecs:
        order.extend(getattr(codec, "scalars", []))

    return order


async def pipeline_encode(
    pipeline: CodecPipeline, chunk_array: NDBuffer, array_spec: ArraySpec
) -> Buffer:
    encoded = await pipeline.encode([(chunk_array, array_spec)])
    scalar_order = extract_scalar_order(pipeline.array_array_codecs)
    scalars = [array_spec.config.scalars[name] for name in scalar_order]

    return pack_buffers(array_spec.prototype, [*encoded, *scalars])


async def pipeline_decode(
    pipeline: CodecPipeline, chunk_data: Buffer, array_spec: ArraySpec
) -> NDBuffer:
    array_buffer, *scalar_buffers = unpack_buffers(array_spec.prototype, chunk_data)
    scalar_order = extract_scalar_order(pipeline.array_array_codecs)
    array_spec.config.scalars.update(dict(zip(scalar_order, scalar_buffers)))

    decoded = await pipeline.decode([(array_buffer, array_spec)])

    return decoded[0]


@dataclass(frozen=True)
class HccDecompose(ArrayBytesCodec):
    name: ClassVar[str] = "hcc-decompose"

    factor: int

    codecs: dict[str, list[Codec] | None] = field(default_factory=list)

    def __post_init__(self):
        for name in self.codecs:
            codec = self.codecs[name]
            if codec is not None:
                continue

            self.codecs[name] = default_codec_pipeline(
                dtype=Float32(endianness="little")
            )

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, cls.name, require_configuration=True
        )
        if not configuration_parsed:
            configuration_parsed = {}

        if "codecs" in configuration_parsed:
            configuration_parsed["codecs"] = {
                name: [parse_codec(subcodec) for subcodec in subcodecs]
                for name, subcodecs in configuration_parsed["codecs"].items()
            }

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
        subarray_dtype = Float32(endianness="little")
        config = ArrayConfigWithScalars(order="C", write_empty_chunks=False)

        subarray_spec = ArraySpec(
            shape=array_spec.shape,
            dtype=subarray_dtype,
            fill_value=np.nan,
            config=config,
            prototype=array_spec.prototype,
        )

        new_codecs = {
            name: [codec.evolve_from_array_spec(subarray_spec) for codec in pipeline]
            for name, pipeline in self.codecs.items()
        }

        return replace(self, codecs=new_codecs)

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
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
        pipeline_cls = get_pipeline_class()
        prototype = chunk_spec.prototype

        nside = int(np.sqrt(chunk_spec.shape[0]))
        downsampled_nside = nside // self.factor

        unpacked_buffers = unpack_buffers(chunk_spec.prototype, chunk_data)
        downsampled_buffer = await pipeline_decode(
            pipeline_cls.from_codecs(self.codecs["downsampled"]),
            unpacked_buffers[0],
            array_spec_from_array_spec(
                chunk_spec,
                dtype=Float32(endianness="little"),
                shape=(downsampled_nside, downsampled_nside),
            ),
        )
        downsampled = downsampled_buffer.as_numpy_array()

        residuals_buffer = await pipeline_decode(
            pipeline_cls.from_codecs(self.codecs["residuals"]),
            unpacked_buffers[1],
            array_spec_from_array_spec(
                chunk_spec,
                dtype=Float32(endianness="little"),
                shape=(nside, nside),
            ),
        )
        residuals = residuals_buffer.as_numpy_array()

        upsampled = upsample(downsampled, self.factor)

        reconstructed = flatten_tile(upsampled + residuals)

        return prototype.nd_buffer.from_ndarray_like(reconstructed)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer:
        pipeline_cls = get_pipeline_class()
        prototype = chunk_spec.prototype

        data = expand_tile(chunk_array.as_numpy_array())

        downsampled = downsample(data, self.factor)
        upsampled = upsample(downsampled, self.factor)

        residuals = data - upsampled

        print(
            "encode:", downsampled.shape, upsampled.shape, residuals.shape, self.factor
        )

        # encode downsampled
        downsampled_bytes = await pipeline_encode(
            pipeline_cls.from_codecs(self.codecs["downsampled"]),
            prototype.nd_buffer.from_numpy_array(downsampled),
            array_spec_from_array_spec(
                chunk_spec,
                dtype=Float32(endianness="little"),
                shape=downsampled.shape,
            ),
        )

        # encode residuals
        residuals_bytes = await pipeline_encode(
            pipeline_cls.from_codecs(self.codecs["residuals"]),
            prototype.nd_buffer.from_numpy_array(residuals),
            array_spec_from_array_spec(
                chunk_spec,
                dtype=Float32(endianness="little"),
                shape=residuals.shape,
            ),
        )

        # pack into a single buffer
        return pack_buffers(prototype, [downsampled_bytes, residuals_bytes])
