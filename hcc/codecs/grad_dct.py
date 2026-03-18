from dataclasses import dataclass, field, replace
from typing import ClassVar, Self

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.buffer.cpu import numpy_buffer_prototype
from zarr.codecs import register_codec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.dtype import Float32, UInt8, UInt64, ZDType

from hcc.codecs.packing import pack_array, unpack_array
from hcc.compression import grad_dct


@dataclass(frozen=True)
class HccGradDctCodec(ArrayBytesCodec):
    name: ClassVar[str] = "hcc-grad-dct"

    window: int = 8

    dtypes: list[ZDType] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, cls.name, require_configuration=True
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": self.name,
            "configuration": {
                "window": self.window,
            },
        }

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        """dtypes of the 6 arrays are fixed, regardless of the input dtype

        dtypes:

        - n : uint64
        - low : float32
        - q_lo : float32
        - q_hi : float32
        - sigma_floor : float32
        - residual_bytes : uint8
        """
        f32 = Float32(endianness="little")
        u8 = UInt8()
        dtypes = [
            UInt64(endianness="little"),  # n           -> uint64 (1 element)
            f32,  # low         -> float32
            f32,  # q_lo        -> float32 (1 element)
            f32,  # q_hi        -> float32 (1 element)
            f32,  # sigma_floor -> float32 (1 element)
            u8,  # residual_bytes -> uint8 (variable size)
        ]

        return replace(self, dtypes=dtypes)

    def compress(self, data: np.ndarray) -> list[np.ndarray]:
        return grad_dct.compress(data, window=self.window)

    def decompress(self, components: list[np.ndarray]) -> np.ndarray:
        return grad_dct.decompress(components, window=self.window)

    async def _decode_single(self, chunk_data: Buffer, chunk_spec: ArraySpec):
        prototype = numpy_buffer_prototype()

        compressed = unpack_array(prototype, chunk_data, self.dtypes)
        chunk_array = self.decompress(compressed)
        return prototype.nd_buffer.from_ndarray_like(chunk_array)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        compressed = self.compress(chunk_array.as_numpy_array())

        prototype = numpy_buffer_prototype()

        return pack_array(prototype, compressed)


register_codec(HccGradDctCodec.name, HccGradDctCodec)
