from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Self

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.buffer.cpu import numpy_buffer_prototype
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import Buffer, NDBuffer
from zarr.core.common import JSON, parse_named_configuration
from zarr.dtype import UInt64, ZDType, parse_dtype

from hcc.codecs.packing import pack_array, unpack_array


@dataclass(frozen=True)
class HccBaseCodec(ArrayBytesCodec):
    delta: int
    top_k: int

    quantization_dtype: np.typing.DTypeLike = "int16"

    dtypes: list[ZDType] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        _, configuration_parsed = parse_named_configuration(
            data, "hcc", require_configuration=True
        )
        configuration_parsed = configuration_parsed or {}

        return cls(**configuration_parsed)

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        value_dtype = array_spec.dtype
        dtypes = [
            value_dtype,
            value_dtype,
            parse_dtype(self.quantization_dtype, zarr_format=3),
            UInt64(endianness="little"),
        ]

        return replace(self, dtypes=dtypes)

    def to_dict(self) -> dict[str, JSON]:
        return {
            "name": "hcc",
            "configuration": {
                "delta": self.delta,
                "top_k": self.top_k,
                "quantization_dtype": self.quantization_dtype,
            },
        }

    def compress(self, chunk: np.ndarray) -> list[np.ndarray]:
        raise NotImplementedError("not implemented")

    def decompress(self, components: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError("not implemented")

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer | None:
        prototype = numpy_buffer_prototype()

        # To decode, we need: dtypes, endianness, and shape (but all arrays are 1D, so shape is not needed)
        compressed = unpack_array(prototype, chunk_data, self.dtypes)

        chunk_array = self.decompress(compressed)

        return prototype.nd_buffer.from_ndarray_like(chunk_array)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        # encoding scheme:
        # 1. extract arrays
        # 2. convert arrays to bytes (TODO: use codec pipeline)
        # 3. compute byte lengths
        # 4. convert byte length array to bytes
        # 5. concatenate

        compressed = self.compress(chunk_array.as_numpy_array())

        prototype = numpy_buffer_prototype()

        return pack_array(prototype, compressed)
