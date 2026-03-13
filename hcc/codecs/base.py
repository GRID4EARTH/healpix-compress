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


def pack_array(prototype, arrays):
    length_dtype = np.dtype("uint64")
    array_buffers = [
        prototype.buffer.from_array_like(array.ravel().view(dtype="B"))
        for array in arrays
    ]
    lengths = [x.as_array_like().size for x in array_buffers]
    length_buffer = prototype.buffer.from_array_like(
        np.asarray(lengths, dtype=length_dtype).view(dtype="B")
    )

    return prototype.buffer.create_zero_length().combine(
        array_buffers + [length_buffer]
    )


def unpack_array(prototype, chunk_data, dtypes):
    length_dtype = np.dtype("uint64")
    n_arrays = len(dtypes)
    n_footer_bytes = n_arrays * length_dtype.nbytes

    footer = chunk_data[-n_footer_bytes:]
    lengths = footer.as_numpy_array().view(dtype=length_dtype)

    offsets = np.cumulative_sum(lengths, include_initial=True)

    slices = [
        slice(offset, offset + size) for offset, size in zip(offsets[:-1], lengths)
    ]

    return [
        chunk_data[slice_].as_numpy_array().view(dtype=dtype.to_native_dtype())
        for slice_, dtype in zip(slices, dtypes)
    ]


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
