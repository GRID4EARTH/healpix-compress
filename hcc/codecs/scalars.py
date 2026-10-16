import numpy as np
from zarr.abc.buffer import Buffer, BufferPrototype, NDBuffer
from zarr.dtype import ZDType


def decode_scalar(
    prototype: BufferPrototype, encoded: Buffer, dtype: ZDType
) -> NDBuffer:
    binary_data = encoded.as_numpy_array()

    decoded = binary_data.view(dtype=dtype.to_native_dtype())
    return prototype.nd_buffer.from_ndarray_like(decoded)


def encode_scalar(prototype: BufferPrototype, encoded: NDBuffer) -> Buffer:
    data = encoded.as_numpy_array()

    encoded = np.ascontiguousarray(data).reshape(-1).view(dtype="B")
    return prototype.buffer.from_array_like(encoded)
