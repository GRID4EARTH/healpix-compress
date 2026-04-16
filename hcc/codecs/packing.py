import numpy as np
from zarr.abc.buffer import Buffer, BufferPrototype


def pack_buffers(prototype: BufferPrototype, array_buffers: list[Buffer]) -> Buffer:
    if len(array_buffers) > 255:
        raise ValueError("More than the maximum number of buffers (255) found.")

    length_dtype = np.dtype("uint64")
    lengths = [x.as_array_like().size for x in array_buffers]
    length_buffer = prototype.buffer.from_array_like(
        np.asarray(lengths, dtype=length_dtype).view(dtype="B")
    )
    size_buffer = prototype.buffer.from_bytes(bytes([len(array_buffers)]))

    return prototype.buffer.create_zero_length().combine(
        array_buffers + [length_buffer, size_buffer]
    )


def unpack_buffers(prototype: BufferPrototype, packed: Buffer) -> list[Buffer]:
    length_dtype = np.dtype("uint64")
    n_buffers = packed[-1:].to_bytes()[0]
    n_footer_bytes = n_buffers * length_dtype.itemsize
    footer = packed[-n_footer_bytes - 1 : -1]

    lengths = footer.as_numpy_array().view(dtype=length_dtype)

    offsets = np.cumulative_sum(lengths, include_initial=True)

    slices = [
        slice(offset, offset + size) for offset, size in zip(offsets[:-1], lengths)
    ]

    return [packed[slice_] for slice_ in slices]


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
    n_footer_bytes = n_arrays * length_dtype.itemsize

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
