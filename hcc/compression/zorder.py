import math

import numpy as np
import numpy.typing as npt


def _decode_zorder(delta: int) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    size = 4**delta
    x = np.zeros((size,), dtype="uint64")
    y = np.zeros((size,), dtype="uint64")

    code = np.arange(size, dtype="uint64")
    for bit in range(delta):
        x |= ((code >> (bit << 1)) & 1) << bit
        y |= ((code >> ((bit << 1) + 1)) & 1) << bit

    return x, y


def flatten_tile[T](data: npt.NDArray[T]) -> npt.NDArray[T]:
    delta = int(np.log(data.size) / np.log(4))
    x, y = _decode_zorder(delta)

    return np.squeeze(data[..., x, y])


def expand_tile[T](data: npt.NDArray[T]) -> npt.NDArray[T]:
    nside = int(math.sqrt(data.shape[-1]))
    delta = int(math.log2(nside))

    x, y = _decode_zorder(delta)

    shape = (*data.shape[:-1], nside, nside)

    # TODO: figure out how to invert x, y to become a 2D array of indexes into data
    out = np.empty(shape, dtype=data.dtype)
    out[..., x, y] = data

    return out
