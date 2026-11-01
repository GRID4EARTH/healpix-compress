import functools
import math
from typing import TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=np.floating)


@functools.lru_cache(maxsize=16)
def _decode_zorder(delta: int) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    # cached: a chunk size has one Morton order, and it is asked for at every chunk
    # (the arrays are returned read-only so that no caller can alter the cache)
    size = 4**delta
    x = np.zeros((size,), dtype="uint64")
    y = np.zeros((size,), dtype="uint64")

    code = np.arange(size, dtype="uint64")
    for bit in range(delta):
        x |= ((code >> (bit << 1)) & 1) << bit
        y |= ((code >> ((bit << 1) + 1)) & 1) << bit

    x.flags.writeable = False
    y.flags.writeable = False
    return x, y


@functools.lru_cache(maxsize=16)
def _linear_zorder(delta: int) -> npt.NDArray[np.intp]:
    # the Morton order as positions in the C-ordered raster: one take instead of
    # two fancy indices (cached and read-only, as _decode_zorder)
    x, y = _decode_zorder(delta)
    lin = (x << np.uint64(delta) | y).astype(np.intp)
    lin.flags.writeable = False
    return lin


def flatten_tile(data: npt.NDArray[T]) -> npt.NDArray[T]:
    delta = int(np.log(data.size) / np.log(4))
    if delta >= 1 and data.ndim == 2 and data.shape == (2**delta, 2**delta):
        return data.reshape(-1)[_linear_zorder(delta)]

    x, y = _decode_zorder(delta)

    return np.squeeze(data[..., x, y])


def expand_tile(data: npt.NDArray[T]) -> npt.NDArray[T]:
    nside = int(math.sqrt(data.shape[-1]))
    delta = int(math.log2(nside))

    x, y = _decode_zorder(delta)

    shape = (*data.shape[:-1], nside, nside)

    # TODO: figure out how to invert x, y to become a 2D array of indexes into data
    out = np.empty(shape, dtype=data.dtype)
    if data.ndim == 1:
        out.reshape(-1)[_linear_zorder(delta)] = data
        return out
    out[..., x, y] = data

    return out
