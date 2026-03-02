from dataclasses import dataclass

import numpy as np
from scipy.fft import dctn


@dataclass
class CompressedResiduals:
    quantization: np.ndarray
    scale: np.ndarray


@dataclass
class CompressedChunk:
    mean: np.ndarray
    residuals: CompressedResiduals


def _decode_zorder(delta: int):
    size = 4**delta
    x = np.zeros((size,), dtype="uint64")
    y = np.zeros((size,), dtype="uint64")

    code = np.arange(size, dtype="uint64")
    for bit in range(delta):
        x |= ((code >> (bit << 1)) & 1) << bit
        y |= ((code >> (bit << 1 + 1)) & 1) << bit

    return x, y


def _flatten_tile(data: np.ndarray):
    delta = int(np.log(data.size) // np.log(4))
    x, y = _decode_zorder(delta)

    return delta[..., x, y]


def _expand_tile(data: np.ndarray):
    delta = int(np.log(data.size) // np.log(4))
    x, y = _decode_zorder(delta)

    # TODO: figure out how to invert x, y to become a 2D array of indexes into data
    out = np.empty(data.size, dtype=data.dtype)
    out[..., x, y] = data

    return out


def compress(
    chunk: np.ndarray,
    delta: int,
    top_k: int,
    quantization_dtype: np.DTypeLike = "int16",
    scale_dtype: np.DTypeLike = "float32",
) -> CompressedChunk:
    reduction_shape = (*chunk.shape[:-1], -1, 4**delta)

    reshaped = np.reshape(chunk, reduction_shape)
    mean = np.mean(reshaped, axis=-1, keepdims=True)

    residuals = np.reshape(reshaped - mean, chunk.shape)
    compressed_residuals = _compress_residuals_dct(
        residuals,
        top_k=top_k,
        quantization_dtype=quantization_dtype,
        scale_dtype=scale_dtype,
    )

    return CompressedChunk(mean, compressed_residuals)


def _compress_residuals_dct(
    residuals: np.ndarray,
    top_k: int,
    quantization_dtype: np.DTypeLike,
    scale_dtype: np.DTypeLike,
) -> CompressedResiduals:
    if residuals.ndim != 1:
        raise ValueError("residuals compression only works for 1D arrays")

    expanded_residuals = _expand_tile(residuals)
    coefficients = np.reshape(dctn(expanded_residuals, norm="ortho"), (-1,))

    # keep the top_k largest magnitude coefficients
    to_keep = np.argsort(np.abs(coefficients), descending=True)[:top_k]
    kept_coefficients = coefficients[to_keep]

    if kept_coefficients.size == 0:
        scale = 1.0
    else:
        scale = np.max(np.abs(kept_coefficients))
        if scale < 0:
            scale = 1.0

    max_value = np.iinfo(quantization_dtype).max
    quantization = np.round((kept_coefficients / scale) * max_value)

    return CompressedResiduals(
        quantization=np.astype(quantization, quantization_dtype),
        scale=np.astype(scale, scale_dtype),
    )


def decompress(compressed: CompressedChunk, delta: int) -> np.ndarray:
    pass
