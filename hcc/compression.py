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


def compress(
    chunk: np.ndarray,
    delta: int,
    top_k: int,
    quantization_dtype: np.DTypeLike = "int16",
    scale_dtype: np.DTypeLike = "float32",
) -> CompressedChunk:
    reduction_shape = (-1, 4**delta, *chunk.shape[1:])

    reshaped = np.reshape(chunk, reduction_shape)
    mean = np.mean(reshaped, axis=1, keepdims=True)

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

    coefficients = np.reshape(dctn(residuals, norm="ortho"), (-1,))

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
