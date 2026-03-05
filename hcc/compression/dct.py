from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.fft import dctn, idctn


@dataclass
class CompressedResiduals:
    quantization: np.ndarray
    scale: np.ndarray
    coefficient_positions: np.ndarray


@dataclass
class CompressedChunk:
    delta: int
    mean: np.ndarray
    residuals: CompressedResiduals

    @classmethod
    def from_arrays(cls, delta: int, arrays: list[np.ndarray]) -> Self:
        mean, scale, quantization, coefficient_positions = arrays

        residuals = CompressedResiduals(quantization, scale, coefficient_positions)

        return cls(delta, mean, residuals)

    def extract_arrays(self) -> tuple[np.ndarray]:
        return [
            self.mean,
            self.residuals.scale,
            self.residuals.quantization,
            self.residuals.coefficient_positions,
        ]


def _decode_zorder(delta: int):
    size = 4**delta
    x = np.zeros((size,), dtype="uint64")
    y = np.zeros((size,), dtype="uint64")

    code = np.arange(size, dtype="uint64")
    for bit in range(delta):
        x |= ((code >> (bit << 1)) & 1) << bit
        y |= ((code >> ((bit << 1) + 1)) & 1) << bit

    return x, y


def _flatten_tile(data: np.ndarray):
    delta = int(np.log(data.size) / np.log(4))
    x, y = _decode_zorder(delta)

    return np.squeeze(data[..., x, y])


def _expand_tile(data: np.ndarray):
    nside = int(np.sqrt(data.shape[-1]))
    delta = int(np.log2(nside))

    x, y = _decode_zorder(delta)

    shape = (*data.shape[:-1], 2**delta, 2**delta)

    # TODO: figure out how to invert x, y to become a 2D array of indexes into data
    out = np.empty(shape, dtype=data.dtype)
    out[..., x, y] = data

    return out


def _compress_residuals_dct(
    residuals: np.ndarray,
    top_k: int,
    quantization_dtype: np.typing.DTypeLike,
    scale_dtype: np.typing.DTypeLike,
) -> CompressedResiduals:
    expanded_residuals = _expand_tile(residuals)
    coefficients = np.reshape(dctn(expanded_residuals, norm="ortho"), (-1,))

    # keep the top_k largest magnitude coefficients
    to_keep = np.argsort(np.abs(coefficients))[::-1][:top_k]
    kept_coefficients = coefficients[to_keep]

    if kept_coefficients.size == 0:
        scale = np.array(1.0, dtype=scale_dtype)
    else:
        scale = np.asarray(np.max(np.abs(kept_coefficients)), dtype=scale_dtype)
        if scale < 0:
            scale = np.array(1.0, dtype=scale_dtype)

    max_value = np.iinfo(quantization_dtype).max
    quantization = np.round((kept_coefficients / scale) * max_value)

    return CompressedResiduals(
        quantization=np.astype(quantization, quantization_dtype),
        scale=np.astype(scale, scale_dtype),
        coefficient_positions=to_keep,
    )


def _decompress_residuals_dct(
    compressed_residuals: CompressedResiduals,
    shape: tuple[int, int],
    residuals_dtype: np.typing.DTypeLike,
    top_k: int,
):
    quantization = compressed_residuals.quantization
    scale = compressed_residuals.scale
    coefficient_positions = compressed_residuals.coefficient_positions

    kept_coefficients = np.astype(
        (quantization / np.iinfo(quantization.dtype).max) * scale,
        scale.dtype,
    )

    coefficients = np.reshape(np.zeros(shape, dtype=residuals_dtype), -1)
    coefficients[coefficient_positions] = kept_coefficients

    tile = idctn(np.reshape(coefficients, shape), norm="ortho")

    return np.astype(_flatten_tile(tile), residuals_dtype)


def compress(
    chunk: np.ndarray,
    delta: int,
    top_k: int,
    quantization_dtype: np.typing.DTypeLike = "int16",
    scale_dtype: np.typing.DTypeLike = "float32",
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

    return CompressedChunk(
        delta=delta,
        mean=np.astype(mean[..., 0], chunk.dtype),
        residuals=compressed_residuals,
    )


def decompress(compressed: CompressedChunk, delta: int, top_k: int) -> np.ndarray:
    delta_mean = int(np.log(compressed.mean.shape[-1]) / np.log(4))
    chunksize = 4 ** (delta_mean + delta)
    nside = int(np.sqrt(chunksize))

    dtype = compressed.mean.dtype
    residuals = _decompress_residuals_dct(
        compressed.residuals,
        shape=(nside, nside),
        residuals_dtype=dtype,
        top_k=top_k,
    )

    reconstructed = np.astype(
        compressed.mean[..., :, None]
        + np.reshape(residuals, (*residuals.shape[:-1], -1, 4**delta)),
        dtype,
    )

    resulting_shape = (*compressed.mean.shape[:-1], -1)
    return np.reshape(reconstructed, resulting_shape)
