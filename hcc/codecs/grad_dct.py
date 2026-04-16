from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import parse_named_configuration
from zarr.dtype import Float32, Int8, Int16

from hcc.codecs.scalars import decode_scalar, encode_scalar
from hcc.compression import grad_dct

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.buffer import Buffer, NDBuffer
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import JSON
    from zarr.dtype import ZDType

    from hcc.codecs.decompose import HccDecomposeCodec


@dataclass(frozen=True)
class HccGradDctCodec(ArrayArrayCodec):
    name: ClassVar[str] = "hcc-grad-dct"

    window: int = 8
    quantize_bits: int = 8
    sigma_floor_ratio: float = 0.05
    clip_percentile: float = 0.5

    dtypes: ClassVar[dict[str, ZDType]] = {
        "q_lo": Float32(endianness="little"),
        "q_hi": Float32(endianness="little"),
        "sigma_floor": Float32(endianness="little"),
    }

    @property
    def scalars(self):
        """Order of storage of the scalars calculated by this codec."""
        return list(self.dtypes.keys())

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
                "quantize_bits": self.quantize_bits,
                "sigma_floor_ratio": self.sigma_floor_ratio,
                "clip_percentile": self.clip_percentile,
            },
        }

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        target_dtypes = {
            8: Int8(),
            16: Int16(endianness="little"),
        }
        target_zdtype = target_dtypes.get(self.quantize_bits)
        if target_zdtype is None:
            raise ValueError(f"unknown bitsize: {self.quantize_bits}.")

        new_fill = np.iinfo(target_zdtype.to_native_dtype()).min

        return replace(chunk_spec, dtype=target_zdtype, fill_value=new_fill)

    async def _decode_single(
        self, chunk_data: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        prototype = chunk_spec.prototype

        scalars = {
            name: decode_scalar(
                prototype, buffer, dtype=self.dtypes[name]
            ).as_numpy_array()
            for name, buffer in chunk_spec.config.scalars.items()
        }
        chunk_array = grad_dct.decompress_residuals(
            chunk_data.as_numpy_array(),
            prediction=chunk_spec.config.context["prediction"],
            window=self.window,
            quantize_bits=self.quantize_bits,
            **scalars,
        )
        return prototype.nd_buffer.from_ndarray_like(chunk_array)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        prototype = chunk_spec.prototype

        compressed = grad_dct.compress_residuals(
            chunk_array.as_numpy_array(),
            prediction=chunk_spec.config.context["prediction"],
            window=self.window,
            quantize_bits=self.quantize_bits,
            clip_percentile=self.clip_percentile,
            sigma_floor_ratio=self.sigma_floor_ratio,
        )
        encoded_data, q_lo, q_hi, sigma_floor = map(
            prototype.nd_buffer.from_ndarray_like, compressed
        )

        chunk_spec.config.scalars.update(
            {
                "q_lo": encode_scalar(prototype, q_lo),
                "q_hi": encode_scalar(prototype, q_hi),
                "sigma_floor": encode_scalar(prototype, sigma_floor),
            }
        )

        return encoded_data


def lookup_preset(name: str) -> HccDecomposeCodec:
    from zarr.codecs import BytesCodec, GzipCodec

    from hcc.codecs.decompose import HccDecomposeCodec
    from hcc.compression.grad_dct import lookup_preset

    presets = lookup_preset(name)
    factor = presets.pop("downsample_factor")
    level = presets.pop("zlib_level")

    return HccDecomposeCodec(
        factor=factor,
        codecs={
            "downsampled": [BytesCodec(), GzipCodec(level=level)],
            "residuals": [
                HccGradDctCodec(**presets),
                BytesCodec(),
                GzipCodec(level=level),
            ],
        },
    )
