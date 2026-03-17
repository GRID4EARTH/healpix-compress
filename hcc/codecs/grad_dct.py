from typing import ClassVar, Self
from dataclasses import dataclass, field, replace

import numpy as np
from zarr.codecs import register_codec
from zarr.core.common import JSON, parse_named_configuration

from hcc.codecs.base import HccBaseCodec
from hcc.compression import grad_dct
from zarr.dtype import UInt64
from zarr.core.array_spec import ArraySpec
from zarr.core.dtype import parse_dtype


@dataclass(frozen=True)
class HccDctCodec(HccBaseCodec):
    name: ClassVar[str] = "hcc-grad-dct"

    window: int = 8

    # Champs hérités de HccBaseCodec — inutilisés ici, valeurs par défaut obligatoires
    delta: int = 0
    top_k: int = 0

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
            },
        }

    def _build_dtypes(self):
        """Dtypes des 6 tableaux retournés par encode() — toujours fixes,
        indépendants du dtype d'entrée zarr.

        encode() retourne TOUJOURS :
          [uint64, float32, float32, float32, float32, uint8]
        peu importe le dtype de la donnée d'entrée.
        """
        f32 = parse_dtype("float32", zarr_format=3)
        u8  = parse_dtype("uint8",   zarr_format=3)
        return [
            UInt64(endianness="little"),  # n           -> uint64 (1 element)
            f32,                          # low         -> float32
            f32,                          # q_lo        -> float32 (1 element)
            f32,                          # q_hi        -> float32 (1 element)
            f32,                          # sigma_floor -> float32 (1 element)
            u8,                           # residual_bytes -> uint8 (variable size)
        ]

    def evolve_from_array_spec(self, array_spec: ArraySpec) -> Self:
        return replace(self, dtypes=self._build_dtypes())

    def compress(self, data: np.ndarray) -> list[np.ndarray]:
        return grad_dct.compress(data, window=self.window)

    def decompress(self, components: list[np.ndarray]) -> np.ndarray:
        return grad_dct.decompress(components, window=self.window)

    async def _decode_single(self, chunk_data, chunk_spec: ArraySpec):
        from zarr.buffer.cpu import numpy_buffer_prototype
        from hcc.codecs.packing import unpack_array

        prototype = numpy_buffer_prototype()

        # Dtypes fixes — independants de chunk_spec.dtype (qui est le dtype zarr
        # d'entree, pas le dtype interne des tableaux compresses)
        dtypes = self._build_dtypes()

        compressed = unpack_array(prototype, chunk_data, dtypes)
        chunk_array = self.decompress(compressed)
        return prototype.nd_buffer.from_ndarray_like(chunk_array)


register_codec(HccDctCodec.name, HccDctCodec)
