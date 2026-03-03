import numpy as np
from zarr.codecs import register_codec

from hcc.codecs.base import HccBaseCodec
from hcc.compression import dct


class HccDctCodec(HccBaseCodec):
    def compress(self, data: np.ndarray) -> list[np.ndarray]:
        compressed = dct.compress(
            data,
            delta=self.delta,
            top_k=self.top_k,
            quantization_dtype=self.quantization_dtype,
            scale_dtype=data.dtype,
        )

        return compressed.extract_arrays()

    def decompress(self, components: list[np.ndarray]) -> np.ndarray:
        compressed = dct.CompressedChunk.from_arrays(self.delta, components)
        return dct.decompress(compressed, delta=self.delta, top_k=self.top_k)


register_codec("hcc-dct", HccDctCodec)
