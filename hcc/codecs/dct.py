import numpy as np
from zarr.codecs import register_codec

from hcc import compression
from hcc.codecs.base import HccBaseCodec


class HccDctCodec(HccBaseCodec):
    def compress(self, data: np.ndarray) -> list[np.ndarray]:
        compressed = compression.dct.compress(
            data,
            delta=self.delta,
            top_k=self.top_k,
            quantization_dtype=self.quantization_dtype,
            scale_dtype=data.dtype,
        )

        return compressed.extract_arrays()

    def decompress(self, components: list[np.ndarray]) -> np.ndarray:
        compressed = compression.dct.CompressedChunk.from_arrays(self.delta, components)
        return compression.dct.decompress(
            compressed, delta=self.delta, top_k=self.top_k
        )


register_codec("hcc-dct", HccDctCodec)
