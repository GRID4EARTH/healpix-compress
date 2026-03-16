from dataclasses import dataclass

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import scipy.signal


def is_noise(x):
    autocorrelation = scipy.signal.correlate(x, x)
    indices, _ = scipy.signal.find_peaks(autocorrelation, height=x.size * 0.001)

    # TODO: figure out a better criterion for detecting a approximate dirac function
    return len(indices) == 1


@dataclass
class ChunkData:
    data: np.ndarray
    chunksize: tuple[int]


def chunk_data():
    chunk_delta = st.integers(min_value=3, max_value=6)
    chunk_sizes = chunk_delta.map(lambda x: 4**x)

    chunksizes_sizes = st.shared(chunk_sizes, key="chunksize")
    chunksizes_shapes = st.shared(chunk_sizes, key="chunksize")

    spatial_size = st.builds(
        lambda chunk_size, n_chunks: chunk_size // n_chunks,
        chunksizes_shapes,
        st.integers(min_value=0, max_value=2).map(lambda x: 4**x),
    )
    shapes = spatial_size.map(lambda s: (s,))
    # orthogonal_shapes = npst.array_shapes(min_dims=0, max_dims=2)
    # shapes = st.builds(
    #     lambda other, spatial: other + (spatial,), orthogonal_shapes, spatial_size
    # )

    chunk_data = npst.arrays(
        shape=shapes,
        dtype=npst.floating_dtypes(sizes=(32, 64)),
        elements={
            "allow_nan": False,
            "allow_infinity": False,
            "min_value": 0,
            "max_value": 1,
            "allow_subnormal": False,
        },
        unique=True,
    )

    return st.builds(
        lambda arr, s: ChunkData(arr, arr.shape[:-1] + (s,)),
        chunk_data,
        chunksizes_sizes,
    )
