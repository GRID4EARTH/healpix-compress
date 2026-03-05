import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import scipy.signal
from hypothesis import given, note

from hcc.compression import dct


def is_noise(x):
    autocorrelation = scipy.signal.correlate(x, x)
    indices, _ = scipy.signal.find_peaks(autocorrelation, height=x.size * 0.001)

    # TODO: figure out a better criterion for detecting a approximate dirac function
    return len(indices) == 1


def chunk_data():
    chunk_delta = st.integers(min_value=3, max_value=6)
    chunk_sizes = chunk_delta.map(lambda x: 4**x)

    # shapes = st.builds(
    #     lambda chunk_size, shape: shape + (chunk_size,),
    #     chunk_sizes,
    #     npst.array_shapes(min_dims=0, max_dims=2, max_side=5)
    # )
    shapes = chunk_sizes.map(lambda x: x)

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

    return chunk_data


@given(chunk_data())
def test_roundtrip_residuals(residuals):
    top_k = 30
    tile_size = residuals.shape[-1]
    nside = int(np.sqrt(tile_size))

    note(f"dtype: {residuals.dtype}")

    compressed = dct._compress_residuals_dct(
        residuals, top_k, quantization_dtype=np.int16, scale_dtype=residuals.dtype
    )

    decompressed = dct._decompress_residuals_dct(
        compressed, shape=(nside, nside), residuals_dtype=residuals.dtype, top_k=top_k
    )

    assert compressed.scale.dtype == residuals.dtype
    assert compressed.quantization.size == min(top_k, residuals.size)
    assert compressed.coefficient_positions.size == min(top_k, residuals.size)

    assert decompressed.size == residuals.size
    assert decompressed.dtype == residuals.dtype

    # properties of a lossy compression algorithm:
    # - if no signal in input data → no signal in output data
    # - if signal in the input: strong correlation between input and output data
    assert (is_noise(residuals) and is_noise(decompressed)) or np.corrcoef(
        decompressed, residuals
    )[0, 1] >= 0.5


@given(chunk_data(), st.integers(min_value=1, max_value=10).map(lambda x: x * 10))
def test_roundtrip_chunk(chunk_data, top_k):
    note(f"chunk shape: {chunk_data.shape}")

    delta = 1
    compressed = dct.compress(
        chunk_data,
        delta=delta,
        top_k=top_k,
        quantization_dtype="int16",
        scale_dtype=chunk_data.dtype,
    )
    decompressed = dct.decompress(compressed, delta=delta, top_k=top_k)

    assert compressed.mean.dtype == chunk_data.dtype
    assert compressed.mean.size == chunk_data.size / 4**delta
    assert compressed.residuals.quantization.size == min(top_k, chunk_data.size)
    assert compressed.residuals.coefficient_positions.size == min(
        top_k, chunk_data.size
    )

    assert decompressed.shape == chunk_data.shape
    assert decompressed.dtype == chunk_data.dtype

    # properties of a lossy compression algorithm:
    # - if no signal in input data → no signal in output data
    # - if signal in the input: strong correlation between input and output data
    assert (is_noise(chunk_data) and is_noise(decompressed)) or np.corrcoef(
        decompressed, chunk_data
    )[0, 1] >= 0.5
