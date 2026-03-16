import hypothesis.strategies as st
import numpy as np
from hypothesis import given, note

from hcc.compression import dct
from hcc.tests import chunk_data, is_noise


@given(chunk_data())
def test_roundtrip_residuals(chunk_data):
    residuals = chunk_data.data
    note(f"dtype: {residuals.dtype}")
    note(f"chunksize: {chunk_data.chunksize}")
    note(f"shape: {residuals.shape}")

    top_k = 30
    tile_size = residuals.shape[-1]
    nside = int(np.sqrt(tile_size))

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
    data = chunk_data.data
    note(f"dtype: {data.dtype}")
    note(f"chunksize: {chunk_data.chunksize}")
    note(f"shape: {data.shape}")

    delta = 1
    compressed = dct.compress(
        data,
        delta=delta,
        top_k=top_k,
        quantization_dtype="int16",
        scale_dtype=data.dtype,
    )
    decompressed = dct.decompress(compressed, delta=delta, top_k=top_k)

    assert compressed.mean.dtype == data.dtype
    assert compressed.mean.size == data.size / 4**delta
    assert compressed.residuals.quantization.size == min(top_k, data.size)
    assert compressed.residuals.coefficient_positions.size == min(top_k, data.size)

    assert decompressed.shape == data.shape
    assert decompressed.dtype == data.dtype

    # properties of a lossy compression algorithm:
    # - if no signal in input data → no signal in output data
    # - if signal in the input: strong correlation between input and output data
    assert (is_noise(data) and is_noise(decompressed)) or np.corrcoef(
        decompressed, data
    )[0, 1] >= 0.5
