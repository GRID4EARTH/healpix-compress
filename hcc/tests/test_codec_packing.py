import numpy as np
import zarr
from zarr.buffer.cpu import numpy_buffer_prototype

from hcc.codecs import packing


def test_pack_arrays():
    arrays = [
        np.linspace(0, 1, 200, dtype="float64"),
        np.arange(50, dtype="uint64"),
        np.array([2, 8, 16], dtype="uint8"),
    ]

    packed = packing.pack_array(
        numpy_buffer_prototype(),
        arrays,
    )
    actual = packed.as_array_like()

    footer_size = len(arrays) * 8
    assert actual.size == sum(array.nbytes for array in arrays) + footer_size

    footer = actual[-footer_size:].view(dtype="uint64")
    expected_footer = np.array([arr.nbytes for arr in arrays], dtype="uint64")
    np.testing.assert_equal(footer, expected_footer)


def test_unpack_arrays():
    footer = np.array([16, 40, 8], dtype="uint64").view(dtype="B")
    data = [
        np.array([45.1, 60.4, 15.2, 92.776], dtype="float32"),
        np.array([42, 31, 83, 190, 320], dtype="uint64"),
        np.array([-1], dtype="int64"),
    ]
    dtypes = [zarr.dtype.Float32(), zarr.dtype.UInt64(), zarr.dtype.Int64()]
    packed = np.concatenate([x.view(dtype="B") for x in data] + [footer])

    prototype = numpy_buffer_prototype()
    unpacked = packing.unpack_array(
        prototype, prototype.buffer.from_array_like(packed), dtypes
    )

    for actual, expected in zip(unpacked, data):
        np.testing.assert_equal(actual, expected)
