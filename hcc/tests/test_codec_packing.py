import numpy as np
import zarr
from zarr.buffer.cpu import numpy_buffer_prototype

from hcc.codecs import packing


def test_pack_arrays():
    arrays = [
        np.linspace(0, 1, 6, dtype="float16"),
        np.arange(4, dtype="uint64"),
        np.array([2, 8, 16], dtype="uint8"),
    ]

    prototype = numpy_buffer_prototype()
    packed = packing.pack_array(prototype, arrays)

    expected_bytes = b"".join(
        [
            b"\x00\x00\x66\x32\x66\x36\xcd\x38\x66\x3a\x00\x3c",
            b"\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00",
            b"\x02\x00\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00",
            b"\x02\x08\x10",
        ]
    )

    expected_footer = b"".join(
        [
            b"\x0c\x00\x00\x00\x00\x00\x00\x00",
            b"\x20\x00\x00\x00\x00\x00\x00\x00",
            b"\x03\x00\x00\x00\x00\x00\x00\x00",
        ]
    )

    footer_size = len(arrays) * np.dtype("uint64").itemsize

    assert packed[:-footer_size].to_bytes() == expected_bytes
    assert packed[-footer_size:].to_bytes() == expected_footer


def test_unpack_arrays():
    expected_arrays = [
        np.array([45.1, 60.4, 15.2, 92.776], dtype="float16"),
        np.array([42, 31, 83, 190, 320], dtype="uint64"),
        np.array([-1], dtype="int64"),
    ]
    dtypes = [zarr.dtype.Float16(), zarr.dtype.UInt64(), zarr.dtype.Int64()]

    footer = b"".join(
        [
            b"\x08\x00\x00\x00\x00\x00\x00\x00",
            b"\x28\x00\x00\x00\x00\x00\x00\x00",
            b"\x08\x00\x00\x00\x00\x00\x00\x00",
        ]
    )
    data = b"".join(
        [
            b"\xa3\x51\x8d\x53\x9a\x4b\xcc\x55",
            b"\x2a\x00\x00\x00\x00\x00\x00\x00\x1f\x00\x00\x00\x00\x00\x00\x00",
            b"\x53\x00\x00\x00\x00\x00\x00\x00\xbe\x00\x00\x00\x00\x00\x00\x00",
            b"\x40\x01\x00\x00\x00\x00\x00\x00",
            b"\xff\xff\xff\xff\xff\xff\xff\xff",
        ]
    )
    prototype = numpy_buffer_prototype()

    packed = prototype.buffer.from_bytes(data + footer)
    unpacked = packing.unpack_array(prototype, packed, dtypes)

    for actual, expected in zip(unpacked, expected_arrays):
        np.testing.assert_equal(actual, expected)
