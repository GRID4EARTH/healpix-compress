"""The optional native kernels of :mod:`hcc.compression.uint16_jxl_pyramid`.

They come with the separate package ``healpix-compress-native`` (module
``hcc_native``, built from ``rust/`` in this repository). When it is not
installed, or when the environment variable ``HCC_NATIVE`` is ``0``, the codec
runs its Python implementation, which stays the reference: the native
kernels give the same codestreams and the same decoded cells, bit for bit.
"""

from __future__ import annotations

import os

#: The ``HS2P`` codestream version the kernels must implement.
PYRAMID_FORMAT = 1


def _load():
    if os.environ.get("HCC_NATIVE", "1") == "0":
        return None
    try:
        import hcc_native
    except ImportError:
        return None
    if getattr(hcc_native, "PYRAMID_FORMAT", None) != PYRAMID_FORMAT:
        return None
    return hcc_native


#: The ``hcc_native`` module, or ``None`` when the Python implementation runs.
module = _load()
