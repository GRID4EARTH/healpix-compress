"""Deprecated: renamed :mod:`hcc.compression.uint16_jxl` in 0.5.0."""

import warnings

from hcc.compression import uint16_jxl as _module

warnings.warn(
    "hcc.compression.sentinel2 is deprecated: it was renamed hcc.compression.uint16_jxl "
    "in healpix-compress 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_module, name)
