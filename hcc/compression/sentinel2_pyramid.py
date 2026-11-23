"""Deprecated: renamed :mod:`hcc.compression.uint16_jxl_pyramid` in 0.5.0."""

import warnings

from hcc.compression import uint16_jxl_pyramid as _module

warnings.warn(
    "hcc.compression.sentinel2_pyramid is deprecated: it was renamed hcc.compression.uint16_jxl_pyramid "
    "in healpix-compress 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_module, name)
