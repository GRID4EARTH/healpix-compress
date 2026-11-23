"""Deprecated: renamed :mod:`hcc.codecs.uint16_jxl_pyramid` in 0.5.0.

``HccSentinel2PyramidCodec`` is the codec under its old id,
``hcc-sentinel2-pyramid``, which stays registered so that existing stores
still open.
"""

import warnings

from hcc.codecs import uint16_jxl_pyramid as _module
from hcc.codecs._aliases import (
    LegacyHccUint16JxlPyramidCodec as HccSentinel2PyramidCodec,
)
from hcc.codecs._aliases import lookup_pyramid_preset as lookup_preset

__all__ = ["HccSentinel2PyramidCodec", "lookup_preset"]

warnings.warn(
    "hcc.codecs.sentinel2_pyramid is deprecated: it was renamed "
    "hcc.codecs.uint16_jxl_pyramid (HccUint16JxlPyramidCodec, id "
    "'hcc-uint16-jxl-pyramid') in healpix-compress 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_module, name)
