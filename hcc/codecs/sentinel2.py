"""Deprecated: renamed :mod:`hcc.codecs.uint16_jxl` in 0.5.0.

``HccSentinel2Codec`` is the codec under its old id, ``hcc-sentinel2``, which
stays registered so that existing stores still open.
"""

import warnings

from hcc.codecs import uint16_jxl as _module
from hcc.codecs._aliases import LegacyHccUint16JxlCodec as HccSentinel2Codec
from hcc.codecs._aliases import lookup_preset

__all__ = ["HccSentinel2Codec", "lookup_preset"]

warnings.warn(
    "hcc.codecs.sentinel2 is deprecated: it was renamed hcc.codecs.uint16_jxl "
    "(HccUint16JxlCodec, id 'hcc-uint16-jxl') in healpix-compress 0.5.0",
    DeprecationWarning,
    stacklevel=2,
)


def __getattr__(name):
    return getattr(_module, name)
