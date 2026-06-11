from importlib.metadata import version

import hcc.codecs  # noqa: F401
from hcc.codecs.presets import preset

try:
    __version__ = version("hcc")
except Exception:
    __version__ = "9999"

__all__ = ["preset"]
