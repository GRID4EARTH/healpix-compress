from importlib.metadata import version

import hcc.codecs  # noqa: F401

try:
    __version__ = version("hcc")
except Exception:
    __version__ = "9999"

__all__ = []
