from importlib.metadata import version

from hcc.codec import HccCodec

try:
    __version__ = version("hcc")
except Exception:
    __version__ = "9999"

__all__ = ["HccCodec"]
