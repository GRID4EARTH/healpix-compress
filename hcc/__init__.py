from importlib.metadata import version

try:
    __version__ = version("hcc")
except Exception:
    __version__ = "9999"
