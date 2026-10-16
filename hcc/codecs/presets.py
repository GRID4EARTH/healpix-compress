from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zarr.abc.codec import Codec


def preset(codec: str, name: str) -> list[Codec]:
    """lookup codec presets and generate a codec pipeline

    Parameters
    ----------
    codec : str
        The name of the codec as declared by its `name` class attribute.
    name : str
        The name of the preset.

    Returns
    -------
    codecs : list of codec
        The codec pipeline as a list of codecs.

    See Also
    --------
    :doc:`codecs`
        The codec documentation
    """

    match codec:
        case "hcc-grad-dct":
            from hcc.codecs.grad_dct import lookup_preset

            return lookup_preset(name)
        case "hcc-grad-dwt":
            from hcc.codecs.grad_dwt import lookup_preset

            return lookup_preset(name)
        case "hcc-sentinel2":
            from hcc.codecs.sentinel2 import lookup_preset

            return lookup_preset(name)
        case "hcc-sentinel2-pyramid":
            from hcc.codecs.sentinel2_pyramid import lookup_preset

            return lookup_preset(name)
        case _:
            raise ValueError(f"unknown codec: {codec}")
