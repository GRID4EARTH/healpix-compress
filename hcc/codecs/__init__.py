from hcc.codecs.packing import pack_array, unpack_array  # noqa: F401


def _register_zarr_codecs() -> None:
    """Register the 16-bit JPEG XL serializers with zarr at import time.

    The ``zarr.codecs`` entry points only exist in an environment where the
    package was installed (not when ``hcc`` is imported from a source tree or
    vendored), and zarr resolves a codec name when it opens an array. Explicit
    registration makes ``import hcc`` sufficient to write and read arrays that
    use these serializers by name; re-registering a name the entry points
    already provided is a no-op. The ids used before 0.5.0 are registered too
    (:mod:`hcc.codecs._aliases`), so that existing stores still open.
    """
    from zarr.registry import register_codec

    from hcc.codecs._aliases import LEGACY_CODECS
    from hcc.codecs.uint16_jxl import HccUint16JxlCodec
    from hcc.codecs.uint16_jxl_pyramid import HccUint16JxlPyramidCodec

    for codec in (HccUint16JxlCodec, HccUint16JxlPyramidCodec, *LEGACY_CODECS):
        register_codec(codec.name, codec)


_register_zarr_codecs()
