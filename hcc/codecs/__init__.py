from hcc.codecs.packing import pack_array, unpack_array  # noqa: F401


def _register_zarr_codecs() -> None:
    """Register the Sentinel-2 serializers with zarr at import time.

    The ``zarr.codecs`` entry points only exist in an environment where the
    package was installed (not when ``hcc`` is imported from a source tree or
    vendored), and zarr resolves a codec name when it opens an array. Explicit
    registration makes ``import hcc`` sufficient to write and read arrays that
    use these serializers by name; re-registering a name the entry points
    already provided is a no-op.
    """
    from zarr.registry import register_codec

    from hcc.codecs.sentinel2 import HccSentinel2Codec
    from hcc.codecs.sentinel2_pyramid import HccSentinel2PyramidCodec

    for codec in (HccSentinel2Codec, HccSentinel2PyramidCodec):
        register_codec(codec.name, codec)


_register_zarr_codecs()
