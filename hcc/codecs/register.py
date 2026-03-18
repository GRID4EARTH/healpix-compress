from zarr.codecs import register_codec as zarr_register_codec


def register_codec(cls):
    zarr_register_codec(cls.name, cls)

    return cls
