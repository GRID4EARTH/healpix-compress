API Reference — grad_dct
========================

.. currentmodule:: hcc.compression.grad_dct

.. automodule:: hcc.compression.grad_dct
   :no-members:
   :no-undoc-members:

Public interface
----------------

Main entry points for compressing and decompressing HEALPix chunks.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: function.rst

   compress
   decompress
   compression_info

Low-level codec
---------------

Operate directly on 2-D ``float32`` tiles (no Z-order conversion).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: function.rst

   encode
   decode

Signal processing primitives
-----------------------------

Used internally by the encoder/decoder; exposed for testing and experimentation.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: function.rst

   block_mean_2x2
   downsample
   upsample
   local_std

Data
----

.. autosummary::
   :toctree: generated/
   :nosignatures:

   QUALITY_PRESETS

API Reference — sentinel2
=========================

.. currentmodule:: hcc.compression.sentinel2

.. automodule:: hcc.compression.sentinel2
   :no-members:
   :no-undoc-members:

The Sentinel-2 product chain on a nested chunk: digital numbers, Morton
de-interleaving, lossless JPEG XL.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   quantize
   dequantize
   encode
   decode
   check_chunk_shape
   check_rows_shape
   encode_rows
   decode_rows

Zarr serializer
---------------

.. currentmodule:: hcc.codecs.sentinel2

.. autosummary::
   :toctree: generated/

   HccSentinel2Codec
   lookup_preset

API Reference — sentinel2_pyramid
=================================

.. currentmodule:: hcc.compression.sentinel2_pyramid

.. automodule:: hcc.compression.sentinel2_pyramid
   :no-members:
   :no-undoc-members:

A lossless parent level on the digital-number lattice and the children
predicted from it, with a quantised detail whose step follows the chunk.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   encode
   decode
   stream_info

Zarr serializer
---------------

.. currentmodule:: hcc.codecs.sentinel2_pyramid

.. autosummary::
   :toctree: generated/

   HccSentinel2PyramidCodec
   lookup_preset
