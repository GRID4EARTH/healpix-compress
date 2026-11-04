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

API Reference — grad_dwt
========================

.. currentmodule:: hcc.compression.grad_dwt

.. automodule:: hcc.compression.grad_dwt
   :no-members:
   :no-undoc-members:

Needs PyWavelets (``healpix-compress[wavelets]``).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: function.rst

   compress
   decompress
   compression_info
   lookup_preset
   compress_residuals
   decompress_residuals
   dwt_forward
   dwt_inverse

Zarr codec
----------

.. currentmodule:: hcc.codecs.grad_dwt

.. autosummary::
   :toctree: generated/

   HccGradDwtCodec
   lookup_preset

API Reference — uint16_jxl
==========================

.. currentmodule:: hcc.compression.uint16_jxl

.. automodule:: hcc.compression.uint16_jxl
   :no-members:
   :no-undoc-members:

Data stored as 16-bit integers with a scale and an offset, on a nested chunk:
digital numbers, Morton de-interleaving, lossless JPEG XL. Up to 0.4.x this
module was ``hcc.compression.sentinel2`` (still importable, deprecated).

.. autosummary::
   :toctree: generated/
   :template: function.rst

   quantize
   dequantize
   value_range
   encode
   decode
   check_chunk_shape
   check_rows_shape
   encode_rows
   decode_rows

Zarr serializer
---------------

.. currentmodule:: hcc.codecs.uint16_jxl

The codec id is ``hcc-uint16-jxl``; ``hcc-sentinel2``, its id up to 0.4.x,
stays registered so that existing stores open.

.. autosummary::
   :toctree: generated/

   HccUint16JxlCodec
   lookup_preset

API Reference — uint16_jxl_pyramid
==================================

.. currentmodule:: hcc.compression.uint16_jxl_pyramid

.. automodule:: hcc.compression.uint16_jxl_pyramid
   :no-members:
   :no-undoc-members:

Up to 0.4.x this module was ``hcc.compression.sentinel2_pyramid`` (still
importable, deprecated). A lossless parent level on the digital-number lattice and the children
predicted from it, with a quantised detail whose step follows the chunk.

.. autosummary::
   :toctree: generated/
   :template: function.rst

   encode
   decode
   stream_info

Zarr serializer
---------------

.. currentmodule:: hcc.codecs.uint16_jxl_pyramid

The codec id is ``hcc-uint16-jxl-pyramid``; ``hcc-sentinel2-pyramid``, its id
up to 0.4.x, stays registered so that existing stores open.

.. autosummary::
   :toctree: generated/

   HccUint16JxlPyramidCodec
   lookup_preset
