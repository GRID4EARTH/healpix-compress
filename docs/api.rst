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
