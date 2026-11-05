Codecs
======

healpix-compress registers these codecs with zarr (entry point group
``zarr.codecs``). The name is the ``name`` stored in the array metadata.

.. list-table::
   :header-rows: 1
   :widths: 22 14 14 50

   * - Name
     - Kind
     - Extra
     - What it does
   * - ``hcc-dct``
     - serializer
     -
     - Averages each chunk over ``delta`` refinement levels and keeps the
       ``top_k`` largest DCT coefficients of the residual (lossy).
   * - ``hcc-decompose``
     - serializer
     -
     - Splits a chunk into a downsampled array and the residual to it, each
       with its own codec pipeline (``codecs={"downsampled": ...,
       "residuals": ...}``).
   * - ``hcc-grad-dct``
     - array to array
     -
     - Normalises the residual of ``hcc-decompose`` by a local standard
       deviation map and quantises it to 8- or 16-bit integers (lossy).
   * - ``hcc-grad-dwt``
     - array to array
     - ``wavelets``
     - Quantises the residual of ``hcc-decompose`` per wavelet subband; the
       approximation subband stays lossless (lossy).
   * - ``hcc-sentinel2``
     - serializer
     - ``jpegxl``
     - Rounds the values to 16-bit integers with a scale and an offset and
       stores them with lossless JPEG XL; the rounding is the only error.
   * - ``hcc-sentinel2-pyramid``
     - serializer
     - ``jpegxl``
     - Keeps the parent level lossless on the 16-bit integer lattice and
       quantises the children predicted from it, with a step that follows the
       chunk (``relative_step``; ``0`` is lossless).

Presets
-------

:func:`hcc.preset` builds a codec pipeline from a named preset:

.. list-table::
   :header-rows: 1

   * - Codec
     - Presets
   * - ``hcc-grad-dct``
     - ``low``, ``medium``, ``high``, ``ultra``
   * - ``hcc-grad-dwt``
     - ``low``, ``medium``, ``high``, ``ultra``
   * - ``hcc-sentinel2``
     - ``lossless``
   * - ``hcc-sentinel2-pyramid``
     - ``default``, ``transfer-floor``, ``lossless``

.. code-block:: python

   import hcc

   [serializer] = hcc.preset("hcc-grad-dct", "high")

The ``hcc-grad-dct`` and ``hcc-grad-dwt`` presets return an ``hcc-decompose``
serializer whose residual pipeline holds the gradient codec.

See the :doc:`api` for the parameters of each codec.
