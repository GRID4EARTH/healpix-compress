Installation
============

healpix-compress needs Python 3.11 or later and zarr 3. Install it from PyPI:

.. code-block:: bash

   pip install healpix-compress

Optional dependencies
---------------------

Some codecs need an extra package:

``jpegxl``
    `imagecodecs <https://github.com/cgohlke/imagecodecs>`_, for the lossless
    JPEG XL serializers (``hcc-sentinel2`` and ``hcc-sentinel2-pyramid``).

``wavelets``
    `PyWavelets <https://pywavelets.readthedocs.io>`_, for ``hcc-grad-dwt``.

.. code-block:: bash

   pip install "healpix-compress[jpegxl,wavelets]"

Native kernels
--------------

The pyramid serializer can run on optional native kernels, the
``healpix-compress-native`` package in ``rust/``. Building it needs a Rust
toolchain; from a clone of the repository:

.. code-block:: bash

   pip install ./rust

The codec uses the native kernels when they are installed, unless the
environment variable ``HCC_NATIVE`` is ``0``. See :doc:`caveats` for when the
native kernels and the Python implementation give the same cells.

From source
-----------

.. code-block:: bash

   git clone https://github.com/GRID4EARTH/healpix-compress
   cd healpix-compress
   pip install -e ".[jpegxl,wavelets]"
