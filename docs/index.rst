healpix-compress
================

Zarr codecs for chunks aligned with HEALPix cells.

A chunk in the nested HEALPix scheme holds the children of one parent cell.
The codecs use that structure: they average the chunk to coarser refinement
levels and code the residuals, or store the cells as a raster with lossless
JPEG XL.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   codecs
   caveats
   api
