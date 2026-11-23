Usage
=====

The codecs plug into the codec pipeline of a zarr 3 array. Each chunk must be
aligned with HEALPix cells: in the nested scheme, the cells of a chunk are the
children of one parent cell, so a chunk holds a power of four cells.

Importing ``hcc`` registers the codecs with zarr, so reading a store only
needs the import.

With zarr-python
----------------

Write an array with a preset pipeline:

.. code-block:: python

   import numpy as np
   import zarr

   import hcc

   # values on 4**10 nested cells, 4**8 cells (one parent cell) per chunk
   data = np.sin(np.linspace(0, 20, 4**10)).astype("float32")

   [serializer] = hcc.preset("hcc-grad-dwt", "medium")

   root = zarr.create_group(store="example.zarr", zarr_format=3)
   z = root.create_array(
       "values",
       shape=data.shape,
       dtype=data.dtype,
       chunks=(4**8,),
       fill_value=float("nan"),
       serializer=serializer,
       compressors=None,
       dimension_names=["cells"],
   )
   z[:] = data

Read it back:

.. code-block:: python

   import zarr

   import hcc  # noqa: F401  (registers the codecs)

   z = zarr.open_group("example.zarr", mode="r")["values"]
   values = z[:]

The array metadata keeps the full codec configuration, not the preset name,
so a later change to a preset does not change how existing data is read.

With xarray
-----------

Pass the codec configuration in the encoding:

.. code-block:: python

   import xarray as xr

   import hcc

   ds = xr.Dataset({"values": ("cells", data)})
   ds.to_zarr(
       "example-xr.zarr",
       encoding={
           "values": {
               "serializer": {
                   "name": "hcc-dct",
                   "configuration": {"delta": 2, "top_k": 4},
               },
               "compressors": None,
               "chunks": (4**8,),
           }
       },
   )

   ds = xr.open_zarr("example-xr.zarr")

See :doc:`codecs` for the codecs and their presets.
