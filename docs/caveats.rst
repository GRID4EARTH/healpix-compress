Things to know
==============

Native kernels: where they match the Python implementation
-----------------------------------------------------------

The pyramid serializer can run on the optional native kernels
(``healpix-compress-native``, built from ``rust/``). The decoder computes the
same prediction as the encoder, so both must sum in the same order. The
native kernels follow the summation order of numpy with OpenBLAS:

- on Linux x86_64 with numpy's OpenBLAS, the native kernels and the Python
  implementation give the same codestreams and the same decoded cells, bit
  for bit (checked on CI, with and without the native package);
- with another BLAS (Apple Accelerate, MKL, ...) numpy can sum in another
  order, so a chunk encoded on one platform and decoded on another may not
  give the same cells bit for bit.

Encode and decode with the same implementation on the same platform when
that matters. ``HCC_NATIVE=0`` turns the native kernels off.

Which versions read ``decoding_speed``
--------------------------------------

The default ``decoding_speed=2`` is not written to the array metadata, so
healpix-compress 0.3.x reads those stores. A store written with
``decoding_speed=0`` or ``1`` needs healpix-compress 0.4.0 or later: 0.3.x
raises a ``TypeError`` on the unknown key.
