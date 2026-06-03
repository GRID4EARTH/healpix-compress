"""Zarr ArrayArrayCodec that applies DWT compression to the residual layer.

This module exposes :class:`HccGradDwtCodec`, a Zarr v3 ``ArrayArrayCodec``
that sits in the residual pipeline of
:class:`~hcc.codecs.decompose.HccDecomposeCodec`.  It implements the
wavelet-based uniform-quantisation scheme defined in
:mod:`hcc.compression.grad_dwt`.

Pipeline position
-----------------
::

    float32 residuals  (from HccDecomposeCodec)
        │
        ▼
    HccGradDwtCodec   ← this codec
        │
        ▼
    int8 / int16 DWT coefficients  (+  sidecar scalar: q_scale)
        │
        ▼
    BytesCodec → GzipCodec / ZstdCodec

Compared to :class:`~hcc.codecs.grad_dct.HccGradDctCodec`:

* Uses a **multi-level 2-D DWT** instead of a block DCT, providing better
  energy compaction and no block-boundary artefacts.
* Stores the **same sidecar scalar** (``q_scale``), so the packing
  infrastructure in :class:`~hcc.codecs.decompose.HccDecomposeCodec` is
  reused without modification.
* Two extra parameters: ``wavelet`` (wavelet family) and ``levels``
  (decomposition depth), replacing the ``window`` (block size) of the DCT
  codec.

Public API
----------
HccGradDwtCodec : Zarr ArrayArrayCodec implementing DWT residual coding.
lookup_preset   : Construct a full :class:`~hcc.codecs.decompose.HccDecomposeCodec`
                  pipeline from a named quality preset.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from zarr.abc.codec import ArrayArrayCodec
from zarr.core.common import parse_named_configuration
from zarr.dtype import Float32, Int8, Int16

from hcc.codecs.scalars import decode_scalar, encode_scalar
from hcc.compression import grad_dwt

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.buffer import Buffer, NDBuffer
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import JSON
    from zarr.dtype import ZDType

    from hcc.codecs.decompose import HccDecomposeCodec


@dataclass(frozen=True)
class HccGradDwtCodec(ArrayArrayCodec):
    """Zarr ArrayArrayCodec — DWT uniform-quantisation of residuals.

    This codec transforms a float32 residual array into a quantised DWT
    coefficient array (int8 or int16) by:

    1. Applying a multi-level 2-D DWT (``'periodization'`` boundary mode)
       to the residual.  The coefficient array has the same shape as the
       input because periodization preserves the total element count.
    2. Computing a global quantisation scale ``q_scale`` from the
       ``(100 − clip_percentile)``-th percentile of the absolute
       coefficients.
    3. Uniformly quantising all coefficients to int8 or int16.

    The scalar ``q_scale`` is stored as a sidecar in the packed chunk buffer
    and recovered automatically during decoding.

    Parameters
    ----------
    wavelet : str, default ``'db4'``
        PyWavelets wavelet name.  Recommended choices:

        * ``'db2'`` — fast, moderate quality.
        * ``'db4'`` — good default (Daubechies 4).
        * ``'db6'`` — high quality, slightly slower.
        * ``'bior4.4'`` — biorthogonal, symmetric; close to JPEG 2000's CDF
          9/7 wavelet.
    levels : int, default 4
        Number of DWT decomposition levels.  Higher values push more energy
        into the LL approximation subband, improving compressibility of the
        remaining detail coefficients.  Must not exceed
        ``floor(log2(n / filter_length))``.
    quantize_bits : {8, 16}, default 8
        Quantisation depth.

        * ``8``  → int8, 254 usable levels (±127), 1 byte per coefficient.
        * ``16`` → int16, 65 534 usable levels (±32 767), 2 bytes per
          coefficient.
    clip_percentile : float, default 0.5
        Tail percentile for ``q_scale`` derivation.  A larger value allows
        more coefficient saturation in exchange for finer resolution of the
        majority.  Typical range: 0.02–5.0.

    Notes
    -----
    This codec is designed to be used *inside* a
    :class:`~hcc.codecs.decompose.HccDecomposeCodec` pipeline, not
    standalone.  It receives the residual array (``original −
    bilinear_prediction``) via ``chunk_array`` and reads the prediction from
    ``chunk_spec.config.context["prediction"]`` for API symmetry (though the
    prediction is not used in the DWT path).

    For standalone compression of a full tile without a separate
    low-frequency layer, use :func:`hcc.compression.grad_dwt.compress`
    directly.

    See Also
    --------
    hcc.compression.grad_dwt.compress_residuals : Core compression function.
    hcc.compression.grad_dwt.decompress_residuals : Core decompression function.
    hcc.codecs.decompose.HccDecomposeCodec : Outer codec that manages the
        low-frequency layer and calls this codec for the residual.
    hcc.codecs.grad_dct.HccGradDctCodec : Sibling codec using block DCT.
    lookup_preset : Convenience factory building a full pipeline from a
        named preset.

    Examples
    --------
    Build a codec directly:

    >>> codec = HccGradDwtCodec(wavelet='db4', levels=4,
    ...                         quantize_bits=16, clip_percentile=0.1)
    >>> codec.to_dict()   # doctest: +NORMALIZE_WHITESPACE
    {'name': 'hcc-grad-dwt', 'configuration': {'wavelet': 'db4', 'levels': 4, 'quantize_bits': 16, 'clip_percentile': 0.1}}

    Build a full pipeline from a preset:

    >>> pipeline = lookup_preset("high")
    >>> pipeline.factor
    2
    >>> pipeline.codecs["residuals"][0].wavelet
    'db4'
    """

    name: ClassVar[str] = "hcc-grad-dwt"

    wavelet: str = "db4"
    levels: int = 4
    quantize_bits: int = 8
    clip_percentile: float = 0.5

    #: Sidecar scalar produced/consumed by this codec.
    #: ``q_scale`` is the uniform quantisation step size derived from the
    #: DWT coefficient distribution.  It is the only per-chunk data-dependent
    #: value that cannot be recomputed without the original residual.
    dtypes: ClassVar[dict[str, ZDType]] = {
        "q_scale": Float32(endianness="little"),
    }

    @property
    def scalars(self) -> list[str]:
        """Ordered list of sidecar scalar names produced/consumed by this codec."""
        return list(self.dtypes.keys())

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Deserialise from a Zarr codec JSON object.

        Parameters
        ----------
        data : dict
            Codec JSON as stored in ``zarr.json``, e.g.::

                {
                    "name": "hcc-grad-dwt",
                    "configuration": {
                        "wavelet": "db4",
                        "levels": 4,
                        "quantize_bits": 16,
                        "clip_percentile": 0.1
                    }
                }

        Returns
        -------
        HccGradDwtCodec
        """
        _, configuration_parsed = parse_named_configuration(
            data, cls.name, require_configuration=True
        )
        configuration_parsed = configuration_parsed or {}
        return cls(**configuration_parsed)

    def to_dict(self) -> dict[str, JSON]:
        """Serialise to a Zarr codec JSON object.

        Returns
        -------
        dict
            JSON-serialisable representation for storage in ``zarr.json``.
        """
        return {
            "name": self.name,
            "configuration": {
                "wavelet": self.wavelet,
                "levels": self.levels,
                "quantize_bits": self.quantize_bits,
                "clip_percentile": self.clip_percentile,
            },
        }

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """Compute the output ``ArraySpec`` produced by this codec.

        The output dtype is int8 (for ``quantize_bits=8``) or int16 (for
        ``quantize_bits=16``); the fill value is the minimum of the integer
        dtype.  Shape is unchanged because the DWT in ``'periodization'``
        mode preserves the total element count.

        Parameters
        ----------
        chunk_spec : ArraySpec
            Input spec (float32 residuals).

        Returns
        -------
        ArraySpec
            Output spec (int8 or int16 DWT coefficients).

        Raises
        ------
        ValueError
            If :attr:`quantize_bits` is not 8 or 16.
        """
        target_dtypes: dict[int, ZDType] = {
            8: Int8(),
            16: Int16(endianness="little"),
        }
        target_zdtype = target_dtypes.get(self.quantize_bits)
        if target_zdtype is None:
            raise ValueError(
                f"quantize_bits must be 8 or 16; got {self.quantize_bits!r}."
            )
        new_fill = np.iinfo(target_zdtype.to_native_dtype()).min
        return replace(chunk_spec, dtype=target_zdtype, fill_value=new_fill)

    async def _decode_single(
        self, chunk_data: NDBuffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        """Decode a single chunk: dequantise DWT coefficients → residuals.

        Reads ``q_scale`` from ``chunk_spec.config.scalars``, dequantises
        the DWT coefficients, and applies a multi-level IDWT to recover the
        float32 residual.

        Parameters
        ----------
        chunk_data : NDBuffer
            Quantised DWT coefficients (int8 or int16).
        chunk_spec : ArraySpec
            Must carry ``scalars["q_scale"]`` and
            ``config.context["prediction"]`` populated by the surrounding
            pipeline.

        Returns
        -------
        NDBuffer
            Reconstructed float32 residuals.
        """
        prototype = chunk_spec.prototype

        scalars = {
            name: decode_scalar(
                prototype, buffer, dtype=self.dtypes[name]
            ).as_numpy_array()
            for name, buffer in chunk_spec.config.scalars.items()
        }

        chunk_array = grad_dwt.decompress_residuals(
            chunk_data.as_numpy_array(),
            prediction=chunk_spec.config.context["prediction"],
            wavelet=self.wavelet,
            levels=self.levels,
            quantize_bits=self.quantize_bits,
            **scalars,
        )
        return prototype.nd_buffer.from_ndarray_like(chunk_array)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        """Encode a single chunk: residuals → quantised DWT coefficients.

        Applies a multi-level DWT and uniform quantisation to the float32
        residual, writes ``q_scale`` into ``chunk_spec.config.scalars``, and
        returns the quantised coefficient array.

        Parameters
        ----------
        chunk_array : NDBuffer
            Float32 residual array (``original − bilinear_prediction``).
        chunk_spec : ArraySpec
            Must carry ``config.context["prediction"]`` populated by the
            surrounding :class:`~hcc.codecs.decompose.HccDecomposeCodec`.

        Returns
        -------
        Buffer or None
            Quantised DWT coefficients (int8 or int16), or ``None`` to skip.
        """
        prototype = chunk_spec.prototype

        coeffs_q_arr, q_scale_arr = grad_dwt.compress_residuals(
            chunk_array.as_numpy_array(),
            prediction=chunk_spec.config.context["prediction"],
            wavelet=self.wavelet,
            levels=self.levels,
            quantize_bits=self.quantize_bits,
            clip_percentile=self.clip_percentile,
        )

        encoded_data = prototype.nd_buffer.from_ndarray_like(coeffs_q_arr)
        q_scale_buf = prototype.nd_buffer.from_ndarray_like(q_scale_arr)

        chunk_spec.config.scalars.update(
            {"q_scale": encode_scalar(prototype, q_scale_buf)}
        )
        return encoded_data


def lookup_preset(name: str) -> HccDecomposeCodec:
    """Construct a full :class:`~hcc.codecs.decompose.HccDecomposeCodec`
    pipeline from a named quality preset.

    The returned pipeline uses:

    * A ``BytesCodec → GzipCodec`` sub-pipeline for the lossless
      low-frequency layer.
    * An ``HccGradDwtCodec → BytesCodec → GzipCodec`` sub-pipeline for the
      DWT-compressed residual.

    .. important::

       Store the explicit parameter dict (from
       :func:`hcc.compression.grad_dwt.lookup_preset`) in your array
       metadata rather than the preset name.  This ensures future changes to
       preset defaults cannot make existing data unreadable.

    Parameters
    ----------
    name : {'low', 'medium', 'high', 'ultra'}
        Quality preset name.

        +----------+---------+--------+---------------+-----------------+------------+-------------------+
        | name     | wavelet | levels | quantize_bits | clip_percentile | zlib_level | downsample_factor |
        +==========+=========+========+===============+=================+============+===================+
        | low      | db2     | 3      | 8             | 3.0             | 9          | 4                 |
        +----------+---------+--------+---------------+-----------------+------------+-------------------+
        | medium   | db4     | 4      | 8             | 0.5             | 9          | 2                 |
        +----------+---------+--------+---------------+-----------------+------------+-------------------+
        | high     | db4     | 5      | 16            | 0.1             | 6          | 2                 |
        +----------+---------+--------+---------------+-----------------+------------+-------------------+
        | ultra    | db6     | 6      | 16            | 0.02            | 1          | 1                 |
        +----------+---------+--------+---------------+-----------------+------------+-------------------+

    Returns
    -------
    HccDecomposeCodec
        Fully assembled pipeline.

    Raises
    ------
    ValueError
        If *name* is not a known preset.

    See Also
    --------
    hcc.compression.grad_dwt.lookup_preset : Returns the raw parameter dict.

    Examples
    --------
    >>> pipeline = lookup_preset("medium")
    >>> pipeline.factor
    2
    >>> pipeline.codecs["residuals"][0].wavelet
    'db4'
    >>> pipeline.codecs["residuals"][0].levels
    4
    """
    from zarr.codecs import BytesCodec, GzipCodec

    from hcc.codecs.decompose import HccDecomposeCodec
    from hcc.compression.grad_dwt import lookup_preset as lookup_raw_preset

    presets = lookup_raw_preset(name, codec=True)
    factor = presets.pop("downsample_factor")
    level = presets.pop("zlib_level")

    return HccDecomposeCodec(
        factor=factor,
        codecs={
            "downsampled": [BytesCodec(), GzipCodec(level=level)],
            "residuals": [
                HccGradDwtCodec(**presets),
                BytesCodec(),
                GzipCodec(level=level),
            ],
        },
    )
