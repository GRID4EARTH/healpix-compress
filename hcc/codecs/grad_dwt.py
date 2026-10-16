"""Zarr ArrayArrayCodec that applies DWT compression to a residual.

This module exposes :class:`HccGradDwtCodec`, a Zarr v3 ``ArrayArrayCodec``
implementing the wavelet-based per-subband quantisation scheme defined in
:mod:`hcc.compression.grad_dwt`.

Pipeline position
-----------------
Like :class:`~hcc.codecs.grad_dct.HccGradDctCodec`, this codec sits in the
``"residuals"`` sub-pipeline of :class:`~hcc.codecs.decompose.HccDecomposeCodec`,
which packs its sidecar scalars into the chunk::

    float32 chunk
        │
        ▼
    HccDecomposeCodec
        ├── "downsampled": BytesCodec → GzipCodec   (lossless low frequencies)
        └── "residuals":
                HccGradDwtCodec   ← this codec
                    │
                    ▼
                int8 / int16 DWT *detail* coefficients
                (+ sidecar scalars: q_scale, ll_values)
                    │
                    ▼
                BytesCodec → GzipCodec

See :func:`lookup_preset` for a ready-made ``codecs=[...]`` list.

Compared to :class:`~hcc.codecs.grad_dct.HccGradDctCodec`:

* Uses a **multi-level 2-D DWT** instead of a block DCT, providing better
  energy compaction and no block-boundary artefacts.
* Stores the LL approximation subband **losslessly** (float32 sidecar).
* ``wavelet`` (wavelet family) and ``levels`` (decomposition depth) replace
  the ``window`` (block size) of the DCT codec.

The DWT needs PyWavelets (``pip install healpix-compress[wavelets]``).

Public API
----------
HccGradDwtCodec : Zarr ArrayArrayCodec implementing the DWT tile coding.
lookup_preset   : Codec pipeline for a named quality preset.
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
    """Zarr ArrayArrayCodec — DWT per-subband quantisation of a residual.

    This codec transforms a float32 array into quantised DWT detail
    coefficients (int8 or int16), plus two sidecars, by:

    1. Applying a multi-level 2-D DWT (``'periodization'`` boundary mode).
       The coefficient array has the same shape as the input because
       periodization preserves the total element count.
    2. Storing the LL approximation subband **losslessly** as a float32
       sidecar (``ll_values``).
    3. Computing an independent quantisation scale per *detail* subband
       (sidecar ``q_scale``, one value per subband) from each subband's own
       ``(100 − clip_percentile)``-th percentile of absolute coefficients.
    4. Uniformly quantising each detail subband to int8 or int16 using its
       own scale.

    Both sidecar scalars (``q_scale``, ``ll_values``) are written to
    ``chunk_spec.config.scalars``, which
    :class:`~hcc.codecs.decompose.HccDecomposeCodec` packs into the chunk
    and restores before decoding.

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
        Number of DWT decomposition levels.  More levels push more energy
        into the (lossless) LL subband and shrink it.  Must not exceed
        ``floor(log2(n / filter_length))``.
    quantize_bits : {8, 16}, default 8
        Quantisation depth for detail subbands (does not affect LL, which
        is always lossless).

        * ``8``  → int8, 254 usable levels (±127), 1 byte per coefficient.
        * ``16`` → int16, 65 534 usable levels (±32 767), 2 bytes per
          coefficient.
    clip_percentile : float, default 0.5
        Tail percentile for each detail subband's ``q_scale`` derivation. A
        larger value allows more coefficient saturation in exchange for
        finer resolution of the majority.  Typical range: 0.02–5.0.

    See Also
    --------
    hcc.compression.grad_dwt.compress_residuals : Core compression function.
    hcc.compression.grad_dwt.decompress_residuals : Core decompression function.
    lookup_preset : Codec pipeline for a named preset.
    hcc.codecs.grad_dct.HccGradDctCodec : Sibling codec using block DCT.

    Examples
    --------
    >>> codec = HccGradDwtCodec(wavelet='db4', levels=4,
    ...                         quantize_bits=16, clip_percentile=0.1)
    >>> codec.to_dict()   # doctest: +NORMALIZE_WHITESPACE
    {'name': 'hcc-grad-dwt', 'configuration': {'wavelet': 'db4', 'levels': 4, 'quantize_bits': 16, 'clip_percentile': 0.1}}
    """

    name: ClassVar[str] = "hcc-grad-dwt"

    wavelet: str = "db4"
    levels: int = 4
    quantize_bits: int = 8
    clip_percentile: float = 0.5

    #: Sidecar scalars produced/consumed by this codec: the per-detail-subband
    #: quantisation steps and the lossless LL subband.
    dtypes: ClassVar[dict[str, ZDType]] = {
        "q_scale": Float32(endianness="little"),
        "ll_values": Float32(endianness="little"),
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
            Input spec (float32 tile).

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
        """Decode a single chunk: dequantise DWT coefficients → tile.

        Parameters
        ----------
        chunk_data : NDBuffer
            Quantised DWT detail coefficients (int8 or int16).
        chunk_spec : ArraySpec
            Must carry ``scalars["q_scale"]`` and ``scalars["ll_values"]``
            populated by the surrounding pipeline.

        Returns
        -------
        NDBuffer
            Reconstructed float32 tile.
        """
        prototype = chunk_spec.prototype

        scalars = {
            name: decode_scalar(
                prototype, buffer, dtype=self.dtypes[name]
            ).as_numpy_array()
            for name, buffer in chunk_spec.config.scalars.items()
        }

        coefficients = chunk_data.as_numpy_array()
        # `prediction` is unused by the DWT path
        chunk_array = grad_dwt.decompress_residuals(
            coefficients,
            prediction=coefficients,
            wavelet=self.wavelet,
            levels=self.levels,
            quantize_bits=self.quantize_bits,
            **scalars,
        )
        return prototype.nd_buffer.from_ndarray_like(chunk_array)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        """Encode a single chunk: tile → quantised DWT detail coefficients.

        Writes ``q_scale`` and ``ll_values`` into
        ``chunk_spec.config.scalars`` and returns the quantised detail
        coefficients (LL region zeroed -- it is carried in ``ll_values``).

        Parameters
        ----------
        chunk_array : NDBuffer
            Float32 residual.
        chunk_spec : ArraySpec
            Output scalars are written to ``chunk_spec.config.scalars``.

        Returns
        -------
        Buffer or None
            Quantised DWT detail coefficients (int8 or int16).
        """
        prototype = chunk_spec.prototype

        tile = chunk_array.as_numpy_array()
        # `prediction` is unused by the DWT path
        coeffs_q, q_scale, ll_values = grad_dwt.compress_residuals(
            tile,
            prediction=tile,
            wavelet=self.wavelet,
            levels=self.levels,
            quantize_bits=self.quantize_bits,
            clip_percentile=self.clip_percentile,
        )

        chunk_spec.config.scalars.update(
            {
                "q_scale": encode_scalar(
                    prototype, prototype.nd_buffer.from_ndarray_like(q_scale)
                ),
                "ll_values": encode_scalar(
                    prototype, prototype.nd_buffer.from_ndarray_like(ll_values)
                ),
            }
        )
        return prototype.nd_buffer.from_ndarray_like(coeffs_q)


def lookup_preset(name: str) -> list[HccDecomposeCodec]:
    """Build the codec pipeline for a named quality preset.

    .. important::

       Store the explicit codec configuration in your array metadata (Zarr
       does this when the pipeline is passed as ``codecs=``), not the preset
       name, so future changes to preset defaults cannot make existing data
       unreadable.

    Parameters
    ----------
    name : {'low', 'medium', 'high', 'ultra'}
        Quality preset name, see
        :func:`hcc.compression.grad_dwt.lookup_preset` with ``codec=True``.

    Returns
    -------
    list of HccDecomposeCodec
        A single :class:`~hcc.codecs.decompose.HccDecomposeCodec` with a
        lossless ``"downsampled"`` layer and an ``HccGradDwtCodec``
        ``"residuals"`` layer, to pass as the ``codecs=`` argument when
        creating a Zarr array.

    Raises
    ------
    ValueError
        If *name* is not a known preset.

    Examples
    --------
    >>> [pipeline] = lookup_preset("medium")
    >>> pipeline.factor
    2
    >>> pipeline.codecs["residuals"][0].wavelet
    'db4'
    """
    from zarr.codecs import BytesCodec, GzipCodec

    from hcc.codecs.decompose import HccDecomposeCodec
    from hcc.compression.grad_dwt import lookup_preset as lookup_raw_preset

    presets = lookup_raw_preset(name, codec=True)
    factor = presets.pop("downsample_factor")
    level = presets.pop("zlib_level")

    return [
        HccDecomposeCodec(
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
    ]
