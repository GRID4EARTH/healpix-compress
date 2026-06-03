"""Zarr ArrayArrayCodec that applies block-DCT compression to the residual layer.

This module exposes :class:`HccGradDctCodec`, a Zarr v3 ``ArrayArrayCodec``
that sits in the residual pipeline of :class:`~hcc.codecs.decompose.HccDecomposeCodec`.
It implements the block-DCT uniform-quantisation scheme defined in
:mod:`hcc.compression.grad_dct`.

Pipeline position
-----------------
::

    float32 residuals
        │
        ▼
    HccGradDctCodec   ← this codec
        │
        ▼
    int8 / int16 DCT coefficients  (+ sidecar scalar: q_scale)
        │
        ▼
    BytesCodec → GzipCodec / ZstdCodec

The codec stores one data-dependent scalar alongside the coefficient array:

* **q_scale** (float32) — the uniform quantisation step size computed from
  the residual DCT coefficients.  This scalar is embedded in the packed
  chunk buffer by the surrounding :class:`~hcc.codecs.decompose.HccDecomposeCodec`
  via :func:`~hcc.codecs.decompose.pipeline_encode` /
  :func:`~hcc.codecs.decompose.pipeline_decode`.

Public API
----------
HccGradDctCodec : Zarr ArrayArrayCodec implementing block-DCT residual coding.
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
from hcc.compression import grad_dct

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.buffer import Buffer, NDBuffer
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import JSON
    from zarr.dtype import ZDType

    from hcc.codecs.decompose import HccDecomposeCodec


@dataclass(frozen=True)
class HccGradDctCodec(ArrayArrayCodec):
    """Zarr ArrayArrayCodec — block-DCT uniform-quantisation of residuals.

    This codec transforms a float32 residual array into a quantised DCT
    coefficient array (int8 or int16) by:

    1. Partitioning the residual into non-overlapping ``window × window``
       blocks.
    2. Applying a 2-D DCT-II (orthonormal normalisation) to each block.
    3. Computing a global quantisation scale ``q_scale`` from the coefficient
       distribution (controlled by *clip_percentile*).
    4. Uniformly quantising all coefficients to int8 or int16.

    The scalar ``q_scale`` is stored as a sidecar in the packed chunk buffer;
    it is recovered automatically during decoding.

    Parameters
    ----------
    window : int, default 8
        DCT block side length.  Must divide the residual array side length
        evenly.  Typical values: 4, 8, 16.
    quantize_bits : {8, 16}, default 8
        Quantisation depth.

        * ``8``  → int8, 254 usable levels (±127), 1 byte per coefficient.
        * ``16`` → int16, 65 534 usable levels (±32 767), 2 bytes per
          coefficient.
    clip_percentile : float, default 0.5
        Symmetric tail percentile for ``q_scale`` derivation.  The
        ``(100 − clip_percentile)``-th percentile of the absolute DCT
        coefficients is mapped to the maximum quantised level.  A larger
        value allows more saturation of large coefficients in exchange for
        finer resolution of the majority.  Typical range: 0.02–5.0.

    Notes
    -----
    This codec is designed to be used *inside* a
    :class:`~hcc.codecs.decompose.HccDecomposeCodec` pipeline, not
    standalone.  It receives the residual (``original − bilinear_prediction``)
    and reads the prediction from ``chunk_spec.config.context["prediction"]``
    for API consistency (though the prediction is not used in the DCT path).

    See Also
    --------
    hcc.compression.grad_dct.compress_residuals : Core compression function.
    hcc.compression.grad_dct.decompress_residuals : Core decompression function.
    hcc.codecs.decompose.HccDecomposeCodec : Outer codec that manages the
        low-frequency layer and calls this codec for the residual.
    lookup_preset : Convenience factory that builds a full pipeline from a
        named preset.

    Examples
    --------
    Build a codec directly:

    >>> codec = HccGradDctCodec(window=8, quantize_bits=16, clip_percentile=0.1)
    >>> codec.to_dict()  # doctest: +NORMALIZE_WHITESPACE
    {'name': 'hcc-grad-dct', 'configuration': {'window': 8, 'quantize_bits': 16, 'clip_percentile': 0.1}}

    Or build an entire pipeline from a preset:

    >>> pipeline = lookup_preset("high")
    >>> pipeline.factor
    2
    """

    name: ClassVar[str] = "hcc-grad-dct"

    window: int = 8
    quantize_bits: int = 8
    clip_percentile: float = 0.5

    #: Sidecar scalar names and their Zarr dtypes.
    #: ``q_scale`` is the only data-dependent value that must be stored per
    #: chunk — it is the uniform quantisation step size derived from the DCT
    #: coefficient distribution.
    dtypes: ClassVar[dict[str, ZDType]] = {
        "q_scale": Float32(endianness="little"),
    }

    @property
    def scalars(self) -> list[str]:
        """Ordered list of sidecar scalar names produced/consumed by this codec.

        The order determines the packing layout in the chunk buffer managed by
        :class:`~hcc.codecs.decompose.HccDecomposeCodec`.
        """
        return list(self.dtypes.keys())

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Deserialise from a Zarr codec JSON object.

        Parameters
        ----------
        data : dict
            Codec JSON as stored in ``zarr.json``, e.g.::

                {
                    "name": "hcc-grad-dct",
                    "configuration": {
                        "window": 8,
                        "quantize_bits": 16,
                        "clip_percentile": 0.1
                    }
                }

        Returns
        -------
        HccGradDctCodec
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
            JSON-serialisable representation suitable for storage in
            ``zarr.json``.
        """
        return {
            "name": self.name,
            "configuration": {
                "window": self.window,
                "quantize_bits": self.quantize_bits,
                "clip_percentile": self.clip_percentile,
            },
        }

    def resolve_metadata(self, chunk_spec: ArraySpec) -> ArraySpec:
        """Compute the output ``ArraySpec`` produced by this codec.

        The output dtype is int8 (for ``quantize_bits=8``) or int16 (for
        ``quantize_bits=16``); the fill value is set to the minimum of the
        integer dtype.  Shape is unchanged.

        Parameters
        ----------
        chunk_spec : ArraySpec
            Input spec (float32 residuals).

        Returns
        -------
        ArraySpec
            Output spec (int8 or int16 DCT coefficients).

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
        """Decode a single chunk: dequantise DCT coefficients → residuals.

        Reads ``q_scale`` from ``chunk_spec.config.scalars``, dequantises the
        DCT coefficients, and applies a block IDCT to recover the float32
        residual.

        Parameters
        ----------
        chunk_data : NDBuffer
            Quantised DCT coefficients (int8 or int16).
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

        chunk_array = grad_dct.decompress_residuals(
            chunk_data.as_numpy_array(),
            prediction=chunk_spec.config.context["prediction"],
            window=self.window,
            quantize_bits=self.quantize_bits,
            **scalars,
        )
        return prototype.nd_buffer.from_ndarray_like(chunk_array)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        """Encode a single chunk: residuals → quantised DCT coefficients.

        Applies block-DCT and uniform quantisation to the float32 residual,
        writes ``q_scale`` into ``chunk_spec.config.scalars``, and returns
        the quantised coefficient array.

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
            Quantised DCT coefficients (int8 or int16) as a ``Buffer``, or
            ``None`` if the chunk should be skipped (e.g. all fill values).
        """
        prototype = chunk_spec.prototype

        encoded_data_arr, q_scale_arr = grad_dct.compress_residuals(
            chunk_array.as_numpy_array(),
            prediction=chunk_spec.config.context["prediction"],
            window=self.window,
            quantize_bits=self.quantize_bits,
            clip_percentile=self.clip_percentile,
        )

        encoded_data = prototype.nd_buffer.from_ndarray_like(encoded_data_arr)
        q_scale_buf = prototype.nd_buffer.from_ndarray_like(q_scale_arr)

        chunk_spec.config.scalars.update(
            {"q_scale": encode_scalar(prototype, q_scale_buf)}
        )

        return encoded_data


def lookup_preset(name: str) -> HccDecomposeCodec:
    """Construct a full :class:`~hcc.codecs.decompose.HccDecomposeCodec`
    pipeline from a named quality preset.

    This is a convenience factory.  It resolves the preset parameters,
    assembles a ``HccDecomposeCodec`` with the appropriate sub-pipelines for
    the downsampled layer and the residual layer, and returns the fully
    configured codec.

    .. important::

       Always resolve and **store the explicit parameter dict** (via
       :func:`hcc.compression.grad_dct.lookup_preset`) rather than the preset
       name in your array metadata.  This ensures that future changes to
       preset defaults cannot make existing data unreadable.

    Parameters
    ----------
    name : {'low', 'medium', 'high', 'ultra'}
        Quality preset name.  See
        :func:`hcc.compression.grad_dct.lookup_preset` for the parameter
        table.

    Returns
    -------
    HccDecomposeCodec
        A fully assembled codec pipeline:

        * ``downsampled`` sub-pipeline: ``BytesCodec → GzipCodec(level)``
        * ``residuals`` sub-pipeline:
          ``HccGradDctCodec(window, quantize_bits, clip_percentile)
          → BytesCodec → GzipCodec(level)``

    Raises
    ------
    ValueError
        If *name* is not a known preset.

    See Also
    --------
    hcc.compression.grad_dct.lookup_preset : Returns the raw parameter dict.

    Examples
    --------
    >>> pipeline = lookup_preset("medium")
    >>> pipeline.factor
    2
    >>> pipeline.codecs["residuals"][0].quantize_bits
    8
    """
    from zarr.codecs import BytesCodec, GzipCodec

    from hcc.codecs.decompose import HccDecomposeCodec
    from hcc.compression.grad_dct import lookup_preset as lookup_raw_preset

    presets = lookup_raw_preset(name)
    factor = presets.pop("downsample_factor")
    level = presets.pop("zlib_level")

    return HccDecomposeCodec(
        factor=factor,
        codecs={
            "downsampled": [BytesCodec(), GzipCodec(level=level)],
            "residuals": [
                HccGradDctCodec(**presets),
                BytesCodec(),
                GzipCodec(level=level),
            ],
        },
    )
