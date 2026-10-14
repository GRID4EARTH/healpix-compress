"""Zarr serializer for the two-layer Sentinel-2 coding of HEALPix chunks.

See :mod:`hcc.compression.sentinel2_pyramid`: a lossless parent level on the
digital-number lattice, and the children predicted from it with a quantised
detail whose step follows the chunk's own detail.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.common import parse_named_configuration

from hcc.compression import sentinel2, sentinel2_pyramid

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.buffer import Buffer, NDBuffer
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import JSON


@dataclass(frozen=True)
class HccSentinel2PyramidCodec(ArrayBytesCodec):
    """Zarr ArrayBytesCodec: lossless parent level, predicted and quantised children.

    Each chunk must hold nested chunks of :math:`4^\\delta` floating-point
    reflectances, :math:`\\delta \\ge 1`, along its last axis; leading axes
    (time, band, ...) are free and each row is coded on its own (see
    :func:`hcc.compression.sentinel2.encode_rows`).

    Parameters
    ----------
    relative_step : float, default 0.57
        The detail step as a fraction of the chunk's detail standard deviation;
        ``0`` uses ``min_step``. For Sentinel-2 B04 resampled with the
        legacy-converters PSF preset, the default gives about 1.3 times the
        native tile under lossless JPEG 2000, for an added error of about 1.5
        times the transfer error; ``0.36`` puts the added error at the
        transfer error, for about 1.5 times the native size.
    min_step : float, default 1.0
        The smallest step, in digital numbers.
    predictor_size : int, default 5
        Odd side of the parent neighbourhood the children are predicted from.
    quantification_value : float, default 10000.0
        The :math:`Q` of the product metadata (``QUANTIFICATION_VALUE``).
    add_offset : float, default -1000.0
        The :math:`O` of the product metadata (``RADIO_ADD_OFFSET`` or
        ``BOA_ADD_OFFSET``).
    effort : int, default 7
        JPEG XL encoder effort in :math:`[1, 9]`.

    See Also
    --------
    hcc.compression.sentinel2_pyramid.encode : The encoding function.
    hcc.codecs.sentinel2.HccSentinel2Codec : The single-layer lossless codec.

    Examples
    --------
    >>> codec = HccSentinel2PyramidCodec(relative_step=0.5)
    >>> codec.to_dict()["configuration"]["relative_step"]
    0.5
    """

    name: ClassVar[str] = "hcc-sentinel2-pyramid"

    relative_step: float = sentinel2_pyramid.DEFAULT_RELATIVE_STEP
    min_step: float = 1.0
    predictor_size: int = 5
    quantification_value: float = sentinel2.DEFAULT_QUANTIFICATION_VALUE
    add_offset: float = sentinel2.DEFAULT_ADD_OFFSET
    effort: int = 7

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Deserialise from a Zarr codec JSON object.

        Parameters
        ----------
        data : dict
            Codec JSON as stored in ``zarr.json``.

        Returns
        -------
        HccSentinel2PyramidCodec

        Examples
        --------
        >>> HccSentinel2PyramidCodec.from_dict({"name": "hcc-sentinel2-pyramid"}).predictor_size
        5
        """
        _, configuration_parsed = parse_named_configuration(
            data, cls.name, require_configuration=False
        )
        return cls(**(configuration_parsed or {}))

    def to_dict(self) -> dict[str, JSON]:
        """Serialise to a Zarr codec JSON object.

        Returns
        -------
        dict
            JSON-serialisable representation for storage in ``zarr.json``.

        Examples
        --------
        >>> HccSentinel2PyramidCodec().to_dict()["name"]
        'hcc-sentinel2-pyramid'
        """
        return {
            "name": self.name,
            "configuration": {
                "relative_step": float(self.relative_step),
                "min_step": float(self.min_step),
                "predictor_size": int(self.predictor_size),
                "quantification_value": float(self.quantification_value),
                "add_offset": float(self.add_offset),
                "effort": int(self.effort),
            },
        }

    def validate(self, *, shape, dtype, chunk_grid) -> None:
        """Reject arrays this codec cannot store.

        Parameters
        ----------
        shape : tuple of int
            Array shape.
        dtype : ZDType
            Array dtype; must be a floating dtype.
        chunk_grid : ChunkGrid
            Chunk grid; the last axis of the chunks must have a power-of-four
            size of at least 4.
        """
        native = dtype.to_native_dtype()
        if not np.issubdtype(native, np.floating):
            raise ValueError(
                f"{self.name} stores reflectances: expected a floating dtype, got {native}"
            )
        chunk_shape = getattr(chunk_grid, "chunk_shape", None)
        if (
            chunk_shape is not None
            and sentinel2.check_rows_shape(tuple(chunk_shape)) < 1
        ):
            raise ValueError(
                f"{self.name} needs chunks of at least 4 cells, got {tuple(chunk_shape)}"
            )

    def compute_encoded_size(
        self, input_byte_length: int, chunk_spec: ArraySpec
    ) -> int:
        raise NotImplementedError(
            "the size of a JPEG XL codestream depends on the content"
        )

    async def _decode_single(
        self, chunk_data: Buffer, chunk_spec: ArraySpec
    ) -> NDBuffer:
        # in a worker thread, like zarr's own compressors: the event loop stays free
        # and chunks written or read concurrently are coded in parallel
        cells = await asyncio.to_thread(
            sentinel2.decode_rows,
            chunk_data.to_bytes(),
            tuple(chunk_spec.shape),
            partial(
                sentinel2_pyramid.decode,
                quantification_value=self.quantification_value,
                add_offset=self.add_offset,
                dtype=chunk_spec.dtype.to_native_dtype(),
            ),
        )
        return chunk_spec.prototype.nd_buffer.from_ndarray_like(cells)

    async def _encode_single(
        self, chunk_array: NDBuffer, chunk_spec: ArraySpec
    ) -> Buffer | None:
        stream = await asyncio.to_thread(
            sentinel2.encode_rows,
            chunk_array.as_numpy_array(),
            partial(
                sentinel2_pyramid.encode,
                relative_step=self.relative_step,
                min_step=self.min_step,
                predictor_size=self.predictor_size,
                quantification_value=self.quantification_value,
                add_offset=self.add_offset,
                effort=self.effort,
            ),
        )
        return chunk_spec.prototype.buffer.from_bytes(stream)


def lookup_preset(name: str) -> list[HccSentinel2PyramidCodec]:
    """Build the codec pipeline of a named preset.

    Parameters
    ----------
    name : {'default', 'transfer-floor', 'lossless'}
        ``'default'`` is the codec's defaults (``relative_step=0.57``, about
        1.3 times the native size); ``'transfer-floor'`` adds an error close to
        the transfer error of the legacy-converters Sentinel-2 PSF preset
        (``relative_step=0.36``); ``'lossless'`` keeps the children on the
        digital-number lattice.

    Returns
    -------
    list of HccSentinel2PyramidCodec
        The codec pipeline, to pass as ``serializer`` or ``codecs``.

    Examples
    --------
    >>> lookup_preset("lossless")[0].relative_step
    0.0
    """
    match name:
        case "default":
            return [HccSentinel2PyramidCodec()]
        case "transfer-floor":
            return [
                HccSentinel2PyramidCodec(
                    relative_step=sentinel2_pyramid.TRANSFER_FLOOR_RELATIVE_STEP
                )
            ]
        case "lossless":
            return [HccSentinel2PyramidCodec(relative_step=0.0)]
        case _:
            raise ValueError(
                f"unknown preset for {HccSentinel2PyramidCodec.name}: {name!r}"
            )
