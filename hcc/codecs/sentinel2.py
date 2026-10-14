"""Zarr serializer applying the Sentinel-2 product chain to HEALPix chunks.

See :mod:`hcc.compression.sentinel2` for the chain itself: reflectance
→ 16-bit digital numbers → Morton de-interleaving → lossless JPEG XL.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.common import parse_named_configuration

from hcc.compression import sentinel2

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.buffer import Buffer, NDBuffer
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import JSON


@dataclass(frozen=True)
class HccSentinel2Codec(ArrayBytesCodec):
    """Zarr ArrayBytesCodec: Sentinel-2 digital numbers plus lossless JPEG XL.

    Each chunk must hold nested chunks of :math:`4^\\delta` floating-point
    reflectances along its last axis; leading axes (time, band, ...) are free.
    Each row is stored as the lossless JPEG XL codestream of the
    :math:`2^\\delta \\times 2^\\delta` raster of its digital numbers, so the
    only error is the rounding to the DN lattice; ``NaN`` cells round-trip
    through the no-data number ``0``. A 1-D chunk is its single codestream;
    see :func:`hcc.compression.sentinel2.encode_rows` for the others.

    Parameters
    ----------
    quantification_value : float, default 10000.0
        The :math:`Q` of the product metadata (``QUANTIFICATION_VALUE``).
    add_offset : float, default -1000.0
        The :math:`O` of the product metadata (``RADIO_ADD_OFFSET`` or
        ``BOA_ADD_OFFSET``).
    effort : int, default 7
        JPEG XL encoder effort in :math:`[1, 9]`.

    See Also
    --------
    hcc.compression.sentinel2.encode : The encoding function.
    hcc.compression.sentinel2.decode : The decoding function.

    Examples
    --------
    >>> codec = HccSentinel2Codec()
    >>> codec.to_dict()
    {'name': 'hcc-sentinel2', 'configuration': {'quantification_value': 10000.0, 'add_offset': -1000.0, 'effort': 7}}
    """

    name: ClassVar[str] = "hcc-sentinel2"

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
        HccSentinel2Codec

        Examples
        --------
        >>> HccSentinel2Codec.from_dict({"name": "hcc-sentinel2", "configuration": {"effort": 3}})
        HccSentinel2Codec(quantification_value=10000.0, add_offset=-1000.0, effort=3)
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
        >>> HccSentinel2Codec(effort=3).to_dict()["configuration"]["effort"]
        3
        """
        return {
            "name": self.name,
            "configuration": {
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
            size.
        """
        native = dtype.to_native_dtype()
        if not np.issubdtype(native, np.floating):
            raise ValueError(
                f"{self.name} stores reflectances: expected a floating dtype, got {native}"
            )
        chunk_shape = getattr(chunk_grid, "chunk_shape", None)
        if chunk_shape is not None:
            sentinel2.check_rows_shape(tuple(chunk_shape))

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
                sentinel2.decode,
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
                sentinel2.encode,
                quantification_value=self.quantification_value,
                add_offset=self.add_offset,
                effort=self.effort,
            ),
        )
        return chunk_spec.prototype.buffer.from_bytes(stream)


def lookup_preset(name: str) -> list[HccSentinel2Codec]:
    """Build the codec pipeline of a named preset.

    Parameters
    ----------
    name : {'lossless'}
        The preset: ``'lossless'`` is the Sentinel-2 product chain with the
        L1C/L2A metadata defaults (:math:`Q = 10000`, :math:`O = -1000`).

    Returns
    -------
    list of HccSentinel2Codec
        The codec pipeline, to pass as ``serializer`` or ``codecs``.

    Examples
    --------
    >>> lookup_preset("lossless")
    [HccSentinel2Codec(quantification_value=10000.0, add_offset=-1000.0, effort=7)]
    """
    match name:
        case "lossless":
            return [HccSentinel2Codec()]
        case _:
            raise ValueError(f"unknown preset for {HccSentinel2Codec.name}: {name!r}")
