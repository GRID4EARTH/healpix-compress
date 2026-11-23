"""Zarr serializer storing HEALPix chunks as 16-bit integers in lossless JPEG XL.

See :mod:`hcc.compression.uint16_jxl` for the chain itself: values
→ 16-bit digital numbers (scale and offset) → Morton de-interleaving
→ lossless JPEG XL.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from zarr.abc.codec import ArrayBytesCodec
from zarr.core.common import parse_named_configuration

from hcc.compression import uint16_jxl

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.buffer import Buffer, NDBuffer
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import JSON


@dataclass(frozen=True)
class HccUint16JxlCodec(ArrayBytesCodec):
    """Zarr ArrayBytesCodec: 16-bit digital numbers plus lossless JPEG XL.

    For data stored as unsigned or signed 16-bit integers with a scale and an
    offset, :math:`x = (DN + O) / Q`: such data round-trip exactly. Each chunk
    must hold nested chunks of :math:`4^\\delta` floating-point values along
    its last axis; leading axes (time, band, ...) are free.
    Each row is stored as the lossless JPEG XL codestream of the
    :math:`2^\\delta \\times 2^\\delta` raster of its digital numbers, so the
    only error is the rounding to the DN lattice; ``NaN`` cells round-trip
    through the no-data number ``0``. A 1-D chunk is its single codestream;
    see :func:`hcc.compression.uint16_jxl.encode_rows` for the others.

    Parameters
    ----------
    quantification_value : float, default 10000.0
        The scale :math:`Q`, in steps per unit of the values.
    add_offset : float, default -1000.0
        The offset :math:`O`, in digital numbers; e.g. ``-32768`` for a
        signed variable.
    effort : int, default 7
        JPEG XL encoder effort in :math:`[1, 9]`.
    out_of_range : {'warn', 'raise', 'clip'}, default 'warn'
        What to do with finite values outside the range the scale and offset
        can hold (:func:`hcc.compression.uint16_jxl.value_range`): clip them
        with a warning, raise, or clip them silently. Written to ``zarr.json``
        only when it is not the default, so that the metadata stays readable
        by earlier versions; reading does not use it.

    See Also
    --------
    hcc.compression.uint16_jxl.encode : The encoding function.
    hcc.compression.uint16_jxl.decode : The decoding function.

    Examples
    --------
    >>> codec = HccUint16JxlCodec()
    >>> codec.to_dict()
    {'name': 'hcc-uint16-jxl', 'configuration': {'quantification_value': 10000.0, 'add_offset': -1000.0, 'effort': 7}}

    A signed variable with steps of ``0.01``, from -327.67 to 327.67:

    >>> HccUint16JxlCodec(quantification_value=100.0, add_offset=-32768.0)
    HccUint16JxlCodec(quantification_value=100.0, add_offset=-32768.0, effort=7, out_of_range='warn')
    """

    name: ClassVar[str] = "hcc-uint16-jxl"

    quantification_value: float = uint16_jxl.DEFAULT_QUANTIFICATION_VALUE
    add_offset: float = uint16_jxl.DEFAULT_ADD_OFFSET
    effort: int = 7
    out_of_range: str = uint16_jxl.DEFAULT_OUT_OF_RANGE

    def __post_init__(self) -> None:
        uint16_jxl.check_out_of_range(self.out_of_range)

    @classmethod
    def from_dict(cls, data: dict[str, JSON]) -> Self:
        """Deserialise from a Zarr codec JSON object.

        Parameters
        ----------
        data : dict
            Codec JSON as stored in ``zarr.json``.

        Returns
        -------
        HccUint16JxlCodec

        Examples
        --------
        >>> HccUint16JxlCodec.from_dict({"name": "hcc-uint16-jxl", "configuration": {"effort": 3}})
        HccUint16JxlCodec(quantification_value=10000.0, add_offset=-1000.0, effort=3, out_of_range='warn')
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
        >>> HccUint16JxlCodec(effort=3).to_dict()["configuration"]["effort"]
        3
        """
        configuration = {
            "quantification_value": float(self.quantification_value),
            "add_offset": float(self.add_offset),
            "effort": int(self.effort),
        }
        if self.out_of_range != uint16_jxl.DEFAULT_OUT_OF_RANGE:
            configuration["out_of_range"] = self.out_of_range
        return {"name": self.name, "configuration": configuration}

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
                f"{self.name} stores floating-point values: expected a floating dtype, got {native}"
            )
        chunk_shape = getattr(chunk_grid, "chunk_shape", None)
        if chunk_shape is not None:
            uint16_jxl.check_rows_shape(tuple(chunk_shape))

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
            uint16_jxl.decode_rows,
            chunk_data.to_bytes(),
            tuple(chunk_spec.shape),
            partial(
                uint16_jxl.decode,
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
            uint16_jxl.encode_rows,
            chunk_array.as_numpy_array(),
            partial(
                uint16_jxl.encode,
                quantification_value=self.quantification_value,
                add_offset=self.add_offset,
                effort=self.effort,
                out_of_range=self.out_of_range,
            ),
        )
        return chunk_spec.prototype.buffer.from_bytes(stream)


def lookup_preset(name: str) -> list[HccUint16JxlCodec]:
    """Build the codec pipeline of a named preset.

    Parameters
    ----------
    name : {'lossless', 'reflectance'}
        The preset: ``'reflectance'`` stores values in steps of ``1e-4`` from
        about ``-0.1`` (:math:`Q = 10000`, :math:`O = -1000`), the codec's
        defaults; ``'lossless'`` is an earlier name of the same pipeline.

    Returns
    -------
    list of HccUint16JxlCodec
        The codec pipeline, to pass as ``serializer`` or ``codecs``.

    Examples
    --------
    >>> lookup_preset("reflectance")
    [HccUint16JxlCodec(quantification_value=10000.0, add_offset=-1000.0, effort=7, out_of_range='warn')]
    """
    return _lookup_preset(HccUint16JxlCodec, name)


def _lookup_preset(cls: type[HccUint16JxlCodec], name: str) -> list[HccUint16JxlCodec]:
    match name:
        case "lossless" | "reflectance":
            return [
                cls(
                    quantification_value=uint16_jxl.DEFAULT_QUANTIFICATION_VALUE,
                    add_offset=uint16_jxl.DEFAULT_ADD_OFFSET,
                )
            ]
        case _:
            raise ValueError(f"unknown preset for {cls.name}: {name!r}")
