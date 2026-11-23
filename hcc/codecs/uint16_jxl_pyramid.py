"""Zarr serializer for the two-layer coding of HEALPix chunks of 16-bit integer data.

See :mod:`hcc.compression.uint16_jxl_pyramid`: a lossless parent level on the
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

from hcc.compression import uint16_jxl, uint16_jxl_pyramid

if TYPE_CHECKING:
    from typing import Self

    from zarr.abc.buffer import Buffer, NDBuffer
    from zarr.core.array_spec import ArraySpec
    from zarr.core.common import JSON


@dataclass(frozen=True)
class HccUint16JxlPyramidCodec(ArrayBytesCodec):
    """Zarr ArrayBytesCodec: lossless parent level, predicted and quantised children.

    For data stored as 16-bit integers with a scale and an offset (see
    :class:`hcc.codecs.uint16_jxl.HccUint16JxlCodec`). Each chunk must hold
    nested chunks of :math:`4^\\delta` floating-point values,
    :math:`\\delta \\ge 1`, along its last axis; leading axes
    (time, band, ...) are free and each row is coded on its own (see
    :func:`hcc.compression.uint16_jxl.encode_rows`).

    Parameters
    ----------
    relative_step : float, default 0.57
        The detail step as a fraction of the chunk's detail standard deviation;
        ``0`` uses ``min_step``. For a 10 m reflectance band resampled with
        a point-spread-function kernel, the default gives about 1.3 times the
        native tile under lossless JPEG 2000, for an added error of about 1.5
        times the transfer error; ``0.36`` puts the added error at the
        transfer error, for about 1.5 times the native size.
    min_step : float, default 1.0
        The smallest step, in digital numbers.
    predictor_size : int, default 5
        Odd side of the parent neighbourhood the children are predicted from.
    quantification_value : float, default 10000.0
        The scale :math:`Q`, in steps per unit of the values.
    add_offset : float, default -1000.0
        The offset :math:`O`, in digital numbers; e.g. ``-32768`` for a
        signed variable.
    effort : int, default 7
        JPEG XL encoder effort in :math:`[1, 9]`.
    decoding_speed : int, default 2
        JPEG XL decoding speed tier in :math:`[0, 4]`; higher decodes faster,
        for larger chunks, and the decoded cells do not depend on it. The
        default is about 3 % larger than ``0`` for half the JPEG XL encoding
        and decoding times. It is written to ``zarr.json`` only when it is not
        the default, so that the metadata of the default stays readable by
        earlier versions (which only decode, the same way, whatever it was).
    out_of_range : {'warn', 'raise', 'clip'}, default 'warn'
        What to do with parent means outside the range the scale and offset
        can hold (:func:`hcc.compression.uint16_jxl.value_range`). Written to
        ``zarr.json`` only when it is not the default, like
        ``decoding_speed``.

    See Also
    --------
    hcc.compression.uint16_jxl_pyramid.encode : The encoding function.
    hcc.codecs.uint16_jxl.HccUint16JxlCodec : The single-layer lossless codec.

    Examples
    --------
    >>> codec = HccUint16JxlPyramidCodec(relative_step=0.5)
    >>> codec.to_dict()["configuration"]["relative_step"]
    0.5
    """

    name: ClassVar[str] = "hcc-uint16-jxl-pyramid"

    relative_step: float = uint16_jxl_pyramid.DEFAULT_RELATIVE_STEP
    min_step: float = 1.0
    predictor_size: int = 5
    quantification_value: float = uint16_jxl.DEFAULT_QUANTIFICATION_VALUE
    add_offset: float = uint16_jxl.DEFAULT_ADD_OFFSET
    effort: int = 7
    decoding_speed: int = uint16_jxl_pyramid.DEFAULT_DECODING_SPEED
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
        HccUint16JxlPyramidCodec

        Examples
        --------
        >>> HccUint16JxlPyramidCodec.from_dict({"name": "hcc-uint16-jxl-pyramid"}).predictor_size
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
        >>> HccUint16JxlPyramidCodec().to_dict()["name"]
        'hcc-uint16-jxl-pyramid'
        """
        configuration = {
            "relative_step": float(self.relative_step),
            "min_step": float(self.min_step),
            "predictor_size": int(self.predictor_size),
            "quantification_value": float(self.quantification_value),
            "add_offset": float(self.add_offset),
            "effort": int(self.effort),
        }
        if self.decoding_speed != uint16_jxl_pyramid.DEFAULT_DECODING_SPEED:
            configuration["decoding_speed"] = int(self.decoding_speed)
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
            size of at least 4.
        """
        native = dtype.to_native_dtype()
        if not np.issubdtype(native, np.floating):
            raise ValueError(
                f"{self.name} stores floating-point values: expected a floating dtype, got {native}"
            )
        chunk_shape = getattr(chunk_grid, "chunk_shape", None)
        if (
            chunk_shape is not None
            and uint16_jxl.check_rows_shape(tuple(chunk_shape)) < 1
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
            uint16_jxl.decode_rows,
            chunk_data.to_bytes(),
            tuple(chunk_spec.shape),
            partial(
                uint16_jxl_pyramid.decode,
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
                uint16_jxl_pyramid.encode,
                relative_step=self.relative_step,
                min_step=self.min_step,
                predictor_size=self.predictor_size,
                quantification_value=self.quantification_value,
                add_offset=self.add_offset,
                effort=self.effort,
                decoding_speed=self.decoding_speed,
                out_of_range=self.out_of_range,
            ),
        )
        return chunk_spec.prototype.buffer.from_bytes(stream)


def lookup_preset(name: str) -> list[HccUint16JxlPyramidCodec]:
    """Build the codec pipeline of a named preset.

    Parameters
    ----------
    name : {'default', 'transfer-floor', 'lossless'}
        ``'default'`` is the codec's defaults (``relative_step=0.57``, about
        1.3 times the native size); ``'transfer-floor'`` adds an error close to
        the transfer error of a point-spread-function resampling
        (``relative_step=0.36``); ``'lossless'`` keeps the children on the
        digital-number lattice.

    Returns
    -------
    list of HccUint16JxlPyramidCodec
        The codec pipeline, to pass as ``serializer`` or ``codecs``.

    Examples
    --------
    >>> lookup_preset("lossless")[0].relative_step
    0.0
    """
    return _lookup_preset(HccUint16JxlPyramidCodec, name)


def _lookup_preset(
    cls: type[HccUint16JxlPyramidCodec], name: str
) -> list[HccUint16JxlPyramidCodec]:
    match name:
        case "default":
            return [cls()]
        case "transfer-floor":
            return [cls(relative_step=uint16_jxl_pyramid.TRANSFER_FLOOR_RELATIVE_STEP)]
        case "lossless":
            return [cls(relative_step=0.0)]
        case _:
            raise ValueError(f"unknown preset for {cls.name}: {name!r}")
