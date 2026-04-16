from typing import TypeVar

import numpy as np
import numpy.typing as npt

T = TypeVar("T", bound=np.floating)


def downsample(data: npt.NDArray[T], factor: int) -> npt.NDArray[T]:
    """Downsample a square array by an integer factor.

    Parameters
    ----------
    data: numpy.ndarray
        Input array of shape ``(n, n)``, where ``n`` must be divisible by ``factor``.
    factor : {1, 2, 4}
        Spatial reduction factor. Must be a power of two.

        * ``1``: the input as a float32 array without modification.
        * ``2``: 2x2 block average of the input array.
        * ``4`` and greater: bilinear interpolation using :py:func:`scipy.ndimage.zoom`.

    Returns
    -------
    numpy.ndarray
        Downsampled array. Has a shape of ``(n // factor, n // factor)`` and a
        dtype of ``float32``.

    See Also
    --------
    upsample : Inverse operation.
    """
    if factor == 1:
        result = data
    elif factor == 2:
        size = data.shape[0]
        shape = (size // 2, 2, size // 2, 2)
        result = np.mean(np.reshape(data, shape), axis=(1, 3))
    else:
        from scipy.ndimage import zoom

        result = zoom(np.astype(data, np.float64), 1.0 / factor, order=1)

    return np.astype(result, np.float32)


def upsample(data: npt.NDArray[T], factor: int) -> npt.NDArray[T]:
    """Upsample a square array by an integer factor using bilinear interpolation.

    Parameters
    ----------
    data : numpy.ndarray
        Square input array of shape ``(m, m)`` and dtype ``float32``.
    factor : {1, 2, 4}
        Spatial magnification factor. Must be a power of 2.

        * ``1``: the input as a float32 array without modification.
        * ``2`` or greater: bilinear interpolation via :py:func:`scipy.ndimage.zoom`.

    Returns
    -------
    numpy.ndarray
        Upsampled array of shape ``(m * factor, m * factor)`` and dtype ``float32``.

    See Also
    --------
    downsample : Inverse operation.
    """
    if factor == 1:
        result = data
    else:
        from scipy.ndimage import zoom

        result = zoom(np.astype(data, np.float64), factor, order=1)

    return np.astype(result, np.float32)
