from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.fft import dctn, idctn

from scipy.ndimage import zoom as sp_zoom, uniform_filter
import zlib

def _decode_zorder(delta: int):
    size = 4**delta
    x = np.zeros((size,), dtype="uint64")
    y = np.zeros((size,), dtype="uint64")

    code = np.arange(size, dtype="uint64")
    for bit in range(delta):
        x |= ((code >> (bit << 1)) & 1) << bit
        y |= ((code >> ((bit << 1) + 1)) & 1) << bit

    return x, y


def _flatten_tile(data: np.ndarray):
    delta = int(np.log(data.size) / np.log(4))
    x, y = _decode_zorder(delta)

    return np.squeeze(data[..., x, y])


def _expand_tile(data: np.ndarray):
    nside = int(np.sqrt(data.shape[-1]))
    delta = int(np.log2(nside))

    x, y = _decode_zorder(delta)

    shape = (*data.shape[:-1], 2**delta, 2**delta)

    # TODO: figure out how to invert x, y to become a 2D array of indexes into data
    out = np.empty(shape, dtype=data.dtype)
    out[..., x, y] = data

    return out


def _compress_residuals_grad_dct(
    residuals: np.ndarray,
    top_k: int,
    quantization_dtype: np.typing.DTypeLike,
    scale_dtype: np.typing.DTypeLike,
) -> list[np.ndarray]:
    expanded_residuals = _expand_tile(residuals)
    coefficients = np.reshape(dctn(expanded_residuals, norm="ortho"), (-1,))

    # keep the top_k largest magnitude coefficients
    to_keep = np.argsort(np.abs(coefficients))[::-1][:top_k]
    kept_coefficients = coefficients[to_keep]

    if kept_coefficients.size == 0:
        scale = np.array(1.0, dtype=scale_dtype)
    else:
        scale = np.asarray(np.max(np.abs(kept_coefficients)), dtype=scale_dtype)
        if scale < 0:
            scale = np.array(1.0, dtype=scale_dtype)

    max_value = np.iinfo(quantization_dtype).max
    quantization = np.round((kept_coefficients / scale) * max_value)

    return CompressedResiduals(
        quantization=np.astype(quantization, quantization_dtype),
        scale=np.astype(scale, scale_dtype),
        coefficient_positions=to_keep,
    )


def _decompress_residuals_grad_dct(
    compressed_residuals: list[np.ndarray],
    shape: tuple[int, int],
    residuals_dtype: np.typing.DTypeLike,
    top_k: int,
):
    quantization = compressed_residuals.quantization
    scale = compressed_residuals.scale
    coefficient_positions = compressed_residuals.coefficient_positions

    kept_coefficients = np.astype(
        (quantization / np.iinfo(quantization.dtype).max) * scale,
        scale.dtype,
    )

    coefficients = np.reshape(np.zeros(shape, dtype=residuals_dtype), -1)
    coefficients[coefficient_positions] = kept_coefficients

    tile = idctn(np.reshape(coefficients, shape), norm="ortho")

    return np.astype(_flatten_tile(tile), residuals_dtype)


# ─────────────────────────────────────────────────────────────────
#  Encodeur
# ─────────────────────────────────────────────────────────────────

def block_mean_2x2(data: np.ndarray) -> np.ndarray:
    """[n,n] → [n//2, n//2] : moyenne sur blocs 2×2 non-chevauchants."""
    n = data.shape[0]
    return data.reshape(n // 2, 2, n // 2, 2).mean(axis=(1, 3)).astype(np.float32)


def upsample2x(data: np.ndarray) -> np.ndarray:
    """[m,m] → [2m,2m] : interpolation bilinéaire (order=1)."""
    return sp_zoom(data.astype(np.float64), 2, order=1).astype(np.float32)


def local_std(data: np.ndarray, window: int = 8) -> np.ndarray:
    """
    Carte d'écart-type local calculée par filtre uniforme.
    Exploite l'identité  Var(X) = E[X²] − E[X]²  pour rester O(n).
    """
    f = data.astype(np.float64)
    mu  = uniform_filter(f,     size=window)
    mu2 = uniform_filter(f**2,  size=window)
    return np.sqrt(np.maximum(mu2 - mu**2, 0)).astype(np.float32)
    
    
def encode(data: np.ndarray, window: int = 8) -> dict:
    """
    Encode data [n,n] float32 en dictionnaire compressé.

    Paramètres
    ----------
    data   : np.ndarray, shape (n,n), dtype float32 — données à compresser
    window : int — taille de la fenêtre pour le calcul de la variance locale

    Retourne
    --------
    dict avec clés :
      n, low, q_lo, q_hi, sigma_floor, residual_bytes
    """
    data = data.astype(np.float32)
    n = data.shape[0]
    assert data.shape == (n, n),         "data doit être carré"
    assert (n & (n - 1)) == 0 and n > 1, "n doit être une puissance de 2"

    # ── 1. Basse résolution (basses fréquences, stockée lossless) ──────────
    low  = block_mean_2x2(data)           # [n//2, n//2]

    # ── 2. Prédiction par upsampling bilinéaire ────────────────────────────
    pred = upsample2x(low)                # [n, n]

    # ── 3. Résidu ──────────────────────────────────────────────────────────
    R    = data - pred                    # [n, n]

    # ── 4. Carte de variance locale (contexte haute fréquence prédit) ──────
    #     sigma(x,y) ∝ activité locale dans la prédiction
    #     → zones lisses : faible sigma  → peu de résidu attendu
    #     → zones actives : grand sigma → fort résidu attendu
    sigma       = local_std(pred, window=window)
    sigma_floor = float(np.std(R)) * 0.05          # plancher anti-div-zéro
    sigma       = np.maximum(sigma, sigma_floor)   # in-place ne marche pas sur view

    # ── 5. Normalisation adaptative en variance ────────────────────────────
    #     Après normalisation, R_norm est statistiquement plus blanc/homogène
    #     → meilleure compressibilité entropique
    R_norm = (R / sigma).astype(np.float32)

    # ── 6. Quantification → int8  (1 octet/valeur = 25 % du float32 brut) ──
    q_lo = float(np.percentile(R_norm,  0.5))
    q_hi = float(np.percentile(R_norm, 99.5))
    scale      = (q_hi - q_lo) / 255.0
    R_clipped  = np.clip(R_norm, q_lo, q_hi)
    R_q        = np.round((R_clipped - q_lo) / scale - 128).astype(np.int8)

    # ── 7. Compression entropique zlib du tableau int8 ─────────────────────
    #     La normalisation whitens le signal → meilleur ratio zlib
    residual_bytes = zlib.compress(R_q.tobytes(), level=9)

    return [
        np.array([n],           dtype=np.uint64),
        low.ravel(),
        np.array([q_lo],        dtype=np.float32),
        np.array([q_hi],        dtype=np.float32),
        np.array([sigma_floor], dtype=np.float32),
        np.frombuffer(residual_bytes, dtype=np.uint8),
    ]


# ─────────────────────────────────────────────────────────────────
#  Décodeur
# ─────────────────────────────────────────────────────────────────

def decode(enc: list, window: int) -> np.ndarray:
    """
    Reconstruit data [n,n] float32 depuis la liste de ndarrays compressés.
    La carte sigma est recalculée à la volée (non stockée).
    """
    n           = int(enc[0][0])
    low         = enc[1].reshape(n // 2, n // 2)
    q_lo        = float(enc[2][0])
    q_hi        = float(enc[3][0])
    sigma_floor = float(enc[4][0])
    residual_bytes = enc[5].tobytes()

    # Prédiction (identique à l'encodeur)
    pred  = upsample2x(low)
    sigma = local_std(pred, window=window)
    sigma = np.maximum(sigma, sigma_floor)

    # Décompression zlib
    raw = zlib.decompress(residual_bytes)
    R_q = np.frombuffer(raw, dtype=np.int8).reshape(n, n).astype(np.float32)

    # Déquantification + dénormalisation variance
    scale = (q_hi - q_lo) / 255.0
    R     = ((R_q + 128) * scale + q_lo) * sigma

    return pred + R


def compress(
    chunk: np.ndarray,
    window: int,
) -> list[np.ndarray]:
    
    l_delta = int(np.log2(chunk.shape[0])/2)
    x,y = _decode_zorder(l_delta)
    
    internal_data = np.zeros([2**l_delta,2**l_delta])
    internal_data[x,y]=chunk

    return encode(internal_data, window=window)


def decompress(compressed: list,window: int) -> np.ndarray:
    
    reconstructed=decode(compressed,window)
    
    return _flatten_tile(reconstructed)
