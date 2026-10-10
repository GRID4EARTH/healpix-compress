"""Level 19 lossless, level 20 predicted from level 19 and quantised: bits and errors.

Runs on the output of ``eopf_s2_to_healpix.py`` for a configuration with
``raw_reference`` set (the PSF output without codec is needed), one level-11
chunk at a time. For that chunk:

1. the level-19 base is the mean of the four children of every parent,
   rounded to the DN lattice and stored losslessly (JPEG XL on its DN raster);
2. the level-20 cells are predicted from the level-19 raster by one of several
   predictors: parent replication, bilinear interpolation, least-squares
   finite-difference filters over a k x k parent neighbourhood (one filter per
   child phase, their coefficients counted as side information), optionally
   with quadratic terms of the first-order differences;
3. the residual is quantised with a step ``s`` (in DN), fixed or made
   adaptive to the gradient magnitude of the level-19 base at the parent
   (known to the decoder, so free), and coded losslessly (JPEG XL on the
   integer raster);
4. the reconstruction is scored per cell and on the UTM samples, read back
   through the PSF forward operator of ``healpix_resample.PSFResampler``
   (the row-normalised ``MT``, a Gaussian-weighted average of the cells),
   overall and by gradient class, against the transfer floor of each class
   (the uncompressed cells read back against the source).

Bits per cell include the base and the side information; the ratio to UTM
converts them to bits per 100 m2 (2.587 cells per 10 m pixel) and divides by
the bits per pixel of the same band under lossless JPEG 2000, taken from the
``sizes.json`` of the conversion.

Usage
-----
    python eopf_s2_level20_tuning.py results/eopf_s2_paris_chunk
    python eopf_s2_level20_tuning.py results/eopf_s2_paris_chunk --steps 32 64 128 --laws fixed

Writes ``level20_tuning.json`` next to the stores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

LEVEL, CHUNK_LEVEL = 20, 11
CELL_AREA_M2 = 4 * np.pi * 6371000.0**2 / (12 * 4**LEVEL)
CELLS_PER_PIXEL = 100.0 / CELL_AREA_M2


# --- rasters and coding --------------------------------------------------------
def to_dn(x, q, o):
    return np.clip(np.rint(x * q - o), 1, 65535).astype(np.uint16)


def jxl_bits(u16) -> int:
    import imagecodecs

    return 8 * len(imagecodecs.jpegxl_encode(np.ascontiguousarray(u16), lossless=True, effort=7))


def replicate(m):
    return np.repeat(np.repeat(m, 2, axis=0), 2, axis=1)


def bilinear(m):
    from scipy.ndimage import map_coordinates

    c = (np.arange(2 * m.shape[0]) - 0.5) / 2.0  # child centre in parent coordinates
    yy, xx = np.meshgrid(c, c, indexing="ij")
    return map_coordinates(m, [yy, xx], order=1, mode="nearest")


def neighbourhood(m, k):
    """(n, n, k*k) parent neighbourhoods, edge-padded by reflection."""
    r = k // 2
    p = np.pad(m, r, mode="reflect")
    n = m.shape[0]
    return np.stack([p[i : i + n, j : j + n] for i in range(k) for j in range(k)], axis=-1)


def ls_filter(m, target, k, quadratic=False):
    """Least-squares filter from the k x k parent neighbourhood to each of the
    four child phases; returns the prediction and the number of coefficients."""
    feats = neighbourhood(m, k)
    n = m.shape[0]
    if quadratic:
        r = k // 2
        centre = feats[..., r * k + r]
        gx = feats[..., r * k + r + 1] - feats[..., r * k + r - 1]
        gy = feats[..., (r + 1) * k + r] - feats[..., (r - 1) * k + r]
        extra = np.stack([gx * gx, gy * gy, gx * gy, np.abs(gx), np.abs(gy), centre * gx, centre * gy], axis=-1)
        feats = np.concatenate([feats, extra], axis=-1)
    design = np.concatenate([feats.reshape(n * n, -1), np.ones((n * n, 1))], axis=1)
    pred = np.empty_like(target)
    n_coef = 0
    for di in (0, 1):
        for dj in (0, 1):
            coef, *_ = np.linalg.lstsq(design, target[di::2, dj::2].reshape(-1), rcond=None)
            pred[di::2, dj::2] = (design @ coef).reshape(n, n)
            n_coef += coef.size
    return pred, n_coef


def gradient_magnitude(m, q):
    """Central-difference gradient magnitude of the base, in DN per parent cell."""
    p = np.pad(m, 1, mode="reflect")
    return np.hypot((p[1:-1, 2:] - p[1:-1, :-2]) / 2, (p[2:, 1:-1] - p[:-2, 1:-1]) / 2) * q


# --- the forward operator on the chunk's UTM samples -----------------------------
class ReadBack:
    """UTM samples of the chunk (with the converter's 60 m buffer) and the PSF
    forward operator that predicts them from the cells."""

    def __init__(self, out_dir: Path, group: str, band: str, chunk_id: int, cell_ids: np.ndarray,
                 q: float, o: float, resampler: dict):
        import healpix_geo.nested as hn
        import healpix_resample as hr
        import pyproj
        import shapely
        import torch
        import zarr

        g = zarr.open_group(out_dir / "source_subset.zarr", mode="r")
        src, g = g, g[group]
        crs = pyproj.CRS.from_user_input(src.attrs["stac_discovery"]["properties"].get("proj:code", "EPSG:32631"))
        to_utm = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
        to_lonlat = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
        vlon, vlat = hn.vertices(np.array([chunk_id]), CHUNK_LEVEL, ellipsoid="WGS84")
        poly = shapely.Polygon(list(zip(*to_utm.transform(np.asarray(vlon[0]), np.asarray(vlat[0])))))
        xmin, ymin, xmax, ymax = shapely.bounds(shapely.buffer(poly, 60.0))
        x, y = np.asarray(g["x"][:], dtype=np.float64), np.asarray(g["y"][:], dtype=np.float64)
        cx = np.nonzero((x >= xmin) & (x <= xmax))[0]
        cy = np.nonzero((y >= ymin) & (y <= ymax))[0]
        dn = np.asarray(g[band][cy.min() : cy.max() + 1, cx.min() : cx.max() + 1])
        xx, yy = np.meshgrid(x[cx.min() : cx.max() + 1], y[cy.min() : cy.max() + 1])
        keep = shapely.within(shapely.points(xx.ravel(), yy.ravel()), shapely.buffer(poly, 60.0)) & (dn.ravel() > 0)
        sx, sy = xx.ravel()[keep], yy.ravel()[keep]
        self.samples = (dn.ravel()[keep].astype(np.float64) + o) / q
        lon, lat = (np.asarray(v) for v in to_lonlat.transform(sx, sy))
        interior = shapely.within(shapely.points(sx, sy), shapely.buffer(poly, -40.0))

        init = dict(resampler.get("init_params", {}))
        self.op = hr.PSFResampler(lon_deg=lon, lat_deg=lat, level=LEVEL, out_cell_ids=cell_ids, verbose=False,
                                  nest=True, ellipsoid="WGS84", **init)
        op_ids = np.asarray(self.op.get_cell_ids() if hasattr(self.op, "get_cell_ids") else self.op.cell_ids,
                            dtype=np.int64)
        self.pos = op_ids - int(cell_ids[0])
        if np.any((self.pos < 0) | (self.pos >= cell_ids.size)):
            raise RuntimeError("the operator retained cells outside the chunk")
        self.norm = (torch.ones(1, op_ids.size, dtype=self.op.MT.dtype) @ self.op.MT).squeeze(0)
        self.good = (self.norm > 0.5).numpy() & interior  # samples fully covered by the chunk's cells
        parent = np.asarray(hn.lonlat_to_healpix(lon, lat, LEVEL - 1, ellipsoid="WGS84"), dtype=np.int64)
        self.parent_pos = parent - (chunk_id << (2 * (LEVEL - 1 - CHUNK_LEVEL)))

    def __call__(self, raster: np.ndarray) -> np.ndarray:
        import torch

        from hcc.compression.zorder import flatten_tile

        v = torch.as_tensor(flatten_tile(raster)[self.pos], dtype=self.op.MT.dtype)[None, :]
        return ((v @ self.op.MT).squeeze(0) / self.norm).numpy()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", type=Path, help="output directory of eopf_s2_to_healpix.py (one chunk, raw_reference)")
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 100, 150, 200, 300])
    ap.add_argument("--predictors", nargs="+", default=["replicate", "bilinear", "ls3x3", "ls5x5", "ls7x7", "ls5x5+quad"])
    ap.add_argument("--laws", nargs="+", default=["fixed", "sqrt", "linear", "strong"],
                    help="step laws: fixed, sqrt = s*(g/p50)^0.5 capped 4, linear = s*(g/p50) capped 4, "
                         "strong = s*(g/p25) capped 8")
    ap.add_argument("--targets", type=float, nargs="+", default=[1.3, 1.5, 1.75])
    args = ap.parse_args(argv)

    import zarr

    from hcc.compression.zorder import expand_tile, flatten_tile

    out_dir = args.out_dir
    sizes = json.loads((out_dir / "sizes.json").read_text())
    band = next(iter(sizes["bands"]))
    conf = sizes["codec"]["configuration"]
    q, o = float(conf["quantification_value"]), float(conf["add_offset"])
    utm_bpp = 8 * sizes["bands"][band]["utm"]["jpeg2000_lossless_bytes"] / sizes["bands"][band]["utm"]["valid_pixels"]

    raw = zarr.open_group(out_dir / "healpix_raw.zarr", mode="r")
    arrays = {p.rsplit("/", 1)[-1]: p for p, node in raw.members(max_depth=None) if isinstance(node, zarr.Array)}
    cells = np.asarray(raw[arrays[band]][:], dtype=np.float64)
    cell_ids = np.asarray(raw[arrays["cell_ids"]][:], dtype=np.int64)
    n = 4 ** (LEVEL - CHUNK_LEVEL)
    if cells.size != n or not np.all(np.isfinite(cells)):
        raise SystemExit("this needs one complete level-11 chunk (eopf_s2_paris_chunk.json)")
    chunk_id = int(cell_ids[0] >> (2 * (LEVEL - CHUNK_LEVEL)))
    side = int(np.sqrt(n))

    x20 = expand_tile(cells)
    m19 = x20.reshape(side // 2, 2, side // 2, 2).mean(axis=(1, 3))
    base_dn = to_dn(m19, q, o)
    m19q = (base_dn.astype(np.float64) + o) / q
    base_bits = jxl_bits(base_dn)
    grad = gradient_magnitude(m19q, q)
    pct = {k: float(np.percentile(grad, k)) for k in (25, 50, 90)}
    print(f"chunk {chunk_id}: {n} cells; base {base_bits / n:.3f} bits per cell; UTM JPEG 2000 {utm_bpp:.3f} bits per pixel")

    predictors = {}
    for name in args.predictors:
        if name == "replicate":
            predictors[name] = (replicate(m19q), 0)
        elif name == "bilinear":
            predictors[name] = (bilinear(m19q), 0)
        elif name.startswith("ls"):
            k = int(name[2])
            predictors[name] = ls_filter(m19q, x20, k, quadratic=name.endswith("+quad"))
        else:
            raise SystemExit(f"unknown predictor {name!r}")
    var0 = np.var(x20 - replicate(m19q))
    print("\npredictor        R2 vs replicate   residual std (DN)   coefficients")
    for name, (pred, n_coef) in predictors.items():
        r = x20 - pred
        print(f"  {name:<14} {1 - np.var(r) / var0:8.3f}         {r.std() * q:9.1f}            {n_coef}")

    t0 = time.perf_counter()
    read_back = ReadBack(out_dir, sizes["group"], band, chunk_id, cell_ids, q, o, sizes["resampler"])
    y_ref = read_back(x20)
    good = read_back.good
    grad_nested = flatten_tile(grad)
    ok = (read_back.parent_pos >= 0) & (read_back.parent_pos < grad_nested.size)
    sample_grad = np.where(ok, grad_nested[np.clip(read_back.parent_pos, 0, grad_nested.size - 1)], np.nan)
    classes = {"all": np.ones_like(sample_grad, dtype=bool), "flat": sample_grad < pct[50], "edge": sample_grad > pct[90]}
    floor = (y_ref - read_back.samples) * q
    floors = {k: float(np.sqrt(np.mean(floor[good & m] ** 2))) for k, m in classes.items()}
    print(f"\nforward operator: {read_back.samples.size} samples ({good.sum()} interior), built in "
          f"{time.perf_counter() - t0:.1f} s; transfer floor (DN RMS) all/flat/edge: "
          f"{floors['all']:.1f} / {floors['flat']:.1f} / {floors['edge']:.1f}")

    laws = {"fixed": (0.0, 50, 1), "sqrt": (0.5, 50, 4), "linear": (1.0, 50, 4), "strong": (1.0, 25, 8)}
    rows = []
    print("\npredictor      law      s0   bits/cell  x UTM(JP2)  cell RMS   UTM RMS all/flat/edge (DN)")
    for pname, (pred, n_coef) in predictors.items():
        resid = x20 - pred
        for law in args.laws:
            alpha, p, cap = laws[law]
            factor = np.minimum(cap, np.maximum(1.0, grad / pct[p]) ** alpha) if alpha > 0 else np.ones_like(grad)
            step = replicate(factor)
            for s0 in args.steps:
                s = s0 * step
                rq = np.rint(resid * q / s).astype(np.int64)
                recon = pred + rq * s / q
                bits = (base_bits + jxl_bits((rq - rq.min()).astype(np.uint16)) + 32 * n_coef) / n
                d = (read_back(recon) - y_ref) * q
                rms = {k: float(np.sqrt(np.mean(d[good & m] ** 2))) for k, m in classes.items()}
                rows.append(dict(predictor=pname, law=law, s0=s0, bits_per_cell=bits,
                                 ratio_utm_jpeg2000=bits * CELLS_PER_PIXEL / utm_bpp,
                                 cell_rms_dn=float(np.sqrt(np.mean(((recon - x20) * q) ** 2))), utm_rms_dn=rms,
                                 utm_max_dn=float(np.abs(d[good]).max())))
                print(f"  {pname:<12} {law:<7} {s0:4d}   {bits:8.3f}   {rows[-1]['ratio_utm_jpeg2000']:8.3f}   "
                      f"{rows[-1]['cell_rms_dn']:6.1f}   {rms['all']:6.1f} / {rms['flat']:6.1f} / {rms['edge']:6.1f}")

    summary = {}
    print("\nat equal bits (interpolated), UTM RMS all/flat/edge (DN) and s0:")
    for pname in predictors:
        for law in args.laws:
            sel = sorted((r for r in rows if r["predictor"] == pname and r["law"] == law), key=lambda r: r["ratio_utm_jpeg2000"])
            xs = [r["ratio_utm_jpeg2000"] for r in sel]
            for target in args.targets:
                if not xs[0] <= target <= xs[-1]:
                    continue
                v = {k: float(np.interp(target, xs, [r["utm_rms_dn"][k] for r in sel])) for k in classes}
                v["s0"] = float(np.interp(target, xs, [r["s0"] for r in sel]))
                summary[f"{pname} {law} x{target}"] = v
                print(f"  {pname:<12} {law:<7} x{target}: {v['all']:5.1f} / {v['flat']:5.1f} / {v['edge']:5.1f}   (s0 {v['s0']:.0f})")

    (out_dir / "level20_tuning.json").write_text(json.dumps(dict(
        chunk=chunk_id, band=band, base_bits_per_cell=base_bits / n, utm_jpeg2000_bits_per_pixel=utm_bpp,
        transfer_floor_dn=floors, gradient_percentiles_dn=pct,
        predictors={k: {"r2": float(1 - np.var(x20 - v[0]) / var0), "coefficients": v[1]} for k, v in predictors.items()},
        rows=rows, summary=summary), indent=1))
    print(f"\nwritten: {out_dir / 'level20_tuning.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
