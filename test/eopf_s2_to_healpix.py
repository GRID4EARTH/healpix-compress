"""Convert a Sentinel-2 band from an EOPF Zarr tile to HEALPix with the
GRID4EARTH chain and compare the bytes on disk.

The question this answers is the only realistic measure of a compression
scheme: how many bytes does the same band take in the product as ESA
distributes it (an EOPF Zarr tile: uint16 digital numbers under blosc-zstd),
and on a HEALPix level-20 grid filled by the PSF transfer and stored with the
``hcc-sentinel2`` serializer (the same digital numbers, lossless JPEG XL per
nested chunk)?

The conversion itself is the operational one: ``healpix_convert.create_healpix_dataset``
with the Sentinel-2 presets of ``legacy_converters`` (level 20, dense chunks at
level 11, PSF resampler), the preset's codecs replaced by the serializer under
test. Everything is driven by one JSON file (see ``eopf_s2_paris.json``):

- ``input``: an EOPF store given by ``path`` (a local copy, or an ``s3://``
  location such as the eopf-mirror bucket, with ``storage_options`` for
  s3fs) or, when ``path`` is null, a STAC search on the EOPF sample service
  that selects the product to fetch over HTTPS;
- ``product``, ``group``, ``bands``: the preset name (``sentinel-2-l2a``),
  the group and the bands to convert; the other bands of the group are left
  out by writing a local subset of the source store;
- ``resampler``: PSFResampler parameters that override the preset's
  (``init_params``, ``resample_params``);
- ``codec``: the serializer and its configuration; ``quantification_value``
  and ``add_offset`` are derived from the band's ``scale_factor`` and
  ``add_offset`` unless given;
- ``output_extent``: null for the whole tile, ``{"point": [lon, lat]}`` for
  the one level-11 chunk holding that point, or ``{"bbox": [w, s, e, n]}``;
- ``raw_reference``: also convert with the preset's default codecs, to score
  the PSF output before the serializer clips it to the DN range.

The sizes reported are: the source band as stored (sum of its chunk objects),
the same digital numbers under lossless JPEG XL and JPEG 2000 per source
chunk, and the HEALPix band under the serializer, under the source's own
blosc-zstd (same digital numbers, for a like-for-like codec comparison) and
its ``cell_ids``.

Usage
-----
    python eopf_s2_to_healpix.py eopf_s2_paris.json
    python eopf_s2_to_healpix.py eopf_s2_paris.json --force

Requires: healpix-compress[jpegxl] (this branch), healpix-convert with
PR 24 (half-pixel shift) and PR 25 (chunks without input), legacy-converters,
healpix-resample (torch), healpix-geo, pystac-client, requests, numcodecs.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# --- the source band -----------------------------------------------------------
def find_item(stac: dict) -> dict:
    """The STAC item selected by the configuration, as a dict."""
    import pystac_client

    catalog = pystac_client.Client.open(stac["url"])
    if stac.get("item_id"):
        items = list(catalog.search(collections=[stac["collection"]], ids=[stac["item_id"]]).items())
        if not items:
            raise RuntimeError(f"item {stac['item_id']!r} not found in {stac['collection']!r}")
        return items[0].to_dict()
    lon, lat = stac["point"]
    items = list(
        catalog.search(
            collections=[stac["collection"]],
            intersects={"type": "Point", "coordinates": [lon, lat]},
            datetime=stac.get("datetime"),
            query=stac.get("query"),
            sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
            max_items=1,
        ).items()
    )
    if not items:
        raise RuntimeError("the STAC search returned no item; widen datetime or the cloud query")
    return items[0].to_dict()


class EopfSource:
    """One EOPF product, Zarr v2 or v3, local, over HTTP or on S3.

    The arrays are decoded by zarr; the bytes each stored chunk occupies are
    read from the store itself (one ``info`` per chunk key), so the source size
    is the one the object store holds, not an estimate. ``storage_options`` go
    to fsspec; for the eopf-mirror bucket: ``{"profile": "grid4earth",
    "client_kwargs": {"endpoint_url": "https://s3.gra.io.cloud.ovh.net"}}``.
    """

    def __init__(self, location: str, storage_options: dict | None = None):
        import fsspec
        import zarr

        self.location = location.rstrip("/")
        self.storage_options = dict(storage_options or {})
        self.root = zarr.open_group(self.location, mode="r", storage_options=self.storage_options or None)
        self.fs, self.root_path = fsspec.core.url_to_fs(self.location, **self.storage_options)
        self.root_path = self.root_path.rstrip("/")
        self.root_attrs = self.root.attrs.asdict()

    @staticmethod
    def chunk_keys(arr) -> list[str]:
        grid = [math.ceil(s / c) for s, c in zip(arr.shape, arr.chunks)]
        meta = arr.metadata
        if meta.zarr_format == 3:
            return [meta.chunk_key_encoding.encode_chunk_key(idx) for idx in np.ndindex(*grid)]
        sep = meta.dimension_separator or "."
        return [sep.join(str(i) for i in idx) for idx in np.ndindex(*grid)]

    def read_array(self, path: str) -> tuple[np.ndarray, dict]:
        """Decode a whole array; also returns its description and the stored
        bytes per chunk."""
        arr = self.root[path]
        data = arr[:]
        prefix = f"{self.root_path}/{path}"
        chunk_bytes = {}
        for key in self.chunk_keys(arr):
            try:
                chunk_bytes[key] = int(self.fs.info(f"{prefix}/{key}")["size"])
            except FileNotFoundError:
                continue
        meta = arr.metadata.to_dict()
        if arr.metadata.zarr_format == 2:
            codec = {"zarr_format": 2, "compressor": meta.get("compressor"), "filters": meta.get("filters")}
        else:
            codec = {"zarr_format": 3, "codecs": meta["codecs"]}
        fill = arr.fill_value
        info = dict(
            shape=list(arr.shape), chunks=list(arr.chunks), dtype=str(arr.dtype),
            fill_value=None if fill is None else fill.item(), codec=codec, attrs=arr.attrs.asdict(),
            chunk_bytes=chunk_bytes, stored_bytes=int(sum(chunk_bytes.values())),
        )
        return data, info


def codec_label(codec: dict) -> str:
    """A short name for the codec a source array is stored with."""
    if codec["zarr_format"] == 2:
        comp = codec.get("compressor") or {}
        if comp.get("id") == "blosc":
            shuffle = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}.get(comp.get("shuffle"), "?")
            return f"blosc-{comp.get('cname')}({comp.get('clevel')}, {shuffle})"
        return f"{comp.get('id', 'none')}({comp.get('level', comp.get('clevel', ''))})"
    names = [c["name"] for c in codec["codecs"] if c["name"] != "bytes"]
    confs = [c.get("configuration", {}) for c in codec["codecs"] if c["name"] != "bytes"]
    return "+".join(f"{n}({c.get('level', c.get('clevel', ''))})" for n, c in zip(names, confs)) or "none"


def write_subset_store(src: EopfSource, group: str, bands: list[str], dest: Path) -> dict:
    """A local Zarr v3 store holding the root STAC attributes and the chosen
    bands of one group, with their x/y coordinates: the input of the converter.

    Returns the source description (stored bytes, chunking, scaling) per band.
    """
    import zarr

    if dest.exists():
        shutil.rmtree(dest)
    root = zarr.create_group(dest, zarr_format=3)
    root.attrs.update(src.root_attrs)
    g = root.create_group(group)
    for axis in ("x", "y"):
        values, axis_info = src.read_array(f"{group}/{axis}")
        g.create_array(axis, data=values, chunks=values.shape, dimension_names=[axis],
                       attributes={k: v for k, v in axis_info["attrs"].items() if k != "_ARRAY_DIMENSIONS"})
    info = {}
    for band in bands:
        t0 = time.perf_counter()
        dn, arr_info = src.read_array(f"{group}/{band}")
        attrs = {k: v for k, v in arr_info["attrs"].items() if k not in ("_ARRAY_DIMENSIONS", "_eopf_attrs")}
        fill = arr_info["fill_value"] or 0
        g.create_array(band, data=dn, chunks=tuple(arr_info["chunks"]), dimension_names=["y", "x"],
                       fill_value=fill, attributes=attrs)
        info[band] = dict(
            shape=arr_info["shape"], chunks=arr_info["chunks"], dtype=arr_info["dtype"],
            codec=arr_info["codec"], fill_value=fill,
            scale_factor=float(attrs.get("scale_factor", 1.0)),
            add_offset=float(attrs.get("add_offset", 0.0)),
            stored_bytes=arr_info["stored_bytes"], raw_bytes=int(dn.nbytes),
            pixels=int(dn.size), valid_pixels=int((dn != fill).sum()),
            chunk_bytes=arr_info["chunk_bytes"],
        )
        print(f"  {band}: {dn.shape} {dn.dtype}, stored {info[band]['stored_bytes'] / 1e6:.1f} MB "
              f"({codec_label(arr_info['codec'])}), read in {time.perf_counter() - t0:.0f} s")
    zarr.consolidate_metadata(dest)
    return info


def lossless_sizes(dn: np.ndarray, chunks: tuple[int, int]) -> dict:
    """The digital numbers under lossless JPEG XL and JPEG 2000, encoded per
    chunk of the source array (the granularity the store uses)."""
    import imagecodecs

    sizes = {"jpegxl_lossless_bytes": 0, "jpeg2000_lossless_bytes": 0}
    for r0 in range(0, dn.shape[0], chunks[0]):
        for c0 in range(0, dn.shape[1], chunks[1]):
            block = np.ascontiguousarray(dn[r0 : r0 + chunks[0], c0 : c0 + chunks[1]])
            sizes["jpegxl_lossless_bytes"] += len(imagecodecs.jpegxl_encode(block, lossless=True))
            sizes["jpeg2000_lossless_bytes"] += len(imagecodecs.jpeg2k_encode(block, level=0))
    return sizes


# --- the conversion ------------------------------------------------------------
def serializer_config(cfg: dict, band_source: dict) -> dict:
    """The serializer of the configuration, with Q and O derived from the
    band's own ``scale_factor`` and ``add_offset`` unless given."""
    codec = copy.deepcopy(cfg["codec"])
    conf = codec.setdefault("configuration", {})
    conf.setdefault("quantification_value", 1.0 / band_source["scale_factor"])
    conf.setdefault("add_offset", band_source["add_offset"] / band_source["scale_factor"])
    return codec


def make_settings(cfg: dict, codecs: list[dict] | None) -> dict:
    """The legacy-converters preset of the product, restricted to the group,
    with the resampler parameters and the codecs of the configuration."""
    from legacy_converters.settings import sentinel2 as presets

    base = {
        "sentinel-2-l1c": presets.SENTINEL2_L1C_CONVERT_SETTINGS,
        "sentinel-2-l2a": presets.SENTINEL2_L2A_CONVERT_SETTINGS,
    }[cfg["product"]]
    settings = copy.deepcopy(base)
    group = cfg["group"]
    settings["group_settings"] = {group: settings["group_settings"][group]}
    gs = settings["group_settings"][group]
    gs["resampler"] = {
        "name": "psf",
        "init_params": dict(cfg["resampler"].get("init_params", {})),
        "resample_params": dict(cfg["resampler"].get("resample_params", {})),
    }
    gs["codecs"] = codecs
    return settings


def output_extent(spec: dict | None, chunk_level: int):
    """None (whole input), one level-``chunk_level`` cell around a point, or a bbox."""
    if spec is None:
        return None
    import shapely

    if "point" in spec:
        import healpix_geo.nested as hn

        lon, lat = spec["point"]
        cell = hn.lonlat_to_healpix(np.array([lon]), np.array([lat]), chunk_level, ellipsoid="WGS84")
        clon, clat = hn.healpix_to_lonlat(cell, chunk_level, ellipsoid="WGS84")
        return shapely.Point(float(clon[0]), float(clat[0])).buffer(1e-4)
    if "bbox" in spec:
        return shapely.box(*spec["bbox"])
    raise ValueError(f"output_extent must be null, {{'point': [lon, lat]}} or {{'bbox': [...]}}, got {spec}")


def convert_worker(input_store: Path, cfg: dict, codecs: list[dict] | None, dest: Path, start: int, stop: int) -> int:
    """One batch of chunks, in its own process, so that an interrupted
    conversion resumes from the last completed batch (a whole 10 m tile takes
    about half an hour on a laptop CPU). Each batch rebuilds the staging cache
    (cheap), the first one also creates the output store.

    Prints ``N_CHUNKS=<n>`` for the driver and returns the number of chunks.
    """
    from healpix_convert import (
        convert_group_to_healpix,
        create_staging_cache,
        prepare_healpix_dataset,
    )
    from healpix_convert.healpix_converters import init_converter

    settings = make_settings(cfg, codecs)
    chunk_level = settings["group_settings"][cfg["group"]]["chunk"]["healpix"]["refinement_level"]
    group = f"/{cfg['group'].lstrip('/')}"  # the converter filters groups on DataTree paths
    cache = create_staging_cache([str(input_store)], settings, groups=[group],
                                 output_extent=output_extent(cfg.get("output_extent"), chunk_level))
    if start == 0:
        prepare_healpix_dataset(cache, settings, str(dest))
    import zarr
    from healpix_convert.settings.common import validate_convert_settings

    converter = init_converter(Path(cfg["group"]), cache=cache, settings=validate_convert_settings(settings),
                               output_store=zarr.open_group(str(dest), mode="a").store)
    n_chunks = math.ceil(converter.cell_dim_size / converter.cell_dim_chunk_size)
    print(f"N_CHUNKS={n_chunks}", flush=True)
    indices = list(range(start, min(stop, n_chunks)))
    if indices:
        convert_group_to_healpix(group, cache=cache, settings=settings, output_path=str(dest), chunk_indices=indices)
    return n_chunks


def convert(config: Path, cfg: dict, raw: bool, dest: Path, force: bool, batch: int) -> Path:
    """Drive the conversion batch by batch in fresh processes, resumably."""
    import subprocess

    import zarr

    progress_path = dest.with_name(dest.name + ".progress.json")
    progress = json.loads(progress_path.read_text()) if progress_path.exists() and not force else {}
    if progress.get("complete"):
        print(f"  {dest.name}: complete, kept (use --force to rebuild)")
        return dest
    if force and dest.exists():
        shutil.rmtree(dest)
    t0 = time.perf_counter()
    start = progress.get("next", 0)
    n_chunks = progress.get("n_chunks")
    while n_chunks is None or start < n_chunks:
        stop = start + batch if n_chunks is None else min(start + batch, n_chunks)
        cmd = [sys.executable, str(Path(__file__).resolve()), str(config), "--worker", dest.name,
               str(start), str(stop)] + (["--raw"] if raw else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout[-3000:], out.stderr[-3000:], sep="\n")
            raise RuntimeError(f"conversion batch {start}-{stop} failed (exit {out.returncode})")
        n_chunks = int(next(line for line in out.stdout.splitlines() if line.startswith("N_CHUNKS=")).split("=")[1])
        start = min(stop, n_chunks)
        progress.update(n_chunks=n_chunks, next=start, complete=start >= n_chunks)
        progress_path.write_text(json.dumps(progress))
        print(f"  {dest.name}: {start}/{n_chunks} chunks, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    zarr.consolidate_metadata(str(dest))
    return dest


# --- the sizes -------------------------------------------------------------------
def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file() and p.name != "zarr.json")


def output_arrays(store: Path) -> dict[str, Path]:
    """Path of every array of the converted store, keyed by its name in the group."""
    import zarr

    root = zarr.open_group(store, mode="r")
    return {path.rsplit("/", 1)[-1]: store / path
            for path, node in root.members(max_depth=None) if isinstance(node, zarr.Array)}


def source_codec_pipeline(codec: dict) -> dict:
    """``filters``/``serializer``/``compressors`` reproducing the codec a
    source array is stored with, for a zarr v3 array."""
    from zarr.abc.codec import ArrayArrayCodec, ArrayBytesCodec
    from zarr.codecs import BloscCodec, BytesCodec, ZstdCodec
    from zarr.registry import get_codec_class

    if codec["zarr_format"] == 2:
        comp = codec.get("compressor") or {}
        if comp.get("id") == "blosc":
            shuffle = {0: "noshuffle", 1: "shuffle", 2: "bitshuffle"}.get(int(comp.get("shuffle", 1)), "shuffle")
            compressors = [BloscCodec(cname=comp.get("cname", "zstd"), clevel=int(comp.get("clevel", 5)),
                                      shuffle=shuffle)]
        elif comp.get("id") == "zstd":
            compressors = [ZstdCodec(level=int(comp.get("level", 0)))]
        else:
            raise ValueError(f"cannot reproduce the source compressor {comp!r} in a zarr v3 array")
        return {"filters": None, "serializer": BytesCodec(), "compressors": compressors}
    codecs = [get_codec_class(c["name"]).from_dict(c) for c in codec["codecs"]]
    return {
        "filters": [c for c in codecs if isinstance(c, ArrayArrayCodec)] or None,
        "serializer": next(c for c in codecs if isinstance(c, ArrayBytesCodec)),
        "compressors": [c for c in codecs if not isinstance(c, (ArrayArrayCodec, ArrayBytesCodec))],
    }


def write_dn_reference(cells: np.ndarray, chunk: int, codec: dict, q: float, o: float, dest: Path) -> int:
    """The HEALPix band as digital numbers under the source's own codec."""
    import zarr

    from hcc.compression.sentinel2 import quantize

    if dest.exists():
        shutil.rmtree(dest)
    z = zarr.create_array(dest, shape=cells.shape, chunks=(chunk,), dtype="uint16", fill_value=0,
                          **source_codec_pipeline(codec))
    z[:] = quantize(cells, quantification_value=q, add_offset=o)
    return dir_bytes(dest)


def band_stats(values: np.ndarray, q: float, o: float) -> dict:
    """What the PSF output looks like and how much of it the serializer clips."""
    ok = np.isfinite(values)
    v = values[ok]
    floor, ceil = (1 + o) / q, (65535 + o) / q
    return {
        "cells": int(values.size), "valid_cells": int(ok.sum()),
        "min": float(v.min()) if v.size else None, "max": float(v.max()) if v.size else None,
        "pct_below_0": float(100 * (v < 0).mean()) if v.size else None,
        "pct_above_1": float(100 * (v > 1).mean()) if v.size else None,
        "n_clipped_low": int((v < floor).sum()), "n_clipped_high": int((v > ceil).sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--force", action="store_true", help="refetch the source and rebuild the stores")
    ap.add_argument("--batch", type=int, default=100, help="chunks per conversion process")
    ap.add_argument("--worker", nargs=3, metavar=("STORE", "START", "STOP"), help=argparse.SUPPRESS)
    ap.add_argument("--raw", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    import zarr

    cfg = json.loads(args.config.read_text())
    out_dir = Path(cfg["output"]["dir"])
    out_dir = out_dir if out_dir.is_absolute() else HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    group, bands = cfg["group"], list(cfg["bands"])

    if args.worker:
        store, start, stop = args.worker
        source = json.loads((out_dir / "source.json").read_text())
        codec = None if args.raw else serializer_config(cfg, source[bands[0]])
        import hcc  # noqa: F401  -- registers the serializer with zarr

        convert_worker(out_dir / "source_subset.zarr", cfg, None if codec is None else [codec],
                       out_dir / store, int(start), int(stop))
        return 0

    # 1. the source product, and a local subset with the bands under test
    if cfg["input"].get("path"):
        location = cfg["input"]["path"]
        item_id = Path(location).name.removesuffix(".zarr")
    else:
        item_path = out_dir / "stac_item.json"
        item = json.loads(item_path.read_text()) if item_path.exists() and not args.force else find_item(cfg["input"]["stac"])
        item_path.write_text(json.dumps(item, indent=1))
        location, item_id = item["assets"]["product"]["href"], item["id"]
    print(f"source {item_id}\n  {location}")

    subset = out_dir / "source_subset.zarr"
    source_path = out_dir / "source.json"
    source = json.loads(source_path.read_text()) if source_path.exists() else {}
    cached = (subset.exists() and source.get("_location") == location
              and all(b in source for b in bands) and not args.force)
    if not cached:
        print("reading the source band(s)")
        source = write_subset_store(EopfSource(location, cfg["input"].get("storage_options")), group, bands, subset)
        source["_location"] = location
        source_path.write_text(json.dumps(source, indent=1))

    # 2. the serializer, with Q and O from the band's own scaling
    codec = serializer_config(cfg, source[bands[0]])
    q, o = codec["configuration"]["quantification_value"], codec["configuration"]["add_offset"]
    import hcc  # noqa: F401  -- registers the serializer with zarr

    # 3. the conversions
    print("conversion")
    hpx = convert(args.config, cfg, False, out_dir / "healpix_codec.zarr", args.force, args.batch)
    raw = (convert(args.config, cfg, True, out_dir / "healpix_raw.zarr", args.force, args.batch)
           if cfg.get("raw_reference") else None)

    # 4. the sizes
    arrays = output_arrays(hpx)
    cell_ids_bytes = dir_bytes(arrays["cell_ids"]) if "cell_ids" in arrays else None
    report = {"source": item_id, "group": group, "codec": codec, "resampler": cfg["resampler"],
              "output_extent": cfg.get("output_extent"), "bands": {}}
    for band in bands:
        src = source[band]
        sizes_path = out_dir / f"{band}_utm_lossless_sizes.json"
        if sizes_path.exists() and not args.force:
            utm_lossless = json.loads(sizes_path.read_text())
        else:
            dn = zarr.open_array(subset / group / band, mode="r")[:]
            utm_lossless = lossless_sizes(dn, tuple(src["chunks"]))
            sizes_path.write_text(json.dumps(utm_lossless))
        cells = np.asarray(zarr.open_array(arrays[band], mode="r")[:], dtype="float64")
        chunk = zarr.open_array(arrays[band], mode="r").chunks[0]
        dn_ref_bytes = write_dn_reference(cells, chunk, src["codec"], q, o, out_dir / f"{band}_dn_source_codec.zarr")
        row = {
            "utm": {"pixels": src["pixels"], "valid_pixels": src["valid_pixels"],
                    "raw_uint16_bytes": src["raw_bytes"], "stored_bytes": src["stored_bytes"],
                    "stored_codec": codec_label(src["codec"]), "chunks": src["chunks"], **utm_lossless},
            "healpix": {"chunk_cells": int(chunk), "raw_float32_bytes": int(cells.size * 4),
                        "raw_uint16_bytes": int(cells.size * 2), f"{codec['name']}_bytes": dir_bytes(arrays[band]),
                        "dn_source_codec_bytes": dn_ref_bytes, "cell_ids_bytes": cell_ids_bytes,
                        **band_stats(cells, q, o)},
        }
        if raw is not None:
            raw_cells = np.asarray(zarr.open_array(output_arrays(raw)[band], mode="r")[:], dtype="float64")
            row["healpix_raw"] = {"bytes": dir_bytes(output_arrays(raw)[band]), **band_stats(raw_cells, q, o)}
            inside = (raw_cells >= (1 + o) / q) & (raw_cells <= (65535 + o) / q) & np.isfinite(cells)
            row["healpix_raw"]["max_abs_err_in_range"] = float(np.abs(cells - raw_cells)[inside].max()) if inside.any() else None
        u, h = row["utm"], row["healpix"]
        key = f"{codec['name']}_bytes"
        partial = cfg.get("output_extent") is not None
        # on a partial extent the UTM figures are for the whole tile, so the
        # comparison is made per sample: bits per valid cell against bits per
        # valid pixel, scaled by the geometric cell count per 10 m pixel
        cells_per_pixel = (h["valid_cells"] / max(1, u["valid_pixels"]) if not partial
                           else 100.0 / (4 * math.pi * 6371000.0**2 / (12 * 4**20)))
        bpc = 8 * h[key] / max(1, h["valid_cells"])
        bits = {k: 8 * u[f"{k}_bytes"] / max(1, u["valid_pixels"]) for k in ("stored", "jpegxl_lossless", "jpeg2000_lossless")}
        row["ratios"] = {
            "cells_per_pixel": round(cells_per_pixel, 3),
            f"{codec['name']}_bits_per_cell": round(bpc, 3),
            "utm_bits_per_pixel": {k: round(v, 3) for k, v in bits.items()},
            f"{codec['name']} / utm_stored": round(bpc * cells_per_pixel / bits["stored"], 3),
            f"{codec['name']} / utm_jpegxl_lossless": round(bpc * cells_per_pixel / bits["jpegxl_lossless"], 3),
            f"{codec['name']} / utm_jpeg2000_lossless": round(bpc * cells_per_pixel / bits["jpeg2000_lossless"], 3),
            "healpix_dn_source_codec / utm_stored": round(
                8 * h["dn_source_codec_bytes"] / max(1, h["valid_cells"]) * cells_per_pixel / bits["stored"], 3),
        }
        report["bands"][band] = row

    (out_dir / "sizes.json").write_text(json.dumps(report, indent=1))

    partial = " (partial extent: UTM figures are for the whole tile)" if cfg.get("output_extent") else ""
    mb = "{:>10.2f} MB".format
    codec_key = f"{codec['name']}_bytes"
    for band, row in report["bands"].items():
        u, h = row["utm"], row["healpix"]
        print(f"\n{group}/{band}{partial}")
        print(f"  UTM     raw uint16                  {mb(u['raw_uint16_bytes'] / 1e6)}  ({u['valid_pixels'] / 1e6:.1f} M valid px)")
        print(f"  UTM     as stored, {u['stored_codec']:<16} {mb(u['stored_bytes'] / 1e6)}")
        print(f"  UTM     JPEG XL lossless            {mb(u['jpegxl_lossless_bytes'] / 1e6)}")
        print(f"  UTM     JPEG 2000 lossless          {mb(u['jpeg2000_lossless_bytes'] / 1e6)}")
        print(f"  HEALPix raw float32                 {mb(h['raw_float32_bytes'] / 1e6)}  ({h['valid_cells'] / 1e6:.1f} M valid cells)")
        print(f"  HEALPix {codec['name']:<20}        {mb(h[codec_key] / 1e6)}")
        print(f"  HEALPix DN, source codec            {mb(h['dn_source_codec_bytes'] / 1e6)}")
        if h["cell_ids_bytes"] is not None:
            print(f"  HEALPix cell_ids                    {mb(h['cell_ids_bytes'] / 1e6)}")
        if "healpix_raw" in row:
            r = row["healpix_raw"]
            print(f"  PSF output (no codec): min {r['min']:.4f}, max {r['max']:.4f}, <0 {r['pct_below_0']:.3f}%, "
                  f">1 {r['pct_above_1']:.3f}%; the serializer clips {r['n_clipped_low']} low / {r['n_clipped_high']} "
                  f"high and errs by at most {r['max_abs_err_in_range']:.2e} in range")
        else:
            print(f"  decoded output: min {h['min']:.4f}, max {h['max']:.4f}, <0 {h['pct_below_0']:.3f}%, "
                  f">1 {h['pct_above_1']:.3f}% (set raw_reference to score the PSF before clipping)")
        for k, v in row["ratios"].items():
            print(f"  {k:<44} {v}")
    print(f"\nwritten: {out_dir / 'sizes.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
