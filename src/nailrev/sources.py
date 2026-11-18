"""Readers for every raw source archive, Han contact-sheet tile extraction and image standardisation.

Every image (file or mosaic tile) becomes one row of data/manifests/index.csv and one standardised JPEG
in data/cache/img<size>/<source>/<image_id>.jpg.  image_id is a deterministic hash of (source, ref).
"""
from __future__ import annotations

import csv
import hashlib
import io
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".gif", ".tif", ".tiff")


def make_image_id(source: str, ref: str) -> str:
    return f"{source}_{hashlib.sha1(ref.encode('utf-8')).hexdigest()[:12]}"


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any mode to RGB, compositing transparency over white."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def standardise(img: Image.Image, size: int) -> Image.Image:
    return to_rgb(img).resize((size, size), Image.BICUBIC)


def kopin_stem(fname: str) -> str:
    """Uploader's source stem before Roboflow's '.rf.<hash>' suffix."""
    return fname.split(".rf.")[0]


# ---------------------------------------------------------------------------------------------
# Han et al. 2018 contact sheets
def parse_mosaic_name(name: str, hcfg: dict) -> str | None:
    """Return the raw label of a Han mosaic PNG, or None if the mosaic must be skipped."""
    toks = Path(name).stem.split("#")
    if len(toks) < 5:
        return None
    batch, label, sub = toks[3], toks[4].strip(), [t.strip() for t in toks[5:]]
    if any(w in batch for w in hcfg["skip_batch_words"]):
        return None
    if sub and not all(s in hcfg["allow_subtags"] for s in sub):
        return None
    return label


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(np.diff(np.concatenate([[0], mask_1d.astype(np.int8), [0]])))
    return [(int(a), int(b) - 1) for a, b in zip(idx[::2], idx[1::2])]


def _merge(runs: list[tuple[int, int]], span: int) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in runs:
        if out and b - out[-1][0] + 1 <= span:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [tuple(r) for r in out]


def extract_tiles(img: Image.Image, hcfg: dict, min_side: int) -> list[tuple[int, int, int, int]]:
    """Bounding boxes (x0, y0, x1, y1), exclusive ends, of thumbnails in a contact sheet."""
    a = np.asarray(img.convert("RGBA"))
    fg = (a[..., 3] > 0) & (a[..., :3].min(-1) < hcfg["white_level"])
    boxes = []
    for y0, y1 in _merge(_runs(fg.any(1)), hcfg["merge_span"]):
        band = fg[y0:y1 + 1]
        for x0, x1 in _merge(_runs(band.any(0)), hcfg["merge_span"]):
            cell = band[:, x0:x1 + 1]
            ys, xs = np.flatnonzero(cell.any(1)), np.flatnonzero(cell.any(0))
            bx0, bx1, by0, by1 = x0 + xs[0], x0 + xs[-1] + 1, y0 + ys[0], y0 + ys[-1] + 1
            if min(bx1 - bx0, by1 - by0) >= min_side:
                boxes.append((int(bx0), int(by0), int(bx1), int(by1)))
    return boxes


# ---------------------------------------------------------------------------------------------
# Work units: each unit is processed by one worker process and returns index rows
def list_units(src: dict, raw_root: Path, hcfg: dict, chunk: int = 400, limit: int = 0) -> list[dict]:
    """Work units for one source. limit > 0 keeps a reproducible random subset (smoke tests only)."""
    units = _list_units(src, raw_root, hcfg, chunk if not limit else 10**9)
    if not limit:
        return units
    rng = np.random.default_rng(0)
    if src["kind"] == "han_mosaic":  # the smallest mosaic of each label keeps smoke tests fast
        with zipfile.ZipFile(raw_root / src["folder"] / src["file"]) as z:
            size = {i.filename: i.file_size for i in z.infolist()}
        best = {}
        for u in units:
            n, lab = u["entries"][0]
            if lab not in best or size[n] < size[best[lab]["entries"][0][0]]:
                best[lab] = u
        return list(best.values())
    entries = units[0]["entries"] if units else []
    keep = sorted(rng.choice(len(entries), min(limit, len(entries)), replace=False)) if entries else []
    return [{**units[0], "entries": [entries[i] for i in keep]}] if entries else []


def _list_units(src: dict, raw_root: Path, hcfg: dict, chunk: int) -> list[dict]:
    zpath = raw_root / src["folder"] / src["file"]
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if not n.endswith("/")]
    labels = set(src.get("label_map", {})) | set(src.get("probes", []))
    units: list[dict] = []
    if src["kind"] == "han_mosaic":
        for n in names:
            if n.lower().endswith(".png"):
                lab = parse_mosaic_name(n, hcfg)
                if lab in labels:
                    units.append({"src": src, "zip": str(zpath), "entries": [(n, lab)]})
        return units
    if src["kind"] == "csv":
        with zipfile.ZipFile(zpath) as z:
            rows = list(csv.DictReader(io.TextIOWrapper(z.open(src["csv_name"]), encoding="utf-8")))
        entries = [(src.get("image_prefix", "") + r[src["csv_image_col"]], r[src["csv_label_col"]]) for r in rows]
    else:  # folder
        inc = re.compile(src["include_regex"]) if src.get("include_regex") else None
        entries = []
        for n in names:
            if not n.lower().endswith(IMG_EXT) or (inc and not inc.search("/" + n)):
                continue
            parts = n.split("/")
            if len(parts) >= 2:
                entries.append((n, parts[-2]))
    entries = [e for e in entries if e[1] in labels]
    for i in range(0, len(entries), chunk):
        units.append({"src": src, "zip": str(zpath), "entries": entries[i:i + chunk]})
    return units


def _row(src: dict, ref: str, raw_label: str, img: Image.Image, md5: str, out_dir: Path, size: int,
         quality: int, vendor_split: str = "", stem: str = "") -> dict:
    image_id = make_image_id(src["id"], ref)
    rel = Path(src["id"]) / f"{image_id}.jpg"
    out = out_dir / rel
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        standardise(img, size).save(out, quality=quality)
    lm, probes = src.get("label_map", {}), src.get("probes", [])
    return {"image_id": image_id, "source": src["id"], "role": src["role"], "ref": ref, "raw_label": raw_label,
            "label": lm.get(raw_label, ""), "is_probe": int(raw_label in probes and raw_label not in lm),
            "vendor_split": vendor_split, "stem": stem, "width": img.size[0], "height": img.size[1],
            "md5": md5, "cache_path": rel.as_posix()}


def process_unit(unit: dict, out_dir: str, size: int, quality: int, min_side: int, hcfg: dict) -> list[dict]:
    src, out_dir = unit["src"], Path(out_dir)
    rows: list[dict] = []
    with zipfile.ZipFile(unit["zip"]) as z:
        for name, raw_label in unit["entries"]:
            try:
                data = z.read(name)
            except KeyError:
                continue
            try:
                img = Image.open(io.BytesIO(data))
                img.load()
            except Exception:
                continue
            if src["kind"] == "han_mosaic":
                rgba = img.convert("RGBA")
                for (x0, y0, x1, y1) in extract_tiles(rgba, hcfg, min_side):
                    tile = to_rgb(rgba.crop((x0, y0, x1, y1)))
                    md5 = hashlib.md5(tile.tobytes()).hexdigest()
                    rows.append(_row(src, f"{name}#{y0}_{x0}", raw_label, tile, md5, out_dir, size, quality))
                continue
            if min(img.size) < min_side:
                continue
            md5 = hashlib.md5(data).hexdigest()
            parts = name.split("/")
            vendor = parts[0] if src.get("vendor_split_from_path") else ""
            stem = kopin_stem(parts[-1]) if src["id"] == "kopin" else ""
            rows.append(_row(src, name, raw_label, img, md5, out_dir, size, quality, vendor, stem))
    return rows
