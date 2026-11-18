"""Stage 01 - read every source archive, cut Han mosaics into tiles, standardise all images to the cache,
and write data/manifests/index.csv (one row per image) plus a per-source summary.

    python scripts/s01_build_index.py [--workers 8] [--only kopin,han_A1]
"""
import argparse
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
from tqdm import tqdm  # noqa: E402

from nailrev.config import load_config, write_meta  # noqa: E402
from nailrev.sources import list_units, process_unit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", default="", help="comma-separated source ids (default: all)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: random subset of N images per source")
    args = ap.parse_args()

    cfg = load_config()
    raw, man = cfg.path("raw"), cfg.path("manifests")
    size, q, ms = cfg.image["cache_size"], cfg.image["jpeg_quality"], cfg.image["min_side"]
    out_dir = cfg.path("cache") / f"img{size}"
    only = set(filter(None, args.only.split(",")))

    units = []
    for src in cfg.sources:
        if only and src["id"] not in only:
            continue
        u = list_units(src, raw, cfg.han_mosaic, limit=args.limit)
        print(f"{src['id']:<16} {len(u):>5} work units")
        units += u

    work = partial(process_unit, out_dir=str(out_dir), size=size, quality=q, min_side=ms, hcfg=cfg.han_mosaic)
    rows = []
    with Pool(args.workers) as pool:
        for r in tqdm(pool.imap_unordered(work, units), total=len(units), desc="index"):
            rows += r
    df = pd.DataFrame(rows).sort_values(["source", "ref"]).reset_index(drop=True)
    idx_path = man / "index.csv"
    if only and idx_path.exists():  # merge with the existing index
        old = pd.read_csv(idx_path, keep_default_na=False)
        df = pd.concat([old[~old.source.isin(only)], df]).sort_values(["source", "ref"]).reset_index(drop=True)
    dup = df.image_id.duplicated().sum()
    assert dup == 0, f"{dup} duplicated image_ids"
    df.to_csv(idx_path, index=False)

    summ = (df.assign(kind=lambda d: d.label.where(d.label != "", "probe:" + d.raw_label))
              .groupby(["source", "role", "kind"]).size().rename("n").reset_index())
    summ.to_csv(man / "index_summary.csv", index=False)
    print(summ.to_string(index=False))
    write_meta(man, cfg, stage="s01_build_index", n_images=len(df))


if __name__ == "__main__":
    main()
