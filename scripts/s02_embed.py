"""Stage 02 - dihedral pHash + SSCD embeddings for every indexed image, then all-pairs candidate search.

Outputs
  data/cache/emb/{order.csv, phash.npy, sscd.npy}      (regenerable, large)
  data/dedup/pairs.parquet                              (candidate pairs: a, b, sscd, ham)

    python scripts/s02_embed.py [--device cuda] [--workers 8]
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from nailrev.config import load_config, write_meta  # noqa: E402
from nailrev.dedup import load_sscd, phash_variants, search_pairs, sscd_embed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="recompute embeddings even if cached")
    args = ap.parse_args()

    cfg = load_config()
    dc = cfg.dedup
    device = torch.device(args.device)
    idx = pd.read_csv(cfg.path("manifests") / "index.csv", keep_default_na=False)
    img_dir = cfg.path("cache") / f"img{cfg.image['cache_size']}"
    emb_dir = cfg.path("cache") / "emb"
    emb_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(img_dir / p) for p in idx.cache_path]

    order_f = emb_dir / "order.csv"
    same_order = order_f.exists() and pd.read_csv(order_f).image_id.tolist() == idx.image_id.tolist()
    if args.force or not same_order or not (emb_dir / "phash.npy").exists():
        with Pool(args.workers) as pool:
            ph = np.stack(list(tqdm(pool.imap(phash_variants, paths, chunksize=256), total=len(paths), desc="pHash")))
        np.save(emb_dir / "phash.npy", ph)
    ph = np.load(emb_dir / "phash.npy")

    if args.force or not same_order or not (emb_dir / "sscd.npy").exists():
        model = load_sscd(dc["sscd_url"], cfg.path("cache") / "models", device)
        emb = sscd_embed(model, paths, dc["sscd_input"], dc["batch_size"], device, args.workers)
        np.save(emb_dir / "sscd.npy", emb)
        idx[["image_id"]].to_csv(order_f, index=False)
    emb = np.load(emb_dir / "sscd.npy")
    assert len(emb) == len(ph) == len(idx)

    ij, sscd, ham = search_pairs(emb, ph, dc["topk"], dc["candidate_sscd"], dc["candidate_phash"], device)
    ids = idx.image_id.to_numpy()
    pairs = pd.DataFrame({"a": ids[ij[:, 0]], "b": ids[ij[:, 1]], "sscd": sscd.round(4), "ham": ham})
    out = cfg.path("dedup")
    pairs.to_parquet(out / "pairs.parquet", index=False)
    print(f"{len(pairs):,} candidate pairs  (sscd>={dc['candidate_sscd']}: {(pairs.sscd >= dc['candidate_sscd']).sum():,};"
          f" ham<={dc['candidate_phash']}: {(pairs.ham <= dc['candidate_phash']).sum():,})")
    write_meta(out, cfg, stage="s02_embed", n_images=len(idx), n_pairs=len(pairs))


if __name__ == "__main__":
    main()
