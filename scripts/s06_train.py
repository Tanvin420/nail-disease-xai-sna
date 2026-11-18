"""Stage 06 - train models and save predictions for validation, test, external and probe sets.

Protocols
  cv      group-disjoint 5-fold CV x 3 repeats (main analysis)           -> results/predictions/cv/<model>/r<r>_f<k>/
  leaky   the submitted image-level 70/15/15 split (leakage quantification) -> results/predictions/leaky/<model>/r0_f0/
  vendor  Roboflow's own train/valid/test folders (optional)              -> results/predictions/vendor/<model>/r0_f0/
  msrc    pooled training on the datasets with published use, leave-one-source-group-out (+ clean ko-pin test)
                                                                          -> results/predictions/msrc/<model>/r0_f<group>/
  xsrc    cross-dataset matrix: train on each dataset alone, test on every other dataset
                                                                          -> results/predictions/xsrc/<model>/r<repeat>_f<dataset>/
Variants (cv only): --fixed-epochs 5  (epoch-count ablation, protocol name cv_fixed5)
                    --lr-mult 0.5|2   (validation-only LR sweep, protocol name cv_lr0.5 / cv_lr2)

Examples
  python scripts/s06_train.py --protocol cv                         # all models, all repeats and folds
  python scripts/s06_train.py --protocol cv --models vit_b16 --repeats 0 --folds 0
  python scripts/s06_train.py --protocol leaky
  python scripts/s06_train.py --protocol cv --repeats 0 --folds 0 --fixed-epochs 5
Runs whose predictions already exist are skipped, so the command can be re-launched after an interruption.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # must precede CUDA initialisation

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nailrev.config import env_report, load_config, write_meta  # noqa: E402
from nailrev.engine import train_run  # noqa: E402


def load_tables(cfg):
    idx = pd.read_csv(cfg.path("manifests") / "index.csv", keep_default_na=False)
    img_dir = cfg.path("cache") / f"img{cfg.image['cache_size']}"
    idx["path"] = [str(img_dir / p) for p in idx.cache_path]
    comp = pd.read_csv(cfg.path("dedup") / "components.csv")
    idx = idx.merge(comp, on="image_id")
    c2i = cfg.class_to_idx
    idx["y"] = idx.label.map(c2i).fillna(-1).astype(int)
    return idx.set_index("image_id", drop=False)


def eval_sets(cfg, idx, split_dir, test_df, protocol):
    sets = {"test": (test_df, "eval")}
    if protocol.startswith("cv"):
        sets["test_degraded"] = (test_df, "degrade")
    for src in cfg.sources:
        f = split_dir / (f"external_{src['id']}.csv" if src["role"] == "external" else f"{src['id']}.csv")
        if src["role"] in ("external", "legacy") and f.exists():
            ids = pd.read_csv(f, keep_default_na=False).image_id
            sets[f"ext_{src['id']}" if src["role"] == "external" else src["id"]] = (idx.loc[ids], "eval")
        full = split_dir / f"full_{src['id']}.csv"            # every mapped image of the source, duplicates kept (flagged)
        if full.exists():
            sets[f"full_{src['id']}"] = (idx.loc[pd.read_csv(full, keep_default_na=False).image_id], "eval")
    return sets


def run_complete(pred_dir: Path, out_dir: Path) -> bool:
    """A run counts as finished only if its files are intact: after a power cut the newest files can be empty or truncated.
    meta.json is written last; every evaluation set must be a readable parquet (magic bytes) and the checkpoint a valid archive."""
    try:
        meta = json.loads((pred_dir / "meta.json").read_text())
        for name in meta["n_eval"]:
            with open(pred_dir / f"{name}.parquet", "rb") as fh:
                head = fh.read(4)
                fh.seek(-4, 2)
                if head != b"PAR1" or fh.read(4) != b"PAR1":
                    return False
        if meta.get("checkpoint"):
            ck = out_dir / meta["checkpoint"]
            return ck.stat().st_size > 0 and zipfile.is_zipfile(ck)
        return True
    except (OSError, ValueError, KeyError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default="cv", choices=["cv", "leaky", "vendor", "msrc", "xsrc"])
    ap.add_argument("--models", default="all")
    ap.add_argument("--repeats", default="")
    ap.add_argument("--folds", default="")
    ap.add_argument("--fixed-epochs", type=int, default=0)
    ap.add_argument("--lr-mult", type=float, default=1.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    device = torch.device(args.device)
    split_dir = cfg.path("splits")
    idx = load_tables(cfg)
    models = list(cfg.models) if args.models == "all" else args.models.split(",")
    K, R = cfg.splits["n_folds"], cfg.splits["n_repeats"]
    repeats = [int(r) for r in args.repeats.split(",")] if args.repeats else list(range(R))
    folds = [int(k) for k in args.folds.split(",")] if args.folds else list(range(K))
    pname = args.protocol
    if args.fixed_epochs:
        pname += f"_fixed{args.fixed_epochs}"
    if args.lr_mult != 1.0:
        pname += f"_lr{args.lr_mult:g}"

    runs = []
    if args.protocol == "cv":
        dev = pd.read_csv(split_dir / "dev_components.csv", keep_default_na=False).set_index("comp")
        kop = idx[(idx.source == "kopin")]
        for r in repeats:
            fold = pd.read_csv(split_dir / f"folds_r{r}.csv").set_index("comp").fold
            for k in folds:
                tr_c = set(fold.index[~fold.isin([k, (k + 1) % K])])
                va_c, te_c = fold.index[fold == (k + 1) % K], fold.index[fold == k]
                train = kop[kop.comp.isin(tr_c)]
                val = idx.loc[dev.loc[va_c, "rep_image_id"]]
                test = idx.loc[dev.loc[te_c, "rep_image_id"]]
                assert not (set(train.comp) & (set(va_c) | set(te_c))), "component leakage!"
                runs.append((r, k, train, val, test, cfg.seed + 1000 * r + k, None, None))
    elif args.protocol in ("msrc", "xsrc"):
        f = split_dir / f"{args.protocol}_sets.csv"
        d = pd.read_csv(f, keep_default_na=False)
        if "repeat" not in d:
            d["repeat"] = 0
        for (r, k), g in d.groupby(["repeat", "run"], sort=True):
            if (args.repeats and r not in repeats) or (args.folds and k not in folds):
                continue
            sets = {n: idx.loc[x.image_id] for n, x in g.groupby("set")}
            empty = idx.iloc[:0]
            ev = {n: (v, "eval") for n, v in sets.items() if n not in ("train", "val")}
            ev.setdefault("test", (empty, "eval"))
            key = "holdout" if args.protocol == "msrc" else "train_set"
            runs.append((r, k, sets["train"], sets["val"], ev["test"][0], cfg.seed + 1000 * r + k, ev, {key: g.name.iloc[0]}))
    else:
        sp = pd.read_csv(split_dir / f"{args.protocol}_split.csv")
        part = {s: idx.loc[sp.image_id[sp.split == s]] for s in ("train", "val", "test")}
        runs.append((0, 0, part["train"], part["val"], part["test"], cfg.seed, None, None))

    print(f"protocol {pname}: {len(runs)} partition(s) x {len(models)} model(s) on {device}")
    res_root = cfg.path("results") / "predictions" / pname
    for m in models:
        for r, k, train, val, test, seed, ev, extra in runs:
            pred_dir = res_root / m / f"r{r}_f{k}"
            out_dir = cfg.path("runs") / pname / m / f"r{r}_f{k}"
            if not args.overwrite and run_complete(pred_dir, out_dir):
                print(f"skip {pname}/{m}/r{r}_f{k} (done)")
                continue
            if (pred_dir / "meta.json").exists() and not args.overwrite:
                print(f"redo {pname}/{m}/r{r}_f{k}: files incomplete or corrupt (interrupted write?)")
            print(f"== {pname}/{m}/r{r}_f{k}: train {len(train)} files / {train.comp.nunique()} components, "
                  f"val {len(val)}, test {len(test)}", flush=True)
            keep = cfg.train["keep_checkpoints"]
            save = keep == "all" or (keep == "r0f0" and r == 0 and k == 0 and pname == "cv")
            train_run(cfg, m, train, val, ev if ev is not None else eval_sets(cfg, idx, split_dir, test, pname),
                      out_dir, pred_dir, seed, device,
                      fixed_epochs=args.fixed_epochs or None, lr_mult=args.lr_mult, save_checkpoint=save, extra_meta=extra)
            torch.cuda.empty_cache() if device.type == "cuda" else None
    write_meta(res_root, cfg, stage=f"s06_train_{pname}", env_full=env_report())


if __name__ == "__main__":
    main()
