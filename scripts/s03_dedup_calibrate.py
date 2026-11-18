"""Stage 03 - calibrate the SSCD link threshold and prepare an (optional) manual audit of borderline pairs.

Positives : synthetic Roboflow-style copies (90 deg + +/-15 deg rotation + +/-10 deg shear) of ko-pin images,
            and real pairs of files sharing a ko-pin (class, stem) group.
Negatives : random ko-pin pairs with different classes and different stems.
Threshold : the (1 - max_fpr) quantile of negative similarities, clipped to sscd_link_bounds.

Outputs data/dedup/{threshold.json, calibration_scores.parquet, audit_pairs.csv, audit_montages/*.png}.
Fill the 'is_duplicate' column of audit_pairs.csv with 1/0 (optional); stage 04 then honours your labels.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from nailrev.config import load_config, write_meta  # noqa: E402
from nailrev.dedup import embed_single, load_sscd, pair_scores, phash_bits, roboflow_like  # noqa: E402


def phash_identity(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    return phash_bits(g)


def montage(rows: pd.DataFrame, img_dir: Path, paths: dict, out: Path, tile: int = 128):
    canvas = Image.new("RGB", (2 * tile + 260, tile * len(rows)), "white")
    d = ImageDraw.Draw(canvas)
    for k, r in enumerate(rows.itertuples()):
        for c, iid in enumerate((r.a, r.b)):
            canvas.paste(Image.open(img_dir / paths[iid]).resize((tile, tile)), (c * tile, k * tile))
        d.text((2 * tile + 8, k * tile + 8), f"pair {r.pair_id}\nsscd={r.sscd:.3f}\nham={r.ham}\n{r.a}\n{r.b}", fill="black")
    canvas.save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = load_config()
    dc, seed = cfg.dedup, cfg.seed
    rng = np.random.default_rng(seed)
    device = torch.device(args.device)
    idx = pd.read_csv(cfg.path("manifests") / "index.csv", keep_default_na=False)
    emb_dir, out = cfg.path("cache") / "emb", cfg.path("dedup")
    img_dir = cfg.path("cache") / f"img{cfg.image['cache_size']}"
    emb, ph = np.load(emb_dir / "sscd.npy"), np.load(emb_dir / "phash.npy")
    kop = idx[idx.source == "kopin"]
    kpos = kop.index.to_numpy()

    # --- synthetic copies
    grp = kop.groupby(["label", "stem"]).apply(lambda g: g.index[0], include_groups=False).to_numpy()
    pick = rng.choice(grp, size=min(dc["calib_n_pos"], len(grp)), replace=False)
    model = load_sscd(dc["sscd_url"], cfg.path("cache") / "models", device)
    syn_s, syn_h = [], []
    for s in range(0, len(pick), 64):
        ids = pick[s:s + 64]
        imgs = [roboflow_like(Image.open(img_dir / idx.cache_path[i]).convert("RGB"), rng) for i in ids]
        e = embed_single(model, imgs, dc["sscd_input"], device).astype(np.float32)
        syn_s += list(np.einsum("nd,ntd->nt", e, emb[ids].astype(np.float32)).max(1))
        syn_h += [int((phash_identity(im)[None] != ph[i]).sum(1).min()) for im, i in zip(imgs, ids)]

    # --- real same-stem pairs
    same = []
    for _, g in kop.groupby(["label", "stem"]):
        if len(g) > 1:
            a = g.index.to_numpy()
            same.append(rng.choice(a, size=(min(3, len(a) - 1), 2)))
    same = np.concatenate(same)
    same = same[same[:, 0] != same[:, 1]]
    same = same[rng.choice(len(same), size=min(5000, len(same)), replace=False)]
    ss_s, ss_h = pair_scores(emb, ph, same, device)

    # --- negatives
    lab, stem = kop.label.to_numpy(), kop.stem.to_numpy()
    ij = rng.choice(len(kpos), size=(dc["calib_n_neg"] * 2, 2))
    ok = (lab[ij[:, 0]] != lab[ij[:, 1]]) & (stem[ij[:, 0]] != stem[ij[:, 1]])
    neg = kpos[ij[ok][: dc["calib_n_neg"]]]
    ng_s, ng_h = pair_scores(emb, ph, neg, device)

    t_fpr = float(np.quantile(ng_s, 1 - dc["calib_max_fpr"]))
    lo, hi = dc["sscd_link_bounds"]
    t = float(np.clip(t_fpr, lo, hi)) if dc.get("sscd_link") is None else float(dc["sscd_link"])
    syn_s, syn_h = np.array(syn_s), np.array(syn_h)
    res = {
        "sscd_threshold": round(t, 4), "sscd_threshold_unclipped": round(t_fpr, 4),
        "phash_threshold": dc["phash_link"],
        "n_synthetic": len(syn_s), "n_same_stem_pairs": len(ss_s), "n_negative_pairs": len(ng_s),
        "recall_synthetic_sscd": float((syn_s >= t).mean()),
        "recall_synthetic_phash": float((syn_h <= dc["phash_link"]).mean()),
        "recall_synthetic_either": float(((syn_s >= t) | (syn_h <= dc["phash_link"])).mean()),
        "recall_same_stem_sscd": float((ss_s >= t).mean()),
        "recall_same_stem_either": float(((ss_s >= t) | (ss_h <= dc["phash_link"])).mean()),
        "fpr_negative_sscd": float((ng_s >= t).mean()),
        "fpr_negative_phash": float((ng_h <= dc["phash_link"]).mean()),
        "fpr_negative_either": float(((ng_s >= t) | (ng_h <= dc["phash_link"])).mean()),
        "synthetic_sscd_quantiles": np.quantile(syn_s, [0.01, 0.05, 0.5]).round(4).tolist(),
        "negative_sscd_quantiles": np.quantile(ng_s, [0.5, 0.99, 0.999]).round(4).tolist(),
    }
    (out / "threshold.json").write_text(json.dumps(res, indent=2))
    pd.concat([
        pd.DataFrame({"kind": "synthetic_copy", "sscd": syn_s, "ham": syn_h}),
        pd.DataFrame({"kind": "same_stem", "sscd": ss_s, "ham": ss_h}),
        pd.DataFrame({"kind": "negative", "sscd": ng_s, "ham": ng_h}),
    ]).to_parquet(out / "calibration_scores.parquet", index=False)
    print(json.dumps(res, indent=2))

    # --- optional manual audit of borderline candidate pairs (never overwrite existing labels)
    audit_f = out / "audit_pairs.csv"
    if not audit_f.exists():
        pairs = pd.read_parquet(out / "pairs.parquet")
        edges = [t - 0.20, t - 0.10, t - 0.05, t, t + 0.05, t + 0.10, 1.01]
        pairs["band"] = pd.cut(pairs.sscd, edges, right=False)
        parts = [g.sample(min(len(g), dc["audit_per_band"]), random_state=seed)
                 for _, g in pairs.dropna(subset=["band"]).groupby("band", observed=True)]
        aud = pd.concat(parts).reset_index(drop=True).rename_axis("pair_id").reset_index()
        aud["band"] = aud.band.astype(str)
        aud["is_duplicate"] = ""
        aud.to_csv(audit_f, index=False)
        mdir = out / "audit_montages"
        mdir.mkdir(exist_ok=True)
        paths = dict(zip(idx.image_id, idx.cache_path))
        for b, g in aud.groupby("band"):
            montage(g, img_dir, paths, mdir / f"band_{b.replace(' ', '').replace(',', '_')}.png")
        print(f"audit sheet: {audit_f} ({len(aud)} pairs) + montages in {mdir}")
    write_meta(out, cfg, stage="s03_dedup_calibrate", **res)


if __name__ == "__main__":
    main()
