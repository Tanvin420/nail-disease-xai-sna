"""Stage 09 - explanations for the repeat-0 / fold-0 model of each architecture, with quantitative evaluation.

Images: the r0/f0 test fold (identical for all architectures). Per model and class, n_correct correctly and up to
n_incorrect wrongly classified images are drawn at random (fixed seed); the explained class is the predicted class.
Methods: Grad-CAM, Integrated Gradients (black baseline, Gauss-Legendre), LIME (quickshift superpixels).
Outputs results/xai/{selection.csv, faithfulness.csv, sanity.csv, summary.csv, examples_<model>.npz}
and figures/si/xai_<model>.png, figures/main/xai_gradcam_all_models.png

Gallery (for choosing paper figures; not used in any statistic): per model, class and outcome the best-ranked test images
(original size >= gallery.min_side_px first, then highest confidence) get a stand-alone panel
original | Grad-CAM | Integrated Gradients | LIME at gallery.dpi:
  figures/gallery/<model>/<class>/<correct|incorrect>/<image_id>__pred-<class>__p<conf>.png, figures/gallery/index.html,
  results/xai/gallery.csv.   --no-gallery skips it.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from nailrev.config import load_config, write_meta  # noqa: E402
from nailrev.data import build_transform  # noqa: E402
from nailrev.models import cam_target, create_model, disable_inplace, load_checkpoint, norm_stats  # noqa: E402
from nailrev.xai import (smooth_random_map, centre_prior, deletion_insertion, gradcam_maps, ig_maps, lime_maps, map_spearman,  # noqa: E402
                         randomised_copy)

METHODS = ["gradcam", "ig", "lime"]


def load_img(path, size):
    from torchvision.io import ImageReadMode, decode_image, read_file

    return decode_image(read_file(path), mode=ImageReadMode.RGB)


def explain(model, key, x, targets, imgs_u8, mean, std, device, xc, seed):
    layer, reshape = cam_target(key, model)
    maps = {"gradcam": np.concatenate([gradcam_maps(model, layer, reshape, x[i:i + 16], targets[i:i + 16])
                                       for i in range(0, len(x), 16)])}
    base = ((torch.zeros(1, 3, 1, 1) - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1))
    base = base.expand(1, 3, x.shape[-2], x.shape[-1]).to(device)
    maps["ig"], deltas = ig_maps(model, x, targets, base, xc["ig_steps"], xc["ig_internal_batch"])
    maps["lime"] = lime_maps(model, imgs_u8, targets, mean, std, device, xc, seed)
    return maps, deltas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", default="all")
    ap.add_argument("--no-gallery", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    xc, classes, K, seed = cfg.analysis["xai"], cfg.classes, len(cfg.classes), cfg.seed
    device = torch.device(args.device)
    rng = np.random.default_rng(seed)
    out = cfg.path("results") / "xai"
    out.mkdir(parents=True, exist_ok=True)
    fig_si, fig_main = cfg.path("figures") / "si", cfg.path("figures") / "main"
    fig_gal = cfg.path("figures") / "gallery"
    fig_si.mkdir(parents=True, exist_ok=True)
    fig_main.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(cfg.path("manifests") / "index.csv", keep_default_na=False).set_index("image_id")
    img_dir = cfg.path("cache") / f"img{cfg.image['cache_size']}"
    models = list(cfg.models) if args.models == "all" else args.models.split(",")
    pred_root, run_root = cfg.path("results") / "predictions" / "cv", cfg.path("runs") / "cv"
    models = [m for m in models if (run_root / m / "r0_f0" / "best.pt").exists()]

    preds = {}
    for m in models:
        d = pd.read_parquet(pred_root / m / "r0_f0" / "test.parquet").sort_values("image_id")
        lg = d.filter(like="logit_").to_numpy().astype(np.float64)
        pr = np.exp(lg - lg.max(1, keepdims=True))
        preds[m] = d.assign(pred=lg.argmax(1), conf=(pr / pr.sum(1, keepdims=True)).max(1))[["image_id", "y", "pred", "conf"]]
    # common figure examples: per class, one image correct for most models and one wrong for most models
    common = None
    if preds:
        allp = pd.concat([p.set_index("image_id").pred.rename(m) for m, p in preds.items()], axis=1)
        y = preds[models[0]].set_index("image_id").y
        ncor = allp.eq(y, axis=0).sum(1)
        common = []
        for c in range(K):
            ids = y.index[y == c]
            if len(ids):
                best = ncor[ids].sort_values(ascending=False, kind="stable").index[0]
                common.append((c, "correct", best))
                rest = ncor[ids].drop(best)
                if len(rest) and (rest < len(models)).any():
                    common.append((c, "incorrect", rest.sort_values(kind="stable").index[0]))

    sel_rows, faith_rows, sanity_rows, gallery_rows = [], [], [], []
    for m in models:
        mc = cfg.models[m]
        model = create_model(mc, K, pretrained=False)
        load_checkpoint(model, run_root / m / "r0_f0" / "best.pt")
        model = disable_inplace(model).to(device).eval()
        mean, std = norm_stats(model)
        S = mc["size"]
        t_eval = build_transform("eval", S, mean, std)
        p = preds[m]
        picks = []
        for c in range(K):
            cor = p[(p.y == c) & (p.pred == c)].image_id.to_numpy()
            inc = p[(p.y == c) & (p.pred != c)].image_id.to_numpy()
            picks += [(i, c, 1) for i in rng.choice(cor, min(len(cor), xc["n_correct_per_class"]), replace=False)]
            picks += [(i, c, 0) for i in rng.choice(inc, min(len(inc), xc["n_incorrect_per_class"]), replace=False)]
        sel = pd.DataFrame(picks, columns=["image_id", "y", "correct"]).merge(p[["image_id", "pred"]], on="image_id")
        sel_rows.append(sel.assign(model=m))
        u8 = [load_img(str(img_dir / idx.cache_path[i]), S) for i in sel.image_id]
        x = torch.stack([t_eval(t) for t in u8]).to(device)
        imgs = [np.asarray(Image.fromarray(t.permute(1, 2, 0).numpy()).resize((S, S), Image.BICUBIC)) for t in u8]
        targets = sel.pred.tolist()
        print(f"== {m}: explaining {len(sel)} images", flush=True)
        maps, deltas = explain(model, m, x, targets, imgs, mean, std, device, xc, seed)
        rnd = rng.random((len(sel), S, S))
        rng_s = np.random.default_rng(seed + 7)          # separate stream: the image selection above must not change
        grid = {"vgg16": 14, "densenet121": 7, "efficientnet_b2": 8, "mobilenetv3": 7, "vit_b16": 14}.get(m, 7)   # Grad-CAM map size
        rnd_s = [smooth_random_map(S, grid, rng_s) for _ in range(len(sel))]
        cen = centre_prior(S)
        for i, r in enumerate(sel.itertuples()):
            row = {"model": m, "image_id": r.image_id, "class": classes[r.y], "correct": r.correct,
                   "ig_delta": deltas[i]}
            for name, mp in [*((k, maps[k][i]) for k in METHODS), ("random", rnd[i]), ("random_smooth", rnd_s[i]), ("centre", cen)]:
                d_auc, i_auc = deletion_insertion(model, x[i:i + 1], mp, int(r.pred), xc["deletion_steps"])
                row[f"{name}_deletion"], row[f"{name}_insertion"] = d_auc, i_auc
            faith_rows.append(row)
        # sanity check: correlation with maps from a model whose top stage + head are re-initialised
        rmodel = disable_inplace(randomised_copy(model, m, seed)).to(device)
        sub = sel[sel.correct == 1].groupby("y").head(xc["sanity_n_per_class"]).index.to_numpy()
        if len(sub):
            layer, reshape = cam_target(m, rmodel)
            g_r = gradcam_maps(rmodel, layer, reshape, x[sub], [targets[i] for i in sub])
            base = ((torch.zeros(1, 3, 1, 1) - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1))
            ig_r, _ = ig_maps(rmodel, x[sub], [targets[i] for i in sub], base.expand(1, 3, S, S).to(device),
                              xc["ig_steps"], xc["ig_internal_batch"])
            for k, i in enumerate(sub):
                sanity_rows.append({"model": m, "image_id": sel.image_id[i], "gradcam_spearman": map_spearman(maps["gradcam"][i], g_r[k]),
                                    "ig_spearman": map_spearman(maps["ig"][i], ig_r[k])})
        # figure examples (common images) for this model
        if common:
            ex_ids = [e[2] for e in common]
            u8e = [load_img(str(img_dir / idx.cache_path[i]), S) for i in ex_ids]
            xe = torch.stack([t_eval(t) for t in u8e]).to(device)
            ime = [np.asarray(Image.fromarray(t.permute(1, 2, 0).numpy()).resize((S, S), Image.BICUBIC)) for t in u8e]
            pe = preds[m].set_index("image_id").loc[ex_ids, "pred"].tolist()
            me, _ = explain(model, m, xe, pe, ime, mean, std, device, xc, seed)
            np.savez_compressed(out / f"examples_{m}.npz", ids=np.array(ex_ids), pred=np.array(pe),
                                **{k: v.astype(np.float16) for k, v in me.items()}, images=np.stack(ime))
            plot_model_grid(m, cfg, common, ime, me, pe, fig_si / f"xai_{m}.png")
        if not args.no_gallery:
            gal = gallery_selection(p, idx, xc["gallery"], K)
            if len(gal):
                u8g = [load_img(str(img_dir / idx.cache_path[i]), S) for i in gal.image_id]
                xg = torch.stack([t_eval(t) for t in u8g]).to(device)
                img_g = [np.asarray(Image.fromarray(t.permute(1, 2, 0).numpy()).resize((S, S), Image.BICUBIC)) for t in u8g]
                print(f"   gallery: explaining {len(gal)} images", flush=True)
                mg, _ = explain(model, m, xg, gal.pred.tolist(), img_g, mean, std, device, xc, seed)
                for i, r in enumerate(gal.itertuples()):
                    d = fig_gal / m / classes[r.y] / ("correct" if r.correct else "incorrect")
                    d.mkdir(parents=True, exist_ok=True)
                    f = d / f"{r.image_id}__pred-{classes[r.pred]}__p{r.conf:.2f}.png"
                    plot_panel(f, u8g[i].permute(1, 2, 0).numpy(), {k: mg[k][i] for k in METHODS}, cfg.display_names[classes[r.y]],
                               cfg.display_names[classes[r.pred]], r.conf, cfg.models[m]["label"], xc["gallery"]["dpi"])
                    gallery_rows.append({"model": m, "class": classes[r.y], "pred": classes[r.pred], "correct": r.correct,
                                         "confidence": r.conf, "min_side_px": r.side, "image_id": r.image_id,
                                         "file": str(f.relative_to(fig_gal))})
        del model, rmodel
        torch.cuda.empty_cache() if device.type == "cuda" else None

    if sel_rows:
        pd.concat(sel_rows).to_csv(out / "selection.csv", index=False)
        fa = pd.DataFrame(faith_rows)
        fa.to_csv(out / "faithfulness.csv", index=False)
        pd.DataFrame(sanity_rows).to_csv(out / "sanity.csv", index=False)
        cols = [c for c in fa.columns if c.endswith(("_deletion", "_insertion"))]
        summ = fa.groupby(["model", "correct"])[cols + ["ig_delta"]].agg(["mean", "std"])
        summ.columns = ["_".join(c) for c in summ.columns]
        summ.reset_index().to_csv(out / "summary.csv", index=False)
        counts = pd.concat(sel_rows).groupby(["model", "y", "correct"]).size().rename("n").reset_index()
        counts.to_csv(out / "selection_counts.csv", index=False)
        if common:
            plot_all_models(cfg, common, models, out, fig_main / "xai_gradcam_all_models.png")
    if gallery_rows:
        g = pd.DataFrame(gallery_rows)
        g.to_csv(out / "gallery.csv", index=False)
        write_gallery_index(g, fig_gal / "index.html", cfg)
        print(f"gallery: {len(g)} panels -> {fig_gal} (open index.html)")
    write_meta(out, cfg, stage="s09_xai", models=models)


def gallery_selection(p: pd.DataFrame, idx: pd.DataFrame, gc: dict, K: int) -> pd.DataFrame:
    """Per class and outcome: images whose original is >= min_side_px first (highest confidence first), then the largest rest."""
    side = np.minimum(idx.loc[p.image_id, "width"].to_numpy(), idx.loc[p.image_id, "height"].to_numpy())
    d = p.assign(side=side)
    parts = []
    for c in range(K):
        for correct, n in ((True, gc["per_class_correct"]), (False, gc["per_class_incorrect"])):
            g = d[(d.y == c) & ((d.pred == c) == correct)]
            big = g[g.side >= gc["min_side_px"]].sort_values("conf", ascending=False, kind="stable")
            rest = g.drop(big.index).sort_values("side", ascending=False, kind="stable")
            parts.append(pd.concat([big, rest]).head(n).assign(correct=int(correct)))
    return pd.concat(parts).reset_index(drop=True)


def plot_panel(path, img, maps, true_name, pred_name, conf, model_label, dpi):
    """original | Grad-CAM | Integrated Gradients | LIME; maps (S x S) are drawn over the 288-px original."""
    fig, axes = plt.subplots(1, 4, figsize=(4 * 1.7, 1.95))
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=8)
    for ax, (k, title) in zip(axes[1:], (("gradcam", "Grad-CAM"), ("ig", "Integrated Gradients"), ("lime", "LIME"))):
        mp = np.clip(maps[k] / max(float(np.percentile(maps[k], 99)), 1e-8), 0, 1)
        ax.imshow(img, extent=(0, 1, 1, 0))
        ax.imshow(mp, cmap="inferno", alpha=0.5, vmin=0, vmax=1, extent=(0, 1, 1, 0))
        ax.set_title(title, fontsize=8)
    for ax in axes:
        ax.set_xticks([]), ax.set_yticks([])
    mark = "✓" if true_name == pred_name else "✗"
    fig.suptitle(f"{model_label} - true: {true_name}, predicted: {pred_name} {mark} (p = {conf:.2f})", fontsize=8, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def write_gallery_index(g: pd.DataFrame, path: Path, cfg) -> None:
    h = ["<!doctype html><meta charset='utf-8'><title>XAI gallery</title>",
         "<style>body{font-family:sans-serif;margin:16px}img{width:520px;margin:4px;border:1px solid #ccc}"
         "h2{margin-top:28px;border-bottom:2px solid #444}h3{margin:14px 0 2px}</style>",
         "<h1>XAI gallery</h1><p>Test images of the repeat-0 / fold-0 models, best-ranked per class and outcome "
         "(original size, then confidence). Not used in any statistic. Click an image for the full-resolution file.</p>"]
    for m, gm in g.groupby("model", sort=False):
        h.append(f"<h2>{cfg.models[m]['label']}</h2>")
        for (c, ok), gc in gm.groupby(["class", "correct"], sort=False):
            h.append(f"<h3>{cfg.display_names[c]} - {'correctly classified' if ok else 'misclassified'} ({len(gc)})</h3>")
            h += [f"<a href='{f}'><img loading='lazy' src='{f}' title='{f}'></a>" for f in gc.file]
    path.write_text("\n".join(h), encoding="utf-8")


def overlay(ax, img, mp, title=None):
    """Display only: maps are scaled to their 99th percentile (IG mass concentrates on few pixels);
    the faithfulness metrics use the unclipped maps."""
    mp = np.clip(mp / max(float(np.percentile(mp, 99)), 1e-8), 0, 1)
    ax.imshow(img)
    ax.imshow(mp, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
    ax.set_xticks([]), ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=7)


def plot_model_grid(m, cfg, common, imgs, maps, preds, path):
    names = [cfg.display_names[c] for c in cfg.classes]
    fig, axes = plt.subplots(len(common), 4, figsize=(4 * 1.3, len(common) * 1.3), squeeze=False)
    for r, ((c, kind, _), img, p) in enumerate(zip(common, imgs, preds)):
        axes[r, 0].imshow(img)
        axes[r, 0].set_xticks([]), axes[r, 0].set_yticks([])
        mark = "✓" if int(p) == int(c) else "✗"
        axes[r, 0].set_ylabel(f"true: {names[c]}\npred: {names[p]} {mark}", fontsize=6)
        for j, k in enumerate(METHODS):
            overlay(axes[r, j + 1], img, maps[k][r].astype(float), {"gradcam": "Grad-CAM", "ig": "Integrated Gradients", "lime": "LIME"}[k] if r == 0 else None)
    fig.tight_layout()
    fig.suptitle(cfg.models[m]["label"], fontsize=9, y=1.0 + 0.25 / len(common))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_all_models(cfg, common, models, out, path):
    names = [cfg.display_names[c] for c in cfg.classes]
    ex = {m: np.load(out / f"examples_{m}.npz") for m in models}
    fig, axes = plt.subplots(len(common), len(models) + 1, figsize=((len(models) + 1) * 1.2, len(common) * 1.2), squeeze=False)
    for r, (c, kind, _) in enumerate(common):
        axes[r, 0].imshow(ex[models[0]]["images"][r])
        axes[r, 0].set_xticks([]), axes[r, 0].set_yticks([])
        axes[r, 0].set_ylabel(f"true: {names[c]}", fontsize=6)
        for j, m in enumerate(models):
            e = ex[m]
            overlay(axes[r, j + 1], e["images"][r], e["gradcam"][r].astype(float),
                    cfg.models[m]["label"] if r == 0 else None)
            p = int(e["pred"][r])
            mark = "✓" if p == c else "✗"
            axes[r, j + 1].set_xlabel(f"pred: {names[p]} {mark}", fontsize=5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
