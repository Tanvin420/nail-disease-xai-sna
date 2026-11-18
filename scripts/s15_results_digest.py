"""Stage 15 - results/RESULTS_GUIDE.md: a self-contained explanation of everything the pipeline produced.

Headline numbers are read from results/metrics, results/sna, results/xai and results/bench (never typed by hand); missing
inputs are skipped. The guide is meant to travel with the results (Google Drive bundle) so the paper can be written away
from the workstation.   python scripts/s15_results_digest.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nailrev.config import load_config  # noqa: E402



def _xsrc_restricted(x, order, cols):
    """Same matrix with the model restricted to the classes present in the test dataset (recall, mean over classes)."""
    if "restricted_bal_acc" not in x:
        return ""
    g = x.groupby(["train_set", "target"]).restricted_bal_acc.mean().unstack().round(2).loc[order, cols]
    g = g.reset_index().rename(columns={"train_set": "train \\ test"})
    own = x[x.target == "own_test"].restricted_bal_acc.mean()
    cross = x[x.target != "own_test"].restricted_bal_acc.mean()
    return (f"\n\n**Restricted to the classes present in each test dataset** (the model chooses only among those classes; mean recall over "
            f"them; mean own-test {own:.3f}, mean cross-dataset {cross:.3f}). Accuracy and macro-F1 of the same evaluation are in "
            "`xsrc_matrix.csv` (`restricted_acc`, `restricted_macro_f1`).\n\n" + md(g))


def md(df: pd.DataFrame) -> str:
    if df is None or not len(df):
        return "_not available_"
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    rows += ["| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def ci(r, k):
    return f"{r[k]:.3f} [{r[k + '_lo']:.3f}, {r[k + '_hi']:.3f}]"


def section(title: str, fn, out: list):
    out.append(f"\n## {title}\n")
    try:
        out.append(fn())
    except Exception as e:  # a missing input must never stop the guide
        out.append(f"_not available ({type(e).__name__}: {e})_")


STATIC_FILES = """
| Where | What it is | Reviewer point |
|---|---|---|
| `results/macros.tex`, `results/tables/tab_*.tex` | Every number and table of the manuscript (LaTeX); the manuscript holds them as plain text (see `manuscript_number_provenance.csv`) | 5 |
| `results/metrics/internal_summary.csv`, `internal_per_class.csv` | Group-disjoint CV metrics: mean ± SD over 15 runs, 95% bootstrap CI, per-class recall | 1, 4 |
| `results/metrics/leakage_table.csv` | Same models under the submitted image-level split (`leaky`), Roboflow split (`vendor`) and group-disjoint CV (`cv`) | 1 |
| `results/metrics/external_summary.csv`, `external_per_class.csv` | De-duplicated per-source external testing (ko-pin-trained models), CIs, out-of-label rate | 3 |
| `results/metrics/full_source_audit.csv` | ko-pin-trained models on EVERY image of every other dataset, split into duplicates of ko-pin / independent | 3 |
| `results/metrics/xsrc_matrix.csv` | Cross-dataset matrix: train on one dataset, test on every other (one row per model, train set, target) | 3 |
| `results/metrics/runs_all_sets.csv` | Per-run metrics for every set of every protocol (msrc results live here: `holdout`, `set` = test / ext_kopin) | 3 |
| `results/metrics/pairwise_tests.csv` | Paired model comparisons (McNemar + Holm, corrected t-test, bootstrap ΔF1, equivalence) | 4, 5 |
| `results/metrics/epochs_and_temperature.csv`, `reliability_bins.csv`, `risk_coverage.csv` | Calibration (temperature per run), reliability, selective prediction | 9 |
| `results/metrics/probes.csv`, `melanoma_subgroup.csv`, `legacy_external_audit.csv` | Out-of-label probes (e.g. psoriasis, melanoma), melanoma-sourced black-line recall, audit of the submitted external set | 3, 9 |
| `results/sna/*` | Confusion networks: `node_metrics`, `edges`, `graph_summary` (3 edge filters), bootstrap stability, `similarity_*.csv` (+ `_sd`), `mst_support`, `graph_summary_compare` (leaky vs CV vs domain shift) | 7 |
| `results/xai/*` | `selection.csv` (which images), `faithfulness.csv` (deletion/insertion AUC vs random/centre), `sanity.csv` (parameter randomisation), `summary.csv`, `gallery.csv` | 6 |
| `results/bench/*` | Controlled GPU latency/memory, ONNX CPU, phone results | 8 |
| `results/predictions/<protocol>/<model>/<run>/*.parquet` | Raw logits of every run on every evaluation set (float16) – everything above can be recomputed from these | 4 |
| `figures/main`, `figures/si` | Paper figures (PDF + 300-dpi PNG + 600-dpi TIFF) | 10 |
| `figures/gallery/index.html` | XAI gallery: stand-alone panels (original / Grad-CAM / IG / LIME) per model, class and outcome – open in a browser and pick | 6 |
| `data/splits`, `data/dedup` | Splits, near-duplicate components, flow counts, calibration, audit labels | 1, 2 |
| `kaggle_release/` | Package of the fixed ko-pin dataset (manifest, README, sources, licences) | 2 |
| `runs/<protocol>/<model>/<run>/best.pt` | Every trained model (kept on the workstation); use `python scripts/infer.py --list` | – |
"""

CAVEATS = """
- Duplicate detection uses SSCD >= 0.75 and pHash <= 2 (chosen from a 150-pair manual audit). Re-uploads of one picture that score
  between about 0.5 and 0.75 can stay in different components, so residual cross-fold leakage is possible: state it as a limitation.
- No patient identifiers exist; independence is at the level of source-picture components, not patients.
- The cross-dataset matrix uses one repeat per training dataset by default (`crosssource.n_repeats`); its per-cell CIs are bootstrap
  CIs over test images and do not include training-seed variance. Small training datasets (17-class: ~135 images) give weak models.
- "Published use" of a dataset is not clinical validation; only Han et al. carries dermatologist/laboratory-confirmed labels.
- Tiny classes in some sources (e.g. Beau's lines, white spot outside ko-pin) make per-class numbers noisy; read the CIs.
- Balanced accuracy / macro-F1 on external sets are computed over the classes present in that source; out-of-label predictions count as errors.
"""


def main():
    cfg = load_config()
    res = cfg.path("results")
    met, sna, xai, bench = res / "metrics", res / "sna", res / "xai", res / "bench"
    models, label = list(cfg.models), {m: cfg.models[m]["label"] for m in cfg.models}
    ev = cfg.multisource["literature"]
    out = [f"# Results guide (generated {pd.Timestamp.now():%Y-%m-%d %H:%M})\n",
           "Auto-generated from `results/` by `scripts/s15_results_digest.py`. Every number below is read from the CSV files named "
           "in the section; numbers in the manuscript are filled in from `results/macros.tex` / `results/tables/` by `tools/expand_macros.py` (plain text; list in `results/manuscript_number_provenance.csv`).\n",
           "Project context: revision of *Interpretable Nail Disease Classification via XAI and Social Network Analysis* "
           "(Healthcare Technology Letters, HTL-2026-05-0084). See `PROJECT_LOG.md` for decisions and `README.md` for the pipeline."]

    def leakage():
        t = read(met / "leakage_table.csv")
        rows = []
        for m in models:
            r = {"Model": label[m]}
            for p, name in (("leaky", "submitted image-level split"), ("vendor", "Roboflow split"), ("cv", "group-disjoint CV")):
                d = t[(t.model == m) & (t.protocol == p)]
                r[f"macro-F1, {name}"] = ci(d.iloc[0], "macro_f1") if len(d) else "n/a"
            rows.append(r)
        return ("Same architectures, three protocols (`results/metrics/leakage_table.csv`). The gap between the first and last column "
                "is the effect of data leakage.\n\n" + md(pd.DataFrame(rows)))

    def internal():
        t = read(met / "internal_summary.csv")
        t = t[(t.protocol == "cv") & (t.set == "test")]
        rows = []
        for m in models:
            d = t[t.model == m]
            if len(d):
                r = d.iloc[0]
                rows.append({"Model": label[m], "Accuracy [95% CI]": ci(r, "acc"), "Macro-F1 [95% CI]": ci(r, "macro_f1"),
                             "Balanced acc.": f"{r.bal_acc:.3f}", "MCC": f"{r.mcc:.3f}",
                             "SD over runs (macro-F1)": f"{r.macro_f1_sd:.3f}"})
        return "Group-disjoint 5-fold × 3 CV, internal test (`internal_summary.csv`).\n\n" + md(pd.DataFrame(rows))

    def external():
        t = read(met / "external_summary.csv")
        t = t[(t.protocol == "cv") & t.source.str.startswith("ext_")]
        g = t.groupby("source").agg(n=("n", "first"), evidence=("evidence", "first"), bal_acc=("bal_acc", "mean"),
                                    macro_f1=("macro_f1", "mean"), out_of_label=("out_of_label_rate", "mean"),
                                    restr_bal_acc=("restricted_bal_acc", "mean"), restr_acc=("restricted_acc", "mean"),
                                    restr_macro_f1=("restricted_macro_f1", "mean")).reset_index()
        g["source"] = g.source.str[4:]
        cols_ = ["bal_acc", "macro_f1", "out_of_label", "restr_bal_acc", "restr_acc", "restr_macro_f1"]
        g[cols_] = g[cols_].round(3)
        return ("ko-pin-trained models on de-duplicated external sources (one image per component, mean over 5 architectures; "
                "`external_summary.csv` has per-model values and CIs).\n\n" + md(g))

    def full():
        t = read(met / "full_source_audit.csv")
        t = t[t.protocol == "cv"]
        g = t.groupby(["source", "subset"]).agg(n=("n", "first"), acc=("acc_mean", "mean")).reset_index()
        p = g.pivot(index="source", columns="subset", values="acc").round(3)
        n = g.pivot(index="source", columns="subset", values="n")
        p["n_all"], p["n_dup"], p["n_indep"] = n["all"], n.get("linked_to_kopin"), n.get("not_linked")
        p = p.rename(columns={"all": "acc_all", "linked_to_kopin": "acc_dup_of_kopin", "not_linked": "acc_independent"}).reset_index()
        return ("Every image of every other dataset (duplicates kept). `dup_of_kopin` images are copies of ko-pin material and inflate "
                "accuracy; `independent` is the honest number (mean over 5 architectures).\n\n" + md(p))

    def xsrc():
        x = read(met / "xsrc_matrix.csv")
        g = x.groupby(["train_set", "target"]).bal_acc.mean().unstack().round(2)
        order = [t for t in cfg.crosssource["train_sets"] if t in g.index]
        cols = ["own_test"] + [c for c in g.columns if c != "own_test"]
        g = g.loc[order, cols].reset_index().rename(columns={"train_set": "train \\ test"})
        own = x[x.target == "own_test"].bal_acc.mean()
        cross = x[x.target != "own_test"].bal_acc.mean()
        return (f"Balanced accuracy over the classes present in each test dataset, mean of 5 architectures (`xsrc_matrix.csv`). "
                f"Mean own-test {own:.3f} vs mean cross-dataset {cross:.3f}: the drop is the domain shift. Blank = same origin as the "
                "training set. Evidence of published use per dataset: " + ", ".join(f"{k}={v}" for k, v in ev.items()) + ".\n\n" + md(g)
                + _xsrc_restricted(x, order, cols))

    def msrc():
        r = read(met / "runs_all_sets.csv")
        r = r[r.protocol == "msrc"]
        rows = []
        for hold, g in r.groupby("holdout", sort=False):
            for s in ("test", "ext_kopin"):
                d = g[g.set == s]
                if len(d):
                    rows.append({"held-out group": hold, "set": s, "n": int(d.n.iloc[0]), "bal_acc (mean of models)": round(d.bal_acc.mean(), 3),
                                 "macro_f1": round(d.macro_f1.mean(), 3)})
        return ("Pooled training on datasets with published use, leave-one-source-group-out (`runs_all_sets.csv`, protocol msrc). "
                "`ext_kopin` = ko-pin images with no link to any pooled dataset (never trained on).\n\n" + md(pd.DataFrame(rows)))

    def sna_s():
        g = read(sna / "graph_summary.csv")
        tmax = max(cfg.analysis["sna"]["edge_thresholds"])
        g = g[g["filter"] == f"w>={tmax:g}"]
        t = g[["model", "n_edges", "top_sink", "top_sink_bootstrap_freq", "top_sink_raw_counts", "tau_counts_vs_rates",
               "kendall_tau_median", "ari_median"]].round(3)
        c = read(sna / "graph_summary_compare.csv")
        txt = f"Confusion networks, edges = row-normalised confusion rate >= {tmax:g} (`graph_summary.csv`).\n\n" + md(t)
        if c is not None:
            cc = c[c["filter"] == f"w>={tmax:g}"].groupby(["protocol", "set"]).agg(
                accuracy=("accuracy", "mean"), edges=("n_edges", "mean"), top_sink=("top_sink", lambda s: s.mode().iloc[0])).round(3).reset_index()
            txt += "\n\nTopology under the leaky split, Roboflow split, group-disjoint CV and domain shift (`graph_summary_compare.csv`):\n\n" + md(cc)
        return txt

    def xai_s():
        s = read(xai / "summary.csv")
        sa = read(xai / "sanity.csv")
        txt = "Deletion/insertion AUC (lower deletion / higher insertion = more faithful) vs random and centre baselines (`summary.csv`):\n\n"
        cols = [c for c in s.columns if c.endswith("_mean") and any(k in c for k in ("gradcam", "ig", "lime", "random", "centre"))]
        txt += md(s[["model", "correct"] + cols].round(3))
        if sa is not None:
            txt += "\n\nParameter-randomisation sanity check, mean Spearman between original and randomised-model maps (lower = passes):\n\n"
            txt += md(sa.groupby("model")[["gradcam_spearman", "ig_spearman"]].mean().round(3).reset_index())
        txt += "\n\nGallery for choosing paper figures: `figures/gallery/index.html` (see `results/xai/gallery.csv`)."
        return txt

    def bench_s():
        fs = sorted(p.name for p in bench.glob("*.csv")) if bench.exists() else []
        return "Files: " + (", ".join(f"`{f}`" for f in fs) if fs else "none") + ". Protocol details are in the `_meta_*.json` and the paper's Methods."

    for title, fn in (("1. Leakage effect (reviewer point 1)", leakage), ("2. Internal performance, group-disjoint CV (points 1, 4)", internal),
                      ("3. External testing per source (point 3)", external), ("4. All images of all datasets, duplicates flagged (point 3)", full),
                      ("5. Cross-dataset matrix: train on each dataset, test on every other (point 3)", xsrc),
                      ("6. Pooled multi-source training (point 3)", msrc), ("7. Confusion-network (SNA) results (point 7)", sna_s),
                      ("8. Explanation (XAI) results (point 6)", xai_s), ("9. Benchmarks (point 8)", bench_s)):
        section(title, fn, out)
    out += ["\n## What every file is\n", STATIC_FILES, "\n## Caveats to state in the paper\n", CAVEATS,
            "\n## Using the saved models\n",
            "```bash\npython scripts/infer.py --list\npython scripts/infer.py --model efficientnet_b2 --images photo.jpg\n"
            "python scripts/infer.py --model efficientnet_b2 --ensemble --images folder/ --csv out.csv\n```"]
    (res / "RESULTS_GUIDE.md").write_text("\n".join(out), encoding="utf-8")
    print("written", res / "RESULTS_GUIDE.md")


if __name__ == "__main__":
    main()
