"""Stage 07b - cross-dataset results on the SHARED labels only.

For every (training dataset X, test dataset Y) pair: keep only the classes that BOTH datasets have, keep only the test images of those
classes, let the model choose only among those classes, and report accuracy, recall (balanced accuracy = mean per-class recall) and macro-F1.
Training datasets: ko-pin (the 15 group-disjoint cv runs per model -> every external source) and each dataset of the `xsrc` protocol
(train on X alone -> every other dataset). Pairs with fewer than 2 shared classes are listed but have no accuracy/F1 (not meaningful).

Outputs  results/metrics/shared_label_matrix.csv (one row per model, pair), shared_label_per_class.csv, and
         results/SHARED_LABEL_RESULTS.md (matrices, mean over models).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nailrev.config import load_config  # noqa: E402
from nailrev.metrics import cm_metrics, confusion  # noqa: E402

cfg = load_config()
classes = list(cfg.classes)
K = len(classes)
pred_root = cfg.path("results") / "predictions"
models = list(cfg.models)
MIN_N = 30                                           # below this many test images a cell is flagged as unreliable


def load(d: Path, name: str):
    f = d / f"{name}.parquet"
    if not f.exists():
        return None
    t = pd.read_parquet(f)
    t = t[t.y >= 0]
    if not len(t):
        return None
    return t.y.to_numpy(), t[[c for c in t.columns if c.startswith("logit_")]].to_numpy(np.float64)


def evaluate(y, lg, shared):
    """shared: list of class indices. Returns metrics on test images of shared classes, prediction restricted to shared."""
    keep = np.isin(y, shared)
    if keep.sum() == 0:
        return None
    mask = np.full(K, -np.inf)
    mask[shared] = 0
    m = cm_metrics(confusion(y[keep], (lg[keep] + mask).argmax(1), K))
    return {"n": int(keep.sum()), "acc": m["acc"], "recall": m["bal_acc"], "macro_f1": m["macro_f1"],
            "per_class": m["recall"]}


rows, cls_rows = [], []


def add(model, train_name, target, y, lg, train_cls, run_metrics):
    present = sorted(set(y.tolist()))
    shared = sorted(set(present) & set(train_cls))
    base = {"model": model, "train_set": train_name, "target": target,
            "train_classes": ";".join(classes[c] for c in train_cls), "target_classes": ";".join(classes[c] for c in present),
            "shared_classes": ";".join(classes[c] for c in shared), "n_shared_classes": len(shared)}
    if len(shared) < 2:
        rows.append({**base, "n": int(np.isin(y, shared).sum()), "status": "single shared class (not meaningful)"})
        return
    res = [r for r in run_metrics(shared) if r is not None]
    if not res:
        return
    n = res[0]["n"]
    rows.append({**base, "n": n, "n_runs": len(res), "acc": np.mean([r["acc"] for r in res]),
                 "recall": np.mean([r["recall"] for r in res]), "macro_f1": np.mean([r["macro_f1"] for r in res]),
                 "status": "ok" if n >= MIN_N else f"small test set (n<{MIN_N})"})
    pc = np.mean([r["per_class"] for r in res], 0)
    for c in shared:
        cls_rows.append({"model": model, "train_set": train_name, "target": target, "class": classes[c],
                         "support": int((y == c).sum()), "recall": float(pc[c])})


# ---- ko-pin (cv runs) -> every external source
for m in models:
    runs = sorted((pred_root / "cv" / m).glob("r*_f*"))
    sets = sorted({f.stem for r in runs[:1] for f in r.glob("ext_*.parquet")})
    for s in sets:
        first = load(runs[0], s)
        if first is None:
            continue
        add(m, "kopin", s[4:], first[0], first[1], list(range(K)),
            lambda shared, s=s, runs=runs: [evaluate(*load(r, s), shared) for r in runs if load(r, s) is not None])

# ---- xsrc: train on X alone -> own test and every other dataset
xs = pd.read_csv(cfg.path("splits") / "xsrc_sets.csv", keep_default_na=False)
tr_cls = {n: [classes.index(c) for c in sorted(g.label.unique()) if c in classes]
          for n, g in xs[xs.set == "train"].groupby("name")}
for m in models:
    for run_dir in sorted((pred_root / "xsrc" / m).glob("r*_f*")):
        meta = pd.read_json(run_dir / "meta.json", typ="series")
        tname = meta["train_set"]
        for f in sorted(run_dir.glob("*.parquet")):
            s = f.stem
            if s == "val" or not (s == "test" or s.startswith("x_")):
                continue
            r = load(run_dir, s)
            if r is None:
                continue
            target = "own_test" if s == "test" else s[2:]
            add(m, tname, target, r[0], r[1], tr_cls[tname], lambda shared, r=r: [evaluate(r[0], r[1], shared)])

R = pd.DataFrame(rows)
out = cfg.path("results") / "metrics"
R.to_csv(out / "shared_label_matrix.csv", index=False)
pd.DataFrame(cls_rows).to_csv(out / "shared_label_per_class.csv", index=False)

# ---- markdown report
order = [t for t in ["kopin", "han", "han_virtualE", "thai", "nikhilgurav21", "kaggle17", "josephrasanjana"] if t in set(R.train_set)]
ok = R[R.status.isin(["ok", f"small test set (n<{MIN_N})"]) & R.acc.notna()]
targets = ["own_test"] + sorted(set(R.target) - {"own_test"})


def matrix(metric):
    g = ok.groupby(["train_set", "target"])[metric].mean().unstack()
    g = g.reindex(index=order, columns=[c for c in targets if c in g.columns])
    flag = ok.groupby(["train_set", "target"]).status.first().unstack().reindex_like(g)
    cells = g.copy().astype(object)
    for i in g.index:
        for j in g.columns:
            v = g.loc[i, j]
            cells.loc[i, j] = "–" if pd.isna(v) else f"{v:.2f}" + ("*" if flag.loc[i, j] != "ok" else "")
    return cells.reset_index().rename(columns={"train_set": "trained on \\ tested on"})


def md(df):
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join(["---"] * len(df.columns)) + "|\n"
    return h + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.to_numpy())


share = R.drop_duplicates(["train_set", "target"]).pivot(index="train_set", columns="target", values="n_shared_classes")
share = share.reindex(index=order, columns=[c for c in targets if c in share.columns]).fillna(0).astype(int).astype(str).replace("0", "–")
share = share.reset_index().rename(columns={"train_set": "trained on \\ tested on"})
txt = ["# Cross-dataset results on the shared labels only\n",
       "_Generated by `scripts/s07b_shared_label_matrix.py` from `results/predictions/`. Provisional until the manuscript numbers are taken "
       "from `results/macros.tex`._\n",
       "**How each cell is computed.** Train on dataset X, test on dataset Y. Only the classes that **both** X and Y have are used: test images "
       "of other classes are dropped, and the model may choose only among the shared classes. Values are the mean over the 5 architectures "
       "(ko-pin row: each architecture first averaged over its 15 group-disjoint cv runs; other rows: one run per architecture). "
       "`own_test` = held-out part of the training dataset itself. `*` = fewer than "
       f"{MIN_N} test images, unreliable. `–` = not available, or fewer than 2 shared classes (accuracy/F1 would be meaningless).\n",
       "Note that images already linked to the training dataset (copies) have been removed from every target, and that the restricted choice is "
       "optimistic compared with letting the model choose among all 7 classes (see `results/metrics/xsrc_matrix.csv`, strict columns).\n",
       "## Number of shared classes\n", md(share),
       "\n## Recall (mean per-class recall over the shared classes)\n", md(matrix("recall")),
       "\n## Accuracy\n", md(matrix("acc")),
       "\n## Macro-F1\n", md(matrix("macro_f1"))]
txt += ["\n## How to read these tables\n",
        "- Compare cells only when the number of shared classes is the same. A 2-class task (for example healthy vs onychomycosis) is much easier than a "
        "5-class task, so a high value from a dataset with few classes does not mean a better model. For this reason no row averages are given.",
        "- Fair comparison for a given test dataset: look down one column and check the shared-class table above for equal counts.",
        "- Own-test values of 2-class datasets (nikhilgurav21, josephrasanjana: 0.98-0.99) are easy binary tasks."]
(cfg.path("results") / "SHARED_LABEL_RESULTS.md").write_text("\n".join(txt) + "\n")
print(f"{len(R)} rows; {len(ok)} with metrics; wrote shared_label_matrix.csv, shared_label_per_class.csv, SHARED_LABEL_RESULTS.md")

# ---- class-wise accuracy (per-class recall): one matrix per class, trained-on x tested-on
PC = pd.DataFrame(cls_rows)
if len(PC):
    cw = ["# Class-wise accuracy across datasets\n",
          "_Generated by `scripts/s07b_shared_label_matrix.py`. Class-wise accuracy = share of the test images of that class that the model "
          "classifies correctly (per-class recall). Rows = dataset the model was trained on, columns = dataset it was tested on. "
          "Each cell is `accuracy (number of test images of that class)`; mean over the 5 architectures (ko-pin row: each architecture first averaged "
          "over its 15 group-disjoint cv runs). `–` = the training or the test dataset does not have the class, or fewer than 2 shared classes. "
          "`*` = fewer than 30 test images, unreliable. The model chooses only among the classes both datasets share, and images "
          "linked to the training dataset are removed. Per-architecture values: `results/metrics/shared_label_per_class.csv`._\n"]
    for c in classes:
        d = PC[PC["class"] == c]
        if d.empty:
            continue
        g = d.groupby(["train_set", "target"]).agg(recall=("recall", "mean"), n=("support", "first")).reset_index()
        tg = ["own_test"] + sorted(set(g.target) - {"own_test"})
        rowsx = []
        for tr in [t for t in order if t in set(g.train_set)]:
            cells = []
            for tt in tg:
                r = g[(g.train_set == tr) & (g.target == tt)]
                cells.append("–" if r.empty else f"{r.recall.iloc[0]:.2f} ({int(r.n.iloc[0]):,})" + ("*" if r.n.iloc[0] < MIN_N else ""))
            rowsx.append([tr] + cells)
        cw += [f"\n## {c}\n", md(pd.DataFrame(rowsx, columns=["trained on \\ tested on"] + tg))]
    (cfg.path("results") / "CLASSWISE_ACCURACY.md").write_text("\n".join(cw) + "\n")
    print("wrote CLASSWISE_ACCURACY.md")
