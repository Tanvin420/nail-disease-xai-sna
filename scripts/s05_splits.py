"""Stage 05 - group-disjoint repeated cross-validation folds, the original (leaky) protocol, Roboflow's own
split, and a leakage audit of the latter two.

  folds_r{r}.csv     comp, label, fold   (stratified K-fold over components = StratifiedGroupKFold with exact
                     stratification, since every component carries exactly one label)
                     run (r, k): test = fold k, validation = fold (k+1) mod K, training = remaining folds.
  leaky_split.csv    the submitted protocol: random image-level stratified 70/15/15 (seed 42) over all files
  vendor_split.csv   Roboflow's export folders (train/valid/test)
  leakage_audit.json share of test images/components with a near-duplicate in the training partition
  msrc_sets.csv      multi-source protocol (config `multisource`): columns run, name, set, image_id, label. Pool = sources with
                     published use. Run j holds out source group j (all components containing one of its images leave
                     the training pool); the last run holds out nothing. Sets: train (all files), val (one image per
                     component, stratified by class), test (held-out group, one image per component), ext_<source>
                     (per held-out source), and fixed clean sets ext_kopin / ext_<evaluation-only source> made of
                     components that contain no pool image.
  msrc_flow.json     images/components per class for every run and set
  xsrc_sets.csv      cross-dataset matrix (config `crosssource`): columns repeat, run, name, set, image_id, label.
                     Run j trains on dataset j alone (train/val/own-test split by component) and is evaluated on every
                     other dataset: x_<source> = one image per component of that source, with every component that touches
                     the training dataset removed.
  xsrc_flow.json     image counts per run and set
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

from nailrev.config import load_config, write_meta  # noqa: E402


def audit(split: pd.DataFrame, comp: pd.Series) -> dict:
    s = split.assign(comp=split.image_id.map(comp))
    train_c = set(s.comp[s.split == "train"])
    out = {}
    for part in ("val", "test"):
        p = s[s.split == part]
        out[part] = {"images": int(len(p)),
                     "images_with_duplicate_in_train": int(p.comp.isin(train_c).sum()),
                     "share_images": round(float(p.comp.isin(train_c).mean()), 4),
                     "components": int(p.comp.nunique()),
                     "share_components_in_train": round(float(pd.Series(p.comp.unique()).isin(train_c).mean()), 4)}
    return out


def comp_split(labels: pd.Series, frac: float, seed: int):
    """Split components (index) into (rest, held) stratified by their class; falls back to unstratified if a class is too small."""
    ids = labels.index.to_numpy()
    try:
        return train_test_split(ids, test_size=frac, stratify=labels.to_numpy(), random_state=seed)
    except ValueError:
        return train_test_split(ids, test_size=frac, random_state=seed)


def build_msrc(cfg, idx: pd.DataFrame, comp: pd.Series, split_dir: Path) -> None:
    ms, seed = cfg.multisource, cfg.seed
    rng = np.random.default_rng(seed + 7)
    idx = idx.assign(comp=idx.image_id.map(comp))
    known = {s["id"] for s in cfg.sources}
    pool_src = [s for s, lv in ms["literature"].items() if lv in ms["pool_evidence"] and s in known]
    lab = idx[idx.label != ""]
    pool_any = lab[lab.source.isin(pool_src)]
    nl = pool_any.groupby("comp").label.nunique()
    conflict = set(nl[nl > 1].index)                      # same picture, different labels in the pool: dropped
    P = pool_any[~pool_any.comp.isin(conflict)]
    comp_label = P.groupby("comp").label.first()
    pool_comps = set(pool_any.comp)                       # includes conflicts: nothing linked to the pool is "clean"

    def pick(df):                                         # one random representative image per component
        reps = df.groupby("comp", sort=True).image_id.apply(lambda s: rng.choice(sorted(s)))
        return df[df.image_id.isin(set(reps))]

    dev = pd.read_csv(split_dir / "dev_components.csv", keep_default_na=False)
    dev = dev[(dev.status == "included") & ~dev.comp.isin(pool_comps)]
    fixed = {"ext_kopin": idx.set_index("image_id", drop=False).loc[dev.rep_image_id]}
    for sid in sorted(known - set(pool_src) - {"kopin", "legacy2272"}):
        g = lab[(lab.source == sid) & ~lab.comp.isin(pool_comps)]
        n_lab = g.groupby("comp").label.nunique()
        g = g[~g.comp.isin(n_lab[n_lab > 1].index)]
        if len(g):
            fixed[f"ext_{sid}"] = pick(g)

    groups = ms["groups"]
    rows, flow = [], {"pool_sources": pool_src, "pool_images": int(len(pool_any)), "pool_components": int(len(comp_label)),
                      "conflict_components_dropped": len(conflict), "runs": {}}

    def add(run, name, df, group=""):
        rows.append(pd.DataFrame({"run": run, "name": group, "set": name, "image_id": df.image_id.to_numpy(),
                                  "label": df.label.to_numpy()}))
        flow["runs"].setdefault(str(run), {})[name] = {"images": int(len(df)), "components": int(df.comp.nunique()),
                                                       "by_class": df.label.value_counts().sort_index().to_dict()}

    for j, gname in enumerate(list(groups) + ["none"]):
        held_src = groups.get(gname, [])
        held = P[P.source.isin(held_src)]
        tr_c = comp_label.index.difference(held.comp.unique()).to_numpy()
        fit_c, val_c = comp_split(comp_label.loc[tr_c], ms["val_fraction"], seed + j)
        add(j, "train", P[P.comp.isin(fit_c)], gname)
        add(j, "val", pick(P[P.comp.isin(val_c)]), gname)
        if len(held):
            add(j, "test", pick(held), gname)
            for sid in held_src:
                add(j, f"ext_{sid}", pick(held[held.source == sid]), gname)
        for name, d in fixed.items():
            add(j, name, d, gname)
        flow["runs"][str(j)]["holdout"] = gname
    pd.concat(rows).to_csv(split_dir / "msrc_sets.csv", index=False)
    (split_dir / "msrc_flow.json").write_text(json.dumps(flow, indent=2, default=int))
    print(f"\nmulti-source protocol: pool {pool_src}\n  {flow['pool_images']} images / {flow['pool_components']} components "
          f"({flow['conflict_components_dropped']} label-conflict components dropped)")
    for r, d in flow["runs"].items():
        print(f"  run {r} (hold out {d['holdout']}): " + ", ".join(f"{k} {v['images']}" for k, v in d.items() if k != "holdout"))


def build_xsrc(cfg, idx: pd.DataFrame, comp: pd.Series, split_dir: Path) -> None:
    """Cross-dataset matrix: one training dataset per run; evaluation on every other dataset (components linked to the
    training dataset removed) and on the training dataset's own held-out components."""
    xs, seed = cfg.crosssource, cfg.seed
    idx = idx.assign(comp=idx.image_id.map(comp))
    lab = idx[idx.label != ""]
    dev = pd.read_csv(split_dir / "dev_components.csv", keep_default_na=False)
    kop_ok = set(dev.comp[dev.status == "included"])
    targets = {}
    for s in cfg.sources:
        g = lab[lab.source == s["id"]]
        g = g[g.comp.isin(kop_ok)] if s["id"] == "kopin" else g
        n_lab = g.groupby("comp").label.nunique()
        g = g[~g.comp.isin(n_lab[n_lab > 1].index)]
        if len(g):
            targets[s["id"]] = g

    def pick(df, rng):
        reps = df.groupby("comp", sort=True).image_id.apply(lambda s: rng.choice(sorted(s)))
        return df[df.image_id.isin(set(reps))]

    rows, flow = [], {}
    for j, (tname, members) in enumerate(xs["train_sets"].items()):
        S = pd.concat([targets[m] for m in members if m in targets])
        if S.label.nunique() < xs["min_classes"] or len(S) < xs["min_images"]:
            print(f"  skip training set {tname}: {len(S)} images, {S.label.nunique()} classes")
            continue
        comp_label = S.groupby("comp").label.first()
        linked = set(lab[lab.source.isin(members)].comp)          # strict: any component touching the training dataset
        rng = np.random.default_rng(seed + 100 + j)
        xsets = {f"x_{sid}": pick(g[~g.comp.isin(linked)], rng) for sid, g in targets.items()
                 if sid not in members and (~g.comp.isin(linked)).any()}
        for r in range(xs["n_repeats"]):
            tr_c, te_c = comp_split(comp_label, xs["test_fraction"], seed + 1000 * r + j)
            fit_c, val_c = comp_split(comp_label.loc[tr_c], xs["val_fraction"] / (1 - xs["test_fraction"]), seed + 1000 * r + j + 1)
            parts = {"train": S[S.comp.isin(fit_c)], "val": pick(S[S.comp.isin(val_c)], rng),
                     "test": pick(S[S.comp.isin(te_c)], rng), **xsets}
            for name, d in parts.items():
                rows.append(pd.DataFrame({"repeat": r, "run": j, "name": tname, "set": name,
                                          "image_id": d.image_id.to_numpy(), "label": d.label.to_numpy()}))
            flow[f"r{r}_f{j}_{tname}"] = {k: int(len(v)) for k, v in parts.items()}
    pd.concat(rows).to_csv(split_dir / "xsrc_sets.csv", index=False)
    (split_dir / "xsrc_flow.json").write_text(json.dumps(flow, indent=2))
    print("\ncross-dataset matrix: training datasets and evaluation sets (images)")
    for k, v in flow.items():
        if k.startswith("r0_"):
            print(f"  {k}: " + ", ".join(f"{a} {b}" for a, b in v.items()))


def main():
    cfg = load_config()
    sp = cfg.splits
    split_dir = cfg.path("splits")
    dev = pd.read_csv(split_dir / "dev_components.csv", keep_default_na=False)
    inc = dev[dev.status == "included"].reset_index(drop=True)
    for r in range(sp["n_repeats"]):
        skf = StratifiedKFold(n_splits=sp["n_folds"], shuffle=True, random_state=cfg.seed + r)
        fold = pd.Series(-1, index=inc.index)
        for k, (_, te) in enumerate(skf.split(inc.comp, inc.label)):
            fold.iloc[te] = k
        f = inc[["comp", "label"]].assign(fold=fold.values)
        f.to_csv(split_dir / f"folds_r{r}.csv", index=False)
        tab = f.groupby(["fold", "label"]).size().unstack(fill_value=0)
        assert (tab > 0).all().all(), f"repeat {r}: a class is missing from some fold\n{tab}"
        print(f"repeat {r}: components per fold x class\n{tab}\n")

    idx = pd.read_csv(cfg.path("manifests") / "index.csv", keep_default_na=False)
    comp = pd.read_csv(cfg.path("dedup") / "components.csv").set_index("image_id").comp
    kop = idx[idx.source == "kopin"][["image_id", "label", "vendor_split"]]
    a, b, c = sp["leaky_fractions"]
    tr, rest = train_test_split(kop, train_size=a, stratify=kop.label, random_state=sp["leaky_seed"])
    va, te = train_test_split(rest, test_size=c / (b + c), stratify=rest.label, random_state=sp["leaky_seed"])
    leaky = pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])
    leaky[["image_id", "label", "split"]].to_csv(split_dir / "leaky_split.csv", index=False)
    vendor = kop.assign(split=kop.vendor_split.replace({"valid": "val"}))
    vendor[["image_id", "label", "split"]].to_csv(split_dir / "vendor_split.csv", index=False)
    res = {"leaky_image_level_701515": audit(leaky, comp), "roboflow_vendor_split": audit(vendor, comp)}
    (split_dir / "leakage_audit.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    build_msrc(cfg, idx, comp, split_dir)
    build_xsrc(cfg, idx, comp, split_dir)
    write_meta(split_dir, cfg, stage="s05_splits", n_components=len(inc))


if __name__ == "__main__":
    main()
