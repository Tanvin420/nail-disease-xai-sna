"""Stage 07 - metrics, uncertainty, paired comparisons, external testing, probes, calibration, ablations.

Reads results/predictions/<protocol>/<model>/r<r>_f<k>/*.parquet and writes CSV/JSON to results/metrics/.
Temperature scaling is fitted per run on that run's validation predictions only.

Result sets: test / test_degraded (internal), ext_<source> (de-duplicated per-source external), full_<source> (every image of the
source, duplicates kept; reported for all / linked-to-ko-pin / not-linked), x_<source> (cross-dataset matrix, protocol xsrc).

Uncertainty layers
  (1) mean +/- SD over runs (15 for cv);
  (2) 95% stratified bootstrap CI (B resamples of test components/images) of the run-averaged metric;
  (3) per-class recall CIs from the same bootstrap.
"""
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nailrev.config import load_config, write_meta  # noqa: E402
from nailrev.metrics import (auroc_ovr, boot_ci, brier, cm_metrics, confusion, corrected_ttest, ece,  # noqa: E402
                             fit_temperature, holm, mcnemar, nll, reliability, risk_coverage, softmax,
                             strat_boot_indices)

MAIN = ["acc", "bal_acc", "macro_f1", "mcc"]


def read_set(run_dir: Path, name: str):
    f = run_dir / f"{name}.parquet"
    if not f.exists():
        return None
    d = pd.read_parquet(f).sort_values("image_id").reset_index(drop=True)
    return d.image_id.to_numpy(), d.y.to_numpy(), d.filter(like="logit_").to_numpy(np.float32)


def load_protocol(root: Path):
    """model -> run -> {'T': float, 'meta': dict, 'sets': {name: (ids, y, logits)}}"""
    out = {}
    for mdir in sorted(p for p in root.iterdir() if p.is_dir()):
        runs = {}
        for rdir in sorted(p for p in mdir.iterdir() if (p / "meta.json").exists()):
            sets = {f.stem: read_set(rdir, f.stem) for f in rdir.glob("*.parquet")}
            ids, yv, lv = sets["val"]
            runs[rdir.name] = {"T": fit_temperature(lv, yv), "meta": json.loads((rdir / "meta.json").read_text()),
                               "sets": sets}
        if runs:
            out[mdir.name] = runs
    return out


def summarise(y, preds: list[np.ndarray], K: int, B: int, rng) -> dict:
    """Point mean/SD over prediction sets aligned with y, and bootstrap CI of the run-averaged metric."""
    pts = [cm_metrics(confusion(y, p, K)) for p in preds]
    res = {}
    for m in MAIN:
        v = np.array([d[m] for d in pts], float)
        res[m], res[m + "_sd"] = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
    rec = np.array([d["recall"] for d in pts])
    bm, br = {m: [] for m in MAIN}, []
    for ix in strat_boot_indices(y, B, rng):
        ds = [cm_metrics(confusion(y[ix], p[ix], K)) for p in preds]
        for m in MAIN:
            bm[m].append(np.mean([d[m] for d in ds]))
        br.append(np.nanmean([d["recall"] for d in ds], 0))
    for m in MAIN:
        res[m + "_lo"], res[m + "_hi"] = boot_ci(bm[m])
    br = np.array(br)
    res["recall_mean"] = np.nanmean(rec, 0)
    res["recall_sd"] = np.nanstd(rec, 0, ddof=1) if len(rec) > 1 else np.zeros(K)
    res["recall_lo"] = np.nanquantile(br, 0.025, 0) if len(br) else np.full(K, np.nan)
    res["recall_hi"] = np.nanquantile(br, 0.975, 0) if len(br) else np.full(K, np.nan)
    res["support"] = pts[0]["support"]
    res["f1_mean"] = np.mean([d["f1"] for d in pts], 0)
    res["precision_mean"] = np.mean([d["precision"] for d in pts], 0)
    return res


def pooled_by_repeat(runs: dict, set_name: str):
    """cv: concatenate the 5 test folds of each repeat -> {repeat: (ids, y, logits, T per row)}"""
    rep = {}
    for rn, r in runs.items():
        s = r["sets"].get(set_name)
        if s is None:
            continue
        rep.setdefault(rn.split("_")[0], []).append((*s, np.full(len(s[0]), r["T"])))
    out = {}
    for k, parts in rep.items():
        ids, y, lg, T = (np.concatenate(z) for z in zip(*parts))
        o = np.argsort(ids, kind="stable")
        out[k] = (ids[o], y[o], lg[o], T[o])
    return out


def main():
    cfg = load_config()
    classes, K = cfg.classes, len(cfg.classes)
    B = cfg.analysis["bootstrap"]
    rng = np.random.default_rng(cfg.seed)
    pred_root, out = cfg.path("results") / "predictions", cfg.path("results") / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    split_dir = cfg.path("splits")
    protocols = {p.name: load_protocol(p) for p in sorted(pred_root.iterdir()) if p.is_dir()}
    print("protocols:", {k: list(v) for k, v in protocols.items()})
    ext_meta = {}
    for src in cfg.sources:
        f = split_dir / f"external_{src['id']}.csv"
        if src["role"] == "external" and f.exists():
            ext_meta[f"ext_{src['id']}"] = pd.read_csv(f, keep_default_na=False).set_index("image_id")
    legacy = pd.read_csv(split_dir / "legacy2272.csv", keep_default_na=False).set_index("image_id") \
        if (split_dir / "legacy2272.csv").exists() else None
    full_meta = {f"full_{s['id']}": pd.read_csv(split_dir / f"full_{s['id']}.csv", keep_default_na=False).set_index("image_id")
                 for s in cfg.sources if (split_dir / f"full_{s['id']}.csv").exists()}
    EXT = ("ext_", "x_", "full_")                       # sets treated as external to the training data
    evidence = cfg.multisource["literature"]            # published-use level per source (research_notes/dataset_usage_in_literature.md)
    B_cell = min(B, cfg.analysis.get("bootstrap_cell", 500))   # bootstrap size for the many per-dataset cells
    dev = pd.read_csv(split_dir / "dev_components.csv", keep_default_na=False)
    mel_ids = set(dev.rep_image_id[dev.tags.str.contains("melanoma_sourced")])

    run_rows, int_rows, cls_rows, cal_rows, ext_rows, extcls_rows, probe_rows = [], [], [], [], [], [], []
    rel_rows, rc_rows, hist_rows, epoch_rows, mel_rows, leg_rows, full_rows, xs_rows = [], [], [], [], [], [], [], []
    for pname, models in protocols.items():
        for m, runs in models.items():
            # ------------------ per-run internal metrics + calibration
            for rn, r in runs.items():
                meta = r["meta"]
                epoch_rows.append({"protocol": pname, "model": m, "run": rn, "best_epoch": meta["best_epoch"],
                                   "epochs_run": meta["epochs_run"], "train_seconds": meta["train_seconds"],
                                   "T": r["T"], "n_train_components": meta["n_train_components"],
                                   "n_train_images": meta["n_train_images"]})
                hf = cfg.path("runs") / pname / m / rn / "history.csv"
                if hf.exists():
                    hist_rows.append(pd.read_csv(hf).assign(protocol=pname, model=m, run=rn))
                for sname, s in r["sets"].items():
                    ids, y, lg = s
                    keep = y >= 0
                    if keep.sum() == 0:
                        continue
                    yk, lk = y[keep], lg[keep]
                    p0, p1 = softmax(lk), softmax(lk, r["T"])
                    d = cm_metrics(confusion(yk, p1.argmax(1), K))
                    row = {"protocol": pname, "model": m, "run": rn, "train_set": meta.get("train_set", ""),
                           "holdout": meta.get("holdout", ""), "set": sname, "n": int(keep.sum()),
                           **{k: d[k] for k in MAIN}, "auroc": auroc_ovr(yk, p1), "T": r["T"],
                           "ece_pre": ece(p0, yk, cfg.analysis["ece_bins"]), "ece_post": ece(p1, yk, cfg.analysis["ece_bins"]),
                           "aece_post": ece(p1, yk, cfg.analysis["ece_bins"], adaptive=True),
                           "nll_pre": nll(p0, yk), "nll_post": nll(p1, yk), "brier_pre": brier(p0, yk), "brier_post": brier(p1, yk)}
                    # "restricted" = the model may only choose among the classes the evaluated set contains
                    present = np.unique(yk)
                    pr = p1.argmax(1)
                    row["out_of_label_rate"] = float((~np.isin(pr, present)).mean())
                    pres = np.full(K, -np.inf)
                    pres[present] = 0
                    rd = cm_metrics(confusion(yk, (lk + pres).argmax(1), K))
                    row["restricted_acc"], row["restricted_bal_acc"], row["restricted_macro_f1"] = rd["acc"], rd["bal_acc"], rd["macro_f1"]
                    run_rows.append(row)

            # ------------------ cross-dataset matrix: one cell per (training dataset, target dataset)
            if pname == "xsrc":
                by_train = {}
                for r in runs.values():
                    by_train.setdefault(r["meta"]["train_set"], []).append(r)
                for tname, rs in by_train.items():
                    ev_train = ";".join(sorted({evidence.get(sid, "?") for sid in cfg.crosssource["train_sets"][tname]}))
                    for en in sorted({n for r in rs for n in r["sets"] if n.startswith("x_") or n == "test"}):
                        parts = [(r["sets"][en], r["T"]) for r in rs if en in r["sets"]]
                        ys = [p[0][1] for p in parts]
                        same = all(len(a) == len(ys[0]) and (p[0][0] == parts[0][0][0]).all() for a, p in zip(ys, parts))
                        lgs = [p[0][2] for p in parts]
                        if same:
                            y, raw = ys[0], lgs
                            preds = [softmax(l, p[1]).argmax(1) for l, p in zip(lgs, parts)]
                        else:                      # own test split differs between repeats: pool the repeats
                            y, raw = np.concatenate(ys), [np.concatenate(lgs)]
                            preds = [np.concatenate([softmax(l, p[1]).argmax(1) for l, p in zip(lgs, parts)])]
                        if (y >= 0).sum() == 0:
                            continue
                        present = np.unique(y[y >= 0])
                        pres = np.full(K, -np.inf)
                        pres[present] = 0
                        sm = summarise(y, preds, K, B_cell, rng)
                        rest = [cm_metrics(confusion(y, (l + pres).argmax(1), K)) for l in raw]
                        tgt = en[2:] if en.startswith("x_") else "own_test"
                        xs_rows.append({"protocol": pname, "model": m, "train_set": tname, "target": tgt, "n": int(len(y)),
                                        "n_runs": len(parts), "classes_present": ";".join(classes[c] for c in present),
                                        "evidence_train": ev_train, "evidence_target": evidence.get(tgt, "own_test"),
                                        **{k: v for k, v in sm.items() if not isinstance(v, np.ndarray)},
                                        "out_of_label_rate": float(np.mean([(~np.isin(p, present)).mean() for p in preds])),
                                        "restricted_acc": float(np.mean([d["acc"] for d in rest])),
                                        "restricted_bal_acc": float(np.mean([d["bal_acc"] for d in rest])),
                                        "restricted_macro_f1": float(np.mean([d["macro_f1"] for d in rest]))})
                continue

            # ------------------ internal summary (pooled per repeat for cv; single run otherwise)
            for sname in ("test", "test_degraded"):
                pooled = pooled_by_repeat(runs, sname)
                if not pooled:
                    continue
                reps = list(pooled.values())
                y = reps[0][1]
                assert all((p[0] == reps[0][0]).all() for p in reps), "repeats must cover the same test images"
                preds = [softmax(p[2], p[3][:, None]).argmax(1) for p in reps]
                s = summarise(y, preds, K, B, rng)
                fold_vals = pd.DataFrame([rw for rw in run_rows if rw["protocol"] == pname and rw["model"] == m
                                          and rw["set"] == sname])
                int_rows.append({"protocol": pname, "model": m, "set": sname, "n_test": len(y), "n_repeats": len(reps),
                                 "n_runs": len(fold_vals), **{k: v for k, v in s.items() if not isinstance(v, np.ndarray)},
                                 **{f"{k}_runs_mean": fold_vals[k].mean() for k in MAIN + ["auroc"]},
                                 **{f"{k}_runs_sd": fold_vals[k].std(ddof=1) if len(fold_vals) > 1 else 0 for k in MAIN + ["auroc"]}})
                if sname == "test":
                    for c in range(K):
                        cls_rows.append({"protocol": pname, "model": m, "class": classes[c], "support": int(s["support"][c]),
                                         "recall": s["recall_mean"][c], "recall_sd": s["recall_sd"][c],
                                         "recall_lo": s["recall_lo"][c], "recall_hi": s["recall_hi"][c],
                                         "precision": s["precision_mean"][c], "f1": s["f1_mean"][c]})
                    cm = sum(confusion(y, p, K) for p in preds)
                    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / f"confusion_{pname}_{m}.csv")
                    pp = np.concatenate([softmax(p[2], p[3][:, None]) for p in reps])
                    yy = np.concatenate([p[1] for p in reps])
                    rel_rows += [{"protocol": pname, "model": m, "set": "test", **b} for b in reliability(pp, yy, cfg.analysis["ece_bins"])]
                    cov, risk, aurc, eaurc = risk_coverage(pp, yy)
                    grid = np.linspace(0.01, 1, 100)
                    rc_rows += [{"protocol": pname, "model": m, "set": "test", "coverage": g,
                                 "risk": float(np.interp(g, cov, risk)), "aurc": aurc, "eaurc": eaurc} for g in grid]
                    # melanoma-sourced black-line subgroup
                    if mel_ids:
                        bl = classes.index("black_line")
                        for tag, mask_fn in (("melanoma_sourced", lambda i: np.isin(i, list(mel_ids))),
                                             ("other", lambda i: ~np.isin(i, list(mel_ids)))):
                            rs = []
                            for p, pr in zip(reps, preds):
                                mk = (p[1] == bl) & mask_fn(p[0])
                                rs.append((int(mk.sum()), float((pr[mk] == bl).mean()) if mk.any() else np.nan))
                            mel_rows.append({"protocol": pname, "model": m, "subgroup": tag, "n": rs[0][0],
                                             "black_line_recall": float(np.nanmean([x[1] for x in rs]))})

            # ------------------ external sources and probes
            ext_names = sorted({n for r in runs.values() for n in r["sets"]
                                if n.startswith(("ext_", "full_")) or n == "legacy2272"})
            for en in ext_names:
                parts = [(r["sets"][en], r["T"]) for r in runs.values() if en in r["sets"]]
                ids, y = parts[0][0][0], parts[0][0][1]
                lab = y >= 0
                if lab.sum():
                    preds = [softmax(s[2][lab], T).argmax(1) for s, T in parts]
                    sm = summarise(y[lab], preds, K, B_cell if en.startswith("full_") else B, rng)
                    base = en.split("_", 1)[1] if en.startswith(("ext_", "full_")) else en
                    rr = pd.DataFrame([rw for rw in run_rows if rw["protocol"] == pname and rw["model"] == m and rw["set"] == en])
                    present = [classes[c] for c in np.unique(y[lab])]
                    ext_rows.append({"protocol": pname, "model": m, "source": en, "evidence": evidence.get(base, ""),
                                     "n": int(lab.sum()), "n_runs": len(parts),
                                     "classes_present": ";".join(present),
                                     **{k: v for k, v in sm.items() if not isinstance(v, np.ndarray)},
                                     **{k: rr[k].mean() for k in ("auroc", "out_of_label_rate", "restricted_acc",
                                                                  "restricted_bal_acc", "restricted_macro_f1", "ece_post", "nll_post")}})
                    pmask = np.full(K, -np.inf)
                    pmask[np.unique(y[lab])] = 0
                    rrec = np.mean([cm_metrics(confusion(y[lab], (s[2][lab].astype(np.float64) + pmask).argmax(1), K))["recall"]
                                    for s, _ in parts], 0)
                    for c in np.unique(y[lab]):
                        extcls_rows.append({"protocol": pname, "model": m, "source": en, "class": classes[c],
                                            "support": int(sm["support"][c]), "recall": sm["recall_mean"][c],
                                            "recall_sd": sm["recall_sd"][c], "recall_lo": sm["recall_lo"][c],
                                            "recall_hi": sm["recall_hi"][c],
                                            "recall_restricted": float(rrec[c])})
                    if en.startswith("full_") and en in full_meta:
                        linked = full_meta[en].loc[ids[lab], "linked_dev"].to_numpy() == 1
                        yl = y[lab]
                        for sub, mk in (("all", np.ones_like(linked)), ("linked_to_kopin", linked), ("not_linked", ~linked)):
                            accs = [float((p[mk] == yl[mk]).mean()) if mk.any() else np.nan for p in preds]
                            f1s = [cm_metrics(confusion(yl[mk], p[mk], K))["macro_f1"] if mk.any() else np.nan for p in preds]
                            full_rows.append({"protocol": pname, "model": m, "source": en[5:], "evidence": evidence.get(en[5:], ""),
                                              "subset": sub, "n": int(mk.sum()),
                                              "classes": ";".join(classes[c] for c in np.unique(yl[mk])),
                                              "acc_mean": float(np.nanmean(accs)) if mk.any() else np.nan,
                                              "acc_sd": float(np.nanstd(accs, ddof=1)) if len(accs) > 1 and mk.any() else 0.0,
                                              "macro_f1_mean": float(np.nanmean(f1s)) if mk.any() else np.nan})
                    if en == "legacy2272" and legacy is not None:
                        dup = legacy.loc[ids[lab], "linked_dev"].to_numpy() == 1
                        org = legacy.loc[ids[lab], "origin"].to_numpy() if "origin" in legacy else np.full(len(dup), "all")
                        yl = y[lab]
                        for o in ["all sources"] + sorted(set(org)):
                            om = np.ones_like(dup) if o == "all sources" else org == o
                            for sub, mk in (("all", om), ("duplicated_in_dev", om & dup), ("not_duplicated", om & ~dup)):
                                accs = [float((p[mk] == yl[mk]).mean()) if mk.any() else np.nan for p in preds]
                                f1s = [cm_metrics(confusion(yl[mk], p[mk], K))["macro_f1"] if mk.any() else np.nan for p in preds]
                                leg_rows.append({"protocol": pname, "model": m, "origin": o, "subset": sub, "n": int(mk.sum()),
                                                 "classes": ";".join(classes[c] for c in np.unique(yl[mk])),
                                                 "acc_mean": float(np.nanmean(accs)) if mk.any() else np.nan,
                                                 "acc_sd": float(np.nanstd(accs, ddof=1)) if len(accs) > 1 and mk.any() else 0.0,
                                                 "macro_f1_mean": float(np.nanmean(f1s)) if mk.any() else np.nan})
                if (~lab).sum() and en in ext_meta:
                    raw = ext_meta[en].loc[ids[~lab], "raw_label"].to_numpy()
                    probs = np.stack([softmax(s[2][~lab], T) for s, T in parts])  # runs x n x K
                    for rl in np.unique(raw):
                        mk = raw == rl
                        pr = probs[:, mk]
                        dist = np.stack([np.bincount(p.argmax(1), minlength=K) / mk.sum() for p in pr]).mean(0)
                        probe_rows.append({"protocol": pname, "model": m, "source": en, "probe": rl, "n": int(mk.sum()),
                                           **{f"pred_{c}": dist[i] for i, c in enumerate(classes)},
                                           "mean_confidence": float(pr.max(2).mean()),
                                           "share_conf_gt_0.9": float((pr.max(2) > 0.9).mean())})

    # ------------------ pairwise model comparison (cv, internal test)
    pair_rows = []
    if "cv" in protocols:
        cvm = protocols["cv"]
        pooled = {m: pooled_by_repeat(r, "test") for m, r in cvm.items()}
        names = [m for m in cfg.models if m in pooled]
        for a, b in itertools.combinations(names, 2):
            ra, rb = pooled[a], pooled[b]
            keys = sorted(set(ra) & set(rb))
            y = ra[keys[0]][1]
            pa = [softmax(ra[k][2], ra[k][3][:, None]).argmax(1) for k in keys]
            pb = [softmax(rb[k][2], rb[k][3][:, None]).argmax(1) for k in keys]
            f = lambda ps, ix: np.mean([cm_metrics(confusion(y[ix], p[ix], K))["macro_f1"] for p in ps])  # noqa: E731
            full = np.arange(len(y))
            delta = f(pa, full) - f(pb, full)
            bd = np.array([f(pa, ix) - f(pb, ix) for ix in strat_boot_indices(y, B, rng)])
            bb, cc, p_mc = mcnemar(pa[0] == y, pb[0] == y)
            ra_runs = {rn: r for rn, r in cvm[a].items()}
            rows_a = {rw["run"]: rw["macro_f1"] for rw in run_rows if rw["protocol"] == "cv" and rw["model"] == a and rw["set"] == "test"}
            rows_b = {rw["run"]: rw["macro_f1"] for rw in run_rows if rw["protocol"] == "cv" and rw["model"] == b and rw["set"] == "test"}
            common = sorted(set(rows_a) & set(rows_b))
            mt = next(iter(ra_runs.values()))["meta"]
            t, p_t = corrected_ttest(np.array([rows_a[k] - rows_b[k] for k in common]),
                                     mt["n_train_components"], mt["n_eval"].get("test", 1)) if len(common) > 1 else (np.nan, np.nan)
            lo, hi = boot_ci(bd)
            pair_rows.append({"model_a": a, "model_b": b, "delta_macro_f1": delta, "delta_lo": lo, "delta_hi": hi,
                              "p_boot": float(min(1, 2 * min((bd <= 0).mean(), (bd >= 0).mean()))),
                              "mcnemar_b": bb, "mcnemar_c": cc, "p_mcnemar": p_mc, "t_corrected": t, "p_ttest": p_t,
                              "equivalent": bool(max(abs(lo), abs(hi)) < cfg.analysis["equivalence_margin"])})
        if pair_rows:
            pr = pd.DataFrame(pair_rows)
            pr["p_mcnemar_holm"] = holm(pr.p_mcnemar.tolist())
            pr["p_ttest_holm"] = holm(pr.p_ttest.fillna(1).tolist())
            pr.to_csv(out / "pairwise_tests.csv", index=False)

    def save(rows, name):
        if rows:
            (pd.concat(rows) if isinstance(rows[0], pd.DataFrame) else pd.DataFrame(rows)).to_csv(out / name, index=False)

    save(run_rows, "runs_all_sets.csv")
    save(int_rows, "internal_summary.csv")
    save(cls_rows, "internal_per_class.csv")
    save(ext_rows, "external_summary.csv")
    save(extcls_rows, "external_per_class.csv")
    save(probe_rows, "probes.csv")
    save(rel_rows, "reliability_bins.csv")
    save(rc_rows, "risk_coverage.csv")
    save(hist_rows, "history_all.csv")
    save(epoch_rows, "epochs_and_temperature.csv")
    save(mel_rows, "melanoma_subgroup.csv")
    save(leg_rows, "legacy_external_audit.csv")
    save(full_rows, "full_source_audit.csv")
    save(xs_rows, "xsrc_matrix.csv")

    # ------------------ compact leakage table: protocol x model
    if int_rows:
        it = pd.DataFrame(int_rows)
        it = it[it.set == "test"][["protocol", "model", "n_test", "acc", "acc_sd", "acc_lo", "acc_hi",
                                   "macro_f1", "macro_f1_sd", "macro_f1_lo", "macro_f1_hi"]]
        if ext_rows:
            ex = pd.DataFrame(ext_rows)
            ex = ex[ex.source == "ext_han_A1"][["protocol", "model", "macro_f1", "bal_acc"]].rename(
                columns={"macro_f1": "hanA1_macro_f1", "bal_acc": "hanA1_bal_acc"})
            it = it.merge(ex, on=["protocol", "model"], how="left")
        it[it.protocol.isin(["leaky", "vendor", "cv"])].to_csv(out / "leakage_table.csv", index=False)
    write_meta(out, cfg, stage="s07_evaluate", protocols=list(protocols))
    print("metrics written to", out)


if __name__ == "__main__":
    main()
