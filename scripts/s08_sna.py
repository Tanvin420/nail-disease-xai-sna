"""Stage 08 - confusion-network analysis with bootstrap stability, model-similarity matrices and MST support.

Main analysis: cv test predictions pooled over folds within each repeat (argmax; temperature scaling does not change it).
Edge filters: none, w >= 0.01, and the disparity-filter backbone (alpha from analysis.sna.backbone_alpha, default 0.05).
Model similarity: computed per repeat on identical test images, reported as mean and SD over repeats.
Comparison: topology under the submitted (leaky) split, the Roboflow split, and under domain shift (all de-duplicated
external sources pooled), to show how data leakage / domain shift change the error network.
Outputs results/sna/*.csv
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import kendalltau  # noqa: E402

from nailrev.config import load_config, write_meta  # noqa: E402
from nailrev.metrics import confusion, strat_boot_indices  # noqa: E402
from nailrev.sna import bootstrap_stability, mst_edges, node_metrics, pair_similarity, rates  # noqa: E402

SIM_KEYS = ("cohen_kappa", "error_consistency_kappa", "jaccard_errors", "double_fault", "pearson_confusion")


def pooled_preds(pdir: Path):
    """repeat -> (ids, y, pred) with test folds concatenated and sorted by image_id"""
    out = {}
    for rd in sorted(pdir.glob("r*_f*")):
        f = rd / "test.parquet"
        if f.exists() and (rd / "meta.json").exists():          # meta.json is written last: skip runs still in progress
            d = pd.read_parquet(f)
            out.setdefault(rd.name.split("_")[0], []).append(d)
    res = {}
    for r, parts in out.items():
        d = pd.concat(parts).sort_values("image_id")
        res[r] = (d.image_id.to_numpy(), d.y.to_numpy(), d.filter(like="logit_").to_numpy().argmax(1))
    return res


def single_run_test(mdir: Path):
    """(y, [pred]) of the single test set of a one-partition protocol (leaky, vendor)"""
    f = mdir / "r0_f0" / "test.parquet"
    if not f.exists() or not (mdir / "r0_f0" / "meta.json").exists():
        return None
    d = pd.read_parquet(f).sort_values("image_id")
    return d.y.to_numpy(), [d.filter(like="logit_").to_numpy().argmax(1)]


def external_pooled(mdir: Path):
    """(y, [pred per run]) on all de-duplicated external sources pooled (labelled images only), over all runs of a model"""
    ys, preds = None, []
    for rd in sorted(mdir.glob("r*_f*")):
        if not (rd / "meta.json").exists():                       # run still in progress
            continue
        parts = []
        for f in sorted(rd.glob("ext_*.parquet")):
            d = pd.read_parquet(f).sort_values("image_id")
            parts.append(d[d.y >= 0].assign(src=f.stem))
        if not parts:
            continue
        d = pd.concat(parts)
        y = d.y.to_numpy()
        if ys is None:
            ys = y
        assert len(y) == len(ys) and (y == ys).all(), "external sets differ between runs"
        preds.append(d.filter(like="logit_").to_numpy().argmax(1))
    return (ys, preds) if preds else None


def main():
    cfg = load_config()
    sc, classes, K = cfg.analysis["sna"], cfg.classes, len(cfg.classes)
    rng = np.random.default_rng(cfg.seed)
    out = cfg.path("results") / "sna"
    out.mkdir(parents=True, exist_ok=True)
    pred_root = cfg.path("results") / "predictions"
    root = pred_root / "cv"
    models = [m for m in cfg.models if (root / m).exists()]
    alpha_bb = sc.get("backbone_alpha", 0.05)
    filters = [(f"w>{t:g}" if t == 0 else f"w>={t:g}", t, None) for t in sc["edge_thresholds"]] + \
              [(f"disparity_a{alpha_bb:g}", np.nan, alpha_bb)]
    data = {m: pooled_preds(root / m) for m in models}
    node_rows, graph_rows, stab_rows, edge_rows, rep_rows = [], [], [], [], []
    for m in models:
        reps = list(data[m].values())
        assert all((r[0] == reps[0][0]).all() for r in reps), f"{m}: repeats cover different test images"
        y = reps[0][1]
        preds = [p[2] for p in reps]
        counts = sum(confusion(y, p, K) for p in preds).astype(float)
        R = rates(counts)
        np.fill_diagonal(counts, 0)
        in_counts = counts.sum(0)                         # sensitivity: weighted in-degree from raw error counts instead of rates
        tau_counts = float(kendalltau(in_counts, R.sum(0)).statistic)
        pd.DataFrame(R, index=classes, columns=classes).to_csv(out / f"rates_{m}.csv")
        for label, thr, bb in filters:
            st = bootstrap_stability(y, preds, K, sc["bootstrap"], 0.0 if bb else thr, sc["pagerank_alpha"],
                                     sc["louvain_resolution"], cfg.seed, rng, backbone_alpha=bb)
            f = st["full"]
            sink_name = classes[int(np.argmax(f["in_strength"]))] if np.max(f["in_strength"]) > 0 else "none"   # none = graph has no edges
            graph_rows.append({"model": m, "filter": label, "threshold": thr, "n_edges": f["n_edges"], "density": f["density"],
                               "top_sink": sink_name, "top_sink_bootstrap_freq": st["full_sink_freq"],
                               "top_sink_raw_counts": classes[int(np.argmax(in_counts))], "tau_counts_vs_rates": tau_counts,
                               "n_communities": int(f["community"].max() + 1),
                               "kendall_tau_median": st["kendall_tau_median"], "kendall_tau_q25": st["kendall_tau_q25"],
                               "ari_median": st["ari_median"], "ari_q25": st["ari_q25"]})
            for i, c in enumerate(classes):
                node_rows.append({"model": m, "filter": label, "threshold": thr, "class": c, "in_strength": f["in_strength"][i],
                                  "in_lo": st["in_lo"][i], "in_hi": st["in_hi"][i], "out_strength": f["out_strength"][i],
                                  "pagerank": f["pagerank"][i], "pr_lo": st["pr_lo"][i], "pr_hi": st["pr_hi"][i],
                                  "clustering": f["clustering"][i], "community": int(f["community"][i]),
                                  "top_sink_freq": st["top_sink_freq"][i]})
            for i, j in itertools.permutations(range(K), 2):
                edge_rows.append({"model": m, "filter": label, "threshold": thr, "true": classes[i], "pred": classes[j],
                                  "rate": R[i, j], "rate_lo": st["edge_rate_lo"][i, j], "rate_hi": st["edge_rate_hi"][i, j],
                                  "presence_freq": st["edge_presence"][i, j]})
            tag = f"thr{thr:g}" if bb is None else "disparity"
            pd.DataFrame(st["coassignment"], index=classes, columns=classes).to_csv(out / f"coassignment_{m}_{tag}.csv")
        # stability across training repeats (different partitions and seeds)
        per_rep = [node_metrics(rates(confusion(p[1], p[2], K)), 0.0, sc["pagerank_alpha"], sc["louvain_resolution"], cfg.seed)
                   for p in reps]
        for a, b in itertools.combinations(range(len(per_rep)), 2):
            rep_rows.append({"model": m, "repeat_a": a, "repeat_b": b,
                             "kendall_tau_in_strength": kendalltau(per_rep[a]["in_strength"], per_rep[b]["in_strength"]).statistic,
                             "same_top_sink": int(np.argmax(per_rep[a]["in_strength"]) == np.argmax(per_rep[b]["in_strength"]))})

    pd.DataFrame(node_rows).to_csv(out / "node_metrics.csv", index=False)
    pd.DataFrame(graph_rows).to_csv(out / "graph_summary.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(out / "edges.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(out / "across_repeat_stability.csv", index=False)

    # consensus confusions: edges above the largest threshold (1%) in every model
    e = pd.DataFrame(edge_rows)
    tmax = max(sc["edge_thresholds"])
    cons = e[e.threshold == tmax].groupby(["true", "pred"]).agg(min_rate=("rate", "min"), mean_rate=("rate", "mean"),
                                                                  min_presence=("presence_freq", "min")).reset_index()
    cons[cons.min_rate > tmax].sort_values("mean_rate", ascending=False).to_csv(out / "consensus_edges.csv", index=False)

    # pairwise model similarity: per repeat on identical test images -> mean and SD over repeats; MST support
    if len(models) > 1:
        reps_m = {m: list(data[m].values()) for m in models}
        n_rep = min(len(v) for v in reps_m.values())
        ids0, y = reps_m[models[0]][0][0], reps_m[models[0]][0][1]
        assert all((reps_m[m][r][0] == ids0).all() for m in models for r in range(n_rep)), "models/repeats differ in test images"
        per_rep_sims = []
        for r in range(n_rep):
            P = {m: reps_m[m][r][2] for m in models}
            Rr = {m: rates(confusion(y, P[m], K)) for m in models}
            sims = {k: np.eye(len(models)) for k in SIM_KEYS}
            for (i, a), (j, b) in itertools.combinations(enumerate(models), 2):
                for k, v in pair_similarity(y, P[a], P[b], Rr[a], Rr[b]).items():
                    sims[k][i, j] = sims[k][j, i] = v
            for i, m in enumerate(models):
                sims["double_fault"][i, i] = float((P[m] != y).mean())
            per_rep_sims.append(sims)
        mean_sims = {}
        for k in SIM_KEYS:
            stack = np.stack([s[k] for s in per_rep_sims])
            mean_sims[k] = np.nanmean(stack, 0)
            pd.DataFrame(mean_sims[k], index=models, columns=models).to_csv(out / f"similarity_{k}.csv")
            pd.DataFrame(np.nanstd(stack, 0, ddof=1) if n_rep > 1 else np.zeros_like(mean_sims[k]),
                         index=models, columns=models).to_csv(out / f"similarity_{k}_sd.csv")
        base = mst_edges(1 - mean_sims["pearson_confusion"])
        support = {}
        off = ~np.eye(K, dtype=bool)
        for ix in strat_boot_indices(y, sc["bootstrap"], rng):
            Db = np.zeros((len(models), len(models)))
            for r in range(n_rep):
                Rb = {m: rates(confusion(y[ix], reps_m[m][r][2][ix], K)) for m in models}
                Db += np.array([[1 - np.corrcoef(Rb[a][off], Rb[b][off])[0, 1] for b in models] for a in models])
            for ed in mst_edges(np.nan_to_num(Db / n_rep, nan=1.0)):
                support[ed] = support.get(ed, 0) + 1
        pd.DataFrame([{"model_a": models[i], "model_b": models[j], "in_full_mst": int((i, j) in base),
                       "bootstrap_support": support.get((i, j), 0) / sc["bootstrap"]}
                      for i, j in itertools.combinations(range(len(models)), 2)]).to_csv(out / "mst_support.csv", index=False)

    # topology under the submitted split, the Roboflow split and domain shift
    def cv_test(mdir):
        reps = list(data[mdir.name].values())
        return reps[0][1], [p[2] for p in reps]

    comp_rows = []
    for protocol, sname, loader in (("cv", "test", cv_test),
                                    ("leaky", "test", lambda mdir: single_run_test(mdir)),
                                    ("vendor", "test", lambda mdir: single_run_test(mdir)),
                                    ("cv", "external_pooled", external_pooled)):
        for m in models:
            mdir = pred_root / protocol / m
            if not mdir.exists():
                continue
            got = loader(mdir)
            if got is None:
                continue
            yy, pp = got
            for label, thr, bb in filters:
                st = bootstrap_stability(yy, pp, K, sc["bootstrap"], 0.0 if bb else thr, sc["pagerank_alpha"],
                                         sc["louvain_resolution"], cfg.seed, rng, backbone_alpha=bb)
                f = st["full"]
                comp_rows.append({"protocol": protocol, "set": sname, "model": m, "filter": label, "n": int(len(yy)),
                                  "n_prediction_sets": len(pp), "accuracy": float(np.mean([(p == yy).mean() for p in pp])),
                                  "n_edges": f["n_edges"], "density": f["density"],
                                  "top_sink": classes[int(np.argmax(f["in_strength"]))] if np.max(f["in_strength"]) > 0 else "none",
                                  "top_sink_bootstrap_freq": st["full_sink_freq"],
                                  "n_communities": int(f["community"].max() + 1),
                                  "kendall_tau_median": st["kendall_tau_median"], "ari_median": st["ari_median"]})
            pd.DataFrame(rates(sum(confusion(yy, p, K) for p in pp)), index=classes, columns=classes).to_csv(
                out / f"rates_compare_{protocol}_{sname}_{m}.csv")
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(out / "graph_summary_compare.csv", index=False)
    write_meta(out, cfg, stage="s08_sna", models=models, filters=[f[0] for f in filters])
    print("SNA written to", out)


if __name__ == "__main__":
    main()
