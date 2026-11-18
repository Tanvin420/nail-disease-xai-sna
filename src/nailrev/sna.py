"""Confusion-network ("social network") analysis of misclassification patterns.

Graph G_m for model m: nodes = classes; directed edge i -> j (i != j) with weight
    w_ij = P(pred = j | true = i)   (row-normalised confusion rate, pooled over repeats),
kept if w_ij > threshold.  Node metrics: weighted in-strength (the "diagnostic sink" = class that attracts the
most errors), out-strength, weighted PageRank (alpha = 0.85), Fagiolo's weighted directed clustering
coefficient, and Louvain communities on the symmetrised graph (w_ij + w_ji).
Note: networkx PageRank normalises each node's out-weights, so raw counts and row rates give identical PageRank.

Edge filters (all reported): none (w > 0); threshold w >= 0.01; and the disparity-filter backbone (Serrano, Boguna &
Vespignani 2009, alpha = 0.05; directed: an edge is kept if it is significant for its source's out-edges or its
target's in-edges; nodes with a single edge keep it).  Raw confusion counts are written by stage 07
(results/metrics/confusion_<protocol>_<model>.csv).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import adjusted_rand_score, cohen_kappa_score


def rates(cm: np.ndarray) -> np.ndarray:
    r = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    np.fill_diagonal(r, 0)
    return r


def backbone(R: np.ndarray, alpha: float) -> np.ndarray:
    """Disparity filter on a directed weighted graph given as a rate matrix with zero diagonal."""
    keep = np.zeros(R.shape, bool)
    for axis in (1, 0):                                   # out-edges of each source, then in-edges of each target
        strength = R.sum(axis, keepdims=True)
        k = (R > 0).sum(axis, keepdims=True)
        p = np.divide(R, strength, out=np.zeros_like(R), where=strength > 0)
        significance = np.where(k > 1, (1 - p) ** np.clip(k - 1, 0, None), 0.0)
        keep |= (R > 0) & (significance < alpha)
    return np.where(keep, R, 0.0)


def graph(R: np.ndarray, thr: float) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(range(len(R)))
    for i, j in zip(*np.nonzero(R > thr)):
        G.add_edge(int(i), int(j), weight=float(R[i, j]))
    return G


def communities(G: nx.DiGraph, resolution: float, seed: int) -> np.ndarray:
    U = nx.Graph()
    U.add_nodes_from(G.nodes)
    for i, j, d in G.edges(data=True):
        w = d["weight"] + (U[i][j]["weight"] if U.has_edge(i, j) else 0)
        U.add_edge(i, j, weight=w)
    lab = np.zeros(G.number_of_nodes(), int)
    if U.number_of_edges() == 0:
        return np.arange(len(lab))
    for c, nodes in enumerate(nx.community.louvain_communities(U, weight="weight", resolution=resolution, seed=seed)):
        lab[list(nodes)] = c
    return lab


def node_metrics(R: np.ndarray, thr: float, alpha: float, resolution: float, seed: int) -> dict:
    G = graph(R, thr)
    K = len(R)
    W = nx.to_numpy_array(G, nodelist=range(K), weight="weight")
    pr = nx.pagerank(G, alpha=alpha, weight="weight") if G.number_of_edges() else {i: 1 / K for i in range(K)}
    cl = nx.clustering(G, weight="weight")
    com = communities(G, resolution, seed)
    return {"in_strength": W.sum(0), "out_strength": W.sum(1), "pagerank": np.array([pr[i] for i in range(K)]),
            "clustering": np.array([cl[i] for i in range(K)]), "community": com,
            "n_edges": G.number_of_edges(), "density": G.number_of_edges() / (K * (K - 1))}


def bootstrap_stability(y: np.ndarray, preds: list[np.ndarray], K: int, B: int, thr: float, alpha: float,
                        resolution: float, seed: int, rng, backbone_alpha: float | None = None) -> dict:
    """Stratified bootstrap of test items; confusion pooled over the prediction sets (repeats).
    backbone_alpha: if given, the disparity-filter backbone (thr is then ignored and edges are the surviving ones)."""
    from .metrics import confusion, strat_boot_indices

    prep = (lambda R: backbone(R, backbone_alpha)) if backbone_alpha else (lambda R: R)
    thr = 0.0 if backbone_alpha else thr
    full = node_metrics(prep(rates(sum(confusion(y, p, K) for p in preds))), thr, alpha, resolution, seed)
    none = K                                      # sentinel: a graph without edges has no sink (argmax of all zeros would pick class 0)
    top_of = lambda s: int(np.argmax(s)) if np.max(s) > 0 else none  # noqa: E731
    full_top = top_of(full["in_strength"])
    tops, ins, prs, taus, aris, edges, kept, coassign = [], [], [], [], [], [], [], np.zeros((K, K))
    for ix in strat_boot_indices(y, B, rng):
        R = rates(sum(confusion(y[ix], p[ix], K) for p in preds))
        Rf = prep(R)
        nm = node_metrics(Rf, thr, alpha, resolution, seed)
        tops.append(top_of(nm["in_strength"]))
        ins.append(nm["in_strength"])
        prs.append(nm["pagerank"])
        taus.append(kendalltau(full["in_strength"], nm["in_strength"]).statistic)
        aris.append(adjusted_rand_score(full["community"], nm["community"]))
        edges.append(R)
        kept.append(Rf > thr)
        coassign += nm["community"][:, None] == nm["community"][None, :]
    ins, prs, edges = np.array(ins), np.array(prs), np.array(edges)
    return {"full": full, "top_sink_freq": np.bincount(tops, minlength=K + 1)[:K] / B,
            "full_sink_freq": float("nan") if full_top == none else float((np.array(tops) == full_top).mean()),
            "in_lo": np.quantile(ins, 0.025, 0), "in_hi": np.quantile(ins, 0.975, 0),
            "pr_lo": np.quantile(prs, 0.025, 0), "pr_hi": np.quantile(prs, 0.975, 0),
            "kendall_tau_median": float(np.nanmedian(taus)), "kendall_tau_q25": float(np.nanquantile(taus, 0.25)),
            "ari_median": float(np.median(aris)), "ari_q25": float(np.quantile(aris, 0.25)),
            "edge_rate_lo": np.quantile(edges, 0.025, 0), "edge_rate_hi": np.quantile(edges, 0.975, 0),
            "edge_presence": np.array(kept).mean(0), "coassignment": coassign / B}


def pair_similarity(y, pa, pb, Ra, Rb) -> dict:
    ca, cb = pa == y, pb == y
    p1, p2 = ca.mean(), cb.mean()
    c_obs = (ca == cb).mean()
    c_exp = p1 * p2 + (1 - p1) * (1 - p2)
    ea, eb = ~ca, ~cb
    off = ~np.eye(len(Ra), dtype=bool)
    return {"cohen_kappa": cohen_kappa_score(pa, pb),
            "error_consistency_kappa": (c_obs - c_exp) / (1 - c_exp) if c_exp < 1 else np.nan,
            "jaccard_errors": (ea & eb).sum() / max((ea | eb).sum(), 1),
            "double_fault": (ea & eb).mean(),
            "pearson_confusion": pearsonr(Ra[off], Rb[off]).statistic if Ra[off].std() > 0 and Rb[off].std() > 0 else np.nan}


def mst_edges(D: np.ndarray) -> set[tuple[int, int]]:
    T = minimum_spanning_tree(D + 1e-9 * (D == 0)).tocoo()
    return {tuple(sorted((int(i), int(j)))) for i, j in zip(T.row, T.col)}
