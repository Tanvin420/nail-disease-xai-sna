"""Classification metrics (fast, confusion-matrix based), calibration, risk-coverage, bootstrap and tests."""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats
from sklearn.metrics import roc_auc_score


def softmax(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def confusion(y: np.ndarray, pred: np.ndarray, K: int) -> np.ndarray:
    return np.bincount(y * K + pred, minlength=K * K).reshape(K, K)


def cm_metrics(cm: np.ndarray) -> dict:
    """Metrics over the classes present in the ground truth (rows with support > 0). Predictions into absent
    classes count as errors (false negatives of the true class)."""
    sup = cm.sum(1)
    present = sup > 0
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = sup - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        rec = np.where(sup > 0, tp / sup, np.nan)
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)
    s = cm.sum()
    t, p, c = sup.astype(float), cm.sum(0).astype(float), tp.sum()
    den = np.sqrt((s**2 - p @ p) * (s**2 - t @ t))
    return {"acc": c / s if s else np.nan, "bal_acc": np.nanmean(rec[present]), "macro_f1": f1[present].mean(),
            "mcc": (c * s - t @ p) / den if den > 0 else 0.0, "recall": rec, "precision": prec, "f1": f1,
            "support": sup}


def auroc_ovr(y: np.ndarray, prob: np.ndarray) -> float:
    vals = []
    for c in np.unique(y):
        pos = y == c
        if 0 < pos.sum() < len(y):
            vals.append(roc_auc_score(pos, prob[:, c]))
    return float(np.mean(vals)) if vals else np.nan


# ---------------------------------------------------------------------------------------------
# calibration
def ece(prob: np.ndarray, y: np.ndarray, bins: int = 15, adaptive: bool = False) -> float:
    conf, pred = prob.max(1), prob.argmax(1)
    acc = (pred == y).astype(float)
    if adaptive:
        edges = np.quantile(conf, np.linspace(0, 1, bins + 1))
        b = np.clip(np.searchsorted(edges, conf, side="right") - 1, 0, bins - 1)
    else:
        b = np.clip((conf * bins).astype(int), 0, bins - 1)
    tot = 0.0
    for i in range(bins):
        m = b == i
        if m.any():
            tot += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(tot)


def reliability(prob: np.ndarray, y: np.ndarray, bins: int = 15):
    conf, pred = prob.max(1), prob.argmax(1)
    b = np.clip((conf * bins).astype(int), 0, bins - 1)
    out = []
    for i in range(bins):
        m = b == i
        out.append({"bin": i, "lo": i / bins, "hi": (i + 1) / bins, "n": int(m.sum()),
                    "conf": float(conf[m].mean()) if m.any() else np.nan,
                    "acc": float((pred[m] == y[m]).mean()) if m.any() else np.nan})
    return out


def nll(prob: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-12, 1)).mean())


def brier(prob: np.ndarray, y: np.ndarray) -> float:
    oh = np.zeros_like(prob)
    oh[np.arange(len(y)), y] = 1
    return float(((prob - oh) ** 2).sum(1).mean())


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising validation NLL (Guo et al., 2017)."""
    f = lambda lt: nll(softmax(logits, np.exp(lt)), y)  # noqa: E731
    return float(np.exp(optimize.minimize_scalar(f, bounds=(-3, 3), method="bounded").x))


def risk_coverage(prob: np.ndarray, y: np.ndarray):
    """Selective prediction by maximum softmax probability: (coverage, risk) curve, AURC and E-AURC."""
    conf = prob.max(1)
    err = (prob.argmax(1) != y).astype(float)
    o = np.argsort(-conf, kind="stable")
    risk = np.cumsum(err[o]) / np.arange(1, len(y) + 1)
    cov = np.arange(1, len(y) + 1) / len(y)
    aurc = float(risk.mean())
    r = err.mean()
    opt = r + (1 - r) * np.log(1 - r) if 0 < r < 1 else 0.0
    return cov, risk, aurc, aurc - opt


# ---------------------------------------------------------------------------------------------
# uncertainty and tests
def strat_boot_indices(y: np.ndarray, B: int, rng: np.random.Generator):
    groups = [np.flatnonzero(y == c) for c in np.unique(y)]
    for _ in range(B):
        yield np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])


def boot_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    return (float(np.quantile(v, alpha / 2)), float(np.quantile(v, 1 - alpha / 2))) if len(v) else (np.nan, np.nan)


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (c - h, c + h)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[int, int, float]:
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    if b + c == 0:
        return b, c, 1.0
    if b + c < 25:
        return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue)
    return b, c, float(stats.chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def corrected_ttest(diffs: np.ndarray, n_train: float, n_test: float) -> tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV differences."""
    d = np.asarray(diffs, float)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))
