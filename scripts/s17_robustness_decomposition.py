"""Stage 17 - dataset robustness vs model robustness: variance decomposition of transfer performance.

Two analyses, both on results already computed (no new training):
  A. Cross-dataset matrix (protocol xsrc, shared-label evaluation, results/metrics/shared_label_matrix.csv): one score per
     (architecture, training dataset, test dataset). Two-way-interaction ANOVA (Type II sums of squares)
         score ~ train + test + model + train:test + model:train + model:test     (residual = three-way interaction)
     Score = chance-corrected balanced accuracy on the shared classes, (recall - 1/k) / (1 - 1/k), k = number of shared classes,
     so that easy binary pairs do not dominate; the uncorrected balanced accuracy is reported as a sensitivity analysis.
  B. Development-set models on the de-duplicated external sources (protocol cv, results/metrics/runs_all_sets.csv): one score per
     (architecture, external source, training run); score ~ source + model + source:model, residual = run-to-run variation (folds/seeds).
Effect sizes: eta^2 = SS_term / SS_total. Rank agreement of the five architectures across test settings: Kendall's W.
Outputs: results/metrics/robustness_anova.csv, robustness_marginal_means.csv, results/tables/tab_variance.tex,
         results/macros_robustness.tex (\\R... macros, expanded into the manuscript by tools/expand_macros.py).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import rankdata
from statsmodels.stats.anova import anova_lm

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from nailrev.config import load_config  # noqa: E402
from nailrev.labels import DATASETS  # noqa: E402

MET, TAB = R / "results" / "metrics", R / "results" / "tables"
PARTS = {"han": {"han_A1", "han_A2", "han_BCDE"}, "thai": {"arthitaya", "arthitaya2", "nuttida"}}
TERM = {"C(train)": "Training dataset", "C(test)": "Test dataset", "C(model)": "Architecture",
        "C(train):C(test)": "Training x test dataset", "C(model):C(train)": "Architecture x training dataset",
        "C(model):C(test)": "Architecture x test dataset", "C(source)": "External source",
        "C(model):C(source)": "Architecture x external source", "Residual": "Residual"}


def macro_name(key):
    out, up = [], True
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    for ch in key:
        if ch.isdigit():
            out.append(digits[ch]); up = True
        elif ch.isalpha():
            out.append(ch.upper() if up else ch); up = False
        else:
            up = True
    return "R" + "".join(out)


def kendall_w(M):
    """M: settings x models score matrix (no NaN). Kendall's W of the model ranks across settings."""
    ranks = np.vstack([rankdata(-row) for row in M])
    m, n = ranks.shape
    S = ((ranks.sum(0) - m * (n + 1) / 2) ** 2).sum()
    return 12 * S / (m ** 2 * (n ** 3 - n))


def anova(df, formula):
    fit = smf.ols(formula, data=df).fit()
    a = anova_lm(fit, typ=2)
    sst = ((df.y - df.y.mean()) ** 2).sum()
    a["eta2"] = a["sum_sq"] / sst
    return a, fit


def main():
    cfg = load_config()
    label = {m: cfg.models[m]["label"] for m in cfg.models}
    mac, rows, marg = {}, [], []

    # ---------------------------------------------------------------- A: cross-dataset matrix
    sl = pd.read_csv(MET / "shared_label_matrix.csv")
    sl = sl[(sl.status == "ok") & (sl.target != "own_test") & sl.recall.notna() & (sl.n_runs == 1)]   # xsrc: one model per architecture
    sl = sl[[t not in PARTS.get(tr, set()) and t != tr for tr, t in zip(sl.train_set, sl.target)]]
    k = sl.n_shared_classes.astype(float)
    sl = sl.assign(train=sl.train_set, test=sl.target, raw=sl.recall, cc=(sl.recall - 1 / k) / (1 - 1 / k))
    sl["pair"] = sl.train + "->" + sl.test
    full = sl.groupby("pair").model.nunique()
    sl = sl[sl.pair.isin(full[full == sl.model.nunique()].index)].copy()
    assert not sl.duplicated(["pair", "model"]).any()
    for score in ("cc", "raw"):
        d = sl.assign(y=sl[score])
        # exact two-way decomposition of the complete (pair x architecture) table; residual = architecture x pair interaction
        a, _ = anova(d, "y ~ C(pair) + C(model)")
        sst = ((d.y - d.y.mean()) ** 2).sum()
        ss_pair, ss_model, ss_int = a.loc["C(pair)", "sum_sq"], a.loc["C(model)", "sum_sq"], a.loc["Residual", "sum_sq"]
        assert abs(ss_pair + ss_model + ss_int - sst) < 1e-8 * max(sst, 1)
        # split the pair share: pair means ~ training dataset + test dataset (Type II), remainder = specific pair
        pm = d.groupby(["pair", "train", "test"]).y.mean().reset_index()
        b, _ = anova(pm, "y ~ C(train) + C(test)")
        n_m = d.model.nunique()
        ss_tr, ss_te = n_m * b.loc["C(train)", "sum_sq"], n_m * b.loc["C(test)", "sum_sq"]
        ss_sp = ss_pair - ss_tr - ss_te
        terms = [("Training dataset", ss_tr, b.loc["C(train)", "df"], b.loc["C(train)", "PR(>F)"]),
                 ("Test dataset", ss_te, b.loc["C(test)", "df"], b.loc["C(test)", "PR(>F)"]),
                 ("Training x test dataset (specific pair)", ss_sp, b.loc["Residual", "df"], np.nan),
                 ("Architecture", ss_model, a.loc["C(model)", "df"], a.loc["C(model)", "PR(>F)"]),
                 ("Architecture x dataset pair", ss_int, a.loc["Residual", "df"], np.nan)]
        for name, ss, df_, p_ in terms:
            rows.append(dict(analysis="A_cross_dataset", score=score, term=name, df=df_, sum_sq=ss, F=np.nan, p=p_, eta2=ss / sst))
        sfx = "" if score == "cc" else "_raw"
        mac.update({f"var_xd_dataset_pct{sfx}": 100 * ss_pair / sst, f"var_xd_arch_pct{sfx}": 100 * ss_model / sst,
                    f"var_xd_inter_pct{sfx}": 100 * ss_int / sst, f"var_xd_train_pct{sfx}": 100 * ss_tr / sst,
                    f"var_xd_test_pct{sfx}": 100 * ss_te / sst, f"var_xd_ratio{sfx}": ss_pair / ss_model})
        if score == "cc":
            mac["var_xd_arch_p"] = float(a.loc["C(model)", "PR(>F)"])
    mac["var_xd_n_pairs"] = sl.pair.nunique()
    mac["var_xd_n_train"] = sl.train.nunique()
    mac["var_xd_n_test"] = sl.test.nunique()
    mac["var_xd_n_obs"] = len(sl)
    for fac, col in (("Architecture", "model"), ("Training dataset", "train"), ("Test dataset", "test")):
        for lev, d in sl.groupby(col):
            marg.append(dict(analysis="A_cross_dataset", factor=fac, level=lev, mean_cc=d.cc.mean(), mean_raw=d.raw.mean(), n=len(d)))
    mm = sl.groupby("model").cc.mean()
    mt = sl.groupby("train").cc.mean()
    mac.update({"var_xd_model_mean_min": mm.min(), "var_xd_model_mean_max": mm.max(),
                "var_xd_train_mean_min": mt.min(), "var_xd_train_mean_max": mt.max()})
    mac["var_xd_best_model"] = label[mm.idxmax()]
    mac["var_xd_worst_model"] = label[mm.idxmin()]
    mac["var_xd_best_train"] = DATASETS[mt.idxmax()][1]
    mac["var_xd_best_train_code"] = DATASETS[mt.idxmax()][0]
    mac["var_xd_worst_train_code"] = DATASETS[mt.idxmin()][0]
    piv = sl.pivot_table(index="pair", columns="model", values="cc")
    mac["var_xd_kendall_w"] = kendall_w(piv.to_numpy())
    best = piv.idxmax(axis=1).value_counts()
    mac["var_xd_best_model_wins"] = int(best.max())
    mac["var_xd_best_model_wins_name"] = label[best.idxmax()]

    # ---------------------------------------------------------------- B: development models on external sources (per run)
    ru = pd.read_csv(MET / "runs_all_sets.csv")
    ru = ru[(ru.protocol == "cv") & ru.set.astype(str).str.startswith("ext_") & ru.bal_acc.notna()].copy()
    ru["source"] = ru.set.str[4:]
    ru = ru[ru.groupby("source").model.transform("nunique") == ru.model.nunique()]
    a, _ = anova(ru.assign(y=ru.bal_acc), "y ~ C(source) + C(model) + C(model):C(source)")
    for term, r in a.iterrows():
        rows.append(dict(analysis="B_external_sources", score="bal_acc", term=TERM.get(term, term), df=r.df, sum_sq=r.sum_sq,
                         F=r.F, p=r["PR(>F)"], eta2=r.eta2))
    g = a.eta2
    mac.update({"var_ext_source_pct": 100 * g["C(source)"], "var_ext_arch_pct": 100 * g["C(model)"],
                "var_ext_inter_pct": 100 * g["C(model):C(source)"], "var_ext_run_pct": 100 * g["Residual"],
                "var_ext_n_sources": ru.source.nunique(), "var_ext_n_runs": ru.run.nunique()})
    pe = ru.groupby(["source", "model"]).bal_acc.mean().unstack()
    mac["var_ext_kendall_w"] = kendall_w(pe.to_numpy())
    for fac, col in (("Architecture", "model"), ("External source", "source")):
        for lev, d in ru.groupby(col):
            marg.append(dict(analysis="B_external_sources", factor=fac, level=lev, mean_cc=np.nan, mean_raw=d.bal_acc.mean(), n=len(d)))
    # internal (development) test, same decomposition for contrast: architecture vs run
    it = pd.read_csv(MET / "runs_all_sets.csv")
    it = it[(it.protocol == "cv") & (it.set == "test")]
    a_i, _ = anova(it.assign(y=it.macro_f1), "y ~ C(model)")
    mac["var_int_arch_pct"] = 100 * a_i.eta2["C(model)"]

    # internal AUROC range (development test, group-disjoint CV; quoted in the text next to the decomposition)
    ins = pd.read_csv(MET / "internal_summary.csv")
    au = ins[(ins.protocol == "cv") & (ins.set == "test")].auroc_runs_mean
    mac["cv_auroc_min"], mac["cv_auroc_max"] = float(au.min()), float(au.max())
    # per-level means for the text (chance-corrected for A, balanced accuracy for B)
    mg = pd.DataFrame(marg)
    for r in mg.itertuples():
        if r.analysis == "A_cross_dataset" and r.factor in ("Training dataset", "Architecture"):
            lev = DATASETS[r.level][0] if r.factor == "Training dataset" else r.level
            mac[f"var_xd_mean_{'train' if r.factor == 'Training dataset' else 'model'}_{lev}"] = r.mean_cc
        if r.analysis == "B_external_sources" and r.factor == "Architecture":
            mac[f"var_ext_mean_{r.level}"] = r.mean_raw
    an = pd.DataFrame(rows)
    an.to_csv(MET / "robustness_anova.csv", index=False)
    pd.DataFrame(marg).to_csv(MET / "robustness_marginal_means.csv", index=False)

    # ---------------------------------------------------------------- table
    def fmt_p(p):
        return "--" if pd.isna(p) else ("$<$0.001" if p < 0.001 else f"{p:.3f}")
    lines = [r"\begin{table}[!tbp]", r"\centering",
             r"\caption{Variance decomposition of transfer performance ($\eta^2$: share of the total variation). (A) Cross-dataset matrix "
             r"(chance-corrected balanced accuracy on shared classes); (B) NailGroup-7 models on the external datasets (balanced accuracy "
             r"of each run; residual = run-to-run variation).}",
             r"\label{tab:variance}", r"\footnotesize", r"\begin{tabular}{llrrr}", r"\hline",
             r"Analysis & Source of variation & df & $\eta^2$ (\%) & $p$ \\", r"\hline"]
    for an_name, title in (("A_cross_dataset", "(A) Cross-dataset"), ("B_external_sources", "(B) External sources")):
        d = an[(an.analysis == an_name) & (an.score.isin(["cc", "bal_acc"]))]
        for i, r in enumerate(d.itertuples()):
            lines.append(f"{title if i == 0 else ''} & {r.term.replace(' x ', r' $\times$ ')} & {int(r.df)} & {100 * r.eta2:.1f} & {fmt_p(r.p)} \\\\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table}"]
    (TAB / "tab_variance.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- macros
    out = ["% generated by scripts/s17_robustness_decomposition.py"]
    for k_, v in mac.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer() and "pct" not in k_ and "mean" not in k_
                                                   and "ratio" not in k_ and "_w" not in k_ and "_p" not in k_):
            val = str(int(v))
        elif k_.endswith("_p"):
            val = "<0.001" if v < 0.001 else f"{v:.3f}"
        elif "pct" in k_:
            val = f"{v:.1f}"
        elif "ratio" in k_:
            val = f"{v:.1f}"
        else:
            val = f"{v:.2f}" if "_w" in k_ else f"{v:.3f}"
        out.append(f"\\newcommand{{\\{macro_name(k_)}}}{{{val}}}")
    (R / "results" / "macros_robustness.tex").write_text("\n".join(out) + "\n")
    print(an[an.score.isin(["cc", "bal_acc"])][["analysis", "term", "df", "eta2", "p"]].to_string())
    print("\n".join(out))


if __name__ == "__main__":
    main()
