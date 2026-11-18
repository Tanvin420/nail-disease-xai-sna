"""Stage 18 - two overview tables of the manuscript, generated from data/ and results/ (no hand-typed numbers):

  results/tables/tab_datasets.tex  Table "Datasets": code, reference, licence, images with a 7-class label, independent test images,
                                   number and names of the classes, role in the study.
  results/tables/tab_summary.tex   Table "Results at a glance": one row per test and measure, what the measure tells the reader, and the
                                   value for each architecture (best value in bold where "best" is defined).
Run after s13 and s17 (it reads results/macros*.tex and the result CSVs). usage: python scripts/s18_manuscript_tables.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from nailrev.config import load_config  # noqa: E402
from nailrev.labels import DATASETS  # noqa: E402

MET, TAB = R / "results" / "metrics", R / "results" / "tables"
MAC = {k: v for f in sorted((R / "results").glob("macros*.tex"))
       for k, v in re.findall(r"\\newcommand\{\\(R[A-Za-z]+)\}\{([^}]*)\}", f.read_text())}
PREFIX = {"vgg16": "VggOneSix", "densenet121": "DensenetOneTwoOne", "efficientnet_b2": "EfficientnetBTwo",
          "mobilenetv3": "MobilenetvThree", "vit_b16": "VitBOneSix"}
# source -> (references, licence, role); text only, every number is computed below
META = {
    "kopin": (r"NailGroup-7, curated in this work \cite{nailgrouped2026kaggle} from a Roboflow export \cite{kopin2024dataset}", "CC BY 4.0",
              "development; cross-dataset training"),
    "han_A1": (r"Han et al. A1, hospital thumbnails \cite{han2018onychomycosis,han2017figshareA}", "CC BY 4.0", "external test; training (HAN)"),
    "han_A2": (r"Han et al. A2, hospital \cite{han2017figshareA}", "CC BY 4.0", "training (HAN); provenance"),
    "han_BCDE": (r"Han et al. B--E, hospitals \cite{han2017figshareA}", "CC BY 4.0", "training (HAN); provenance"),
    "han_virtualE": (r"Han et al. Virtual E, web \cite{han2017figshareE}", "CC BY 4.0", "external test; training"),
    "arthitaya": (r"Kaggle arthitaya \cite{kaggleArthitaya1,sogukkuyu2023melanonychia}", "not stated", "external test; training (THA)"),
    "arthitaya2": (r"Kaggle arthitaya-2 \cite{kaggleArthitaya2}", "not stated", "external test; training (THA)"),
    "nuttida": (r"Kaggle nuttidalapthanachai \cite{kaggleNuttida,lapthanachai2023nail}", "not stated", "external test; training (THA)"),
    "josephrasanjana": (r"Kaggle josephrasanjana \cite{kaggleJoseph}", "not stated", "external test; training"),
    "nikhilgurav21": (r"Kaggle nikhilgurav21 (KNM) \cite{kaggleNikhil,1_shandilya2024autonomous,khullar2026federated}", "CC BY-NC-SA 4.0",
                      "external test; training"),
    "kaggle17": (r"Kaggle 17-class \cite{kaggle17class}", "MIT (re-upload)", "external test; training"),
    "farisfaikar": (r"Kaggle nail-melanoma-350 \cite{kaggleFaris}", "not stated", "external test; melanoma as out-of-label probe"),
    "isratjahanmitu": (r"Kaggle isratjahanmitu \cite{kaggleIsrat}", "not stated", "external test"),
    "legacy2272": (r"Combined set: Roboflow mohit \cite{roboflowNailDisease} and nikhilgurav21", "mixed", "duplicate audit (SI Table~S18)"),
}


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def fmt_int(v):
    return f"{int(v):,}"


def datasets_table(cfg):
    names = cfg.display_names
    es = pd.read_csv(MET / "external_summary.csv")
    es = es[es.protocol == "cv"].drop_duplicates("source").set_index("source")
    flow = json.loads((cfg.path("dedup") / "flow.json").read_text())["dev"]
    inc_files = sum(flow["included_files_by_class"].values())
    lg = pd.read_csv(MET / "legacy_external_audit.csv")
    lg_ind = lg[(lg.protocol == "cv") & (lg.origin == "all sources") & (lg.subset == "not_duplicated")].n.iloc[0]

    def classes(cl):
        cl = [c for c in cfg.classes if c in cl]
        return f"{len(cl)}: " + ", ".join(names[c] for c in cl)
    rows = []
    for sid in META:
        ref, lic, role = META[sid]
        if sid == "kopin":
            imgs = f"{fmt_int(flow['kopin_files'])} files; {fmt_int(inc_files)} used"
            test = f"{fmt_int(flow['included_components'])} groups (CV)"
            cl = classes(list(cfg.classes))
        elif sid == "legacy2272":
            imgs = fmt_int(es.loc["legacy2272"].n)
            test = f"{fmt_int(lg_ind)} not duplicated"
            cl = classes(str(es.loc["legacy2272"].classes_present).split(";"))
        else:
            full = es.loc[f"full_{sid}"]
            imgs = fmt_int(full.n)
            test = fmt_int(es.loc[f"ext_{sid}"].n) if f"ext_{sid}" in es.index else "--"
            cl = classes(str(full.classes_present).split(";"))
        rows.append(f"{DATASETS[sid][0]} & {ref} & {lic} & {imgs} & {test} & {esc(cl)} & {role} \\\\")
    L = [r"\begin{table*}[!htbp]", r"\centering",
         r"\caption{Datasets used in this study: images with a seven-class label, independent test images after removing copies of "
         r"NailGroup-7 pictures, classes and role; licences as declared by the hosting page. HAN = HA1 + HA2 + HBE; "
         r"THA = ART1 + ART2 + NUT.}",
         r"\label{tab:datasets}", r"\footnotesize",
         r"\begin{tabular}{" + "".join(r">{\raggedright\arraybackslash}p{%s}" % w for w in ("0.9cm", "4.1cm", "1.7cm", "2.0cm", "1.9cm", "5.0cm", "2.8cm")) + "}", r"\hline",
         r"Code & Dataset (reference) & Licence & Images (7-class labels) & Independent test images & Classes (number: names) & Role \\",
         r"\hline", *rows, r"\hline", r"\end{tabular}", r"\end{table*}"]
    (TAB / "tab_datasets.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def summary_table(cfg):
    models = list(cfg.models)
    label = {m: cfg.models[m]["label"] for m in models}
    mac = lambda m, k: MAC[f"R{PREFIX[m]}{k}"]  # noqa: E731
    ins = pd.read_csv(MET / "internal_summary.csv")
    ins = ins[(ins.protocol == "cv") & (ins.set == "test")].set_index("model")
    ru = pd.read_csv(MET / "runs_all_sets.csv")
    han_auc = ru[(ru.protocol == "cv") & (ru.set == "ext_han_A1")].groupby("model").auroc.mean()
    es = pd.read_csv(MET / "external_summary.csv")
    ext_mean = es[(es.protocol == "cv") & es.source.str.startswith("ext_")].groupby("model").bal_acc.mean()
    msrc = ru[(ru.protocol == "msrc") & (ru.set == "test")].groupby("model").bal_acc.mean()
    fa = pd.read_csv(R / "results" / "xai" / "faithfulness.csv")
    fa = fa[fa.correct == 1]
    del_pct = fa.groupby("model").apply(lambda g: (g.gradcam_deletion < g.random_smooth_deletion).mean() * 100)
    san = pd.read_csv(R / "results" / "xai" / "sanity.csv").groupby("model").gradcam_spearman.mean()
    gs = pd.read_csv(R / "results" / "sna" / "graph_summary.csv")
    gs = gs[gs["filter"] == "w>=0.01"].set_index("model")
    cls_code = {"beau_s": "BEA", "black_line": "BLK", "clubbing": "CLB", "healthy_nail": "HLT", "onychomycosis": "ONM",
                "onycholysis": "ONL", "white_spot": "WSP"}

    rows = [  # (test, measure, what it tells, values per model, best = "max"/"min"/None, format)
        ("T1", "Macro-F1, internal", "accuracy on unseen pictures of NailGroup-7 (1 = perfect)",
         {m: float(mac(m, "CvMacroFOne")) for m in models}, "max", "{:.3f}"),
        ("T1", "AUROC, internal", "separation of the classes, threshold-free (1 = perfect)",
         {m: float(ins.loc[m, "auroc_runs_mean"]) for m in models}, "max", "{:.3f}"),
        ("T1", "Calibration error after temperature scaling (ECE)", "gap between confidence and accuracy (0 = perfect)",
         {m: float(mac(m, "EcePost")) for m in models}, "min", "{:.3f}"),
        ("T2", "Macro-F1, image-level split", "inflated estimate when copies of a picture reach the test set",
         {m: float(mac(m, "LeakyMacroFOne")) for m in models}, None, "{:.3f}"),
        ("T3", "Macro-F1, Han A1 (hospital)", "performance on independent hospital images",
         {m: float(mac(m, "ExtHanAOneMacroFOne")) for m in models}, "max", "{:.3f}"),
        ("T3", "AUROC, Han A1 (hospital)", "class separation on independent hospital images",
         {m: float(han_auc[m]) for m in models}, "max", "{:.3f}"),
        ("T3", "Balanced accuracy, mean of the external datasets", "average performance on independent datasets",
         {m: float(ext_mean[m]) for m in models}, "max", "{:.3f}"),
        ("T4", "Transfer across datasets (chance-corrected)", "0 = chance, 1 = perfect; mean over all dataset pairs",
         {m: float(MAC[f"RVarXdMeanModel{PREFIX[m]}"]) for m in models}, "max", "{:.3f}"),
        ("T5", "Balanced accuracy, held-out source (pooled training)", "transfer after training on several sources",
         {m: float(msrc[m]) for m in models}, "max", "{:.3f}"),
        ("T7", "Change in macro-F1 with 5 epochs (points)", "negative = a short fixed schedule under-trains the model",
         {m: float(mac(m, "FixedFiveDeltaPts")) for m in models}, None, "{:+.1f}"),
        ("T8", "Parameters (millions)", "model size", {m: float(mac(m, "Params")) for m in models}, "min", "{:.1f}"),
        ("T8", "Latency, GPU FP32 (ms per image)", "speed on a graphics card", {m: float(mac(m, "GpuFpThreeTwoBOneMs")) for m in models},
         "min", "{:.2f}"),
        ("T8", "Latency, 1 CPU thread (ms per image)", "speed on an ordinary processor",
         {m: float(mac(m, "CpuFpThreeTwoOneThreadMs")) for m in models}, "min", "{:.1f}"),
        ("T9", "Grad-CAM better than random maps (\\% images)", "how often the map points at pixels that matter",
         {m: float(del_pct[m]) for m in models}, "max", "{:.0f}"),
        ("T9", "Grad-CAM sanity check ($\\rho$)", "near 0 = the map depends on the trained model (passed)",
         {m: float(san[m]) for m in models}, None, "{:.2f}"),
    ]
    L = [r"\begin{table*}[!htbp]", r"\centering",
         r"\caption{Main results of all tests (Table~\ref{tab:tests}) for each architecture; best value in bold where higher or "
         r"lower is clearly better.}",
         r"\label{tab:summary}", r"\footnotesize",
         r"\begin{tabular}{" + "".join(r">{\raggedright\arraybackslash}p{%s}" % w for w in ("0.6cm", "4.0cm", "4.6cm")) + "ccccc}", r"\hline",
         "Test & Measure & What it tells & " + " & ".join(label[m] for m in models) + r" \\", r"\hline"]
    t6 = (r"T6 & Share of cross-dataset variance & how much the data or the architecture matter & \multicolumn{5}{l}{"
          f"datasets {MAC['RVarXdDatasetPct']}\\%, architecture {MAC['RVarXdArchPct']}\\%, interaction {MAC['RVarXdInterPct']}\\%"
          r"} \\")
    for test, meas, tells, vals, best, f in rows:
        arr = np.array([vals[m] for m in models])
        bi = None if best is None else (int(arr.argmax()) if best == "max" else int(arr.argmin()))
        cells = [(r"\textbf{" + f.format(v) + "}") if i == bi else f.format(v) for i, v in enumerate(arr)]
        L.append(f"{test} & {meas} & {tells} & " + " & ".join(cells) + r" \\")
        if test == "T5":
            L.append(t6)
    sink = [f"{cls_code[gs.loc[m, 'top_sink']]} ({gs.loc[m, 'top_sink_bootstrap_freq']:.2f})" for m in models]
    L.append(r"T10 & Top error sink (bootstrap frequency) & class that receives most errors & " + " & ".join(sink) + r" \\")
    L += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    (TAB / "tab_summary.tex").write_text("\n".join(L) + "\n", encoding="utf-8")


def main():
    cfg = load_config()
    datasets_table(cfg)
    summary_table(cfg)
    print("tables -> results/tables/tab_datasets.tex, tab_summary.tex")


if __name__ == "__main__":
    main()
