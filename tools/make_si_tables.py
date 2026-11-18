"""SI tables that are not produced by s13: full class mapping per source (from configs/default.yaml) and the software/hardware
environment (installed package versions). Output: results/tables/tab_class_mapping.tex, tab_software.tex
(written into the SI by tools/inline_tables.py)"""
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from nailrev.config import load_config  # noqa: E402

cfg = load_config()
out = R / "results" / "tables"
out.mkdir(parents=True, exist_ok=True)
esc = lambda s: str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")  # noqa: E731
names = cfg.display_names
rows = []
for s in cfg.sources:
    if s.get("role") not in ("dev", "external", "provenance", "legacy"):
        continue
    lm = s.get("label_map", {}) or {}
    mapping = "; ".join(f"{k} $\\to$ {names.get(v, v)}" for k, v in lm.items())
    probes = ", ".join(map(str, s.get("probes", []) or [])) or "--"
    rows.append(rf"{esc(s.get('display', s['id']))} & {esc(mapping)} & {esc(probes)} \\")
T = [r"\begin{table}[!htb]", r"\centering", r"\caption{Class mapping of every source to the seven development classes; labels without a counterpart "
     r"were kept as out-of-label probes (generated from \texttt{configs/default.yaml}).}", r"\label{tab:classmapping}", r"\footnotesize",
     r"\begin{adjustbox}{max width=\linewidth}", r"\begin{tabular}{p{3.6cm}p{8.4cm}p{4.6cm}}", r"\hline",
     r"Source & Original label $\to$ development class & Out-of-label probes \\", r"\hline", *rows, r"\hline", r"\end{tabular}",
     r"\end{adjustbox}", r"\end{table}"]
(out / "tab_class_mapping.tex").write_text("\n".join(T), encoding="utf-8")

pk = ["torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "scipy", "captum", "grad-cam", "lime", "networkx", "onnx",
      "onnxruntime", "imagehash"]
vv = []
for p in pk:
    try:
        vv.append((p, version(p)))
    except Exception:
        pass
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
except Exception:
    gpu = "n/a"
cpu = "n/a"
try:
    for line in open("/proc/cpuinfo"):
        if line.startswith("model name"):
            cpu = line.split(":", 1)[1].strip()
            break
except Exception:
    pass
rows = [rf"Python & {platform.python_version()} \\", rf"Operating system & {esc(platform.platform(terse=True))} \\",
        rf"GPU (driver) & {esc(gpu)} \\", rf"CPU & {esc(cpu)} \\"] + [rf"{esc(p)} & {esc(v)} \\" for p, v in vv]
T = [r"\begin{table}[!htb]", r"\centering", r"\caption{Software and hardware used for all experiments (read from the analysis environment).}",
     r"\label{tab:software}", r"\footnotesize", r"\begin{tabular}{ll}", r"\hline", r"Component & Version \\", r"\hline", *rows, r"\hline",
     r"\end{tabular}", r"\end{table}"]
(out / "tab_software.tex").write_text("\n".join(T), encoding="utf-8")
print("written tab_class_mapping.tex, tab_software.tex")
