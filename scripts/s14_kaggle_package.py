"""Stage 14 - build the Kaggle release NailGroup-7: the curated (grouped, leakage-controlled) version of the ko-pin Roboflow export.

  python scripts/s14_kaggle_package.py                      # builds everything below (a few seconds on the workstation)
  python scripts/s14_kaggle_package.py --user <kaggle-name> # once the Kaggle account is known: fills the dataset URL into the
                                                            # description, the citation and dataset-metadata.json

Writes kaggle_release/
  nailgroup-7_with_images.zip (+ .sha256)   THE DATASET, one folder nailgroup-7/: README.md (= the dataset description), manifest.csv,
                                            components.csv, representatives.csv, dataset-metadata.json, images/<class>/<file>.jpg
                                            (original bytes of the included files)
  NailGroup-7_Kaggle_upload.md              ONE file for the upload (not part of the dataset): what to upload, step-by-step manual
                                            (web and API), the value of every Kaggle field, "About this file" texts, column
                                            descriptions, checklist, and the complete description text to paste into Kaggle
  NailGroup-7_Kaggle_description.md         the description alone, ready to paste into Kaggle (same text as README.md in the zip)
  cover_image.png                           1120x560 cover image for the Kaggle page (not part of the dataset)
Every count in the generated text comes from data/ and results/ (flow.json, dev_components.csv, folds, internal_summary.csv,
results/macros*.tex); the few hand-written facts (source list, licence notes) are the constants SOURCES and LICENSES below.
(The earlier "manifest only" variant without images was dropped on 2026-11-18: the authors publish the images.)
"""
import argparse
import hashlib
import io
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nailrev.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TITLE = "NailGroup-7: 7-class nail photos"   # Kaggle: 6-50 characters (author decision 2026-11-18: no ", grouped")
SUBTITLE = "Curated nail-disease photos with near-duplicate groups and leakage-free CV folds"   # 20-80 characters
SLUG = "nailgroup-7"
KAGGLE_USER = "jamaluddintanvin"                    # dataset URL: https://www.kaggle.com/datasets/jamaluddintanvin/nailgroup-7
GUIDE_NAME = "NailGroup-7_Kaggle_upload.md"
DESC_NAME = "NailGroup-7_Kaggle_description.md"     # paste-ready description only (= README.md in the zip = guide Section 8)
OLD_OUTPUTS = ("KAGGLE_UPLOAD_MANUAL.md", "UPLOAD_STEPS.md")   # replaced by GUIDE_NAME
KEYWORDS = ["healthcare", "dermatology", "image", "classification", "medical", "data cleaning"]
NAMES = {"beau_s": "Beau's lines", "black_line": "Black line", "clubbing": "Clubbing", "healthy_nail": "Healthy nail",
         "onychomycosis": "Onychomycosis", "onycholysis": "Onycholysis", "white_spot": "White spot"}
CLASS_NOTE = {
    "beau_s": "transverse grooves across the nail plate; pictures that an upstream set labels 'pitting' were excluded",
    "black_line": "longitudinal dark band (longitudinal melanonychia); includes melanoma-sourced pictures (tag `melanoma_sourced`)",
    "clubbing": "bulbous fingertip with increased nail curvature",
    "healthy_nail": "no visible nail abnormality",
    "onychomycosis": "fungal nail infection (thickening, discolouration, crumbling)",
    "onycholysis": "separation of the nail plate from the nail bed",
    "white_spot": "white spots or bands in the nail plate (leukonychia)"}
MAC = {k: v for f in sorted((ROOT / "results").glob("macros*.tex"))
       for k, v in re.findall(r"\\newcommand\{\\R([A-Za-z]+)\}\{([^}]*)\}", f.read_text())}

# column dictionary: name -> (Kaggle field type, description). Also written into dataset-metadata.json (Kaggle data card).
COLUMNS = {
    "file": ("string", "Path of the image inside this dataset, `images/<class>/<name>.jpg`. Image files exist only for rows with "
                       "`status = included`."),
    "image_id": ("string", "Stable identifier of the file (used in the paper's code and result files)."),
    "class": ("string", "Class label = image folder: beau_s, black_line, clubbing, healthy_nail, onychomycosis, onycholysis, white_spot."),
    "original_class": ("string", "Class folder name in the original Roboflow export."),
    "original_path": ("string", "Path of the file in the original export (`<split>/<class>/<name>`)."),
    "roboflow_split": ("string", "train/valid/test split of the original export. Not group-disjoint: do not use it for evaluation."),
    "stem": ("string", "Filename stem shared by the copies of one upload (Roboflow appends `.rf.<hash>` to each copy)."),
    "component_id": ("string", "Near-duplicate group = one independent source picture. Keep every component in a single partition."),
    "component_files": ("integer", "Number of files in the component."),
    "component_stems": ("integer", "Number of distinct filename stems in the component (above 1 = re-uploads under another name were merged)."),
    "is_representative": ("integer", "1 for one fixed image per component: the image used for validation and testing."),
    "status": ("string", "`included` (image provided) or `excluded` (listed for transparency only)."),
    "exclusion_reason": ("string", "Why the file is excluded; empty when included."),
    "tags": ("string", "`melanoma_sourced`: an upstream dataset labels the picture as acral lentiginous / subungual melanoma."),
    "upstream_links": ("string", "Public datasets (`;`-separated) in which a near-duplicate of the component was found."),
    "fold_r0": ("integer", "Cross-validation fold (0-4) of the component in repeat 0; empty when excluded."),
    "fold_r1": ("integer", "Fold (0-4) in repeat 1."),
    "fold_r2": ("integer", "Fold (0-4) in repeat 2."),
    "width": ("integer", "Image width in pixels."),
    "height": ("integer", "Image height in pixels."),
    "sha256": ("string", "SHA-256 checksum of the original file bytes."),
    "md5_file": ("string", "MD5 checksum of the original file bytes."),
}
COMP_COLUMNS = {
    "component_id": ("string", "Near-duplicate group (one independent source picture)."),
    "class": ("string", "Class label; components excluded for `label_conflict` list all their labels, separated by a vertical bar."),
    "status": ("string", "`included` or `excluded`."),
    "exclusion_reason": ("string", "Why the component is excluded; empty when included."),
    "files": ("integer", "Number of files in the component."),
    "stems": ("integer", "Number of distinct filename stems in the component."),
    "upstream_links": ("string", "Public datasets in which a near-duplicate was found (`;`-separated)."),
    "tags": ("string", "`melanoma_sourced` or empty."),
    "representative_file": ("string", "Path of the component's representative image."),
}


def sha256_md5(zf: zipfile.ZipFile, member: str) -> tuple[str, str, bytes]:
    b = zf.read(member)
    return hashlib.sha256(b).hexdigest(), hashlib.md5(b).hexdigest(), b  # noqa: S324 (checksum, not security)


def build_manifest(cfg, zf) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    idx = pd.read_csv(cfg.path("manifests") / "index.csv", keep_default_na=False)
    comp = pd.read_csv(cfg.path("dedup") / "components.csv").set_index("image_id").comp
    dev = pd.read_csv(cfg.path("splits") / "dev_components.csv", keep_default_na=False).set_index("comp")
    k = idx[idx.source == "kopin"].copy()
    k["component_id"] = k.image_id.map(comp)
    sizes = k.groupby("component_id").agg(component_files=("image_id", "size"), component_stems=("stem", "nunique"))
    k = k.join(sizes, on="component_id")
    k["status"] = k.component_id.map(dev.status)
    k["exclusion_reason"] = k.component_id.map(dev.reason)
    k["tags"] = k.component_id.map(dev.tags)
    k["upstream_links"] = k.component_id.map(dev.linked_sources)
    k["is_representative"] = (k.image_id == k.component_id.map(dev.rep_image_id)).astype(int)
    for r in range(cfg.splits["n_repeats"]):
        f = pd.read_csv(cfg.path("splits") / f"folds_r{r}.csv").set_index("comp").fold
        k[f"fold_r{r}"] = k.component_id.map(f).astype("Int64")
    k["class"] = k.label
    k["original_class"] = k.raw_label
    k["file"] = "images/" + k["class"] + "/" + k.ref.str.rsplit("/", n=1).str[-1]
    sha, md5 = [], []
    for ref in k.ref:
        s, m, _ = sha256_md5(zf, ref)
        sha.append(s)
        md5.append(m)
    k["sha256"], k["md5_file"] = sha, md5
    k = k.rename(columns={"ref": "original_path", "vendor_split": "roboflow_split"})
    cols = ["file", "image_id", "class", "original_class", "original_path", "roboflow_split", "stem", "component_id",
            "component_files", "component_stems", "is_representative", "status", "exclusion_reason", "tags", "upstream_links",
            *[f"fold_r{r}" for r in range(cfg.splits["n_repeats"])], "width", "height", "sha256", "md5_file"]
    man = k[cols].reset_index(drop=True)
    inc = man[man.status == "included"]
    comps = (man.groupby("component_id").agg(**{"class": ("class", lambda s: "|".join(sorted(set(s)))), "status": ("status", "first"),
                                                 "exclusion_reason": ("exclusion_reason", "first"), "files": ("image_id", "size"),
                                                 "stems": ("stem", "nunique"), "upstream_links": ("upstream_links", "first"),
                                                 "tags": ("tags", "first")}).reset_index())
    comps["representative_file"] = comps.component_id.map(man[man.is_representative == 1].set_index("component_id").file)
    stats = {"files": len(man), "included_files": len(inc), "excluded_files": len(man) - len(inc),
             "components": man.component_id.nunique(), "included_components": inc.component_id.nunique(),
             "stem_groups": int(man.groupby(["class", "stem"]).ngroups),
             "excluded_reasons": man[man.status != "included"].exclusion_reason.value_counts().to_dict(),
             "excluded_components": man[man.status != "included"].component_id.nunique(),
             "melanoma_files": int(inc.tags.str.contains("melanoma_sourced").sum()),
             "melanoma_components": int(inc[inc.tags.str.contains("melanoma_sourced")].component_id.nunique())}
    return man, comps, stats


def md(df: pd.DataFrame) -> str:
    """Markdown table; integers with thousands separators."""
    df = df.reset_index()
    h = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join(["---"] * len(df.columns)) + "|\n"
    return h + "\n".join("| " + " | ".join(f"{v:,}" if isinstance(v, (int, np.integer)) else str(v) for v in r) + " |"
                         for r in df.to_numpy())


def benchmark_table(cfg) -> str:
    f = cfg.path("results") / "metrics" / "internal_summary.csv"
    if not f.exists():
        return "_Benchmark results are added when the final analysis (`results/`) exists; re-run `scripts/s14_kaggle_package.py`._"
    d = pd.read_csv(f)
    d = d[d.set == "test"]
    rows = ["| Model | Group-disjoint CV: macro-F1 [95% CI] | Group-disjoint CV: balanced accuracy | Group-disjoint CV: AUROC "
            "| Image-level split: macro-F1 [95% CI] |", "|---|---|---|---|---|"]
    for m in cfg.models:
        cv, lk = d[(d.model == m) & (d.protocol == "cv")], d[(d.model == m) & (d.protocol == "leaky")]
        ci = lambda r: f"{r.macro_f1.iloc[0]:.3f} [{r.macro_f1_lo.iloc[0]:.3f}, {r.macro_f1_hi.iloc[0]:.3f}]" if len(r) else "n/a"  # noqa: E731
        rows.append(f"| {cfg.models[m]['label']} | {ci(cv)} | {cv.bal_acc.iloc[0]:.3f} | {cv.auroc_runs_mean.iloc[0]:.3f} | {ci(lk)} |")
    return "\n".join(rows)


SOURCES = """Status: our own analysis, November 2026. "Declared licence" is what the hosting page stated; it does not prove the uploader owned
the rights. Overlap statements come from near-duplicate matching (perceptual hash + SSCD) and are **lower bounds**.

### The collection itself
- Roboflow Universe `ko-pin/dataset-7clxx`, version 1, exported 2024-02-17 (43,144 images, folder format, resize 640x640 stretch,
  3x augmentation: 90-degree rotations, +/-15 degree rotation, +/-10 degree shear). Declared licence CC BY 4.0. Uploader and
  description unknown; the project could not be retrieved in November 2026 (HTTP 403, no web-archive snapshot).
- It is a secondary aggregation of public collections (below); roughly 54% of its source-picture groups could be attributed.

### Identified upstream sources
| Upstream | Link | Declared licence | What it contributes |
|---|---|---|---|
| Han et al. 2018 (Asan / Inje / Hallym / Seoul National University hospitals) | https://doi.org/10.1371/journal.pone.0191493 ; Figshare https://doi.org/10.6084/m9.figshare.5398573 | CC BY 4.0 | about 549 of 1,042 onychomycosis groups (validation sets B1, B2, C, D); about 211 more named after the A2 set |
| Thai "Nails Images Dataset" (Lapthanachai et al., J. Applied Informatics and Technology 2023) | https://www.kaggle.com/datasets/nuttidalapthanachai/nails-new-test ; https://doi.org/10.14456/jait.2023.2 | not stated on Kaggle | the class names black line / white spot / Beau's line; ~100 images each |
| Thai set, Kaggle arthitaya | https://www.kaggle.com/datasets/arthitaya/nail-dataset ; https://www.kaggle.com/datasets/arthitaya/nail-dataset2 | not stated | clubbing (59/60 filenames), some Beau's lines and onycholysis |
| Roboflow KNM "Nail Disease Detection" | https://universe.roboflow.com/knm/nail-disease-detection-mxoqy | CC BY 4.0 | acral lentiginous melanoma pictures (-> black_line), "pitting" pictures (-> beau_s, **excluded here**), clubbing, healthy |
| Kaggle nikhilgurav21 / naeun1 (KNM re-exports) | https://www.kaggle.com/datasets/nikhilgurav21/nail-disease-detection-dataset ; https://www.kaggle.com/datasets/naeun1/nail-dataset-disease | CC BY-NC-SA 4.0 (nikhilgurav21); not stated (naeun1) | same pictures as KNM |
| Kaggle 17-class "Nail Disease Dataset" | https://www.kaggle.com/datasets/saisrushikgovindgari/nail-disease-dataset ; earlier copies chengcris/nail-disease, hunterimam/nail-diseases | MIT (re-upload); not stated (earlier copies) | Beau's lines, clubbing, onycholysis, leukonychia/white nail; "splinter haemorrhage" and "alopecia areata" pictures filed under black line / white spot (**excluded here**) |
| Kaggle josephrasanjana | https://www.kaggle.com/datasets/josephrasanjana/nail-disease-image-classification-dataset | not stated | onychomycosis (DermNet-watermarked and web images) and healthy nails (2023 phone photos) |
| Unidentified | - | - | about 1,300 source groups with letter-coded names (mostly onycholysis and white spot), 91 healthy groups named `20120206_index_right_*`, 470 screenshot-named groups (21 dated 2024-01-31..02-09, the uploader's own) |

Per-file provenance is in `manifest.csv` (`upstream_links`); per-class tracing rates are in Section 5.6.

### Class-label conflicts between upstreams (why exclusions exist)
- "Pitting" (nikhilgurav21) vs Beau's lines (ko-pin): ~200 pictures.
- "Acral lentiginous melanoma" (nikhilgurav21) vs black line (ko-pin): kept, tagged `melanoma_sourced` (safety).
- "Splinter haemorrhage" (17-class) vs black line (Thai set): excluded. "Alopecia areata" (17-class) vs white spot: excluded.
"""

LICENSES = """This is a description of the situation, not legal advice.

1. **Declared licence of the original export:** CC BY 4.0 (Roboflow Universe `ko-pin/dataset-7clxx`). CC BY 4.0 lets you share and adapt
   with attribution and an indication of changes. A licence can only grant rights the licensor holds.
2. **The images come from many places.** Upstream licences are mixed (CC BY 4.0, CC BY-NC-SA 4.0, MIT, "not stated" on Kaggle),
   and the collection contains web screenshots, 5 DermNet-watermarked files (DermNet images are under DermNet's own restrictive licence:
   attribution, non-commercial, no derivatives) and 2 publisher figures. Some images may be in-copyright material the uploader had no
   right to license.
3. **What we license:** the metadata we created (manifest, component and fold assignments, provenance tags, documentation) is released
   under CC BY 4.0. We make no claim over the images.
4. **Recommended use:** research and education, non-commercial, with attribution to the sources in Section 10. Do not assume commercial
   rights. If you are a rights holder and want an image removed, contact the dataset owner on Kaggle; it will be removed in the next version.

**Required attribution:** "NailGroup-7" is derived from Roboflow Universe ko-pin/dataset-7clxx (v1, CC BY 4.0 as declared), which in turn
aggregates the sources listed in Section 10. Changes: near-duplicate grouping, exclusion of conflicting labels, fold assignments;
image content unmodified.
"""

HOWTO = r'''### 6.1 Get the files
- **Kaggle notebook:** *Add Input* (right panel) > search "NailGroup-7" > *Add*. The files appear read-only under `/kaggle/input/`.
- **Browser:** the *Download* button on this page gives `archive.zip`; unzip it.
- **Kaggle command line** (API token: Kaggle > Settings > API > *Create New Token*, saved as `~/.kaggle/kaggle.json`):
```bash
pip install kaggle
kaggle datasets download @@HANDLE@@ -p nailgroup-7 --unzip
```
- **kagglehub (Python):**
```python
import kagglehub
path = kagglehub.dataset_download("@@HANDLE@@")   # local folder with the files
```

### 6.2 Load the manifest
`manifest.csv` is the entry point: one row per file, with label, group, folds and image path.
```python
from pathlib import Path
import pandas as pd

BASE = Path("/kaggle/input")         # Kaggle notebook; on your computer: the unzipped folder or the kagglehub `path`
ROOT = next(p.parent for p in BASE.rglob("representatives.csv"))  # the folder with manifest.csv and images/

m = pd.read_csv(ROOT / "manifest.csv")
data = m[m.status == "included"].copy()                            # rows that have an image file
for r in range(3):
    data[f"fold_r{r}"] = data[f"fold_r{r}"].astype(int)
data["path"] = [str(ROOT / f) for f in data.file]
CLASSES = sorted(data["class"].unique())
print(len(data), "images,", data.component_id.nunique(), "independent pictures")    # @@NINC@@ images, @@NCOMP@@ independent pictures
print(data.groupby("class").component_id.nunique())                                   # pictures per class
```

### 6.3 Leakage-free cross-validation (the protocol of the paper)
Every component (source picture) lies in exactly one of five folds; `fold_r0`, `fold_r1` and `fold_r2` are three independent
repeats. Run (repeat, k) tests on fold k, validates on fold k+1 (mod 5) and trains on the other three folds. Training uses all files of
the training components (the copies act as extra augmentation); validation and testing use one image per component.
```python
def split(data, repeat=0, k=0):
    f = data[f"fold_r{repeat}"]
    train = data[~f.isin([k, (k + 1) % 5])]
    val = data[(f == (k + 1) % 5) & (data.is_representative == 1)]
    test = data[(f == k) & (data.is_representative == 1)]
    assert not set(train.component_id) & (set(val.component_id) | set(test.component_id))   # no picture on two sides
    return train, val, test

train, val, test = split(data, repeat=0, k=0)
print(len(train), len(val), len(test))     # @@SPLITSIZES@@
# full protocol: for repeat in range(3): for k in range(5): ...  -> report mean and SD over the 15 runs
```
For a single quick experiment, use `split(data, 0, 0)`. To make your own split, assign whole components, for example:
```python
from sklearn.model_selection import StratifiedKFold
comps = data[data.is_representative == 1]                        # one row per component
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_of = {c: i for i, (_, te) in enumerate(skf.split(comps, comps["class"])) for c in comps.component_id.iloc[te]}
data["my_fold"] = data.component_id.map(fold_of)                  # every file inherits the fold of its component
```

### 6.4 Train a model (PyTorch)
The sampler below draws each class equally often, each picture equally often within its class and each copy equally often within its
picture, so that large components and large classes do not dominate training (as in the paper).
```python
import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])           # ImageNet statistics
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(0.85, 1.18)), T.RandomHorizontalFlip(),
                      T.RandomVerticalFlip(), T.RandomChoice([T.RandomRotation((a, a)) for a in (0, 90, 180, 270)]),
                      T.ColorJitter(0.2, 0.2, 0.15, 0.02), T.ToTensor(), NORM])
eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), NORM])


class NailGroup7(Dataset):
    def __init__(self, df, tf):
        self.paths, self.y, self.tf = list(df.path), [CLASSES.index(c) for c in df["class"]], tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("RGB")), self.y[i]


w = 1.0 / (train.component_id.map(train.component_id.value_counts())
           * train["class"].map(train.groupby("class").component_id.nunique()))
sampler = WeightedRandomSampler(torch.tensor(w.to_numpy(), dtype=torch.double), num_samples=4 * train.component_id.nunique())
train_dl = DataLoader(NailGroup7(train, train_tf), batch_size=64, sampler=sampler, num_workers=4)
val_dl = DataLoader(NailGroup7(val, eval_tf), batch_size=128, num_workers=4)
test_dl = DataLoader(NailGroup7(test, eval_tf), batch_size=128, num_workers=4)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = timm.create_model("mobilenetv3_large_100", pretrained=True, num_classes=len(CLASSES)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
for epoch in range(10):                         # the paper: up to 40 epochs, early stopping on val_dl
    model.train()
    for x, y in train_dl:
        opt.zero_grad()
        loss_fn(model(x.to(device)), y.to(device)).backward()
        opt.step()
```

### 6.5 Evaluate
```python
import numpy as np
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score, roc_auc_score


@torch.no_grad()
def predict(model, loader):
    model.eval()
    p, t = zip(*[(model(x.to(device)).softmax(1).cpu(), y) for x, y in loader])
    return torch.cat(p).numpy(), torch.cat(t).numpy()


prob, y = predict(model, test_dl)
pred = prob.argmax(1)
print(f"macro-F1 {f1_score(y, pred, average='macro'):.3f}  balanced accuracy {balanced_accuracy_score(y, pred):.3f}  "
      f"AUROC {roc_auc_score(y, prob, multi_class='ovr'):.3f}")
print(classification_report(y, pred, target_names=CLASSES, digits=3))
```

### 6.6 Rules that keep results honest
- **Never split by file.** `ImageFolder` + `random_split`, `train_test_split` on the file list or Keras
  `image_dataset_from_directory(validation_split=...)` put copies of one picture into training and test data. In the paper such an
  image-level split had a near-duplicate in training for @@DUPPCT@@% of the test images and raised macro-F1 from @@CVMIN@@-@@CVMAX@@ to
  @@LKMIN@@-@@LKMAX@@ (Section 7).
- **Do not use `roboflow_split` either:** @@SPANNING@@ components have files in more than one of its splits.
- **Score one image per component** (`is_representative == 1`); otherwise large components (up to @@LARGEST@@ files) dominate.
- **Report macro-F1 or balanced accuracy and per-class results:** the classes differ up to @@IMBAL@@-fold in independent pictures.
- **Files with `status = excluded`** are listed for transparency; their images are not part of the dataset.
- **External testing:** first remove near-duplicates of NailGroup-7 from the external set. `upstream_links` names public datasets
  that already share pictures with NailGroup-7 (for example the Han et al. hospital sets), so those are not independent test data.
- **Safety:** Black line contains melanoma-sourced pictures; a model trained here must never be used to rule out melanoma.

### 6.7 Check the files (optional)
```python
import hashlib
bad = [p for p, h in zip(data.path, data.sha256) if hashlib.sha256(Path(p).read_bytes()).hexdigest() != h]
print(len(bad), "files differ from the original export")       # expected: 0
```
'''


def coltable(cols: dict) -> str:
    return "| Column | Type | Meaning |\n|---|---|---|\n" + "\n".join(f"| `{k}` | {t} | {d} |" for k, (t, d) in cols.items())


def file_info(cfg, man: pd.DataFrame, st: dict) -> dict:
    """Kaggle "About this file" texts (also written into dataset-metadata.json)."""
    inc = man[man.status == "included"]
    per_class = ", ".join(f"{c} {n:,}" for c, n in inc["class"].value_counts().reindex(cfg.classes).items())
    sizes = ", ".join(f"{w}x{h}" for w, h in man[["width", "height"]].drop_duplicates().itertuples(index=False))
    return {
        "README.md": "Complete documentation of NailGroup-7 (same text as the dataset description): files, classes, column dictionary, "
                     "statistics, how to use the data (loading, leakage-free cross-validation, PyTorch example, evaluation), reference "
                     "results, curation, limitations and safety, sources, licence and citation.",
        "manifest.csv": f"Start here. One row per file of the original Roboflow export ({len(man):,} rows: {len(inc):,} included with an "
                        f"image, {len(man) - len(inc):,} excluded and listed for transparency). Columns: image path, class, near-duplicate "
                        "component, representative flag, inclusion status and reason, safety tag, provenance links, cross-validation "
                        "folds of three repeats, image size and checksums.",
        "components.csv": f"One row per near-duplicate component ({st['components']:,} rows, {st['included_components']:,} included). A "
                          "component is one independent source picture with all its copies; never split a component between training "
                          "and test data.",
        "representatives.csv": f"One image per included component ({st['included_components']:,} rows; same columns as manifest.csv): a "
                               "de-duplicated version of the dataset, used for validation and testing.",
        "dataset-metadata.json": "Kaggle metadata (title, subtitle, licence, keywords, file and column descriptions) used for uploads with "
                                 "the Kaggle API.",
        "images/": f"{len(inc):,} JPEG images ({sizes} pixels, original bytes of the export), one sub-folder per class: {per_class}. "
                   "Use manifest.csv for labels, groups and folds.",
    }


def card(cfg, man: pd.DataFrame, st: dict, flow: dict, user: str | None) -> str:
    """The complete dataset card (README.md of the dataset = Kaggle description). Numbers come from the manifest and results/."""
    inc = man[man.status == "included"].copy()
    exc = man[man.status != "included"]
    per_comp = inc.groupby("component_id").size()
    thr = json.loads((cfg.path("dedup") / "threshold.json").read_text())
    handle = f"{user or 'OWNER'}/{SLUG}"
    url = f"https://www.kaggle.com/datasets/{handle}"
    sizes = ", ".join(f"{w}x{h}" for w, h in man[["width", "height"]].drop_duplicates().itertuples(index=False))

    # classes
    pc = man.assign(inc=man.status == "included").groupby("class").agg(original_files=("image_id", "size"), included=("inc", "sum"))
    pc["excluded"] = pc.original_files - pc.included
    pc["pictures"] = inc.groupby("class").component_id.nunique()
    pc["median"] = inc.groupby(["class", "component_id"]).size().groupby("class").median().round(0).astype(int)
    pc["largest"] = inc.groupby(["class", "component_id"]).size().groupby("class").max()
    pc = pc.reindex(cfg.classes).astype(int)
    cls_rows = "\n".join(
        f"| `{c}` | {NAMES[c]} | {CLASS_NOTE[c]} | {r.included:,} | {r.pictures:,} | {100 * r.pictures / pc.pictures.sum():.1f}% |"
        for c, r in pc.iterrows())
    cls_stats = pc.rename(columns={"original_files": "files in original export", "included": "included files",
                                   "excluded": "excluded files", "pictures": "included components",
                                   "median": "median files per component", "largest": "largest component"}).rename_axis("class")
    cls_stats.loc["TOTAL"] = [len(man), len(inc), len(exc), st["included_components"], int(per_comp.median()), int(per_comp.max())]

    # tree of the files
    tree = [f"{SLUG}/", "├── README.md               this description",
            f"├── manifest.csv            {len(man):,} rows: every file of the original export (included and excluded)",
            f"├── components.csv          {st['components']:,} rows: one per near-duplicate component",
            f"├── representatives.csv     {st['included_components']:,} rows: one image per included component",
            "├── dataset-metadata.json   Kaggle metadata (title, licence, file and column descriptions)",
            f"└── images/                 {len(inc):,} JPEG files, one folder per class"]
    for i, c in enumerate(cfg.classes):
        tree.append(f"    {'└──' if i == len(cfg.classes) - 1 else '├──'} {c + '/':<18}{pc.loc[c, 'included']:>6,} images")
    tree = "\n".join(tree)

    # statistics
    overview = pd.DataFrame({"value": {
        "files in the original export": f"{len(man):,}",
        "distinct source-picture groups (class + filename stem)": f"{st['stem_groups']:,}",
        "near-duplicate components found": f"{st['components']:,}",
        "included files / components": f"{len(inc):,} / {st['included_components']:,}",
        "excluded files / components": f"{st['excluded_files']:,} / {st['excluded_components']:,}",
        "files per included component: median / largest": f"{per_comp.median():.0f} / {per_comp.max():,}",
        "included components with more than one filename stem (merged re-uploads)":
            f"{int((inc.drop_duplicates('component_id').component_stems > 1).sum()):,}",
        "components with files in more than one original Roboflow split": f"{flow.get('components_spanning_vendor_splits', 'n/a'):,}",
        "components with files in all three original Roboflow splits": f"{flow.get('components_in_all_three_vendor_splits', 'n/a'):,}",
        "image size (width x height)": sizes,
        "representative images (one per included component)": f"{int(inc.is_representative.sum()):,}"}}).rename_axis("item")
    ex = exc.groupby("exclusion_reason").agg(files=("image_id", "size"), components=("component_id", "nunique"))
    ex["meaning"] = ex.index.map(lambda r: {
        "label_conflict": "the same picture (or a near-duplicate) carries different class labels",
        "same_hospital_as_HanA1": "linked to the Han et al. A2 hospital collection; removed so that Han A1 is a hospital-disjoint external test set",
        "label_policy:nikhilgurav21:pitting": "an upstream dataset labels these pictures 'pitting'; filed under Beau's lines in the export",
        "label_policy:kaggle17:splinter hemmorrage": "an upstream dataset labels these 'splinter haemorrhage'; filed under black line in the export",
        "label_policy:kaggle17:aloperia areata": "an upstream dataset labels these 'alopecia areata'; filed under white spot in the export"}.get(r, ""))
    ex = ex.sort_values("files", ascending=False)
    rf_split = inc.groupby(["class", "roboflow_split"]).size().unstack(fill_value=0).reindex(cfg.classes)
    for r in range(cfg.splits["n_repeats"]):
        inc[f"fold_r{r}"] = inc[f"fold_r{r}"].astype(int)
    fold_c = inc.drop_duplicates("component_id").groupby(["class", "fold_r0"]).size().unstack(fill_value=0).reindex(cfg.classes)
    fold_i = inc.groupby(["class", "fold_r0"]).size().unstack(fill_value=0).reindex(cfg.classes)
    for t in (fold_c, fold_i):
        t.columns = [f"fold {c}" for c in t.columns]
        t.loc["total"] = t.sum()
    cc = inc.drop_duplicates("component_id").assign(traced=lambda d: d.upstream_links.fillna("") != "")
    pv = cc.groupby("class").agg(components=("component_id", "size"), traced=("traced", "sum")).reindex(cfg.classes)
    pv["traced_share"] = (pv.traced / pv.components).map(lambda v: f"{v:.0%}")
    pv["traced"] = pv.traced.astype(int)
    up = (cc.upstream_links.fillna("").str.split(";").explode().loc[lambda s: s != ""].value_counts()
          .rename("components").to_frame().rename_axis("upstream dataset"))

    # numbers for the how-to section
    f0 = inc.fold_r0
    tr = inc[~f0.isin([0, 1])]
    va = inc[(f0 == 1) & (inc.is_representative == 1)]
    te = inc[(f0 == 0) & (inc.is_representative == 1)]
    howto = (HOWTO.replace("@@HANDLE@@", handle)
             .replace("@@NINC@@", f"{len(inc)}").replace("@@NCOMP@@", f"{st['included_components']}")
             .replace("@@SPLITSIZES@@", f"{len(tr)} {len(va)} {len(te)}")
             .replace("@@DUPPCT@@", MAC.get("LeakyTestDupPct", "n/a")).replace("@@CVMIN@@", MAC.get("CvMacroFOneMin", "n/a"))
             .replace("@@CVMAX@@", MAC.get("CvMacroFOneMax", "n/a")).replace("@@LKMIN@@", MAC.get("LeakyMacroFOneMin", "n/a"))
             .replace("@@LKMAX@@", MAC.get("LeakyMacroFOneMax", "n/a"))
             .replace("@@SPANNING@@", f"{flow.get('components_spanning_vendor_splits', 'n/a'):,}")
             .replace("@@LARGEST@@", f"{per_comp.max():,}").replace("@@IMBAL@@", f"{pc.pictures.max() / pc.pictures.min():.1f}"))
    assert "@@" not in howto, re.findall(r"@@\w+@@", howto)

    mc = cfg.models
    sz = sorted({v["size"] for v in mc.values()})
    tc = cfg.train
    ext_line = ""
    if "HanAoneMacroFOneMin" in MAC:
        ext_line = (f"\nTrained on NailGroup-7 only, the same models reached macro-F1 {MAC['HanAoneMacroFOneMin']}-"
                    f"{MAC['HanAoneMacroFOneMax']} on the independent hospital dataset of Han et al. (set A1, "
                    f"{int(MAC.get('ExtHanAoneN', 0)):,} images): internal scores do not carry over to images from other sources, "
                    "so test on external data before drawing conclusions.\n")
    melan = f"{st['melanoma_files']:,} files ({st['melanoma_components']:,} pictures)"
    paper = "*Interpretable Nail Disease Classification via XAI and Social Network Analysis* (Healthcare Technology Letters, under review)"

    return f"""# {TITLE}

*{SUBTITLE}*

**NailGroup-7** contains **{len(inc):,} nail photographs in seven classes** (Beau's lines, Black line, Clubbing, Healthy nail,
Onychomycosis, Onycholysis, White spot) that come from **{st['included_components']:,} independent source pictures**. It is a curated,
leakage-controlled re-release of the public Roboflow Universe collection `ko-pin/dataset-7clxx` (version 1, exported 2024-02-17,
declared CC BY 4.0). We did **not** create the images. We grouped the near-duplicate copies of every picture, removed pictures with
conflicting labels, and added group identifiers, ready-made cross-validation folds, provenance tags and checksums, so that models can
be trained and evaluated without train/test leakage. NailGroup-7 is the development dataset of the paper {paper}.

> **Not for diagnosis.** Not a medical device or clinical reference. Labels are those of the original uploaders and were not verified
> by a clinician; the images carry no patient identifiers. Read *Known limitations and safety* (Section 9) before use.

## At a glance
| Item | Value |
|---|---|
| Task | single-label image classification, 7 classes |
| Images | {len(inc):,} JPEG photographs, {sizes} pixels, RGB (original bytes of the export) |
| Independent pictures | {st['included_components']:,} near-duplicate components (median {per_comp.median():.0f}, largest {per_comp.max():,} files each) |
| Ready-made evaluation | stratified, group-disjoint 5-fold cross-validation, 3 repeats (`fold_r0`-`fold_r2`); score one image per component |
| Metadata | `manifest.csv` (all {len(man):,} files of the original export, incl. {len(exc):,} excluded), `components.csv`, `representatives.csv` |
| Original source | Roboflow Universe `ko-pin/dataset-7clxx` v1 (no longer publicly accessible) |
| Licence | images: mixed upstream licences (Section 11); our added metadata: CC BY 4.0 |
| Version | 1.0 (November 2026) |

## Contents of this page
1. Why this dataset exists
2. Files
3. Classes
4. Column dictionary
5. Statistics
6. How to use the dataset after downloading
7. Reference results
8. How the dataset was curated
9. Known limitations and safety
10. Sources and provenance
11. Licence and attribution
12. Citation, credits and versions

## 1. Why this dataset exists
The original export contains **{len(man):,} files but only {st['stem_groups']:,} distinct source-picture groups** (class + filename
stem), and fewer independent pictures still: the upload was pre-augmented, and the Roboflow export added three rotated or sheared
copies of every training picture. A random split by file therefore puts copies of the same picture into training and test data. In
our experiments, {MAC.get('LeakyTestDupPct', 'n/a')}% of the test images of such a split had a near-duplicate in training, and the
macro-F1 of five standard models rose from {MAC.get('CvMacroFOneMin', 'n/a')}-{MAC.get('CvMacroFOneMax', 'n/a')} (leakage-free) to
{MAC.get('LeakyMacroFOneMin', 'n/a')}-{MAC.get('LeakyMacroFOneMax', 'n/a')}. Even the export's own train/valid/test split is not
leakage-free ({flow.get('components_spanning_vendor_splits', 0):,} pictures have copies in more than one of its splits). NailGroup-7
gives every file a **`component_id`** (one near-duplicate group = one source picture) and **`fold_r0`-`fold_r2`** assignments that
keep every component in a single fold.

## 2. Files
```
{tree}
```
- Only included files have an image; rows with `status = excluded` document what was removed and why.
- Images are the unmodified original files ({sizes}, stretched to a square by the original export); resize them as your model needs.

## 3. Classes
| Folder (`class`) | Class | Description | Images | Independent pictures | Share of pictures |
|---|---|---|---|---|---|
{cls_rows}
| **Total** | | | **{len(inc):,}** | **{st['included_components']:,}** | 100% |

Image counts overstate how much a class really contains: count independent pictures (components), not files. For example,
`beau_s` has {pc.loc['beau_s', 'included']:,} images but only {pc.loc['beau_s', 'pictures']:,} pictures, while `onycholysis` has
{pc.loc['onycholysis', 'pictures']:,}.

## 4. Column dictionary
**`manifest.csv`** - one row per file of the original export ({len(man):,} rows):

{coltable(COLUMNS)}

**`components.csv`** - one row per near-duplicate component ({st['components']:,} rows):

{coltable(COMP_COLUMNS)}

**`representatives.csv`** - the {st['included_components']:,} rows of `manifest.csv` with `status = included` and
`is_representative = 1` (one image per independent picture; a de-duplicated version of the dataset).

## 5. Statistics
Every number below is computed from `manifest.csv`.

### 5.1 Overview
{md(overview)}

### 5.2 Per class
{md(cls_stats)}

### 5.3 Exclusions
{md(ex)}

Excluded files stay in `manifest.csv` with their reason; their images are not provided.

### 5.4 Included files by original Roboflow split
{md(rf_split.rename_axis("class"))}

Copies of one picture often sit in different original splits, which is why `roboflow_split` must not be used for evaluation.

### 5.5 Folds (repeat 0; repeats 1 and 2 are analogous)
Independent pictures (components) per fold:

{md(fold_c.rename_axis("class"))}

Images per fold (all files of the components):

{md(fold_i.rename_axis("class"))}

### 5.6 Provenance
Share of included pictures with at least one near-duplicate in a named public dataset that we downloaded and compared (a **lower
bound**; untraced does not mean original). It differs from the roughly 54% in Section 10, which attributes source-picture groups of
the whole export by filename and description.

{md(pv.rename_axis("class"))}

Pictures linked to each public dataset (one picture can link to several). `han_*` = Han et al. 2018 hospital data; `arthitaya*`,
`nuttida` = Thai Kaggle sets; `legacy2272` = a combined set of {int(MAC.get('LegacyImages', 0)):,} images from Kaggle nikhilgurav21 and Roboflow mohit (the latter
re-uses the 17-class and josephrasanjana sets), so it is not an independent source:

{md(up)}

### 5.7 Safety flag
`black_line` contains {melan} tagged `melanoma_sourced` (upstream datasets label them acral lentiginous / subungual melanoma). They are
kept and tagged in `manifest.csv`.

## 6. How to use the dataset after downloading
{howto}
## 7. Reference results
Five ImageNet-pretrained models (timm) fine-tuned on the folds of Section 6.3 (15 runs: 3 repeats x 5 folds; 95% confidence intervals
by bootstrap over test pictures), compared with a conventional random 70/15/15 split by file:

{benchmark_table(cfg)}

Settings: input {sz[0]} px ({", ".join(f"{v['label']} {v['size']} px" for v in mc.values() if v['size'] != sz[0])}), AdamW,
component-balanced sampling, augmentation (random resized crop, flips, 90-degree rotations, colour jitter), label smoothing
{tc['label_smoothing']}, early stopping on the validation fold (at most {tc['max_epochs']} epochs). The gap between the first and the
last column is the leakage effect.
{ext_line}
## 8. How the dataset was curated
1. **Near-duplicate components.** Two files are linked if they share a filename stem within a class, are byte-identical, have a
   perceptual-hash distance <= {thr['phash_threshold']} (minimum over the 8 rotations/flips), or an SSCD copy-detection embedding
   similarity >= {thr['sscd_threshold']} (also maximised over the 8 rotations/flips). Connected components of this graph are the units
   of independence ({st['components']:,} found). The SSCD threshold was set from a manual audit of 150 candidate pairs (precision 100%
   above {thr['sscd_threshold']} in the audit; lower thresholds chained look-alike pictures into one giant group).
2. **Exclusions** ({st['excluded_files']:,} files in {st['excluded_components']:,} components; Section 5.3): pictures whose copies carry
   different class labels, pictures that an upstream dataset labels as a different condition ("pitting" filed under Beau's lines,
   "splinter haemorrhage" under black line, "alopecia areata" under white spot), and pictures linked to the Han et al. A2 hospital
   collection (so that Han A1 can serve as a hospital-disjoint external test set).
3. **Folds and representatives.** `fold_r0`-`fold_r2`: stratified, group-disjoint 5-fold assignment of the components, three repeats.
   `is_representative` marks one fixed image per component for unbiased validation and test sets.
4. **Provenance tags** (`upstream_links`): public datasets in which a near-duplicate of the component was found (Section 10).
5. **Safety tag:** melanoma-sourced Black-line pictures are tagged `melanoma_sourced`.
6. **Integrity:** `sha256` and `md5_file` of every original file. **No image was altered.**

## 9. Known limitations and safety
- **Residual duplicates:** re-uploads under another name whose similarity falls below the thresholds can remain in different
  components; residual leakage cannot be excluded.
- **Provenance:** the uploader `ko-pin` is anonymous and the project is no longer publicly reachable. About 45% of the source-picture
  groups (mostly onycholysis, white spot and healthy) could not be traced to a public collection.
- **Labels are creator-assigned, not clinically verified.** Classes mix diseases, clinical signs and visual findings; an image can
  show several. Class-to-source confounding is likely (some classes come mostly from one upstream collection).
- **No patient, site, skin-tone or demographic metadata.** Patient-level independence cannot be guaranteed; bias towards the
  populations of the upstream web and hospital images is expected.
- **Unequal classes:** independent pictures per class range from {pc.pictures.min():,} to {pc.pictures.max():,} (Section 3).
- **Image quality:** some images are web screenshots, contain watermarks (5 DermNet-watermarked files) or are publisher figures (2 files).
- **Safety:** `black_line` includes {melan} that upstream datasets label as acral lentiginous / subungual melanoma. A model trained on
  this class must never be used to rule out melanoma.

## 10. Sources and provenance
{SOURCES}
## 11. Licence and attribution
{LICENSES}
## 12. Citation, credits and versions
If you use NailGroup-7, please cite the paper and the dataset:
```bibtex
@article{{sircar2026nail,
  author  = {{Sircar, Arnab and Tanvin, Jamal Uddin and Islam, Muhammad Nazrul}},
  title   = {{Interpretable Nail Disease Classification via {{XAI}} and Social Network Analysis}},
  journal = {{Healthcare Technology Letters}},
  year    = {{2026}},
  note    = {{Under review; DOI to be added}}
}}
@misc{{nailgroup7_2026,
  author       = {{Sircar, Arnab and Tanvin, Jamal Uddin and Islam, Muhammad Nazrul}},
  title        = {{{{NailGroup-7}}: a curated 7-class nail-photograph dataset with near-duplicate groups and leakage-free cross-validation folds}},
  howpublished = {{Kaggle, {url}}},
  year         = {{2026}},
  note         = {{Version 1.0}}
}}
```
Please also cite the original image sources (Section 10).

**Credits.** Images: the upstream data owners in Section 10 (hospital data of Han et al. 2018; Thai Kaggle sets; Roboflow KNM;
Kaggle nikhilgurav21, 17-class and josephrasanjana sets, and others); the aggregation `ko-pin/dataset-7clxx` was made by an anonymous
Roboflow Universe user. This curated version (grouping, exclusions, folds, provenance tags, statistics, documentation): Arnab Sircar,
Jamal Uddin Tanvin and Muhammad Nazrul Islam (Military Institute of Science and Technology, Dhaka).

**Versions.** 1.0 (November 2026): first release - near-duplicate components, label-policy exclusions, folds r0-r2,
representatives, provenance and safety tags, checksums, documentation.

**Questions and corrections:** please use the *Discussion* tab of this dataset.
"""


GUIDE = """# NailGroup-7 - Kaggle upload guide and dataset description

This one file contains everything needed to publish NailGroup-7 on Kaggle: what to upload, step-by-step instructions, what to enter
in every Kaggle field, the *About this file* texts, the column descriptions, a final checklist and (Section 8) the complete
description text. This guide itself is **not** uploaded.

## 1. What to upload
All files are in `kaggle_release/` on the workstation:

| File | Size | Use |
|---|---|---|
| `nailgroup-7_with_images.zip` | @@ZIPMB@@ MB | **the dataset** - upload its contents |
| `NailGroup-7_Kaggle_description.md` | - | the description alone: open, select all, copy, paste into Kaggle (= Section 8) |
| `cover_image.png` | 1120 x 560 px | cover image of the Kaggle page |
| `nailgroup-7_with_images.zip.sha256` | - | checksum of the zip (optional check after copying it) |
| `NailGroup-7_Kaggle_upload.md` | - | this guide (do not upload) |

The zip contains one folder `nailgroup-7/` with:
- `README.md` - the dataset description (same text as Section 8)
- `manifest.csv`, `components.csv`, `representatives.csv` - labels, near-duplicate groups, folds, provenance, checksums
- `dataset-metadata.json` - Kaggle metadata (only used by the API upload; harmless in the web upload)
- `images/` - @@NINC@@ JPEG images in seven class folders (@@PERCLASS@@)

Optional check after copying the zip to another computer: `sha256sum nailgroup-7_with_images.zip` (Linux/macOS) or
`CertUtil -hashfile nailgroup-7_with_images.zip SHA256` (Windows) must print `@@SHA@@`.

## 2. Before you start
- Sign in at https://www.kaggle.com . Publishing may require a verified phone number (Settings > Phone verification).
- @@OWNERTEXT@@
- Agree with the co-authors that the dataset may be made public (Section 11 of the description explains the licence situation).

## 3. Upload step by step (web browser)
1. Unzip `nailgroup-7_with_images.zip` on your computer. You get the folder `nailgroup-7/`.
2. On Kaggle: **Datasets** (left menu) > **+ New Dataset**.
3. Drag the **contents** of `nailgroup-7/` (the five files and the `images` folder) into the upload area, or click *Browse Files*.
   Wait until all @@NFILES@@ files are uploaded (several minutes). Uploading the zip itself also works: Kaggle unpacks it but keeps
   the folder `nailgroup-7/` as an extra level (the code in the description finds the files either way).
4. **Title:** `@@TITLE@@`. Check that the URL ends in `@@SLUG@@`. Keep the dataset **Private**. Click **Create**.
5. When processing has finished, open the dataset page and fill in the fields of Section 4 (*Edit* / pencil icons).
6. In the **Data Card** (Data Explorer), click each file and add its *About this file* text (Section 5) and the column
   descriptions (Section 6).
7. Go through the checklist (Section 7). Then **Settings > Sharing > Public**. If Kaggle offers a DOI in the settings, create it.
8. @@STEP8@@

### Alternative: upload with the Kaggle API (command line)
1. `pip install kaggle`; Kaggle > Settings > API > *Create New Token* downloads `kaggle.json`; save it as `~/.kaggle/kaggle.json`
   (Linux/macOS, then `chmod 600 ~/.kaggle/kaggle.json`) or `C:\\Users\\<you>\\.kaggle\\kaggle.json` (Windows).
2. On the workstation: `python scripts/s14_kaggle_package.py --user <your-user-name>` (fills the user name into
   `dataset-metadata.json` and the description), then unzip `kaggle_release/nailgroup-7_with_images.zip`.
3. `kaggle datasets create -p nailgroup-7 --dir-mode zip` creates the private dataset with title, subtitle, description, licence,
   tags, file descriptions and column descriptions taken from `dataset-metadata.json` (Sections 5 and 6 are then already done).
4. Finish on the web page: cover image, provenance, update frequency and visibility (Section 4; steps 7-8 above).
5. Later versions: `kaggle datasets version -p nailgroup-7 -m "what changed" --dir-mode zip`.

## 4. What to enter in each Kaggle field
| Kaggle field | Enter |
|---|---|
| Title | `@@TITLE@@` |
| URL (slug) | `@@SLUG@@` |
| Subtitle | `@@SUBTITLE@@` |
| Description (*About Dataset*) | the complete text of `NailGroup-7_Kaggle_description.md` (= Section 8) |
| Licence | **Other (specified in description)** |
| Tags | @@TAGS@@ |
| Cover image | `cover_image.png` |
| Provenance: sources | @@PROV_SOURCES@@ |
| Provenance: collection methodology | @@PROV_METHOD@@ |
| Expected update frequency | Never |
| Collaborators (optional) | the co-authors' Kaggle user names |
| DOI (optional) | Settings > DOI, after the dataset is public |

Kaggle's usability score is higher when these fields and the file and column descriptions (Sections 5 and 6) are filled in; a public
notebook that uses the dataset also helps (the code in Section 6 of the description can serve as a starter notebook).

## 5. File information (*About this file*)
| File | About this file |
|---|---|
@@FILEINFO@@

## 6. Column descriptions (Data Card)
With the API upload these are filled in automatically from `dataset-metadata.json`; in the web upload, copy them per column.

**`manifest.csv`** (the same columns appear in `representatives.csv`):

@@COLS_MANIFEST@@

**`components.csv`**:

@@COLS_COMP@@

## 7. Final checklist
- [ ] Title, subtitle, description, licence, tags, cover image, provenance and update frequency filled in (Section 4)
- [ ] @@OWNERCHECK@@
- [ ] *About this file* and column descriptions added (Sections 5 and 6)
- [ ] Kaggle shows README.md, manifest.csv, components.csv, representatives.csv, dataset-metadata.json and @@NINC@@ images in 7 class folders
- [ ] The description renders correctly (tables, code blocks); the *Not for diagnosis* note and Section 9 (safety) are visible
- [ ] Co-authors agreed; dataset set to Public
- [ ] Dataset link sent to Claude (data-availability statement, reference list, code release and description are then updated)

## 8. Description text - paste into *About Dataset*
Copy everything inside the box below, from `# @@TITLE@@` to the last line, into Kaggle's Description field. Easier: open
`NailGroup-7_Kaggle_description.md` (same folder), select all, copy and paste. It is also the same text as `README.md` in the zip
(open it in a Markdown preview to see how it will look).

````markdown
@@CARD@@
````
"""


def guide(cfg, man: pd.DataFrame, st: dict, text: str, finfo: dict, user: str | None, zip_path: Path, sha: str) -> str:
    inc = man[man.status == "included"]
    thr = json.loads((cfg.path("dedup") / "threshold.json").read_text())
    n_owner = text.count("OWNER")
    owner_text = (f"**Your Kaggle user name.** The description contains `OWNER` in {n_owner} places (download commands and dataset "
                  "citation). Replace `OWNER` with your Kaggle user name before pasting (search and replace), or tell Claude your user "
                  "name and it rebuilds everything with `python scripts/s14_kaggle_package.py --user <your-user-name>`."
                  if n_owner else f"The description already contains your Kaggle user name (`{user}`).")
    owner_check = "`OWNER` replaced by your Kaggle user name in the description" if n_owner else "Kaggle user name in the description is correct"
    prov_sources = ("Curated from Roboflow Universe ko-pin/dataset-7clxx v1 (exported 2024-02-17, declared CC BY 4.0), an aggregation of "
                    "public nail-image collections (Han et al. 2018 hospital data, Thai Kaggle sets, Roboflow KNM, Kaggle nikhilgurav21, "
                    "17-class and josephrasanjana sets, and others). Full list with links: Section 10 of the description.")
    prov_method = (f"Near-duplicate grouping of all {len(man):,} files (same filename stem, byte-identical, perceptual-hash distance <= "
                   f"{thr['phash_threshold']} or SSCD copy-detection similarity >= {thr['sscd_threshold']}, each over 8 rotations/flips) "
                   f"into {st['components']:,} components; exclusion of {st['excluded_files']:,} files (conflicting labels, upstream "
                   "label-policy conflicts, pictures linked to the Han et al. A2 hospital set); stratified group-disjoint 5-fold "
                   "cross-validation folds with 3 repeats; provenance and safety tags; SHA-256/MD5 checksums. Images unmodified. "
                   "Details: Section 8 of the description.")
    vals = {"ZIPMB": f"{zip_path.stat().st_size / 2**20:,.0f}" if zip_path.exists() else "n/a", "SHA": sha,
            "NINC": f"{len(inc):,}", "NFILES": f"{len(inc) + 5:,}",
            "PERCLASS": ", ".join(f"{c} {n:,}" for c, n in inc["class"].value_counts().reindex(cfg.classes).items()),
            "OWNERTEXT": owner_text, "OWNERCHECK": owner_check,
            "STEP8": ("Send the dataset link to Claude: it updates the manuscript, the code release and this description (`OWNER` -> "
                      "your user name); then paste the updated description into Kaggle again (*Edit* > Description)." if n_owner else
                      f"The dataset URL is https://www.kaggle.com/datasets/{user}/{SLUG} (already in the manuscript and the code release). "
                      "If the dataset was uploaded while the description still contained `OWNER`: paste Section 8 again (*Edit* > "
                      "Description) and replace `README.md` and `dataset-metadata.json` with the files from the current zip "
                      "(*New Version*, keep all other files)."), "TITLE": TITLE, "SLUG": SLUG, "SUBTITLE": SUBTITLE,
            "TAGS": ", ".join(KEYWORDS), "PROV_SOURCES": prov_sources, "PROV_METHOD": prov_method,
            "FILEINFO": "\n".join(f"| `{k}` | {v} |" for k, v in finfo.items()),
            "COLS_MANIFEST": coltable(COLUMNS), "COLS_COMP": coltable(COMP_COLUMNS)}
    g = GUIDE
    for k, v in vals.items():
        g = g.replace(f"@@{k}@@", v)
    g = g.replace("@@CARD@@", text.rstrip("\n"))              # last: the card itself contains no @@ markers
    assert not re.findall(r"@@[A-Z_]+@@", g.split("````markdown")[0]), re.findall(r"@@[A-Z_]+@@", g)
    return g


def metadata(text: str, user: str | None, finfo: dict) -> dict:
    def schema(cols):
        return {"fields": [{"name": k, "description": d.replace("`", ""), "type": t} for k, (t, d) in cols.items()]}
    res = [{"path": "README.md", "description": finfo["README.md"]},
           {"path": "manifest.csv", "description": finfo["manifest.csv"], "schema": schema(COLUMNS)},
           {"path": "components.csv", "description": finfo["components.csv"], "schema": schema(COMP_COLUMNS)},
           {"path": "representatives.csv", "description": finfo["representatives.csv"], "schema": schema(COLUMNS)}]
    return {"title": TITLE, "id": f"{user or 'YOUR_KAGGLE_USERNAME'}/{SLUG}", "subtitle": SUBTITLE, "description": text,
            "licenses": [{"name": "other"}], "keywords": KEYWORDS, "resources": res}


def cover_image(cfg, man: pd.DataFrame, zf, dst: Path):
    """1120x560 cover: one example picture per class (the correctly classified examples of the paper's XAI figure) + title."""
    from PIL import Image, ImageDraw, ImageFont
    sel_f = ROOT / "figures" / "manuscript" / "main" / "fig5_xai_examples_selection.csv"
    picks = {}
    if sel_f.exists():
        sel = pd.read_csv(sel_f)
        for r in sel[sel.kind == "correct"].itertuples():
            m = man[man.image_id == r.image_id]
            if len(m):
                picks[r.row] = m.original_path.iloc[0]
    for c in cfg.classes:                                  # fallback: first included representative of the class
        if c not in picks:
            m = man[(man["class"] == c) & (man.status == "included") & (man.is_representative == 1)]
            picks[c] = m.original_path.iloc[0]
    W, H, s, gap = 1120, 560, 140, 12
    im = Image.new("RGB", (W, H), (247, 247, 244))
    d = ImageDraw.Draw(im)

    def font(sz, bold=False):
        for f in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), ("arialbd.ttf" if bold else "arial.ttf")):
            try:
                return ImageFont.truetype(f, sz)
            except OSError:
                pass
        return ImageFont.load_default()
    d.text((40, 34), "NailGroup-7", fill=(30, 30, 30), font=font(58, True))
    d.text((44, 108), "7-class nail photographs  |  near-duplicate groups  |  leakage-free CV folds", fill=(85, 85, 85), font=font(22))
    x0 = (W - 7 * s - 6 * gap) // 2
    for i, c in enumerate(cfg.classes):
        x = x0 + i * (s + gap)
        thumb = Image.open(io.BytesIO(zf.read(picks[c]))).convert("RGB").resize((s, s))
        im.paste(thumb, (x, 190))
        lab = NAMES[c]
        f = font(16, True)
        w = d.textlength(lab, font=f)
        d.text((x + (s - w) / 2, 190 + s + 10), lab, fill=(30, 30, 30), font=f)
    n_inc = int(man[man.status == "included"].component_id.nunique())
    d.text((44, 420), f"{n_inc:,} independent picture groups  |  manifest, folds, provenance and checksums included",
           fill=(85, 85, 85), font=font(20))
    d.text((44, 470), "Curated from a public Roboflow collection; images unmodified. Not a medical device.", fill=(120, 120, 120),
           font=font(18))
    im.save(dst)
    print("cover image ->", dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kaggle_release")
    ap.add_argument("--user", default=KAGGLE_USER, help="Kaggle user name (fills the dataset URL); empty string = placeholder OWNER")
    ap.add_argument("--limit-images", type=int, default=0, help="testing: put only this many images into the zip")
    ap.add_argument("--with-images", action="store_true", help="ignored (kept for old commands): the image zip is always built")
    ap.add_argument("--zip", action="store_true", help="ignored (kept for old commands): the image zip is always built")
    args = ap.parse_args()
    cfg = load_config()
    out = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    src = cfg.source("kopin")
    zpath = cfg.path("raw") / src["folder"] / src["file"]
    flow = json.loads((cfg.path("dedup") / "flow.json").read_text()).get("dev", {})
    zpath_out = out / f"{SLUG}_with_images.zip"
    with zipfile.ZipFile(zpath) as zf:
        print("hashing", zpath.name, "...")
        man, comps, st = build_manifest(cfg, zf)
        text = card(cfg, man, st, flow, args.user)
        finfo = file_info(cfg, man, st)
        docs = {"README.md": text, "manifest.csv": man.to_csv(index=False), "components.csv": comps.to_csv(index=False),
                "representatives.csv": man[(man.status == "included") & (man.is_representative == 1)].to_csv(index=False),
                "dataset-metadata.json": json.dumps(metadata(text, args.user, finfo), indent=2, ensure_ascii=False)}
        cover_image(cfg, man, zf, out / "cover_image.png")
        inc = man[man.status == "included"]
        inc = inc.head(args.limit_images) if args.limit_images else inc
        tmp = zpath_out.with_suffix(".zip.part")                 # new file, then rename: hard-linked old copies stay intact
        with zipfile.ZipFile(tmp, "w") as zo:
            for name, content in docs.items():
                zo.writestr(f"{SLUG}/{name}", content, compress_type=zipfile.ZIP_DEFLATED)
            for i, r in enumerate(inc.itertuples(), 1):
                zo.writestr(f"{SLUG}/{r.file}", zf.read(r.original_path), compress_type=zipfile.ZIP_STORED)   # JPEGs: no gain
                if i % 10000 == 0:
                    print(f"  {i}/{len(inc)} images")
        tmp.replace(zpath_out)
    h = hashlib.sha256()
    with open(zpath_out, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    sha = h.hexdigest()
    (out / f"{zpath_out.name}.sha256").write_text(f"{sha}  {zpath_out.name}\n")
    (out / DESC_NAME).write_text(text, encoding="utf-8")
    (out / GUIDE_NAME).write_text(guide(cfg, man, st, text, finfo, args.user, zpath_out, sha), encoding="utf-8")
    for name in OLD_OUTPUTS:                                     # superseded by GUIDE_NAME
        (out / name).unlink(missing_ok=True)
    for d in ("manifest_only", "with_images"):                   # earlier variants (dropped 2026-11-18; the zip holds everything)
        if (out / d).is_dir():
            shutil.rmtree(out / d)
    print(f"zip: {zpath_out} ({zpath_out.stat().st_size / 2**20:.0f} MB, {len(inc)} images), sha256 {sha}")
    print(f"included {st['included_files']:,} files / {st['included_components']:,} components; excluded {st['excluded_files']:,} files")
    print("upload guide ->", out / GUIDE_NAME)


if __name__ == "__main__":
    main()
