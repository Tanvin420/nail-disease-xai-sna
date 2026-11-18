# Interpretable Nail Disease Classification via XAI and Social Network Analysis — code, results and figures

Code, complete results and all figures of the paper *"Interpretable Nail Disease Classification via XAI and Social Network Analysis"*
by Arnab Sircar, Jamal Uddin Tanvin and Muhammad Nazrul Islam (Military Institute of Science and Technology, Dhaka), revised version
submitted to *Healthcare Technology Letters* (manuscript HTL-2026-05-0084). Every number, table and figure of the paper can be regenerated
from this repository.

> Research code. The models are not medical devices and must not be used for diagnosis.

---------------------------------------------------------------------------------------------------------------------------------

## 1. What the study does
Five ImageNet-pretrained deep learning architectures — VGG-16, DenseNet-121, EfficientNet-B2, MobileNetV3-Large and ViT-B/16 — are
compared for seven-class nail-photograph classification (Beau's lines, Black line, Clubbing, Healthy nail, Onychomycosis, Onycholysis,
White spot) across eleven public nail datasets. The development dataset, **NailGroup-7**, is a curated version of a public Roboflow
collection in which near-duplicate copies of a picture are grouped so that no picture appears in both training and test data
(published on Kaggle: https://www.kaggle.com/datasets/jamaluddintanvin/nailgroup-7). The study compares accuracy, calibration, speed and robustness, explains the models with Grad-CAM,
Integrated Gradients and LIME (with quantitative faithfulness tests), and analyses misclassifications as networks (Social Network Analysis).

### Tests and main results
| Test | Research question | Main result |
|---|---|---|
| T1 Internal performance | How well does each architecture classify unseen pictures of NailGroup-7? | macro-F1 0.882–0.913 (group-disjoint 5-fold CV x 3); AUROC 0.988–0.993 |
| T2 Partitioning unit | How much does splitting by image file instead of by picture inflate the estimate? | macro-F1 0.997–0.998 with an image-level split (+8.4 to +11.5 points) |
| T3 External testing | Do the models transfer to 10 independent datasets? | macro-F1 0.367–0.420 on hospital images (Han A1) |
| T4 Cross-dataset matrix | Which datasets teach transferable features? | own-dataset balanced accuracy 0.897, much lower on other datasets |
| T5 Pooled sources | Does training on several sources help? | held-out balanced accuracy 0.341 (missing classes are not compensated) |
| T6 Robustness decomposition | Does performance depend on the data or on the architecture? | datasets explain 80.2% of the variance, the architecture 4.5% |
| T7 Training dynamics | Is a short fixed schedule sufficient? | five epochs lower macro-F1 of the CNNs by 2.5–4.7 points |
| T8 Computational cost | How large and fast are the models? | MobileNetV3-Large: 1.22 ms (GPU), 2.2 ms (1 CPU thread) |
| T9 Explanation faithfulness | Do the maps reflect what the models use? | Grad-CAM passes the sanity check (rho -0.04 to 0.14), Integrated Gradients fails (rho 0.65–0.83) |
| T10 Error topology (SNA) | Which conditions are confused? | Onycholysis is the main error sink for 4 of 5 models; errors come mainly from Beau's lines, White spot and Clubbing |

## 2. Repository structure
```
README.md                  this file
LICENSE                    MIT licence of the code
CITATION.cff               how to cite the paper and the dataset
paper/                     the revised manuscript and Supporting Information (PDF) and their LaTeX sources
configs/default.yaml       every setting: data sources and label maps, duplicate-detection thresholds, splits, models, training, analysis
src/nailrev/               library: data loading, near-duplicate detection, models, training engine, metrics, SNA, XAI
scripts/                   pipeline stages s00-s18 (table in section 5) and infer.py
run_all.sh, run_all.ps1    run the whole pipeline (Linux/macOS, Windows)
environment/               requirements.txt, exact versions (requirements.lock.txt), SETUP.md
phone_bench/               browser page for the optional on-device benchmark
tools/make_si_tables.py    SI tables of software versions and class mapping
data/raw/README.md         where to download every source dataset
data/manifests/            raw_sources.csv (URLs, licences, checksums), index.csv (one row per image of all sources)
data/dedup/                near-duplicate components, thresholds, calibration, manual audit (incl. montages), data-flow counts
data/splits/               NailGroup-7 components, cross-validation folds, image-level/vendor splits, external test lists, cross-dataset splits
results/predictions/       raw logits of every trained model on every evaluation set (float16 parquet) + per-run meta.json
results/metrics/           all metrics (internal, external, cross-dataset, pooled, calibration, probes, ablations, robustness ANOVA)
results/sna/, xai/, bench/ confusion networks and stability, explanation faithfulness and sanity, GPU/CPU benchmarks
results/tables/            every LaTeX table of the paper and the SI
results/macros*.tex        every number quoted in the paper
results/manuscript_number_provenance.csv   each number written into the paper and the macro it came from
results/RESULTS_GUIDE.md   description of every result file
figures/manuscript/        figures of the paper (main/ and si/; PDF, PNG and editable draw.io)
figures/main, figures/si   analysis figures (stage 13)
figures/gallery/           XAI panels (original | Grad-CAM | IG | LIME) for every explained image; open index.html
```

## 3. Installation
Python 3.12 and an NVIDIA GPU with CUDA 12.8 are needed for training (results were produced on an RTX 5090); tables, numbers and
most figures can be regenerated without a GPU.
```bash
python3.12 -m venv .venv && source .venv/bin/activate          # Windows: py -3.12 -m venv .venv ; .venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r environment/requirements.txt                    # exact versions: environment/requirements.lock.txt
```

## 4. Data
- **NailGroup-7 (development data):** https://www.kaggle.com/datasets/jamaluddintanvin/nailgroup-7 . It contains the original image bytes of all included files, the near-duplicate
  components, exclusion reasons, the cross-validation folds and checksums. It was curated from the Roboflow Universe export
  `ko-pin/dataset-7clxx` v1 (17 February 2024, declared CC BY 4.0), which is no longer publicly accessible.
- **External datasets:** Han et al. (figshare) and the Kaggle/Roboflow datasets listed in `data/raw/README.md` and
  `data/manifests/raw_sources.csv` (URLs, licences, SHA-256/MD5). Place each archive, unrenamed, in the folder named there.
- Images keep the licences of their sources; see Section 11 (*Licence and attribution*) of the NailGroup-7 `README.md`.

## 5. Pipeline
| Stage | Script | What it does | Paper |
|---|---|---|---|
| 00 | `s00_verify_raw.py` | checks the downloaded archives against `raw_sources.csv` | data availability |
| 01 | `s01_build_index.py` | reads all sources, cuts the Han et al. contact sheets into tiles, standardises images to 288 px | Section 4.1 |
| 02 | `s02_embed.py` | perceptual hashes and SSCD copy-detection embeddings over 8 rotations/flips | Section 4.1.2, SI S1.1 |
| 03 | `s03_dedup_calibrate.py` | calibrates the duplicate threshold (synthetic copies, manual audit) | SI Figure S2 |
| 04 | `s04_components.py` | near-duplicate components, exclusions, de-duplicated external test sets | Sections 4.1.2-4.1.4, Tables 2-3 |
| 05 | `s05_splits.py` | group-disjoint folds, image-level and vendor splits, cross-dataset and pooled splits | Section 4.5 |
| 06 | `s06_train.py --protocol cv/leaky/vendor/xsrc/msrc` | trains and saves all 180 models and their predictions | Sections 4.2-4.4 |
| 07 | `s07_evaluate.py` | metrics, bootstrap CIs, paired tests, calibration, external, cross-dataset, probes, ablations | T1-T5, T7 |
| 07b | `s07b_shared_label_matrix.py` | cross-dataset results on shared labels | Figure 3 |
| 08 | `s08_sna.py` | confusion networks, bootstrap stability, model similarity, minimum spanning tree | T10 |
| 09 | `s09_xai.py` | Grad-CAM, Integrated Gradients, LIME; deletion/insertion; sanity check; gallery | T9 |
| 10, 11 | `s10_bench_gpu.py`, `s11_bench_cpu_onnx.py` | controlled GPU and ONNX/CPU benchmarks | T8 |
| 12 | `s12_phone_server.py` | optional on-device benchmark | (not used) |
| 13 | `s13_figures_tables.py` | LaTeX tables, `results/macros.tex`, analysis figures | all tables, SI figures |
| 17 | `s17_robustness_decomposition.py` | variance decomposition (dataset vs model robustness), Kendall's W | T6 |
| 18 | `s18_manuscript_tables.py` | datasets table and "results at a glance" table | Tables 3 and 5 |
| 16 | `s16_manuscript_figures.py` | figures of the paper (PDF/PNG/TIFF + editable draw.io) | Figures 1-6 |
| 14 | `s14_kaggle_package.py` | builds the NailGroup-7 Kaggle release | data availability |
| 15 | `s15_results_digest.py` | `results/RESULTS_GUIDE.md` | - |

## 6. Reproducing the results
1. **Tables and numbers from the included results (minutes, no GPU):**
   `python scripts/s13_figures_tables.py && python scripts/s17_robustness_decomposition.py && python scripts/s18_manuscript_tables.py`
   regenerates `results/tables/*.tex`, `results/macros.tex` and `results/macros_robustness.tex`; they are identical to the included
   files. `results/manuscript_number_provenance.csv` traces every number of the paper to its macro.
2. **Figures of the paper:** `python scripts/s16_manuscript_figures.py` (uses `results/` and `figures/gallery/`).
3. **Metrics from the raw predictions:** `python scripts/s07_evaluate.py` recomputes `results/metrics/` from `results/predictions/`.
4. **Full re-run:** download the data (section 4), then `bash run_all.sh` (or `bash run_all.sh data|train|analysis`). Training took
   about 10 hours on an RTX 5090. Training is seeded (2026 + 1000 x repeat + fold; deterministic cuDNN and cuBLAS); small numerical
   differences between GPU types are possible.
5. **Trained models:** the 180 trained models (about 34 GB) are available from the authors on request;
   `python scripts/infer.py --list` and `python scripts/infer.py --model efficientnet_b2 --images photo.jpg` predict with them.

## 7. Evaluation design in brief
- **Independence:** images are linked if they share a class and filename stem, are byte-identical, or are visually near-identical
  (perceptual hash, SSCD copy detection); linked images form a component that is always placed in a single partition.
- **Scores:** macro-F1 and balanced accuracy weight every class equally. External datasets are scored over the classes they contain;
  *strict* allows any of the seven classes, *restricted* only the classes of the test set, *shared-label* only classes present in both
  training and test dataset.
- **Uncertainty:** mean ± SD over 15 runs, 95% bootstrap confidence intervals over test components, paired tests with Holm correction.
- **Robustness decomposition:** ANOVA of chance-corrected transfer scores into dataset, architecture and interaction effects (eta²).
Full details: Section 4.5 and SI Section S1 of the paper (`paper/`).

## 8. Citation
See `CITATION.cff`. Paper: Sircar A, Tanvin JU, Islam MN. Interpretable Nail Disease Classification via XAI and Social Network
Analysis. Healthcare Technology Letters (under revision). Dataset: NailGroup-7, Kaggle, https://www.kaggle.com/datasets/jamaluddintanvin/nailgroup-7 . Code: https://github.com/Tanvin420/nail-disease-xai-sna .

## 9. Licence and contact
- Code: MIT licence (`LICENSE`). It covers the code and the material created by the authors in this repository.
- Not covered: the nail images and source datasets, which keep the licences of their sources (section 4). Figures that show nail
  photographs (for example the XAI panels in `figures/gallery/`) reproduce such images.
- Contact: Muhammad Nazrul Islam (corresponding author), nazrul@cse.mist.ac.bd
