# Results guide (generated 2026-11-12 16:47)

Auto-generated from `results/` by `scripts/s15_results_digest.py`. Every number below is read from the CSV files named in the section; numbers in the manuscript are filled in from `results/macros.tex` / `results/tables/` by `tools/expand_macros.py` (plain text; list in `results/manuscript_number_provenance.csv`).

Project context: revision of *Interpretable Nail Disease Classification via XAI and Social Network Analysis* (Healthcare Technology Letters, HTL-2026-05-0084). See `PROJECT_LOG.md` for decisions and `README.md` for the pipeline.

## 1. Leakage effect (reviewer point 1)

Same architectures, three protocols (`results/metrics/leakage_table.csv`). The gap between the first and last column is the effect of data leakage.

| Model | macro-F1, submitted image-level split | macro-F1, Roboflow split | macro-F1, group-disjoint CV |
|---|---|---|---|
| VGG-16 | 0.997 [0.996, 0.998] | 0.996 [0.992, 0.998] | 0.882 [0.870, 0.895] |
| DenseNet-121 | 0.997 [0.996, 0.998] | 0.993 [0.989, 0.996] | 0.895 [0.883, 0.907] |
| EfficientNet-B2 | 0.997 [0.996, 0.998] | 0.994 [0.991, 0.997] | 0.913 [0.902, 0.924] |
| MobileNetV3-Large | 0.998 [0.996, 0.999] | 0.994 [0.990, 0.997] | 0.901 [0.889, 0.912] |
| ViT-B/16 | 0.998 [0.997, 0.999] | 0.995 [0.992, 0.998] | 0.911 [0.902, 0.921] |

## 2. Internal performance, group-disjoint CV (points 1, 4)

Group-disjoint 5-fold × 3 CV, internal test (`internal_summary.csv`).

| Model | Accuracy [95% CI] | Macro-F1 [95% CI] | Balanced acc. | MCC | SD over runs (macro-F1) |
|---|---|---|---|---|---|
| VGG-16 | 0.913 [0.903, 0.922] | 0.882 [0.870, 0.895] | 0.882 | 0.892 | 0.004 |
| DenseNet-121 | 0.922 [0.914, 0.930] | 0.895 [0.883, 0.907] | 0.894 | 0.903 | 0.005 |
| EfficientNet-B2 | 0.934 [0.926, 0.942] | 0.913 [0.902, 0.924] | 0.910 | 0.918 | 0.001 |
| MobileNetV3-Large | 0.925 [0.916, 0.933] | 0.901 [0.889, 0.912] | 0.897 | 0.907 | 0.003 |
| ViT-B/16 | 0.931 [0.924, 0.938] | 0.911 [0.902, 0.921] | 0.909 | 0.914 | 0.003 |

## 3. External testing per source (point 3)

ko-pin-trained models on de-duplicated external sources (one image per component, mean over 5 architectures; `external_summary.csv` has per-model values and CIs).

| source | n | evidence | bal_acc | macro_f1 | out_of_label | restr_bal_acc | restr_acc | restr_macro_f1 |
|---|---|---|---|---|---|---|---|---|
| arthitaya | 53 | peer_reviewed | 0.674 | 0.478 | 0.069 | 0.739 | 0.828 | 0.486 |
| arthitaya2 | 2 | peer_reviewed | 0.64 | 0.636 | 0.347 | 0.913 | 0.913 | 0.884 |
| farisfaikar | 118 | none | 0.761 | 0.862 | 0.239 | 1.0 | 1.0 | 1.0 |
| han_A1 | 17697 | peer_reviewed | 0.471 | 0.39 | 0.191 | 0.53 | 0.476 | 0.394 |
| han_virtualE | 1134 | weak | 0.566 | 0.616 | 0.198 | 0.642 | 0.668 | 0.634 |
| isratjahanmitu | 178 | none | 0.811 | 0.895 | 0.189 | 1.0 | 1.0 | 1.0 |
| josephrasanjana | 367 | weak | 0.772 | 0.861 | 0.226 | 0.967 | 0.962 | 0.958 |
| kaggle17 | 10 | peer_reviewed | 0.471 | 0.58 | 0.547 | 0.893 | 0.863 | 0.855 |
| nikhilgurav21 | 710 | peer_reviewed | 0.603 | 0.681 | 0.324 | 0.882 | 0.884 | 0.882 |
| nuttida | 7 | peer_reviewed | 0.624 | 0.548 | 0.392 | 0.732 | 0.64 | 0.595 |

## 4. All images of all datasets, duplicates flagged (point 3)

Every image of every other dataset (duplicates kept). `dup_of_kopin` images are copies of ko-pin material and inflate accuracy; `independent` is the honest number (mean over 5 architectures).

| source | acc_all | acc_dup_of_kopin | acc_independent | n_all | n_dup | n_indep |
|---|---|---|---|---|---|---|
| arthitaya | 0.864 | 0.879 | 0.784 | 360 | 302 | 58 |
| arthitaya2 | 0.87 | 0.872 | 0.64 | 180 | 178 | 2 |
| farisfaikar | 0.758 | 0.741 | 0.759 | 293 | 24 | 269 |
| han_A1 | 0.425 | 0.412 | 0.427 | 28693 | 3207 | 25486 |
| han_A2 | 0.768 | 0.741 | 0.775 | 1964 | 415 | 1549 |
| han_BCDE | 0.999 | 0.999 | 0.999 | 780 | 568 | 212 |
| han_virtualE | 0.601 | 0.659 | 0.599 | 1732 | 46 | 1686 |
| isratjahanmitu | 0.919 | 1.0 | 0.768 | 556 | 362 | 194 |
| josephrasanjana | 0.878 | 0.979 | 0.814 | 1032 | 400 | 632 |
| kaggle17 | 0.799 | 0.827 | 0.396 | 201 | 188 | 13 |
| nikhilgurav21 | 0.727 | 0.946 | 0.644 | 1126 | 310 | 816 |
| nuttida | 0.848 | 0.855 | 0.415 | 500 | 493 | 7 |

## 5. Cross-dataset matrix: train on each dataset, test on every other (point 3)

Balanced accuracy over the classes present in each test dataset, mean of 5 architectures (`xsrc_matrix.csv`). Mean own-test 0.897 vs mean cross-dataset 0.377: the drop is the domain shift. Blank = same origin as the training set. Evidence of published use per dataset: han_A1=peer_reviewed, han_A2=peer_reviewed, han_BCDE=peer_reviewed, han_virtualE=weak, arthitaya=peer_reviewed, arthitaya2=peer_reviewed, nuttida=peer_reviewed, josephrasanjana=weak, nikhilgurav21=peer_reviewed, kaggle17=peer_reviewed, farisfaikar=none, isratjahanmitu=none, legacy2272=derived, kopin=none.

| train \ test | own_test | arthitaya | arthitaya2 | farisfaikar | han_A1 | han_A2 | han_BCDE | han_virtualE | isratjahanmitu | josephrasanjana | kaggle17 | kopin | legacy2272 | nikhilgurav21 | nuttida |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kopin | 0.9 | 0.75 | 0.7 | 0.79 | 0.49 | 0.66 | 1.0 | 0.55 | 0.8 | 0.77 | 0.4 |  | 0.6 | 0.63 | 0.69 |
| han | 0.83 | 0.41 | 0.47 | 0.77 |  |  |  | 0.71 | 0.89 | 0.87 | 0.24 | 0.47 | 0.34 | 0.14 | 0.32 |
| han_virtualE | 0.82 | 0.36 | 0.4 | 0.92 | 0.54 | 0.67 | 0.95 |  | 0.91 | 0.9 | 0.1 | 0.42 | 0.38 | 0.33 | 0.24 |
| thai | 0.86 |  |  | 0.6 | 0.23 | 0.36 | 0.0 | 0.36 | 0.0 | 0.15 | 0.33 | 0.37 | 0.26 | 0.35 |  |
| nikhilgurav21 | 0.99 | 0.22 | 0.0 | 0.38 | 0.01 | 0.0 | 0.0 | 0.05 | 0.0 | 0.12 | 0.25 | 0.17 | 0.25 |  | 0.2 |
| kaggle17 | 0.9 | 0.31 | 0.35 | 0.0 | 0.13 | 0.13 | 0.0 | 0.09 | 0.0 | 0.0 |  | 0.22 | 0.06 | 0.1 | 0.37 |
| josephrasanjana | 0.99 | 0.15 | 0.0 | 0.9 | 0.34 | 0.33 | 0.98 | 0.41 | 0.96 |  | 0.0 | 0.26 | 0.14 | 0.29 | 0.0 |

**Restricted to the classes present in each test dataset** (the model chooses only among those classes; mean recall over them; mean own-test 0.897, mean cross-dataset 0.591). Accuracy and macro-F1 of the same evaluation are in `xsrc_matrix.csv` (`restricted_acc`, `restricted_macro_f1`).

| train \ test | own_test | arthitaya | arthitaya2 | farisfaikar | han_A1 | han_A2 | han_BCDE | han_virtualE | isratjahanmitu | josephrasanjana | kaggle17 | kopin | legacy2272 | nikhilgurav21 | nuttida |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kopin | 0.9 | 0.81 | 0.9 | 1.0 | 0.53 | 0.74 | 1.0 | 0.62 | 1.0 | 0.97 | 0.82 |  | 0.73 | 0.87 | 0.77 |
| han | 0.83 | 0.48 | 0.62 | 1.0 |  |  |  | 0.72 | 1.0 | 0.95 | 0.45 | 0.47 | 0.36 | 0.41 | 0.46 |
| han_virtualE | 0.82 | 0.43 | 0.61 | 1.0 | 0.54 | 0.83 | 1.0 |  | 1.0 | 0.96 | 0.27 | 0.42 | 0.4 | 0.52 | 0.37 |
| thai | 0.86 |  |  | 1.0 | 0.36 | 0.55 | 1.0 | 0.46 | 1.0 | 0.78 | 0.36 | 0.37 | 0.32 | 0.74 |  |
| nikhilgurav21 | 0.99 | 0.22 | 0.32 | 1.0 | 0.21 | 0.31 | 1.0 | 0.25 | 1.0 | 0.5 | 0.25 | 0.17 | 0.25 |  | 0.2 |
| kaggle17 | 0.91 | 0.31 | 0.46 | 1.0 | 0.23 | 0.34 | 1.0 | 0.26 | 1.0 | 0.55 |  | 0.22 | 0.4 | 0.57 | 0.37 |
| josephrasanjana | 0.99 | 0.18 | 0.32 | 1.0 | 0.34 | 0.33 | 1.0 | 0.41 | 1.0 |  | 0.23 | 0.26 | 0.27 | 0.51 | 0.2 |

## 6. Pooled multi-source training (point 3)

Pooled training on datasets with published use, leave-one-source-group-out (`runs_all_sets.csv`, protocol msrc). `ext_kopin` = ko-pin images with no link to any pooled dataset (never trained on).

| held-out group | set | n | bal_acc (mean of models) | macro_f1 |
|---|---|---|---|---|
| han | test | 19928 | 0.171 | 0.16 |
| han | ext_kopin | 1176 | 0.318 | 0.264 |
| thai | test | 443 | 0.432 | 0.388 |
| thai | ext_kopin | 1176 | 0.378 | 0.365 |
| nikhilgurav21 | test | 795 | 0.205 | 0.292 |
| nikhilgurav21 | ext_kopin | 1176 | 0.579 | 0.556 |
| kaggle17 | test | 137 | 0.554 | 0.655 |
| kaggle17 | ext_kopin | 1176 | 0.426 | 0.419 |
| none | ext_kopin | 1176 | 0.454 | 0.428 |

## 7. Confusion-network (SNA) results (point 7)

Confusion networks, edges = row-normalised confusion rate >= 0.01 (`graph_summary.csv`).

| model | n_edges | top_sink | top_sink_bootstrap_freq | top_sink_raw_counts | tau_counts_vs_rates | kendall_tau_median | ari_median |
|---|---|---|---|---|---|---|---|
| vgg16 | 26 | onycholysis | 0.939 | onycholysis | 0.651 | 0.714 | 0.054 |
| densenet121 | 24 | onycholysis | 0.765 | onycholysis | 0.81 | 0.81 | 0.054 |
| efficientnet_b2 | 24 | onycholysis | 0.876 | onycholysis | 0.81 | 0.81 | 0.417 |
| mobilenetv3 | 24 | onycholysis | 0.862 | onycholysis | 0.81 | 0.81 | 0.417 |
| vit_b16 | 21 | white_spot | 0.707 | onycholysis | 0.429 | 0.714 | 0.432 |

Topology under the leaky split, Roboflow split, group-disjoint CV and domain shift (`graph_summary_compare.csv`):

| protocol | set | accuracy | edges | top_sink |
|---|---|---|---|---|
| cv | external_pooled | 0.451 | 33.0 | onychomycosis |
| cv | test | 0.925 | 23.8 | onycholysis |
| leaky | test | 0.997 | 0.2 | none |
| vendor | test | 0.994 | 1.0 | onycholysis |

## 8. Explanation (XAI) results (point 6)

Deletion/insertion AUC (lower deletion / higher insertion = more faithful) vs random and centre baselines (`summary.csv`):

| model | correct | gradcam_deletion_mean | gradcam_insertion_mean | ig_deletion_mean | ig_insertion_mean | lime_deletion_mean | lime_insertion_mean | random_deletion_mean | random_insertion_mean | random_smooth_deletion_mean | random_smooth_insertion_mean | centre_deletion_mean | centre_insertion_mean | ig_delta_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| densenet121 | 0 | 0.167 | 0.554 | 0.207 | 0.358 | 0.216 | 0.427 | 0.249 | 0.26 | 0.288 | 0.304 | 0.176 | 0.452 | 0.089 |
| densenet121 | 1 | 0.331 | 0.773 | 0.332 | 0.559 | 0.42 | 0.717 | 0.353 | 0.352 | 0.509 | 0.508 | 0.345 | 0.723 | 0.102 |
| efficientnet_b2 | 0 | 0.253 | 0.461 | 0.235 | 0.295 | 0.247 | 0.496 | 0.234 | 0.229 | 0.386 | 0.368 | 0.275 | 0.402 | 0.151 |
| efficientnet_b2 | 1 | 0.418 | 0.686 | 0.344 | 0.501 | 0.423 | 0.705 | 0.339 | 0.34 | 0.565 | 0.555 | 0.452 | 0.641 | 0.174 |
| mobilenetv3 | 0 | 0.223 | 0.555 | 0.228 | 0.251 | 0.236 | 0.516 | 0.199 | 0.197 | 0.352 | 0.367 | 0.339 | 0.385 | 0.17 |
| mobilenetv3 | 1 | 0.393 | 0.695 | 0.302 | 0.443 | 0.406 | 0.714 | 0.281 | 0.282 | 0.549 | 0.558 | 0.44 | 0.639 | 0.15 |
| vgg16 | 0 | 0.233 | 0.515 | 0.185 | 0.282 | 0.275 | 0.49 | 0.18 | 0.178 | 0.331 | 0.353 | 0.289 | 0.449 | 0.057 |
| vgg16 | 1 | 0.435 | 0.67 | 0.271 | 0.501 | 0.434 | 0.713 | 0.28 | 0.277 | 0.535 | 0.532 | 0.414 | 0.697 | 0.063 |
| vit_b16 | 0 | 0.207 | 0.678 | 0.299 | 0.559 | 0.316 | 0.619 | 0.355 | 0.377 | 0.501 | 0.447 | 0.343 | 0.492 | 0.069 |
| vit_b16 | 1 | 0.551 | 0.774 | 0.509 | 0.8 | 0.572 | 0.811 | 0.679 | 0.674 | 0.687 | 0.686 | 0.586 | 0.784 | 0.124 |

Parameter-randomisation sanity check, mean Spearman between original and randomised-model maps (lower = passes):

| model | gradcam_spearman | ig_spearman |
|---|---|---|
| densenet121 | 0.136 | 0.655 |
| efficientnet_b2 | 0.108 | 0.745 |
| mobilenetv3 | 0.115 | 0.762 |
| vgg16 | -0.036 | 0.664 |
| vit_b16 | -0.042 | 0.834 |

Gallery for choosing paper figures: `figures/gallery/index.html` (see `results/xai/gallery.csv`).

## 9. Benchmarks (point 8)

Files: `cpu.csv`, `gpu.csv`, `onnx_checks.csv`. Protocol details are in the `_meta_*.json` and the paper's Methods.

## What every file is


| Where | What it is | Reviewer point |
|---|---|---|
| `results/macros.tex`, `results/tables/tab_*.tex` | Every number and table of the manuscript (LaTeX); the manuscript holds them as plain text (see `manuscript_number_provenance.csv`) | 5 |
| `results/metrics/internal_summary.csv`, `internal_per_class.csv` | Group-disjoint CV metrics: mean ± SD over 15 runs, 95% bootstrap CI, per-class recall | 1, 4 |
| `results/metrics/leakage_table.csv` | Same models under the submitted image-level split (`leaky`), Roboflow split (`vendor`) and group-disjoint CV (`cv`) | 1 |
| `results/metrics/external_summary.csv`, `external_per_class.csv` | De-duplicated per-source external testing (ko-pin-trained models), CIs, out-of-label rate | 3 |
| `results/metrics/full_source_audit.csv` | ko-pin-trained models on EVERY image of every other dataset, split into duplicates of ko-pin / independent | 3 |
| `results/metrics/xsrc_matrix.csv` | Cross-dataset matrix: train on one dataset, test on every other (one row per model, train set, target) | 3 |
| `results/metrics/runs_all_sets.csv` | Per-run metrics for every set of every protocol (msrc results live here: `holdout`, `set` = test / ext_kopin) | 3 |
| `results/metrics/pairwise_tests.csv` | Paired model comparisons (McNemar + Holm, corrected t-test, bootstrap ΔF1, equivalence) | 4, 5 |
| `results/metrics/epochs_and_temperature.csv`, `reliability_bins.csv`, `risk_coverage.csv` | Calibration (temperature per run), reliability, selective prediction | 9 |
| `results/metrics/probes.csv`, `melanoma_subgroup.csv`, `legacy_external_audit.csv` | Out-of-label probes (e.g. psoriasis, melanoma), melanoma-sourced black-line recall, audit of the submitted external set | 3, 9 |
| `results/sna/*` | Confusion networks: `node_metrics`, `edges`, `graph_summary` (3 edge filters), bootstrap stability, `similarity_*.csv` (+ `_sd`), `mst_support`, `graph_summary_compare` (leaky vs CV vs domain shift) | 7 |
| `results/xai/*` | `selection.csv` (which images), `faithfulness.csv` (deletion/insertion AUC vs random/centre), `sanity.csv` (parameter randomisation), `summary.csv`, `gallery.csv` | 6 |
| `results/bench/*` | Controlled GPU latency/memory, ONNX CPU, phone results | 8 |
| `results/predictions/<protocol>/<model>/<run>/*.parquet` | Raw logits of every run on every evaluation set (float16) – everything above can be recomputed from these | 4 |
| `figures/main`, `figures/si` | Paper figures (PDF + 300-dpi PNG + 600-dpi TIFF) | 10 |
| `figures/gallery/index.html` | XAI gallery: stand-alone panels (original / Grad-CAM / IG / LIME) per model, class and outcome – open in a browser and pick | 6 |
| `data/splits`, `data/dedup` | Splits, near-duplicate components, flow counts, calibration, audit labels | 1, 2 |
| `kaggle_release/` | Package of the fixed ko-pin dataset (manifest, README, sources, licences) | 2 |
| `runs/<protocol>/<model>/<run>/best.pt` | Every trained model (kept on the workstation); use `python scripts/infer.py --list` | – |


## Caveats to state in the paper


- Duplicate detection uses SSCD >= 0.75 and pHash <= 2 (chosen from a 150-pair manual audit). Re-uploads of one picture that score
  between about 0.5 and 0.75 can stay in different components, so residual cross-fold leakage is possible: state it as a limitation.
- No patient identifiers exist; independence is at the level of source-picture components, not patients.
- The cross-dataset matrix uses one repeat per training dataset by default (`crosssource.n_repeats`); its per-cell CIs are bootstrap
  CIs over test images and do not include training-seed variance. Small training datasets (17-class: ~135 images) give weak models.
- "Published use" of a dataset is not clinical validation; only Han et al. carries dermatologist/laboratory-confirmed labels.
- Tiny classes in some sources (e.g. Beau's lines, white spot outside ko-pin) make per-class numbers noisy; read the CIs.
- Balanced accuracy / macro-F1 on external sets are computed over the classes present in that source; out-of-label predictions count as errors.


## Using the saved models

```bash
python scripts/infer.py --list
python scripts/infer.py --model efficientnet_b2 --images photo.jpg
python scripts/infer.py --model efficientnet_b2 --ensemble --images folder/ --csv out.csv
```