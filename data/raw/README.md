# data/raw — original, untouched source archives

Rules: never edit or extract inside these folders (extraction goes to `data/cache/`). Each source lives in its
own folder. Integrity is checked against `data/manifests/raw_sources.csv` by `scripts/00_verify_raw.py`.
This folder is git-ignored; on a new device copy it (or set `RAW_ROOT` in `configs/paths.local.yaml`).

## Already downloaded automatically (Figshare, CC BY 4.0 — Han et al. 2018, PLoS ONE 13:e0191493)
| Folder | File | Role |
|---|---|---|
| `han2018_figshare_5398573/` | `A1_thumbnail.zip` (1.83 GB) | **Main new external test**: onychomycosis, onycholysis, melanonychia→black_line, normal→healthy, whitespot→white spot. Contact-sheet mosaics of ~100 px thumbnails (label = last `#` token of filename) — tiles are cut out by a script |
| `han2018_figshare_5398573/` | `A2_thumbnail.zip` (110 MB) | Check whether ko-pin's `L-NAIL-5nailA2…` groups are Han A2 (same hospital as A1) |
| `han2018_figshare_5398573/` | `B1_B2_C_D_E.zip` (376 MB) | Provenance tagging (549 ko-pin onychomycosis groups come from here) |
| `han2018_figshare_5513407/` | `Virtual_E_Dataset.zip` (71 MB) | Secondary external test (web images, CNN-assigned labels) |

## To download manually from Kaggle (log in → "Download" button → put the zip, unrenamed, in the folder)
| Priority | Folder | Kaggle link | Role |
|---|---|---|---|
| **Required** | `kaggle_arthitaya_nail-dataset/` | https://www.kaggle.com/datasets/arthitaya/nail-dataset | Only candidate external test for Beau's lines / white spot / black line |
| **Required** | `kaggle_arthitaya_nail-dataset2/` | https://www.kaggle.com/datasets/arthitaya/nail-dataset2 | Second part of the same Thai set |
| Recommended | `kaggle_nuttidalapthanachai_nails-new-test/` | https://www.kaggle.com/datasets/nuttidalapthanachai/nails-new-test | Upstream of ko-pin black_line/white spot → source tags for leave-one-source-out |
| Recommended | `kaggle_josephrasanjana_nail-disease-image-classification/` | https://www.kaggle.com/datasets/josephrasanjana/nail-disease-image-classification-dataset | Upstream of ko-pin + Roboflow mohit (onychomycosis/healthy/psoriasis) → source tags, psoriasis probe |
| Optional | `kaggle_farisfaikar_nail-melanoma-350/` | https://www.kaggle.com/datasets/farisfaikar/nail-melanoma-350 | Melanoma out-of-label safety probe (does the model call melanoma "black_line"?) |
| Optional | `kaggle_isratjahanmitu_onychomycosis/` | https://www.kaggle.com/datasets/isratjahanmitu/onychomycosis | Extra onychomycosis external images (provenance unknown) |

## Already on disk in the legacy folder (to be moved here, not copied)
| Future folder | Current location | Source / licence |
|---|---|---|
| `roboflow_ko-pin_dataset-7clxx_v1/` | `Nail_Disease/Dataset/train/dataset.v1i.folder.zip` | Roboflow ko-pin/dataset-7clxx v1 (2024-02-17), CC BY 4.0 — **development set, only surviving copy** |
| `kaggle_nikhilgurav21/` | `Nail_Disease/Others Dataset/archive.zip` | Kaggle nikhilgurav21/nail-disease-detection-dataset, CC BY-NC-SA 4.0 |
| `kaggle_17class_saisrushikgovindgari/` | `Nail_Disease/Others Dataset/archive2.zip` | Kaggle saisrushikgovindgari/nail-disease-dataset (MIT; earlier copies chengcris/hunterimam) |
| `roboflow_mohit-fc0fl/` | `Nail_Disease/Dataset/test/test_dataset.zip` | Roboflow mohit-fc0fl/nail-disease-classification-f7gwn, CC BY 4.0 |
