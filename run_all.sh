#!/usr/bin/env bash
# Reproduction (Linux/macOS/Git-Bash), from the HTL_Revision folder with the venv activated.
#   bash run_all.sh            everything (data -> training -> analysis)
#   bash run_all.sh data       stages 00-05
#   bash run_all.sh train      stage 06, all protocols (cv, leaky, vendor, xsrc, msrc, ablations); skips finished runs
#   bash run_all.sh analysis   stages 07-17
# Training/analysis stages continue after a failure and list the failed ones at the end; data stages stop at the first failure.
# Per-stage logs: runs/logs/<stage>.log
set -uo pipefail
PART=${1:-all}
mkdir -p runs/logs
FAILED=()
slug() { echo "$1" | cut -d' ' -f1; }
must() { echo -e "\n===== $1 ====="; local n="$1"; shift; python "$@" 2>&1 | tee "runs/logs/$(slug "$n").log"; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!!!! FAILED: $n"; exit 1; }; }
run()  { echo -e "\n===== $1 ====="; local n="$1"; shift; python "$@" 2>&1 | tee "runs/logs/$(slug "$n").log"; [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "!!!! FAILED: $n"; FAILED+=("$n"); }; }

if [[ $PART == all || $PART == data ]]; then
  must "00 verify raw archives"            scripts/s00_verify_raw.py
  must "01 index + standardise images"     scripts/s01_build_index.py --workers 12
  must "02 pHash + SSCD embeddings"        scripts/s02_embed.py
  must "03 calibrate duplicate threshold"  scripts/s03_dedup_calibrate.py
  must "04 components + curation"          scripts/s04_components.py
  must "05 splits + leakage audit"         scripts/s05_splits.py
fi

if [[ $PART == all || $PART == train ]]; then
  run "06b main CV (5 folds x 3 repeats)"  scripts/s06_train.py --protocol cv
  run "06c image-level split (T2)"         scripts/s06_train.py --protocol leaky
  run "06d Roboflow vendor split"          scripts/s06_train.py --protocol vendor
  run "06h cross-dataset matrix (xsrc)"    scripts/s06_train.py --protocol xsrc
  run "06i pooled multi-source (msrc)"     scripts/s06_train.py --protocol msrc
  run "06e 5-epoch ablation"               scripts/s06_train.py --protocol cv --repeats 0 --fixed-epochs 5
  run "06f LR sweep x0.5"                  scripts/s06_train.py --protocol cv --repeats 0 --folds 0 --lr-mult 0.5
  run "06g LR sweep x2"                    scripts/s06_train.py --protocol cv --repeats 0 --folds 0 --lr-mult 2
fi

if [[ $PART == all || $PART == analysis ]]; then
  run "07 evaluation + statistics"         scripts/s07_evaluate.py
  run "07b shared-label cross-dataset"     scripts/s07b_shared_label_matrix.py
  run "08 confusion networks (SNA)"        scripts/s08_sna.py
  run "09 explanations (XAI + gallery)"    scripts/s09_xai.py
  run "10 GPU benchmark"                   scripts/s10_bench_gpu.py
  run "11 ONNX + CPU benchmark"            scripts/s11_bench_cpu_onnx.py
  run "13 figures, tables, macros"         scripts/s13_figures_tables.py
  run "17 robustness decomposition (T6)"   scripts/s17_robustness_decomposition.py
  run "18 datasets + summary tables"      scripts/s18_manuscript_tables.py
  run "16 manuscript figures"              scripts/s16_manuscript_figures.py
  run "14 Kaggle package (+images)"        scripts/s14_kaggle_package.py --with-images --zip
  run "15 results digest (guide)"          scripts/s15_results_digest.py
fi

if ((${#FAILED[@]})); then echo -e "\nFailed stages: ${FAILED[*]}"; exit 1; fi
echo "Done. Phone benchmark: python scripts/s12_phone_server.py, then re-run s13."
