# Reproduction on a Windows workstation (PowerShell, from the HTL_Revision folder, venv activated).
#   powershell -File run_all.ps1 [-Part all|data|train|analysis]
# Training/analysis stages continue after a failure and are listed at the end; data stages stop at the first failure.
# Every stage can be re-run on its own; training skips runs whose predictions already exist. Logs: runs/logs/<stage>.log
param([ValidateSet("all", "data", "train", "analysis")][string]$Part = "all")
New-Item -ItemType Directory -Force -Path runs/logs | Out-Null
$failed = @()
function Step($name, $cmd, [bool]$must = $false) {
    Write-Host "`n===== $name =====" -ForegroundColor Cyan
    $log = "runs/logs/" + ($name -split " ")[0] + ".log"
    & python @cmd 2>&1 | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host "!!!! FAILED: $name" -ForegroundColor Red
        if ($must) { exit 1 }
        $script:failed += $name
    }
}

if ($Part -in "all", "data") {
    Step "00 verify raw archives"           @("scripts/s00_verify_raw.py") $true
    Step "01 index + standardise images"    @("scripts/s01_build_index.py", "--workers", "12") $true
    Step "02 pHash + SSCD embeddings"       @("scripts/s02_embed.py") $true
    Step "03 calibrate duplicate threshold" @("scripts/s03_dedup_calibrate.py") $true
    Step "04 components + curation"         @("scripts/s04_components.py") $true
    Step "05 splits + leakage audit"        @("scripts/s05_splits.py") $true
}
if ($Part -in "all", "train") {
    Step "06b main CV (5 folds x 3 repeats)" @("scripts/s06_train.py", "--protocol", "cv")
    Step "06c image-level split (T2)"    @("scripts/s06_train.py", "--protocol", "leaky")
    Step "06d Roboflow vendor split"         @("scripts/s06_train.py", "--protocol", "vendor")
    Step "06h cross-dataset matrix (xsrc)"   @("scripts/s06_train.py", "--protocol", "xsrc")
    Step "06i pooled multi-source (msrc)"    @("scripts/s06_train.py", "--protocol", "msrc")
    Step "06e 5-epoch ablation"              @("scripts/s06_train.py", "--protocol", "cv", "--repeats", "0", "--fixed-epochs", "5")
    Step "06f LR sweep x0.5"                 @("scripts/s06_train.py", "--protocol", "cv", "--repeats", "0", "--folds", "0", "--lr-mult", "0.5")
    Step "06g LR sweep x2"                   @("scripts/s06_train.py", "--protocol", "cv", "--repeats", "0", "--folds", "0", "--lr-mult", "2")
}
if ($Part -in "all", "analysis") {
    Step "07 evaluation + statistics"        @("scripts/s07_evaluate.py")
    Step "07b shared-label cross-dataset"     @("scripts/s07b_shared_label_matrix.py")
    Step "08 confusion networks (SNA)"       @("scripts/s08_sna.py")
    Step "09 explanations (XAI + gallery)"   @("scripts/s09_xai.py")
    Step "10 GPU benchmark"                  @("scripts/s10_bench_gpu.py")
    Step "11 ONNX + CPU benchmark"           @("scripts/s11_bench_cpu_onnx.py")
    Step "13 figures, tables, macros"        @("scripts/s13_figures_tables.py")
    Step "17 robustness decomposition (T6)"  @("scripts/s17_robustness_decomposition.py")
    Step "18 datasets + summary tables"      @("scripts/s18_manuscript_tables.py")
    Step "16 manuscript figures"             @("scripts/s16_manuscript_figures.py")
    Step "14 Kaggle package"                 @("scripts/s14_kaggle_package.py", "--with-images")
    Step "15 results digest (guide)"         @("scripts/s15_results_digest.py")
}
if ($failed.Count) { Write-Host "`nFailed stages: $($failed -join ', ')" -ForegroundColor Red; exit 1 }
Write-Host "`nDone. Phone benchmark: python scripts/s12_phone_server.py, then re-run s13." -ForegroundColor Green
