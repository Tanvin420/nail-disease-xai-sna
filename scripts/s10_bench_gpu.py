"""Stage 10 - controlled GPU inference benchmark.

Protocol: one fresh Python process per (model, precision, batch) so no memory or cuDNN state is shared; model in
eval mode under torch.inference_mode; cudnn.benchmark on; FP32 with TF32 disabled (reference) and bf16 autocast;
batch 1 and 32; `warmup` untimed iterations; per-iteration timing with CUDA events + synchronize, `iters`
iterations x `repeats`; peak allocated/reserved memory after torch.cuda.reset_peak_memory_stats();
FLOPs with torch.utils.flop_counter (1 multiply-add = 2 FLOPs).
Outputs results/bench/gpu.csv and gpu_latencies.npz
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402

from nailrev.config import env_report, load_config, write_meta  # noqa: E402


def stats(lat_ms: np.ndarray, batch: int) -> dict:
    q1, med, q3, p90, p95 = np.percentile(lat_ms, [25, 50, 75, 90, 95])
    return {"median_ms": med, "iqr_ms": q3 - q1, "p90_ms": p90, "p95_ms": p95, "mean_ms": lat_ms.mean(),
            "sd_ms": lat_ms.std(ddof=1), "per_image_ms": med / batch, "throughput_img_s": batch * 1000 / med}


def worker(key: str, precision: str, batch: int, out: str):
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from nailrev.models import create_model, load_checkpoint

    cfg = load_config()
    bc, mc = cfg.analysis["bench"], cfg.models[key]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    model = create_model(mc, len(cfg.classes), pretrained=False)
    ck = cfg.path("runs") / "cv" / key / "r0_f0" / "best.pt"
    if ck.exists():
        load_checkpoint(model, ck)
    S = mc["size"]
    n_params = sum(p.numel() for p in model.parameters())
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        model.eval()(torch.randn(1, 3, S, S))
    flops = fc.get_total_flops()
    dev = torch.device("cuda")
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    model = model.eval().to(dev)
    if "vit" not in key:
        model = model.to(memory_format=torch.channels_last)
    weights_mem = torch.cuda.memory_allocated() - base
    x = torch.randn(batch, 3, S, S, device=dev)
    if "vit" not in key:
        x = x.contiguous(memory_format=torch.channels_last)
    amp = precision == "bf16"
    iters = bc["iters_b1"] if batch == 1 else bc["iters_b32"]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    lat = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        for _ in range(bc["warmup"]):
            model(x)
        torch.cuda.synchronize()
        for _ in range(bc["repeats"]):
            for _ in range(iters):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                model(x)
                e.record()
                torch.cuda.synchronize()
                lat.append(s.elapsed_time(e))
            time.sleep(0.5)
    lat = np.array(lat)
    res = {"model": key, "precision": precision, "batch": batch, "size": S, "params_M": n_params / 1e6,
           "gflops": flops / 1e9, "gmacs": flops / 2e9, "weights_mb": weights_mem / 2**20,
           "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
           "peak_reserved_mb": torch.cuda.max_memory_reserved() / 2**20, **stats(lat, batch),
           "repeat_medians_ms": [float(np.median(c)) for c in np.array_split(lat, bc["repeats"])],
           "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "trained_weights": ck.exists()}
    Path(out).write_text(json.dumps(res))
    np.save(Path(out).with_suffix(".npy"), lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", nargs=4, metavar=("MODEL", "PRECISION", "BATCH", "OUT"))
    ap.add_argument("--models", default="all")
    args = ap.parse_args()
    if args.worker:
        m, p, b, o = args.worker
        return worker(m, p, int(b), o)
    import pandas as pd

    cfg = load_config()
    out = cfg.path("results") / "bench"
    tmp = out / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    models = list(cfg.models) if args.models == "all" else args.models.split(",")
    rows, lats = [], {}
    for m in models:
        for prec in ("fp32", "bf16"):
            for b in (1, 32):
                f = tmp / f"{m}_{prec}_b{b}.json"
                print(f"benchmark {m} {prec} batch {b}", flush=True)
                subprocess.run([sys.executable, __file__, "--worker", m, prec, str(b), str(f)], check=True)
                rows.append(json.loads(f.read_text()))
                lats[f"{m}_{prec}_b{b}"] = np.load(f.with_suffix(".npy"))
    pd.DataFrame(rows).to_csv(out / "gpu.csv", index=False)
    np.savez_compressed(out / "gpu_latencies.npz", **lats)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()
    write_meta(out, cfg, stage="s10_bench_gpu", env_full=env_report())
    print(pd.DataFrame(rows)[["model", "precision", "batch", "median_ms", "iqr_ms", "p95_ms", "throughput_img_s",
                              "peak_allocated_mb", "gflops"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
