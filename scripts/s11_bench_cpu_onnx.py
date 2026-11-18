"""Stage 11 - ONNX export (fp32 + dynamic int8), numerical/accuracy checks and CPU latency with ONNX Runtime.

For each architecture (repeat-0/fold-0 weights):
  * export to runs/onnx/<model>.onnx (opset 17, dynamic batch) and verify against PyTorch;
  * dynamic int8 quantisation of MatMul/Gemm layers -> runs/onnx/<model>_int8.onnx;
  * top-1 accuracy / agreement of fp32 vs int8 on the r0/f0 internal test fold (ONNX Runtime CPU);
  * CPU latency at batch 1 for intra-op threads in analysis.bench.cpu_threads (0 = all physical cores).
Writes results/bench/{cpu.csv, onnx_checks.csv} and runs/onnx/models.json (used by the phone benchmark page).
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from nailrev.config import load_config, write_meta  # noqa: E402
from nailrev.data import build_transform  # noqa: E402
from nailrev.models import create_model, load_checkpoint, norm_stats  # noqa: E402


def export(model, S, f: Path):
    x = torch.randn(1, 3, S, S)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, (x,), str(f), dynamo=False, **kw)
    except Exception as e:  # legacy exporter removed in a future PyTorch -> dynamo exporter
        print("legacy exporter failed, trying dynamo:", e)
        torch.onnx.export(model, (x,), str(f), dynamo=True, **kw)


def session(path, threads):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def physical_cores():
    try:
        import psutil

        return psutil.cpu_count(logical=False) or os.cpu_count()
    except ImportError:
        return max(1, (os.cpu_count() or 2) // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args()
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from torchvision.io import ImageReadMode, decode_image, read_file

    cfg = load_config()
    bc, K = cfg.analysis["bench"], len(cfg.classes)
    onnx_dir = cfg.path("runs") / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.path("results") / "bench"
    out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(cfg.path("manifests") / "index.csv", keep_default_na=False).set_index("image_id")
    img_dir = cfg.path("cache") / f"img{cfg.image['cache_size']}"
    models = list(cfg.models) if args.models == "all" else args.models.split(",")
    cores = physical_cores()
    checks, rows, manifest = [], [], []
    for m in models:
        mc = cfg.models[m]
        ck = cfg.path("runs") / "cv" / m / "r0_f0" / "best.pt"
        model = create_model(mc, K, pretrained=False)
        if ck.exists():
            load_checkpoint(model, ck)
        model.eval()
        S = mc["size"]
        f32, i8 = onnx_dir / f"{m}.onnx", onnx_dir / f"{m}_int8.onnx"
        export(model, S, f32)
        # MatMul/Gemm only: ONNX Runtime's dynamic ConvInteger kernels are slower than fp32 convolutions
        quantize_dynamic(str(f32), str(i8), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"])
        x = np.random.default_rng(0).standard_normal((4, 3, S, S)).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        diff = float(np.abs(session(f32, cores).run(None, {"input": x})[0] - ref).max())
        chk = {"model": m, "fp32_mb": f32.stat().st_size / 2**20, "int8_mb": i8.stat().st_size / 2**20,
               "fp32_max_abs_diff_vs_torch": diff, "trained_weights": ck.exists()}
        test_f = cfg.path("results") / "predictions" / "cv" / m / "r0_f0" / "test.parquet"
        if not args.skip_accuracy and test_f.exists():
            t = pd.read_parquet(test_f)
            mean, std = norm_stats(model)
            tf = build_transform("eval", S, mean, std)
            preds = {"fp32": [], "int8": []}
            sess = {"fp32": session(f32, cores), "int8": session(i8, cores)}
            ids = t.image_id.tolist()
            for s in range(0, len(ids), 32):
                xb = torch.stack([tf(decode_image(read_file(str(img_dir / idx.cache_path[i])), mode=ImageReadMode.RGB))
                                  for i in ids[s:s + 32]]).numpy()
                for k in preds:
                    preds[k].append(sess[k].run(None, {"input": xb})[0].argmax(1))
            p32, p8 = np.concatenate(preds["fp32"]), np.concatenate(preds["int8"])
            chk.update(n_test=len(ids), acc_fp32=float((p32 == t.y).mean()), acc_int8=float((p8 == t.y).mean()),
                       top1_agreement=float((p32 == p8).mean()),
                       acc_torch=float((t.filter(like="logit_").to_numpy().argmax(1) == t.y).mean()))
        checks.append(chk)
        for variant, f in (("fp32", f32), ("int8", i8)):
            manifest.append({"model": m, "label": mc["label"], "variant": variant, "file": f.name, "size": S,
                             "mb": round(f.stat().st_size / 2**20, 1)})
            for th in bc["cpu_threads"]:
                n = th or cores
                s = session(f, n)
                xi = np.random.default_rng(1).standard_normal((1, 3, S, S)).astype(np.float32)
                for _ in range(bc["cpu_warmup"]):
                    s.run(None, {"input": xi})
                lat = []
                for _ in range(3):
                    for _ in range(bc["cpu_iters"]):
                        t0 = time.perf_counter()
                        s.run(None, {"input": xi})
                        lat.append((time.perf_counter() - t0) * 1000)
                lat = np.array(lat)
                q1, med, q3, p90, p95 = np.percentile(lat, [25, 50, 75, 90, 95])
                rows.append({"model": m, "variant": variant, "threads": n, "median_ms": med, "iqr_ms": q3 - q1,
                             "p90_ms": p90, "p95_ms": p95, "mean_ms": lat.mean(), "sd_ms": lat.std(ddof=1),
                             "cpu": platform.processor(), "physical_cores": cores})
                print(f"{m:16s} {variant} threads={n:2d}  median {med:8.2f} ms  p90 {p90:8.2f} ms", flush=True)
    pd.DataFrame(checks).to_csv(out / "onnx_checks.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "cpu.csv", index=False)
    (onnx_dir / "models.json").write_text(json.dumps(manifest, indent=2))
    write_meta(out, cfg, stage="s11_bench_cpu_onnx", cpu=platform.processor(), physical_cores=cores)
    print(pd.DataFrame(checks).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
