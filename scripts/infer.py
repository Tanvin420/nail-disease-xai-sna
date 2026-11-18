"""Predict nail conditions with any trained model of this project (every run is saved as runs/<protocol>/<model>/<run>/best.pt).

  python scripts/infer.py --list                                              # available checkpoints
  python scripts/infer.py --model efficientnet_b2 --images photo.jpg          # default: protocol cv, run r0_f0
  python scripts/infer.py --model vit_b16 --protocol xsrc --run r0_f0 --images folder/ --csv out.csv
  python scripts/infer.py --model efficientnet_b2 --ensemble --images folder/ # mean of the softmax over all runs of the model
  python scripts/infer.py --ckpt runs/cv/vgg16/r0_f3 --images a.jpg b.png --top 3 --calibrate

Default is fp32 (slightly more precise than the pipeline's bf16 evaluation; same predicted classes); --amp reproduces the
pipeline's numbers. Preprocessing is rebuilt from the checkpoint's meta.json exactly as in training: stretch to cache_size x cache_size (the
development images were stretched, not cropped), resize to the model's input size, ImageNet-style normalisation.
--calibrate divides the logits by the run's validation temperature (results/metrics/epochs_and_temperature.csv, written by
stage 07). Research prototype only: not a medical device; probabilities are not clinically validated.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms import v2  # noqa: E402

from nailrev.config import load_config  # noqa: E402
from nailrev.data import build_transform  # noqa: E402
from nailrev.models import create_model, load_checkpoint  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}


def find_runs(root: Path, protocol: str, model: str | None) -> list[Path]:
    base = root / protocol
    pat = f"{model}/r*_f*" if model else "*/r*_f*"
    return sorted(p.parent for p in base.glob(pat + "/best.pt"))


def load_run(cfg, run_dir: Path, device):
    meta = json.loads((run_dir / "meta.json").read_text())
    classes = meta.get("classes", cfg.classes)
    model = create_model(cfg.models[meta["model"]], len(classes), pretrained=False)
    load_checkpoint(model, run_dir / "best.pt")
    model = model.to(device).eval()
    cache = meta.get("cache_size", cfg.image["cache_size"])
    tf = v2.Compose([v2.Resize((cache, cache), antialias=True),
                     build_transform("eval", meta["size"], tuple(meta["mean"]), tuple(meta["std"]))])
    return model, tf, classes, meta


def read_image(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy()).permute(2, 0, 1)


def collect(paths: list[str]) -> list[Path]:
    out = []
    for p in map(Path, paths):
        out += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    if not out:
        sys.exit("no images found")
    return out


def temperatures(cfg) -> dict:
    f = cfg.path("results") / "metrics" / "epochs_and_temperature.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    return {(r.protocol, r.model, r.run): float(r.T) for r in d.itertuples()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="*", default=[], help="image files and/or folders")
    ap.add_argument("--model", help="vgg16 | densenet121 | efficientnet_b2 | mobilenetv3 | vit_b16")
    ap.add_argument("--protocol", default="cv", help="cv | leaky | vendor | msrc | xsrc | cv_fixed5 | ...")
    ap.add_argument("--run", default="r0_f0", help="run folder name, e.g. r0_f3 (protocol cv: repeat 0..2, fold 0..4)")
    ap.add_argument("--ckpt", help="run folder (contains best.pt and meta.json); overrides --protocol/--model/--run")
    ap.add_argument("--ensemble", action="store_true", help="average the softmax of all runs of --model in --protocol")
    ap.add_argument("--calibrate", action="store_true", help="temperature-scale logits with each run's validation temperature")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast on GPU = numerically identical to the training pipeline's predictions")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--csv", help="write all class probabilities to this CSV")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = load_config()
    root = cfg.path("runs")

    if args.list:
        rows = []
        for mf in sorted(root.glob("*/*/r*_f*/meta.json")):
            m = json.loads(mf.read_text())
            if (mf.parent / "best.pt").exists():
                rows.append({"protocol": mf.parents[2].name, "model": m["model"], "run": mf.parent.name,
                             "trained_on": m.get("train_set") or m.get("holdout") and f"pool minus {m['holdout']}" or "ko-pin",
                             "best_epoch": m["best_epoch"], "MB": round((mf.parent / "best.pt").stat().st_size / 2**20, 1)})
        print(pd.DataFrame(rows).to_string(index=False) if rows else "no checkpoints found under runs/")
        return
    if not args.images:
        sys.exit("give --images (or --list)")

    if args.ckpt:
        run_dirs = [Path(args.ckpt)]
    elif args.ensemble:
        run_dirs = find_runs(root, args.protocol, args.model)
    else:
        if not args.model:
            sys.exit("give --model (or --ckpt)")
        run_dirs = [root / args.protocol / args.model / args.run]
    run_dirs = [d for d in run_dirs if (d / "best.pt").exists()]
    if not run_dirs:
        sys.exit("no checkpoint found; use --list to see what exists")

    device = torch.device(args.device)
    files = collect(args.images)
    temps = temperatures(cfg) if args.calibrate else {}
    probs, classes = None, None
    for d in run_dirs:
        model, tf, classes, meta = load_run(cfg, d, device)
        T = temps.get((d.parents[2].name, d.parents[1].name, d.name), 1.0) if args.calibrate else 1.0
        out = []
        with torch.inference_mode():
            for i in range(0, len(files), args.batch):
                x = torch.stack([tf(read_image(f)) for f in files[i:i + args.batch]]).to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
                    logits = model(x)
                out.append(torch.softmax(logits.float() / T, 1).cpu().numpy())
        p = np.concatenate(out)
        probs = p if probs is None else probs + p
        del model
    probs /= len(run_dirs)

    names = [cfg.display_names.get(c, c) for c in classes]
    print(f"{len(run_dirs)} model(s): {run_dirs[0].parents[1].name}/{run_dirs[0].parents[0].name}"
          f"{' (ensemble)' if len(run_dirs) > 1 else '/' + run_dirs[0].name}; {len(files)} image(s)\n")
    for f, p in zip(files, probs):
        top = np.argsort(-p)[:args.top]
        print(f"{f.name}: " + "  ".join(f"{names[j]} {p[j]:.3f}" for j in top))
    if args.csv:
        df = pd.DataFrame(probs, columns=[f"p_{c}" for c in classes])
        df.insert(0, "prediction", [classes[j] for j in probs.argmax(1)])
        df.insert(0, "file", [str(f) for f in files])
        df.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
