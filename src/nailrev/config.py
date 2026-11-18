"""Configuration loading, path resolution and run metadata."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]  # .../HTL_Revision


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Config(dict):
    """dict with attribute access and resolved paths."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        p = p if p.is_absolute() else ROOT / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def classes(self) -> list[str]:
        return list(self["classes"])

    @property
    def class_to_idx(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self["classes"])}

    def source(self, sid: str) -> dict:
        for s in self["sources"]:
            if s["id"] == sid:
                return s
        raise KeyError(sid)

    def hash(self) -> str:
        return hashlib.sha256(json.dumps(self, sort_keys=True, default=str).encode()).hexdigest()[:12]


def load_config(path: str | os.PathLike | None = None) -> Config:
    cfg = yaml.safe_load(open(path or ROOT / "configs" / "default.yaml", encoding="utf-8"))
    # machine-specific overrides: configs/local.yaml, then $NAILREV_OVERRIDE (used for smoke tests)
    for extra in [ROOT / "configs" / "local.yaml", os.environ.get("NAILREV_OVERRIDE")]:
        if extra and Path(extra).exists():
            cfg = _merge(cfg, yaml.safe_load(open(extra, encoding="utf-8")) or {})
    return Config(cfg)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def env_report() -> dict:
    """Software/hardware versions for the reproducibility table."""
    rep = {"python": sys.version.split()[0], "platform": platform.platform(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    for mod in ["torch", "torchvision", "timm", "numpy", "pandas", "sklearn", "scipy", "networkx", "captum",
                "pytorch_grad_cam", "lime", "onnx", "onnxruntime", "PIL"]:
        try:
            m = __import__(mod)
            rep[mod] = getattr(m, "__version__", "?")
        except Exception:
            rep[mod] = None
    try:
        import torch

        rep["cuda"] = torch.version.cuda
        rep["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            rep["gpu"] = torch.cuda.get_device_name(0)
            rep["gpu_capability"] = ".".join(map(str, torch.cuda.get_device_capability(0)))
    except Exception:
        pass
    try:
        rep["nvidia_driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        rep["nvidia_driver"] = None
    try:
        rep["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                           timeout=10).stdout.strip() or None
    except Exception:
        rep["git_commit"] = None
    return rep


def write_meta(out_dir: Path, cfg: Config, stage: str, **extra) -> None:
    """Record config hash, software versions and stage outputs next to a stage's results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"stage": stage, "config_hash": cfg.hash(), "env": env_report(), **extra}
    (out_dir / f"_meta_{stage}.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
