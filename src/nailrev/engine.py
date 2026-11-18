"""Training loop with early stopping (validation macro-F1), and batched prediction."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from .data import ImageDS, build_transform, component_balanced_sampler, seed_worker
from .models import create_model, norm_stats, param_groups


def set_determinism(on: bool):
    torch.backends.cudnn.benchmark = not on
    torch.backends.cudnn.deterministic = on
    torch.use_deterministic_algorithms(on, warn_only=True)


def _loader(ds, batch, workers, sampler=None, gen=None, persistent=False):
    return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=persistent and workers > 0,
                      drop_last=sampler is not None, worker_init_fn=seed_worker, generator=gen)


@torch.inference_mode()
def predict_dl(model, dl, device, amp_dtype=None) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            out.append(model(x).float().cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 0), np.float32)


def predict(model, paths, transform, batch, workers, device, amp_dtype=None) -> np.ndarray:
    dl = _loader(ImageDS(paths, np.zeros(len(paths)), transform), batch, workers)
    return predict_dl(model, dl, device, amp_dtype)


def train_run(cfg, model_key: str, train_df: pd.DataFrame, val_df: pd.DataFrame, eval_sets: dict,
              out_dir: Path, pred_dir: Path, seed: int, device, fixed_epochs: int | None = None,
              lr_mult: float = 1.0, save_checkpoint: bool = False, extra_meta: dict | None = None) -> dict:
    """Train one model on one partition and write history, predictions and metadata.

    train_df: image_id, path, y, label, comp (all files of training components)
    val_df  : image_id, path, y (one representative per component; used for early stopping)
    eval_sets: name -> (df[image_id, path, y], transform kind)
    """
    tc, mcfg = cfg.train, cfg.models[model_key]
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    set_determinism(tc["deterministic"])
    amp = getattr(torch, tc["amp_dtype"]) if device.type == "cuda" and tc["amp_dtype"] else None

    model = create_model(mcfg, len(cfg.classes), pretrained=True).to(device)
    if device.type == "cuda" and "vit" not in model_key:
        model = model.to(memory_format=torch.channels_last)
    mean, std = norm_stats(model)
    size = mcfg["size"]
    t_train = build_transform("train", size, mean, std, tc["augment"])
    t_eval = build_transform("eval", size, mean, std)

    n_comp = train_df.comp.nunique()
    bs = tc["batch_size"]
    steps = max(1, min(tc["max_steps_per_epoch"], (tc["samples_per_epoch_factor"] * n_comp) // bs))
    gen = torch.Generator().manual_seed(seed)
    sampler = component_balanced_sampler(train_df, steps * bs, gen)
    dl = _loader(ImageDS(train_df.path.tolist(), train_df.y.to_numpy(), t_train), bs, tc["num_workers"],
                 sampler=sampler, gen=gen, persistent=True)
    yv = val_df.y.to_numpy()
    val_dl = _loader(ImageDS(val_df.path.tolist(), yv, t_eval), bs * 2, tc["num_workers"], persistent=True)
    max_epochs = fixed_epochs or tc["max_epochs"]
    lr = mcfg["lr"] * lr_mult
    opt = torch.optim.AdamW(param_groups(model, lr, mcfg["wd"], tc["head_lr_mult"]), betas=(0.9, 0.999))
    warm, total = tc["warmup_epochs"] * steps, max_epochs * steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    crit = torch.nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])

    hist, best, best_state, bad = [], (-1.0, 0), None, 0
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        te, tl, nb = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if device.type == "cuda" and "vit" not in model_key:
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            tl, nb = tl + loss.item(), nb + 1
        lv = predict_dl(model, val_dl, device, amp)
        vloss = float(torch.nn.functional.cross_entropy(torch.from_numpy(lv), torch.from_numpy(yv)))
        pv = lv.argmax(1)
        f1 = f1_score(yv, pv, average="macro", labels=range(len(cfg.classes)), zero_division=0)
        hist.append({"epoch": ep, "lr": opt.param_groups[0]["lr"], "train_loss": tl / max(nb, 1), "val_loss": vloss,
                     "val_acc": accuracy_score(yv, pv), "val_macro_f1": f1, "seconds": time.time() - te})
        print(f"  ep {ep:2d}  loss {tl / max(nb, 1):.4f}  val_loss {vloss:.4f}  val_F1 {f1:.4f}  ({time.time() - te:.0f}s)", flush=True)
        if fixed_epochs:
            continue
        if (f1, -vloss) > best:
            best, bad = (f1, -vloss), 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_epoch = ep
        else:
            bad += 1
            if ep >= tc["min_epochs"] and bad >= tc["patience"]:
                break
    if fixed_epochs:
        best_epoch = max_epochs
    else:
        model.load_state_dict(best_state)
    train_sec = time.time() - t0
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)
    if save_checkpoint:
        torch.save({k: v.float() if v.is_floating_point() else v for k, v in model.state_dict().items()},
                   out_dir / "best.pt")

    t_kind = {"eval": t_eval, "degrade": build_transform("degrade", size, mean, std, degrade=tc["degrade_size"])}
    n_eval = {}
    for name, (df, kind) in {"val": (val_df, "eval"), **eval_sets}.items():
        if len(df) == 0:
            continue
        lg = predict(model, df.path.tolist(), t_kind[kind], bs * 2, tc["num_workers"], device, amp)
        p = pd.DataFrame(lg.astype(np.float16), columns=[f"logit_{i}" for i in range(lg.shape[1])])
        p.insert(0, "y", df.y.to_numpy())
        p.insert(0, "image_id", df.image_id.to_numpy())
        p.to_parquet(pred_dir / f"{name}.parquet", index=False)
        n_eval[name] = len(df)
    meta = {"model": model_key, "timm": mcfg["timm"], "size": size, "lr": lr, "wd": mcfg["wd"], "seed": seed,
            "best_epoch": best_epoch, "epochs_run": len(hist), "fixed_epochs": fixed_epochs, "lr_mult": lr_mult,
            "n_train_images": len(train_df), "n_train_components": int(n_comp), "steps_per_epoch": steps,
            "n_eval": n_eval, "train_seconds": round(train_sec, 1), "mean": mean, "std": std,
            "classes": list(cfg.classes), "cache_size": cfg.image["cache_size"], "checkpoint": "best.pt" if save_checkpoint else None,
            **(extra_meta or {})}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    (pred_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta
