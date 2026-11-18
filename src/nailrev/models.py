"""Model construction (timm), optimiser parameter groups, and explanation target layers."""
from __future__ import annotations

import timm
import torch


def create_model(mcfg: dict, num_classes: int, pretrained: bool = True):
    return timm.create_model(mcfg["timm"], pretrained=pretrained, num_classes=num_classes)


def norm_stats(model) -> tuple[tuple, tuple]:
    dc = timm.data.resolve_model_data_config(model)
    return tuple(dc["mean"]), tuple(dc["std"])


def param_groups(model, lr: float, wd: float, head_mult: float) -> list[dict]:
    """New classifier head at head_mult x lr; no weight decay on biases and normalisation parameters."""
    head = {id(p) for p in model.get_classifier().parameters()}
    groups = {(h, d): [] for h in (0, 1) for d in (0, 1)}
    for n, p in model.named_parameters():
        if p.requires_grad:
            groups[(int(id(p) in head), int(p.ndim > 1 and not n.endswith(".bias")))].append(p)
    return [{"params": ps, "lr": lr * (head_mult if h else 1.0), "weight_decay": wd if d else 0.0, "base_lr": lr * (head_mult if h else 1.0)}
            for (h, d), ps in groups.items() if ps]


def disable_inplace(model):
    """Gradient-based explanations need non-inplace activations."""
    for m in model.modules():
        if hasattr(m, "inplace"):
            m.inplace = False
    return model


def cam_target(key: str, model):
    """(target layer, reshape_transform) for Grad-CAM; layers verified to give spatial maps
    (VGG16 14x14, DenseNet121 7x7, EfficientNet-B2 8x8 at 256 px, MobileNetV3-L 7x7, ViT-B/16 14x14 tokens)."""
    if key == "vgg16":
        return model.features[-2], None             # ReLU after conv5_3
    if key == "densenet121":
        return model.features.norm5, None           # final BN+ReLU
    if key == "efficientnet_b2":
        return model.bn2, None                      # BN+act after conv_head
    if key == "mobilenetv3":
        return model.blocks[-1], None               # last 960-ch block (timm applies conv_head after pooling)
    if key == "vit_b16":
        npt = getattr(model, "num_prefix_tokens", 1)

        def reshape(t):
            t = t[:, npt:, :]
            s = int(round(t.shape[1] ** 0.5))
            return t.reshape(t.shape[0], s, s, t.shape[2]).permute(0, 3, 1, 2)

        return model.blocks[-1].norm1, reshape
    raise KeyError(key)


def load_checkpoint(model, path, device="cpu"):
    sd = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    return model
