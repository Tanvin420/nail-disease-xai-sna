"""Attribution maps (Grad-CAM, Integrated Gradients, LIME) and their quantitative evaluation.

Faithfulness: deletion / insertion AUC (Petsiuk et al., 2018). Pixels are ranked by saliency and progressively
replaced by (deletion) or restored from (insertion) a Gaussian-blurred copy of the image; the curve is the
softmax probability of the explained class. Lower deletion AUC / higher insertion AUC = more faithful.
Every method is compared with a random map and a centre-prior map on the same images.
Sanity check (Adebayo et al., 2018): Spearman correlation between maps of the trained model and of a model whose
top stage and classifier were re-initialised (low correlation = the map depends on learned weights).
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torchvision.transforms.functional import gaussian_blur


def _norm01(m: np.ndarray) -> np.ndarray:
    m = m - m.min()
    return m / m.max() if m.max() > 0 else m


def gradcam_maps(model, layer, reshape, x: torch.Tensor, targets: list[int]) -> np.ndarray:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

    with GradCAM(model=model, target_layers=[layer], reshape_transform=reshape) as cam:
        return cam(input_tensor=x, targets=[ClassifierOutputTarget(int(t)) for t in targets])


def ig_maps(model, x: torch.Tensor, targets: list[int], baseline: torch.Tensor, steps: int, internal_batch: int):
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(model)
    maps, deltas = [], []
    for i in range(len(x)):
        a, d = ig.attribute(x[i:i + 1], baselines=baseline, target=int(targets[i]), n_steps=steps,
                            internal_batch_size=internal_batch, method="gausslegendre", return_convergence_delta=True)
        maps.append(_norm01(a.abs().sum(1)[0].detach().cpu().numpy()))
        deltas.append(float(d.abs().item()))
    return np.stack(maps), np.array(deltas)


def lime_maps(model, imgs_uint8: list[np.ndarray], targets: list[int], mean, std, device, cfgx: dict, seed: int):
    from lime import lime_image
    from lime.wrappers.scikit_image import SegmentationAlgorithm

    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def clf(batch):
        x = torch.from_numpy(np.asarray(batch, dtype=np.float32) / 255.0).permute(0, 3, 1, 2).to(device)
        return F.softmax(model((x - m) / s), 1).cpu().numpy()

    q = cfgx["lime_quickshift"]
    seg = SegmentationAlgorithm("quickshift", kernel_size=q["kernel_size"], max_dist=q["max_dist"], ratio=q["ratio"],
                                random_seed=seed)
    exp = lime_image.LimeImageExplainer(random_state=seed)
    import inspect

    extra = {"progress_bar": False} if "progress_bar" in inspect.signature(exp.explain_instance).parameters else {}
    out = []
    for img, t in zip(imgs_uint8, targets):
        e = exp.explain_instance(img, clf, labels=(int(t),), top_labels=None, hide_color=None,
                                 num_samples=cfgx["lime_samples"], num_features=cfgx["lime_num_features"],
                                 segmentation_fn=seg, batch_size=128, random_seed=seed, **extra)
        w = dict(e.local_exp[int(t)])
        mp = np.vectorize(lambda k: max(w.get(k, 0.0), 0.0))(e.segments).astype(np.float32)
        out.append(_norm01(mp))
    return np.stack(out)


def centre_prior(S: int) -> np.ndarray:
    yy, xx = np.mgrid[0:S, 0:S]
    return np.exp(-(((xx - S / 2) ** 2 + (yy - S / 2) ** 2) / (2 * (S / 4) ** 2)))


@torch.inference_mode()
def deletion_insertion(model, x: torch.Tensor, sal: np.ndarray, target: int, steps: int) -> tuple[float, float]:
    """x: (1,3,S,S) normalised input; sal: (S,S) map (any scale)."""
    S = x.shape[-1]
    blur = gaussian_blur(x, kernel_size=[51, 51], sigma=[15.0, 15.0])
    order = torch.from_numpy(np.argsort(-sal.reshape(-1), kind="stable").copy()).to(x.device)
    n = S * S
    cuts = [int(round(n * k / steps)) for k in range(steps + 1)]
    masks = torch.zeros(steps + 1, 1, n, device=x.device)
    for k, c in enumerate(cuts):
        masks[k, 0, order[:c]] = 1
    masks = masks.view(steps + 1, 1, S, S)
    dele = x * (1 - masks) + blur * masks
    ins = blur * (1 - masks) + x * masks
    p_del = F.softmax(model(dele), 1)[:, target].float().cpu().numpy()
    p_ins = F.softmax(model(ins), 1)[:, target].float().cpu().numpy()
    fr = np.linspace(0, 1, steps + 1)
    return float(np.trapezoid(p_del, fr)), float(np.trapezoid(p_ins, fr))


def smooth_random_map(S: int, grid: int, rng: np.random.Generator) -> np.ndarray:
    """Random saliency with the same spatial coarseness as a CAM: i.i.d. noise on a grid x grid lattice, bilinearly upsampled to S x S.
    A fair random baseline for Grad-CAM; per-pixel noise (the `random` baseline) deletes scattered pixels, which damages a CNN far
    more than deleting one coherent region and therefore makes any region-based method look worse than chance."""
    g = torch.from_numpy(rng.random((1, 1, grid, grid)).astype(np.float32))
    return F.interpolate(g, size=(S, S), mode="bilinear", align_corners=False)[0, 0].numpy()


TOP_STAGE = {
    "vgg16": lambda m: [m.features[24:], m.pre_logits, m.get_classifier()],
    "densenet121": lambda m: [m.features.denseblock4, m.features.norm5, m.get_classifier()],
    "efficientnet_b2": lambda m: [m.blocks[-1], m.conv_head, m.bn2, m.get_classifier()],
    "mobilenetv3": lambda m: [m.blocks[-1], m.conv_head, m.get_classifier()],
    "vit_b16": lambda m: [m.blocks[-1], m.norm, m.get_classifier()],
}


def randomised_copy(model, key: str, seed: int):
    torch.manual_seed(seed)
    r = copy.deepcopy(model)
    for part in TOP_STAGE[key](r):
        for mod in part.modules():
            if hasattr(mod, "reset_parameters"):
                mod.reset_parameters()
    return r.eval()


def map_spearman(a: np.ndarray, b: np.ndarray, size: int = 56) -> float:
    ta = F.adaptive_avg_pool2d(torch.from_numpy(a)[None, None].float(), size).flatten().numpy()
    tb = F.adaptive_avg_pool2d(torch.from_numpy(b)[None, None].float(), size).flatten().numpy()
    return float(spearmanr(ta, tb).statistic)
