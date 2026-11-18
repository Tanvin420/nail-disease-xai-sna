"""Near-duplicate detection: dihedral-invariant pHash, SSCD copy-detection embeddings, all-pairs search
and connected components.

Similarity of images a, b = max over the 8 dihedral variants t (4 rotations x optional flip) of
<e_0(a), e_t(b)> for SSCD (cosine), and min over t of Hamming(h_0(a), h_t(b)) for pHash.
"""
from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fft import dct
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def dihedral_np(a: np.ndarray, t: int) -> np.ndarray:
    b = np.rot90(a, t % 4)
    return np.fliplr(b) if t >= 4 else b


def dihedral_torch(x, t: int):
    import torch

    y = torch.rot90(x, t % 4, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if t >= 4 else y


# ---------------------------------------------------------------------------------------------
# pHash (same definition as imagehash.phash: 32x32 grey, 2-D DCT-II, 8x8 low band > median)
def phash_bits(gray32: np.ndarray) -> np.ndarray:
    d = dct(dct(gray32, axis=0), axis=1)[:8, :8]
    return (d > np.median(d)).reshape(-1)


def phash_variants(path: str) -> np.ndarray:
    g = np.asarray(Image.open(path).convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    return np.stack([phash_bits(dihedral_np(g, t)) for t in range(8)])  # (8, 64) bool


# ---------------------------------------------------------------------------------------------
# SSCD (Pizzi et al., CVPR 2022) TorchScript model
def load_sscd(url: str, cache_dir: Path, device):
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / Path(url).name
    if not f.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, f)
    return torch.jit.load(str(f), map_location=device).eval()


def _to_tensor(im: Image.Image, size: int):
    import torch

    im = im.convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).float().div_(255)
    return (x - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)


class _ImgDS:
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return _to_tensor(Image.open(self.paths[i]), self.size)


def sscd_embed(model, paths: list[str], size: int, batch: int, device, workers: int = 8):
    """Return float16 array (N, 8, D) of L2-normalised descriptors for the 8 dihedral variants."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    dl = DataLoader(_ImgDS(paths, size), batch_size=batch, num_workers=workers, pin_memory=device.type == "cuda")
    out = []
    with torch.inference_mode():
        for x in tqdm(dl, desc="SSCD"):
            x = x.to(device, non_blocking=True)
            e = torch.stack([torch.nn.functional.normalize(model(dihedral_torch(x, t)), dim=-1) for t in range(8)], 1)
            out.append(e.half().cpu().numpy())
    return np.concatenate(out)


def embed_single(model, imgs: list[Image.Image], size: int, device):
    """Identity-orientation SSCD descriptors for a list of PIL images (used for calibration)."""
    import torch

    with torch.inference_mode():
        e = model(torch.stack([_to_tensor(im, size) for im in imgs]).to(device))
        return torch.nn.functional.normalize(e, dim=-1).half().cpu().numpy()


# ---------------------------------------------------------------------------------------------
# All-pairs search on GPU (falls back to CPU)
def search_pairs(emb: np.ndarray, ph: np.ndarray, topk: int, min_sscd: float, max_ham: int, device,
                 chunk: int = 1024):
    """Candidate pairs (i, j, sscd, ham) with i < j from top-k neighbours under either similarity."""
    import torch

    n = emb.shape[0]
    e0 = torch.from_numpy(emb[:, 0]).to(device)
    e8 = torch.from_numpy(emb.reshape(n * 8, -1)).to(device)
    pm = torch.from_numpy(ph.astype(np.float32) * 2 - 1).to(device).half()  # (N, 8, 64) in {-1, +1}
    p0, p8 = pm[:, 0], pm.reshape(n * 8, 64)
    if device.type == "cpu":
        e0, e8, p0, p8 = e0.float(), e8.float(), p0.float(), p8.float()
    k = min(topk + 1, n)
    rows = []
    for s in range(0, n, chunk):
        q = slice(s, min(s + chunk, n))
        ar = torch.arange(q.start, q.stop, device=device)
        sim = (e0[q] @ e8.T).view(-1, n, 8).amax(-1).float()
        sim[torch.arange(len(ar)), ar] = -2
        ham = ((64 - (p0[q] @ p8.T).float()) / 2).view(-1, n, 8).amin(-1)
        ham[torch.arange(len(ar)), ar] = 99
        sv, si = sim.topk(k, dim=1)
        hv, hi = (-ham).topk(k, dim=1)
        for vals, idx, keep in ((sv, si, sv >= min_sscd), (-hv, hi, -hv <= max_ham)):
            r, c = keep.nonzero(as_tuple=True)
            if len(r):
                j = idx[r, c]
                i = ar[r]
                rows.append(torch.stack([i, j], 1).cpu().numpy())
        del sim, ham
    if not rows:
        return np.zeros((0, 2), int), np.zeros(0), np.zeros(0)
    ij = np.concatenate(rows)
    ij = np.unique(np.sort(ij, 1), axis=0)
    ij = ij[ij[:, 0] != ij[:, 1]]
    sscd, ham = pair_scores(emb, ph, ij, device)
    return ij, sscd, ham


def pair_scores(emb: np.ndarray, ph: np.ndarray, ij: np.ndarray, device, chunk: int = 200_000):
    """Symmetric SSCD (max of both directions, max over variants) and min dihedral Hamming for given pairs."""
    import torch

    sscd, ham = np.empty(len(ij), np.float32), np.empty(len(ij), np.int16)
    E = torch.from_numpy(emb).to(device).float()
    P = torch.from_numpy(ph.astype(np.float32)).to(device)
    for s in range(0, len(ij), chunk):
        a = torch.from_numpy(ij[s:s + chunk, 0]).to(device)
        b = torch.from_numpy(ij[s:s + chunk, 1]).to(device)
        s1 = torch.einsum("nd,ntd->nt", E[a, 0], E[b]).amax(1)
        s2 = torch.einsum("nd,ntd->nt", E[b, 0], E[a]).amax(1)
        sscd[s:s + chunk] = torch.maximum(s1, s2).cpu().numpy()
        h = (P[a, 0].unsqueeze(1) != P[b]).sum(-1).amin(1)
        ham[s:s + chunk] = h.cpu().numpy()
    return sscd, ham


# ---------------------------------------------------------------------------------------------
def components(n: int, edges: np.ndarray) -> np.ndarray:
    """Connected-component label per node for an undirected edge list (E, 2)."""
    if len(edges) == 0:
        return np.arange(n)
    g = coo_matrix((np.ones(len(edges), np.int8), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


def chain_edges(groups) -> np.ndarray:
    """Edges linking consecutive members of each group (iterable of index arrays)."""
    out = [np.stack([g[:-1], g[1:]], 1) for g in groups if len(g) > 1]
    return np.concatenate(out) if out else np.zeros((0, 2), int)


# ---------------------------------------------------------------------------------------------
# Synthetic Roboflow-style copies for threshold calibration
def roboflow_like(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """90-degree rotation, +/-15 degree rotation and +/-10 degree shear (both axes), as in the ko-pin export."""
    img = img.rotate(90 * int(rng.integers(4)), expand=True)
    img = img.rotate(float(rng.uniform(-15, 15)), resample=Image.BICUBIC, fillcolor=(0, 0, 0))
    sx, sy = (math.tan(math.radians(float(rng.uniform(-10, 10)))) for _ in range(2))
    w, h = img.size
    cx, cy = w / 2, h / 2
    # inverse affine for PIL: x_in = a x + b y + c ; y_in = d x + e y + f
    a, b, d, e = 1, sx, sy, 1
    c, f = cx - a * cx - b * cy, cy - d * cx - e * cy
    return img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC, fillcolor=(0, 0, 0))
