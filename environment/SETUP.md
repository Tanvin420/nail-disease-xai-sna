# Environment setup (RTX 5090 workstation)

Tested target: Windows 11 or Linux, Python 3.12, NVIDIA driver >= 570, PyTorch >= 2.7 with CUDA 12.8 wheels
(the RTX 5090 / Blackwell sm_120 is not supported by older wheels).

```powershell
# Windows (PowerShell) - from the HTL_Revision folder
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r environment/requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

```bash
# Linux
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r environment/requirements.txt
```

After the environment works, record the exact versions for the paper:
`pip freeze > environment/requirements.lock.txt` (the stage scripts also write versions into every `_meta_*.json`).

Notes
- First training run downloads the ImageNet checkpoints from the Hugging Face Hub (internet needed once).
- Stage 02 downloads the SSCD copy-detection model (~95 MB) from dl.fbaipublicfiles.com once.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set automatically by `s06_train.py` for deterministic cuBLAS.
- If DataLoader workers crash on Windows, lower `train.num_workers` in `configs/local.yaml`.
- Machine-specific overrides go in `configs/local.yaml` (same keys as `configs/default.yaml`; git-ignored).
