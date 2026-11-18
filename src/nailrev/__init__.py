"""Reproducible pipeline for the HTL-2026-05-0084 revision (nail-condition classification)."""
__version__ = "1.0.0"

import os
from pathlib import Path

# Keep downloaded model weights (timm / Hugging Face Hub, torch hub) inside the project folder, not in ~/.cache.
# Must run before timm / huggingface_hub are imported. Deliberately overrides any machine-wide HF_HOME / TORCH_HOME
# so that this project never writes model weights outside its own folder.
_CACHE = Path(__file__).resolve().parents[2] / "data" / "cache"
os.environ["HF_HOME"] = str(_CACHE / "hf")
os.environ["TORCH_HOME"] = str(_CACHE / "torch")
