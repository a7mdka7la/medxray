"""Central configuration and path constants.

Everything that depends on disk layout, model names, or environment variables
goes through this module so the rest of the package stays decoupled.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    # dotenv is optional — env vars set in the shell still work.
    pass


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
SAMPLE_IMAGES_DIR = DATA_DIR / "sample_images"
SAMPLE_REPORTS_CSV = DATA_DIR / "sample_reports.csv"
QA_DATASET_DIR = DATA_DIR / "qa_dataset"
INDEX_DIR = DATA_DIR / "index"

OUTPUTS_DIR = ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def _pick_device() -> str:
    requested = os.getenv("DEVICE", "auto").lower()
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class ModelConfig:
    medgemma: str = field(default_factory=lambda: os.getenv("MEDGEMMA_MODEL", "google/medgemma-4b-it"))
    colpali: str = field(default_factory=lambda: os.getenv("COLPALI_MODEL", "vidore/colpali-v1.3"))
    clip: str = field(default_factory=lambda: os.getenv("CLIP_MODEL", "openai/clip-vit-base-patch32"))


@dataclass
class RuntimeConfig:
    device: str = field(default_factory=_pick_device)
    hf_token: str | None = field(default_factory=lambda: os.getenv("HUGGINGFACE_TOKEN") or None)
    google_api_key: str | None = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY") or None)
    max_new_tokens: int = 512
    retrieval_top_k: int = 4
    rng_seed: int = 13


MODELS = ModelConfig()
RUNTIME = RuntimeConfig()


REPORT_TEMPLATE = (
    "FINDINGS:\n{findings}\n\n"
    "IMPRESSION:\n{impression}\n\n"
    "RECOMMENDATIONS:\n{recommendations}\n"
)


REPORT_SECTIONS = ("findings", "impression", "recommendations")


def ensure_dirs() -> None:
    for d in (DATA_DIR, SAMPLE_IMAGES_DIR, QA_DATASET_DIR, INDEX_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
