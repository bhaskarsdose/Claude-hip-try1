"""Central configuration for the hip-reconstruction pipeline.

Paths are resolved relative to the repo root so the package works whether
imported from a checkout or an installed wheel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = REPO_ROOT / "models"
OUTPUTS_DIR = REPO_ROOT / "outputs"
RUNS_DIR = REPO_ROOT / "runs"

CHECKPOINT_PATH = MODELS_DIR / "unet3d_hip.pt"

VOXEL_SIZE = 128
TARGET_SPACING_MM = 1.5


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_fraction: float = 0.1
    num_workers: int = 2
    amp: bool = True
    log_every: int = 10
    save_every_epochs: int = 5
    unet_channels: tuple[int, ...] = field(default_factory=lambda: (32, 64, 128, 256, 512))
    unet_strides: tuple[int, ...] = field(default_factory=lambda: (2, 2, 2, 2))


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR, OUTPUTS_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)
