"""PyTorch Dataset for shape-completion training.

Two flavours:

    HipDefectDataset  -- loads pre-processed 128**3 .npy volumes from disk
                         and synthesises a defect on-the-fly each __getitem__.
    SyntheticEllipsoidDataset
                      -- generates random ellipsoid "bones" in memory; used
                         by the smoke-training mode and unit tests so the
                         pipeline can run without any real CT data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import VOXEL_SIZE
from .defects import random_defect


def _augment(volume: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # Random axis flips (cheap, label-preserving for shape completion when
    # applied symmetrically to defective + complete)
    if rng.random() < 0.5:
        volume = volume[::-1]
    if rng.random() < 0.5:
        volume = volume[:, ::-1]
    if rng.random() < 0.5:
        volume = volume[:, :, ::-1]
    # Random 90-deg rotations
    k = int(rng.integers(0, 4))
    axes = ((0, 1), (1, 2), (0, 2))[int(rng.integers(0, 3))]
    if k:
        volume = np.rot90(volume, k=k, axes=axes)
    return np.ascontiguousarray(volume)


def _make_ellipsoid(grid: int, rng: np.random.Generator) -> np.ndarray:
    z, y, x = np.indices((grid, grid, grid))
    cz, cy, cx = grid / 2, grid / 2, grid / 2
    rz = rng.uniform(grid * 0.18, grid * 0.32)
    ry = rng.uniform(grid * 0.20, grid * 0.36)
    rx = rng.uniform(grid * 0.18, grid * 0.32)
    vol = (((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2) <= 1.0
    return vol.astype(np.float32)


class SyntheticEllipsoidDataset(Dataset):
    """In-memory dataset of random ellipsoids with random defects.

    Useful for smoke-tests: lets us verify the model + training loop converge
    before any real data is available.
    """

    def __init__(self, n: int = 32, grid: int = VOXEL_SIZE, seed: int = 0):
        self.n = n
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        self._cache: list[np.ndarray] = [_make_ellipsoid(grid, self.rng) for _ in range(n)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        complete = self._cache[idx]
        rng = np.random.default_rng(idx + 1)
        defective, _, _ = random_defect(complete, rng=rng)
        return (
            torch.from_numpy(defective[None]).float(),
            torch.from_numpy(complete[None]).float(),
        )


class HipDefectDataset(Dataset):
    """Loads pre-processed 128**3 hip volumes and produces defects on-the-fly.

    `deterministic=True` (used for validation) seeds the per-sample RNG from
    the file index alone, so every epoch sees the same defect on the same
    file. Without this, val_dice oscillates randomly because each epoch
    re-rolls the defect cut on the val files.
    """

    def __init__(
        self,
        root: str | Path,
        augment: bool = True,
        grid: int = VOXEL_SIZE,
        seed: int = 0,
        files: list[Path] | None = None,
        deterministic: bool = False,
    ):
        self.root = Path(root)
        if files is not None:
            self.files = list(files)
        else:
            self.files = sorted(self.root.glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npy volumes found in {self.root}. "
                "Run scripts/prepare_dataset.py first."
            )
        self.augment = augment
        self.grid = grid
        self.rng = np.random.default_rng(seed)
        self.deterministic = deterministic
        self.seed = seed

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        complete = np.load(self.files[idx]).astype(np.float32)
        if complete.shape != (self.grid,) * 3:
            raise ValueError(
                f"Expected {(self.grid,) * 3}, got {complete.shape} for {self.files[idx]}"
            )
        if self.deterministic:
            # Same defect for the same file every epoch
            rng = np.random.default_rng(self.seed * 100003 + idx)
        else:
            rng = np.random.default_rng(self.rng.integers(0, 2**31 - 1))
        if self.augment:
            complete = _augment(complete, rng)
        defective, _, _ = random_defect(complete, rng=rng)
        return (
            torch.from_numpy(defective[None]).float(),
            torch.from_numpy(complete[None]).float(),
        )
