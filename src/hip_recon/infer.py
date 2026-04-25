"""End-to-end inference: NIfTI mask in -> implant STL out.

If no checkpoint is available we fall back to a no-op "placeholder" model so
the web stack is fully runnable before training completes. The web app uses
this module unconditionally; the only thing that changes when you drop a
trained checkpoint into models/ is the quality of the prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy import ndimage as ndi

from .config import CHECKPOINT_PATH, VOXEL_SIZE
from .data.preprocessing import (
    canonicalize,
    crop_to_content,
    load_nifti,
    resample_iso,
    voxels_to_mesh,
)
from .models.unet3d import build_unet, load_checkpoint


@dataclass
class ReconstructionResult:
    input_mesh: trimesh.Trimesh
    implant_mesh: trimesh.Trimesh
    used_trained_model: bool
    canonical_input: np.ndarray   # 128**3 float32, the model's input
    canonical_pred: np.ndarray    # 128**3 float32, the model's output (>=0.5)


_MODEL: torch.nn.Module | None = None
_DEVICE: torch.device | None = None
_USED_TRAINED: bool = False


def _get_model() -> tuple[torch.nn.Module, torch.device, bool]:
    """Lazily load the model the first time inference runs."""
    global _MODEL, _DEVICE, _USED_TRAINED
    if _MODEL is not None:
        assert _DEVICE is not None
        return _MODEL, _DEVICE, _USED_TRAINED

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device).eval()
    used_trained = False
    if Path(CHECKPOINT_PATH).is_file():
        try:
            load_checkpoint(model, CHECKPOINT_PATH, map_location=device)
            used_trained = True
            print(f"[infer] loaded checkpoint {CHECKPOINT_PATH}")
        except Exception as exc:  # malformed / mismatched checkpoint
            print(f"[infer] WARN: could not load checkpoint ({exc}); using untrained net")
    else:
        print("[infer] no checkpoint at", CHECKPOINT_PATH, "- running placeholder mode")

    _MODEL, _DEVICE, _USED_TRAINED = model, device, used_trained
    return model, device, used_trained


@torch.no_grad()
def _predict(model: torch.nn.Module, device: torch.device, canonical: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(canonical[None, None]).float().to(device)
    logits = model(x)
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    return (prob > 0.5).astype(np.float32)


def reconstruct_from_nifti(
    nifti_path: str | Path, grid: int = VOXEL_SIZE
) -> ReconstructionResult:
    volume, _affine, spacing = load_nifti(nifti_path)
    iso = resample_iso(volume, spacing)
    cropped, _ = crop_to_content(iso)
    canonical_input, _xform = canonicalize(cropped, grid_size=grid)

    model, device, used_trained = _get_model()
    canonical_pred = _predict(model, device, canonical_input)

    if not used_trained:
        # Placeholder mode: return the input shape, no implant. The UI still
        # renders something useful (the uploaded bone) so the UX is testable.
        canonical_pred = canonical_input.copy()

    implant_vol = ((canonical_pred > 0.5) & ~(canonical_input > 0.5)).astype(np.float32)
    if implant_vol.any():
        implant_vol = ndi.binary_closing(implant_vol > 0, iterations=1).astype(np.float32)

    input_mesh = voxels_to_mesh(canonical_input)
    implant_mesh = voxels_to_mesh(implant_vol)

    return ReconstructionResult(
        input_mesh=input_mesh,
        implant_mesh=implant_mesh,
        used_trained_model=used_trained,
        canonical_input=canonical_input,
        canonical_pred=canonical_pred,
    )
