"""3D U-Net wrapper around MONAI for hip-bone shape completion.

Input:  (B, 1, D, H, W) defective binary volume
Output: (B, 1, D, H, W) logits over the *complete* bone volume
"""

from __future__ import annotations

from pathlib import Path

import torch
from monai.networks.nets import UNet

from ..config import TrainConfig


def build_unet(cfg: TrainConfig | None = None) -> UNet:
    cfg = cfg or TrainConfig()
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=cfg.unet_channels,
        strides=cfg.unet_strides,
        num_res_units=2,
        norm="batch",
    )


def save_checkpoint(model: torch.nn.Module, path: str | Path, **extra) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"state_dict": model.state_dict(), **extra}
    torch.save(payload, str(path))


def load_checkpoint(model: torch.nn.Module, path: str | Path, map_location="cpu"):
    payload = torch.load(str(path), map_location=map_location)
    model.load_state_dict(payload["state_dict"])
    return payload
