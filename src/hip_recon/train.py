"""Training entry point for the 3D U-Net hip shape-completion model.

Usage:
    # Smoke test on synthetic ellipsoids (CPU OK)
    python -m hip_recon.train --smoke

    # Real training (after scripts/prepare_dataset.py has produced .npy files)
    python -m hip_recon.train --data-dir data/processed --epochs 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from monai.losses import DiceLoss
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import CHECKPOINT_PATH, RUNS_DIR, TrainConfig, ensure_dirs
from .data.datasets import HipDefectDataset, SyntheticEllipsoidDataset
from .models.unet3d import build_unet, save_checkpoint


class DiceBCE(torch.nn.Module):
    def __init__(self, dice_w: float = 1.0, bce_w: float = 1.0):
        super().__init__()
        self.dice = DiceLoss(sigmoid=True)
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dw, self.bw = dice_w, bce_w

    def forward(self, logits, target):
        return self.dw * self.dice(logits, target) + self.bw * self.bce(logits, target)


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum().item()
    denom = pred.sum().item() + target.sum().item()
    return (2.0 * inter + eps) / (denom + eps)


def build_loaders(args, cfg: TrainConfig):
    if args.smoke:
        ds = SyntheticEllipsoidDataset(n=8)
        n_val = 2
    else:
        ds = HipDefectDataset(args.data_dir, augment=True)
        n_val = max(1, int(len(ds) * cfg.val_fraction))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, val_loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="Synthetic ellipsoid smoke test")
    p.add_argument("--data-dir", type=str, default="data/processed")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--out", type=str, default=str(CHECKPOINT_PATH))
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-amp", action="store_true", help="Disable mixed-precision (use if val_dice stays 0)")
    args = p.parse_args()

    cfg = TrainConfig()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr = args.lr
    if args.no_amp:
        cfg.amp = False
    if args.smoke:
        cfg.epochs = min(cfg.epochs, 3)
        cfg.batch_size = 1
        cfg.num_workers = 0

    ensure_dirs()
    device = torch.device(args.device)
    model = build_unet(cfg).to(device)
    loss_fn = DiceBCE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    train_loader, val_loader = build_loaders(args, cfg)

    run_dir = RUNS_DIR / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=str(run_dir))
    print(f"[train] device={device} epochs={cfg.epochs} runs={run_dir}")

    step = 0
    best_dice = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_skipped = 0  # AMP optimizer-skip counter
        for defective, complete in tqdm(train_loader, desc=f"epoch {epoch:03d}", leave=False):
            defective = defective.to(device, non_blocking=True)
            complete = complete.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=cfg.amp and device.type == "cuda"
            ):
                logits = model(defective)
                loss = loss_fn(logits, complete)
            if torch.isnan(loss):
                print(f"[warn] NaN loss at step {step} — skipping batch")
                continue
            scaler.scale(loss).backward()
            # Unscale before clipping so the clip threshold is in true gradient units
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            # Detect AMP-skipped steps (scale drops when inf/NaN gradients were found)
            if scaler.get_scale() < scale_before:
                n_skipped += 1
            epoch_loss += loss.item()
            if step % cfg.log_every == 0:
                writer.add_scalar("train/loss", loss.item(), step)
            step += 1

        avg_loss = epoch_loss / max(len(train_loader), 1)
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)

        # Validation
        model.eval()
        dices = []
        for defective, complete in val_loader:
            defective = defective.to(device)
            complete = complete.to(device)
            with torch.autocast(
                device_type=device.type, enabled=cfg.amp and device.type == "cuda"
            ):
                logits = model(defective)
            dices.append(dice_score(logits, complete))
        val_dice = sum(dices) / max(len(dices), 1)
        writer.add_scalar("val/dice", val_dice, epoch)
        skip_info = f"  amp_skipped={n_skipped}" if n_skipped else ""
        print(f"[epoch {epoch:03d}] train_loss={avg_loss:.4f}  val_dice={val_dice:.4f}{skip_info}")

        if val_dice > best_dice:
            best_dice = val_dice
            save_checkpoint(model, Path(args.out), epoch=epoch, val_dice=val_dice, cfg=vars(cfg))
            print(f"  -> saved best checkpoint  dice={best_dice:.4f}")

    # Always save a final checkpoint, even on smoke runs where val_dice may
    # never improve over the initial value.
    final_path = Path(args.out).with_name(Path(args.out).stem + "_last.pt")
    save_checkpoint(model, final_path, epoch=cfg.epochs - 1, val_dice=best_dice, cfg=vars(cfg))
    print(f"[done] best_val_dice={best_dice:.4f}  last_ckpt={final_path}")


if __name__ == "__main__":
    main()
