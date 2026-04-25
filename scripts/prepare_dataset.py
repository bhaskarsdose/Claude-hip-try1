"""Convert raw CTPelvic1K (or any folder of binary hip masks) into 128**3 .npy.

For each `*.nii.gz` mask under --in-dir we:
    1. binarise the label (CTPelvic1K labels: 1=sacrum, 2=L-hip, 3=R-hip,
       4=L5; pass --label to filter, default keeps everything > 0)
    2. resample to isotropic mm
    3. crop to bone bounding box
    4. PCA-canonicalise into a 128**3 cube
    5. save as data/processed/<stem>.npy

Usage:
    python scripts/prepare_dataset.py --in-dir data/raw/ctpelvic1k/labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from hip_recon.config import PROCESSED_DIR, VOXEL_SIZE, ensure_dirs
from hip_recon.data.preprocessing import (
    canonicalize,
    crop_to_content,
    resample_iso,
)


def _binarise(arr: np.ndarray, keep_label: int | None) -> np.ndarray:
    if keep_label is None:
        return (arr > 0).astype(np.float32)
    return (arr == keep_label).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=str, required=True)
    p.add_argument("--out-dir", type=str, default=str(PROCESSED_DIR))
    p.add_argument(
        "--label",
        type=int,
        default=None,
        help="Keep only this label value (1=sacrum, 2=L-hip, 3=R-hip in CTPelvic1K)",
    )
    p.add_argument("--grid", type=int, default=VOXEL_SIZE)
    args = p.parse_args()

    ensure_dirs()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.rglob("*.nii.gz")) + list(in_dir.rglob("*.nii")))
    if not files:
        raise SystemExit(f"No NIfTI files under {in_dir}")

    skipped = 0
    for f in tqdm(files, desc="prepare"):
        try:
            img = nib.load(str(f))
            arr = np.asanyarray(img.dataobj)
            spacing = tuple(float(s) for s in img.header.get_zooms()[:3])

            binary = _binarise(arr, args.label)
            if binary.sum() < 1000:
                skipped += 1
                continue

            iso = resample_iso(binary, spacing)
            cropped, _ = crop_to_content(iso)
            canonical, _ = canonicalize(cropped, grid_size=args.grid)
            np.save(out_dir / (f.stem.replace(".nii", "") + ".npy"), canonical.astype(np.float32))
        except Exception as exc:
            print(f"[warn] {f}: {exc}")
            skipped += 1

    print(f"[done] wrote {len(files) - skipped} volumes to {out_dir} (skipped {skipped})")


if __name__ == "__main__":
    main()
