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
    python scripts/prepare_dataset.py --in-dir data/raw --out-dir data/processed --label 2 --workers 4
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
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


def _process_one(args_tuple) -> tuple[str, str | None]:
    """Process a single NIfTI file. Returns (filename, error_or_None)."""
    f, out_dir, keep_label, grid = args_tuple
    out_path = Path(out_dir) / (Path(f).stem.replace(".nii", "") + ".npy")
    if out_path.exists():
        return (str(f), "skip:exists")
    try:
        img = nib.load(str(f))
        arr = np.asanyarray(img.dataobj)
        spacing = tuple(float(s) for s in img.header.get_zooms()[:3])

        binary = _binarise(arr, keep_label)
        if binary.sum() < 1000:
            return (str(f), "skip:empty")

        iso = resample_iso(binary, spacing)
        cropped, _ = crop_to_content(iso)
        canonical, _ = canonicalize(cropped, grid_size=grid)
        np.save(out_path, canonical.astype(np.float32))
        return (str(f), None)
    except Exception as exc:
        return (str(f), f"error:{exc}")


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
    p.add_argument(
        "--workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel worker processes (default: min(4, cpu_count))",
    )
    args = p.parse_args()

    ensure_dirs()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(list(in_dir.rglob("*.nii.gz")) + list(in_dir.rglob("*.nii")))
    if not files:
        raise SystemExit(f"No NIfTI files under {in_dir}")

    print(f"[prepare] {len(files)} files  label={args.label}  workers={args.workers}  out={out_dir}")

    tasks = [(str(f), str(out_dir), args.label, args.grid) for f in files]
    written = skipped = errors = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, t): t[0] for t in tasks}
        with tqdm(total=len(files), desc="prepare") as bar:
            for fut in as_completed(futures):
                _, result = fut.result()
                if result is None:
                    written += 1
                elif result.startswith("skip"):
                    skipped += 1
                else:
                    errors += 1
                    print(f"\n[warn] {futures[fut]}: {result}")
                bar.update(1)
                bar.set_postfix(written=written, skipped=skipped, errors=errors)

    print(f"[done] written={written}  skipped={skipped}  errors={errors}  out={out_dir}")


if __name__ == "__main__":
    main()
