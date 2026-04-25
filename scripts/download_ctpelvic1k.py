"""Download a CTPelvic1K subset from Zenodo into data/raw/.

CTPelvic1K is split across multiple Zenodo records. The full dataset is
several hundred GB; for hip-implant training we only need the labels (binary
pelvis masks) plus enough cases for a good model. This script downloads the
subset URLs you give it and expects you to organise the resulting NIfTI
files under `data/raw/ctpelvic1k/{cases}/<case_id>.nii.gz` (the labels) for
`scripts/prepare_dataset.py` to consume.

Usage:
    python scripts/download_ctpelvic1k.py URL1 URL2 ...
    python scripts/download_ctpelvic1k.py --urls-from urls.txt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from hip_recon.config import RAW_DIR, ensure_dirs


def _download(url: str, out_dir: Path) -> Path:
    out = out_dir / Path(url).name.split("?")[0]
    if out.is_file():
        print(f"[skip] {out.name} (already on disk)")
        return out
    print(f"[get ] {url} -> {out}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, out.open("wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("urls", nargs="*")
    p.add_argument("--urls-from", type=str, help="text file, one URL per line")
    p.add_argument("--out", type=str, default=str(RAW_DIR / "ctpelvic1k"))
    args = p.parse_args()

    urls = list(args.urls)
    if args.urls_from:
        urls += [
            line.strip()
            for line in Path(args.urls_from).read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    if not urls:
        sys.exit("No URLs supplied. See script docstring for usage.")

    ensure_dirs()
    out_dir = Path(args.out)
    for u in urls:
        _download(u, out_dir)
    print(f"[done] downloaded {len(urls)} archives to {out_dir}")
    print("Next: extract them and run scripts/prepare_dataset.py")


if __name__ == "__main__":
    main()
