# Hip Bone Reconstruction (Colab-only)

Deep-learning pipeline that takes a partial / defective hip-bone segmentation
(NIfTI mask from
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator))
and predicts the missing piece, exporting an STL implant for 3D-printing as a
prosthetic.

The approach mirrors cranial-implant work (AutoImplant Challenge,
SkullBreak / SkullFix): a 3D U-Net is trained on synthetically-defected hip
volumes to map *defective → complete*; at inference the implant is the
difference between prediction and input.

The whole workflow — training, uploading a patient scan, viewing the
reconstruction, downloading the STL — runs **inside one Colab notebook**.
There is no local server or GUI to set up.

## How to use

1. Open **`notebooks/train_colab.ipynb`** in Google Colab
   ([colab.research.google.com](https://colab.research.google.com) → File →
   Upload notebook, or open from GitHub).
2. Switch to a GPU runtime: *Runtime → Change runtime type → GPU* (T4 is fine).
3. Run the cells top to bottom. The notebook covers, in order:
   1. Clone this repo and install dependencies.
   2. Mount Google Drive so your data + checkpoint persist between sessions.
   3. Download / point at a hip-CT dataset (CTPelvic1K by default).
   4. Preprocess it into 128³ training tensors.
   5. Train the 3D U-Net and save `unet3d_hip.pt` to Drive.
   6. Upload your patient `.nii.gz` from TotalSegmentator.
   7. Reconstruct, view inline (Plotly), and download `input.stl` +
      `implant.stl`.
4. Send `implant.stl` to your slicer / 3D printer.

To reconstruct another patient later, re-open the notebook and run only
cells 7–10 — the checkpoint is reloaded from Drive automatically.

## Pipeline

```
TotalSegmentator output (.nii.gz)
  ──▶  preprocess (resample to 1.5 mm iso, crop to bone, PCA-canonicalise,
                   voxelise to 128³)
  ──▶  3D U-Net  ──▶  predicted complete hip
  ──▶  implant = predicted − input
  ──▶  marching cubes + smoothing
  ──▶  STL  ──▶  3D-print
```

## Repository layout

```
src/hip_recon/
  config.py                 # paths, voxel size, training hyperparams
  data/
    preprocessing.py        # NIfTI ↔ voxel ↔ mesh, canonicalisation
    defects.py              # synthetic sphere / box / plane defects
    datasets.py             # PyTorch Dataset (real + synthetic ellipsoid)
  models/unet3d.py          # MONAI 3D U-Net + checkpoint helpers
  train.py                  # CLI training entry point
  infer.py                  # NIfTI → implant Trimesh end-to-end
scripts/
  download_ctpelvic1k.py    # Zenodo downloader for raw CT labels
  prepare_dataset.py        # raw NIfTI → 128³ .npy training tensors
notebooks/
  train_colab.ipynb         # the end-to-end Colab notebook
tests/                      # unit tests on synthetic ellipsoids
```

## Tests (optional, runs on CPU)

```bash
pip install -r requirements.txt
pip install -e .
pytest             # 10 tests on synthetic ellipsoids
python -m hip_recon.train --smoke   # CPU smoke training (~1 min)
```

## License

MIT.
