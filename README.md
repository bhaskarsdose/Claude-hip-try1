# Hip Bone Reconstruction (Colab-only)

Two independent workflows for building a 3D-printable hip implant from a
patient's CT segmentation. Pick the one that matches your case.

## Two workflows

### 1. `notebooks/mirror_colab.ipynb` — contralateral mirror (recommended for unilateral defects)

If the patient has **one healthy hip and one defective hip**, mirror the
healthy contralateral side and use it as the implant template. No model, no
GPU, no training. Runs in ~10 seconds on Colab CPU. Surgical gold standard
for unilateral cases — uses the patient's own anatomy as ground truth.

**Inputs:** two STL files (defective hip + healthy contralateral hip).
**Output:** `implant.stl`.

### 2. `notebooks/train_colab.ipynb` — 3D U-Net shape completion (for bilateral / no-reference cases)

When the contralateral hip isn't available (bilateral defects, asymmetric
pathology), train a 3D U-Net on synthetic defects of CTPelvic1K hips. Trains
on Colab GPU in ~30–60 minutes; inference is then instant.

**Inputs:** patient's NIfTI mask or STL of the defective hip.
**Output:** `implant.stl`.

The training notebook covers, in order: clone repo, mount Drive, download +
preprocess CTPelvic1K, train the U-Net, save checkpoint, then upload + run
inference on a patient scan.

## How to use

1. Open the notebook you need in Google Colab (File → Open notebook → GitHub
   tab → paste this repo's URL).
2. For the training notebook, switch the runtime to GPU (Runtime → Change
   runtime type → GPU; A100 if you have Pro+, T4 otherwise).
3. Run the cells top to bottom.

## Pipeline (mirror)

```
Defective hip STL  +  Healthy contralateral STL
  ──▶  mirror healthy across the medial-lateral axis
  ──▶  ICP-align the mirror to the defective bone
  ──▶  voxelise both, implant = mirrored ∖ defective
  ──▶  marching cubes + smoothing
  ──▶  STL  ──▶  3D-print
```

## Pipeline (AI)

```
TotalSegmentator output (.nii.gz) or STL
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
  mirror.py                 # contralateral mirror reconstruction
  data/
    preprocessing.py        # NIfTI ↔ voxel ↔ mesh, canonicalisation
    defects.py              # synthetic sphere / box / plane / slab defects
    datasets.py             # PyTorch Dataset (real + synthetic ellipsoid)
  models/unet3d.py          # MONAI 3D U-Net + checkpoint helpers
  train.py                  # CLI training entry point
  infer.py                  # NIfTI/STL → implant Trimesh end-to-end
scripts/
  download_ctpelvic1k.py    # Zenodo downloader for raw CT labels
  prepare_dataset.py        # raw NIfTI → 128³ .npy training tensors
notebooks/
  mirror_colab.ipynb        # mirror reconstruction (no training)
  train_colab.ipynb         # 3D U-Net training + inference
tests/                      # unit tests on synthetic ellipsoids
```

## Splitting the workflows into separate GitHub repos

If you want each workflow in its own GitHub repo (e.g. to share the mirror
tool without the AI training code), the **mirror workflow only needs**:

- `src/hip_recon/__init__.py`
- `src/hip_recon/mirror.py`
- `src/hip_recon/data/__init__.py`
- `src/hip_recon/data/preprocessing.py` (only for `voxels_to_mesh`)
- `notebooks/mirror_colab.ipynb`
- A minimal `requirements.txt` with: `trimesh numpy scipy scikit-image plotly nibabel SimpleITK`

The **AI training workflow needs everything else** (`train.py`, `infer.py`,
`models/`, `data/datasets.py`, `data/defects.py`, `scripts/`,
`train_colab.ipynb`).

To create a standalone mirror repo:

```bash
# Create a new empty repo on github.com first, then locally:
mkdir hip-mirror && cd hip-mirror
git init
# Copy the files listed above from this repo into the new structure
# Update mirror_colab.ipynb's clone URL to point at the new repo
git add . && git commit -m "Initial: mirror reconstruction"
git remote add origin git@github.com:YOUR-USER/hip-mirror.git
git push -u origin main
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
