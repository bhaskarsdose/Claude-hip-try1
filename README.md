# Hip Bone Reconstruction

Deep-learning pipeline that takes a partial / defective hip-bone segmentation
(NIfTI mask from [TotalSegmentator](https://github.com/wasserth/TotalSegmentator))
and predicts the missing piece, exporting a watertight STL implant suitable
for 3D-printing as a prosthetic.

The approach mirrors cranial-implant work (AutoImplant Challenge,
SkullBreak/SkullFix): a 3D U-Net is trained on synthetically-defected hip
volumes to produce *(defective → complete)*, and at inference the implant is
the difference between prediction and input.

## Pipeline

```
TotalSegmentator output (.nii.gz)  -->  preprocess (resample, crop,
                                                    canonicalise, voxelise)
                                  -->  3D U-Net  -->  predicted complete hip
                                  -->  implant = predicted − input
                                  -->  marching cubes + smoothing
                                  -->  STL download   -->  3D print
```

## Layout

```
src/hip_recon/
  config.py
  data/{preprocessing,defects,datasets}.py
  models/unet3d.py
  train.py
  infer.py
  web/{app.py, jobs.py, static/}
scripts/{download_ctpelvic1k.py, prepare_dataset.py}
notebooks/train_colab.ipynb
tests/
```

## Quickstart

```bash
# 1) Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2) Run the web app (uses an untrained model; output is a placeholder
#    until you complete steps 3 and 4)
uvicorn hip_recon.web.app:app --reload
# open http://localhost:8000

# 3) Smoke-train on synthetic ellipsoids (CPU OK, ~1 min)
python -m hip_recon.train --smoke

# 4) Run unit tests
pytest
```

## Real training (Colab / RunPod)

Open `notebooks/train_colab.ipynb` in Colab. It mounts Google Drive,
downloads a CTPelvic1K subset, runs `hip_recon.train`, and saves the
checkpoint back to Drive. Place the resulting `unet3d_hip.pt` in `models/`
to enable real reconstruction in the web app.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Scaffold + FastAPI + three.js viewer | done |
| 2 | Preprocessing + defect simulation | done |
| 3 | 3D U-Net + training loop (smoke OK) | done |
| 4 | CTPelvic1K prep + Colab notebook | done |
| 5 | Wire trained model into web app | done |
| 6 | Hardening (Docker, queue, mesh repair) | future |

## License

MIT.
