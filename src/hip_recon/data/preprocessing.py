"""NIfTI / volume / mesh utilities used by training and inference.

The pipeline operates on binary occupancy volumes. Concretely:

  load_nifti      : read a TotalSegmentator-style mask from disk
  resample_iso    : SimpleITK isotropic resampling
  crop_to_content : tight bbox around the bone
  fit_to_grid     : pad/scale into a fixed grid_size**3 cube
  canonicalize    : PCA-align the bone to a canonical pose (returns inverse)
  voxels_to_mesh  : marching cubes + Taubin smoothing
"""

from __future__ import annotations

from dataclasses import dataclass

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import trimesh
from scipy import ndimage as ndi
from skimage import measure

from ..config import TARGET_SPACING_MM, VOXEL_SIZE


@dataclass
class CanonicalTransform:
    """Stores everything needed to invert canonicalize() back to source space."""

    rotation: np.ndarray  # 3x3, applied as (R @ x) on zero-centred coords
    centroid: np.ndarray  # 3, in source-grid voxel coordinates
    grid_size: int
    scale: float          # source -> canonical voxel scale factor
    crop_origin: np.ndarray  # 3, voxel origin of the source crop
    source_shape: tuple[int, int, int]


def load_nifti(path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    """Return (volume[float32 binary], affine, spacing_mm)."""
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    affine = img.affine.astype(np.float64)
    spacing = tuple(float(s) for s in img.header.get_zooms()[:3])
    binary = (data > 0).astype(np.float32)
    return binary, affine, spacing


def resample_iso(
    volume: np.ndarray,
    spacing: tuple[float, float, float],
    target: float = TARGET_SPACING_MM,
) -> np.ndarray:
    """Resample to isotropic `target` mm voxels using nearest-neighbour."""
    img = sitk.GetImageFromArray(volume.astype(np.uint8))
    img.SetSpacing(tuple(float(s) for s in spacing[::-1]))  # SITK wants (x,y,z)

    in_size = np.array(img.GetSize())
    in_spacing = np.array(img.GetSpacing())
    out_spacing = np.array([target] * 3)
    out_size = np.maximum(1, np.round(in_size * in_spacing / out_spacing)).astype(int)

    res = sitk.ResampleImageFilter()
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    res.SetOutputSpacing(tuple(out_spacing.tolist()))
    res.SetSize([int(s) for s in out_size])
    res.SetOutputOrigin(img.GetOrigin())
    res.SetOutputDirection(img.GetDirection())
    out = res.Execute(img)
    arr = sitk.GetArrayFromImage(out).astype(np.float32)
    return arr


def crop_to_content(volume: np.ndarray, padding: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Tight axis-aligned crop around non-zero voxels. Returns (cropped, origin)."""
    nz = np.argwhere(volume > 0)
    if nz.size == 0:
        return volume.copy(), np.zeros(3, dtype=int)
    lo = np.maximum(nz.min(axis=0) - padding, 0)
    hi = np.minimum(nz.max(axis=0) + padding + 1, volume.shape)
    cropped = volume[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    return cropped, lo


def fit_to_grid(volume: np.ndarray, grid_size: int = VOXEL_SIZE) -> tuple[np.ndarray, float]:
    """Scale + pad the volume into a `grid_size`-cube, preserving aspect ratio.

    Returns (out_volume, scale) where scale = grid_size / max(volume.shape).
    """
    src_max = max(volume.shape)
    if src_max == 0:
        return np.zeros((grid_size,) * 3, dtype=np.float32), 1.0
    scale = grid_size / src_max
    new_shape = tuple(max(1, int(round(s * scale))) for s in volume.shape)
    zoomed = ndi.zoom(volume, np.array(new_shape) / np.array(volume.shape), order=0)
    out = np.zeros((grid_size,) * 3, dtype=np.float32)
    pads = [(grid_size - n) // 2 for n in zoomed.shape]
    out[
        pads[0] : pads[0] + zoomed.shape[0],
        pads[1] : pads[1] + zoomed.shape[1],
        pads[2] : pads[2] + zoomed.shape[2],
    ] = zoomed
    return out.astype(np.float32), scale


def canonicalize(
    volume: np.ndarray, grid_size: int = VOXEL_SIZE
) -> tuple[np.ndarray, CanonicalTransform]:
    """PCA-align occupied voxels to a canonical pose centred in a grid_size cube."""
    coords = np.argwhere(volume > 0).astype(np.float64)
    if coords.shape[0] < 3:
        # Fallback: just centre + rescale
        out, scale = fit_to_grid(volume, grid_size)
        return out, CanonicalTransform(
            rotation=np.eye(3),
            centroid=np.array(volume.shape) / 2.0,
            grid_size=grid_size,
            scale=scale,
            crop_origin=np.zeros(3, dtype=int),
            source_shape=tuple(int(s) for s in volume.shape),
        )

    centroid = coords.mean(axis=0)
    centred = coords - centroid
    # PCA via SVD of covariance
    cov = np.cov(centred.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    rot = eigvecs[:, order].T  # rows = principal axes
    if np.linalg.det(rot) < 0:
        rot[2] *= -1

    rotated = (centred @ rot.T)
    extents = rotated.max(axis=0) - rotated.min(axis=0)
    src_max = float(extents.max())
    scale = (grid_size - 4) / src_max if src_max > 0 else 1.0
    rotated *= scale
    rotated += grid_size / 2.0

    out = np.zeros((grid_size,) * 3, dtype=np.float32)
    idx = np.clip(rotated.astype(int), 0, grid_size - 1)
    out[idx[:, 0], idx[:, 1], idx[:, 2]] = 1.0
    # Close tiny holes from rasterisation
    out = ndi.binary_closing(out > 0, iterations=1).astype(np.float32)

    transform = CanonicalTransform(
        rotation=rot,
        centroid=centroid,
        grid_size=grid_size,
        scale=scale,
        crop_origin=np.zeros(3, dtype=int),
        source_shape=tuple(int(s) for s in volume.shape),
    )
    return out, transform


def voxels_to_mesh(
    volume: np.ndarray,
    level: float = 0.5,
    smooth_iters: int = 8,
) -> trimesh.Trimesh:
    """Marching cubes + Taubin smoothing. Returns an empty mesh if volume is empty."""
    if volume.max() < level:
        return trimesh.Trimesh()
    # `measure.marching_cubes` requires the surface to be strictly inside the volume.
    padded = np.pad(volume, 1, mode="constant", constant_values=0)
    verts, faces, normals, _ = measure.marching_cubes(padded, level=level)
    verts -= 1.0  # undo the pad shift
    # skimage returns verts in (row, col, slice) order; treating that as
    # (x, y, z) flips chirality. Reversing face winding restores outward
    # normals so STLs render correctly in three.js.
    faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)
    if smooth_iters > 0 and len(mesh.vertices) > 0:
        try:
            trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iters)
        except Exception:  # smoothing is best-effort
            pass
    return mesh
