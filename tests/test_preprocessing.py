"""Tests for preprocessing utilities — run on a synthetic ellipsoid so they
don't need any real CT data."""

from __future__ import annotations

import numpy as np

from hip_recon.data.preprocessing import (
    canonicalize,
    crop_to_content,
    fit_to_grid,
    voxels_to_mesh,
)


def _ellipsoid(grid: int = 64) -> np.ndarray:
    z, y, x = np.indices((grid, grid, grid))
    cz, cy, cx = grid / 2, grid / 2, grid / 2
    rz, ry, rx = grid * 0.30, grid * 0.20, grid * 0.25
    return ((((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2) <= 1).astype(
        np.float32
    )


def test_crop_to_content_empty_returns_zero_origin():
    z = np.zeros((10, 10, 10), dtype=np.float32)
    cropped, origin = crop_to_content(z)
    assert cropped.shape == z.shape
    assert origin.tolist() == [0, 0, 0]


def test_crop_to_content_tight():
    v = np.zeros((20, 20, 20), dtype=np.float32)
    v[5:8, 6:9, 4:7] = 1.0
    cropped, origin = crop_to_content(v, padding=0)
    assert cropped.shape == (3, 3, 3)
    assert origin.tolist() == [5, 6, 4]
    assert cropped.sum() == 27


def test_fit_to_grid_centers_and_scales():
    e = _ellipsoid(64)
    out, scale = fit_to_grid(e, grid_size=128)
    assert out.shape == (128, 128, 128)
    # The scaled bone should fit comfortably inside the cube
    nz = np.argwhere(out > 0)
    assert nz.size > 0
    lo, hi = nz.min(axis=0), nz.max(axis=0)
    assert (lo >= 0).all() and (hi < 128).all()
    assert scale == 128 / 64  # since src_max == 64


def test_canonicalize_produces_valid_volume():
    e = _ellipsoid(64)
    out, xform = canonicalize(e, grid_size=64)
    assert out.shape == (64, 64, 64)
    assert out.sum() > 0
    # Rotation matrix should be (approximately) orthonormal with det +1
    R = xform.rotation
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) > 0.99


def test_voxels_to_mesh_round_trip_volume():
    e = _ellipsoid(64)
    mesh = voxels_to_mesh(e, smooth_iters=0)
    assert len(mesh.vertices) > 0
    # Mesh volume should be close (within 15 %) to the voxel count.
    if mesh.is_watertight:
        ratio = mesh.volume / float(e.sum())
        assert 0.85 < ratio < 1.15


def test_voxels_to_mesh_empty_volume():
    z = np.zeros((32, 32, 32), dtype=np.float32)
    mesh = voxels_to_mesh(z)
    assert len(mesh.vertices) == 0
