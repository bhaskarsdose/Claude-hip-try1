"""Tests for the synthetic defect generators."""

from __future__ import annotations

import numpy as np

from hip_recon.data.defects import (
    DEFECT_FNS,
    random_box_defect,
    random_plane_cut,
    random_sphere_defect,
)


def _ellipsoid(grid: int = 64) -> np.ndarray:
    z, y, x = np.indices((grid, grid, grid))
    return (
        (((z - grid / 2) / (grid * 0.30)) ** 2)
        + (((y - grid / 2) / (grid * 0.22)) ** 2)
        + (((x - grid / 2) / (grid * 0.25)) ** 2)
        <= 1.0
    ).astype(np.float32)


def test_sphere_defect_is_consistent():
    rng = np.random.default_rng(0)
    e = _ellipsoid()
    defective, implant = random_sphere_defect(e, rng=rng)
    # implant + defective == complete (no overlap, no leakage)
    assert np.array_equal(defective + implant > 0, e > 0)
    assert (defective * implant).sum() == 0
    assert implant.sum() > 0
    assert defective.sum() > 0


def test_box_defect_is_consistent():
    rng = np.random.default_rng(1)
    e = _ellipsoid()
    defective, implant = random_box_defect(e, rng=rng)
    assert np.array_equal(defective + implant > 0, e > 0)
    assert (defective * implant).sum() == 0


def test_plane_cut_implant_is_smaller_side():
    rng = np.random.default_rng(2)
    e = _ellipsoid()
    defective, implant = random_plane_cut(e, rng=rng)
    assert implant.sum() <= defective.sum()
    assert (defective * implant).sum() == 0
    assert np.array_equal(defective + implant > 0, e > 0)


def test_all_defect_kinds_registered():
    assert set(DEFECT_FNS) == {"sphere", "box", "plane"}
