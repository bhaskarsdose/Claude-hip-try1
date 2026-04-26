"""Synthetic defect generators for shape-completion training.

Each generator takes a complete binary occupancy volume and returns
(defective_volume, implant_gt) where:

    defective_volume = complete - implant_gt   (logical AND-NOT)
    implant_gt       = the carved-out region (still bone, in the complete model)

We generate three defect types so the network sees varied shapes:
  random_sphere_defect : a localised spherical bite, like a focal bone loss
  random_box_defect    : a chunk gone, like a surgical resection
  random_plane_cut     : a planar slice, like a fracture / osteotomy gap
"""

from __future__ import annotations

import numpy as np

Rng = np.random.Generator


def _surface_seed(volume: np.ndarray, rng: Rng) -> np.ndarray:
    """Pick a random voxel near the surface of the bone."""
    coords = np.argwhere(volume > 0)
    if coords.size == 0:
        return np.array(volume.shape) // 2
    return coords[rng.integers(0, len(coords))]


def random_sphere_defect(
    volume: np.ndarray,
    radius_range: tuple[int, int] = (8, 22),
    rng: Rng | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    centre = _surface_seed(volume, rng)
    r = int(rng.integers(radius_range[0], radius_range[1] + 1))

    zz, yy, xx = np.indices(volume.shape)
    sphere = ((zz - centre[0]) ** 2 + (yy - centre[1]) ** 2 + (xx - centre[2]) ** 2) <= r * r
    implant = (volume > 0) & sphere
    defective = (volume > 0) & ~implant
    return defective.astype(np.float32), implant.astype(np.float32)


def random_box_defect(
    volume: np.ndarray,
    size_range: tuple[int, int] = (12, 32),
    rng: Rng | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng()
    centre = _surface_seed(volume, rng)
    sz = rng.integers(size_range[0], size_range[1] + 1, size=3)
    lo = np.maximum(centre - sz // 2, 0)
    hi = np.minimum(lo + sz, volume.shape)

    box = np.zeros_like(volume, dtype=bool)
    box[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = True
    implant = (volume > 0) & box
    defective = (volume > 0) & ~implant
    return defective.astype(np.float32), implant.astype(np.float32)


def random_plane_cut(
    volume: np.ndarray,
    margin: float = 0.15,
    rng: Rng | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cut along a random plane through the bone, removing the smaller side."""
    rng = rng or np.random.default_rng()
    coords = np.argwhere(volume > 0)
    if coords.size == 0:
        return volume.copy(), np.zeros_like(volume)
    centroid = coords.mean(axis=0)
    # Random plane normal, offset slightly off-centre so neither side is empty
    n = rng.normal(size=3)
    n /= np.linalg.norm(n) + 1e-8
    extents = coords.max(axis=0) - coords.min(axis=0)
    offset = rng.uniform(-margin, margin) * extents.max()
    plane_point = centroid + n * offset

    zz, yy, xx = np.indices(volume.shape)
    rel = np.stack([zz - plane_point[0], yy - plane_point[1], xx - plane_point[2]], axis=-1)
    side = (rel @ n) > 0

    side_a = (volume > 0) & side
    side_b = (volume > 0) & ~side
    if side_a.sum() < side_b.sum():
        implant = side_a
    else:
        implant = side_b
    defective = (volume > 0) & ~implant
    return defective.astype(np.float32), implant.astype(np.float32)


def random_slab_defect(
    volume: np.ndarray,
    thickness_range: tuple[float, float] = (0.25, 0.55),
    rng: Rng | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove a slab through the middle of the bone (two parallel planes).

    Models pelvic-discontinuity / massive-osteolysis cases where the central
    region is gone and only two end pieces remain — disconnected.
    """
    rng = rng or np.random.default_rng()
    coords = np.argwhere(volume > 0)
    if coords.size == 0:
        return volume.copy(), np.zeros_like(volume)
    centroid = coords.mean(axis=0)
    n = rng.normal(size=3)
    n /= np.linalg.norm(n) + 1e-8
    extents = coords.max(axis=0) - coords.min(axis=0)
    span = float(extents.max())
    thickness = rng.uniform(*thickness_range) * span
    offset = rng.uniform(-0.15, 0.15) * span

    zz, yy, xx = np.indices(volume.shape)
    rel = np.stack(
        [zz - centroid[0], yy - centroid[1], xx - centroid[2]], axis=-1
    )
    proj = rel @ n
    in_slab = (proj > offset - thickness / 2) & (proj < offset + thickness / 2)
    implant = (volume > 0) & in_slab
    defective = (volume > 0) & ~implant
    return defective.astype(np.float32), implant.astype(np.float32)


DEFECT_FNS = {
    "sphere": random_sphere_defect,
    "box": random_box_defect,
    "plane": random_plane_cut,
    "slab": random_slab_defect,
}


def random_defect(
    volume: np.ndarray, rng: Rng | None = None
) -> tuple[np.ndarray, np.ndarray, str]:
    """Pick one defect type at random."""
    rng = rng or np.random.default_rng()
    kind = rng.choice(list(DEFECT_FNS.keys()))
    fn = DEFECT_FNS[kind]
    defective, implant = fn(volume, rng=rng)
    return defective, implant, kind
