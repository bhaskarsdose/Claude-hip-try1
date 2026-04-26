"""Contralateral mirror reconstruction.

For unilateral hip defects, the gold standard is to mirror the healthy
contralateral side and use it as the implant template. This is vastly more
reliable than any learned model when the defect is large and the healthy
side is available, because we use the patient's own anatomy as ground truth.

Pipeline:
  1. Load both meshes (defective and healthy contralateral)
  2. Mirror the healthy mesh across the medial-lateral axis (X by default)
  3. Coarse-align centroids, then refine with ICP
  4. Voxelize both in a shared grid; implant = mirrored AND NOT defective
  5. Mesh the implant region with marching cubes + smoothing
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from scipy import ndimage as ndi
from skimage import measure


@dataclass
class MirrorResult:
    defective_mesh: trimesh.Trimesh
    mirrored_mesh: trimesh.Trimesh   # the aligned mirror of the healthy side
    implant_mesh: trimesh.Trimesh    # the predicted implant region
    pitch_mm: float                  # voxel pitch used for the boolean


def _voxelize_to_grid(
    mesh: trimesh.Trimesh, pitch: float, bmin: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    """Voxelize a mesh into a fixed grid that starts at world coordinate `bmin`."""
    vg = mesh.voxelized(pitch).fill()
    m = np.asarray(vg.matrix, dtype=bool)
    offset_vox = np.floor((mesh.bounds[0] - bmin) / pitch).astype(int)
    out = np.zeros(tuple(shape), dtype=bool)
    end = offset_vox + np.array(m.shape)
    # Clip to grid bounds in case of small float error at the edges
    src_lo = np.maximum(-offset_vox, 0)
    src_hi = m.shape - np.maximum(end - shape, 0)
    dst_lo = offset_vox + src_lo
    dst_hi = dst_lo + (src_hi - src_lo)
    sl_src = tuple(slice(src_lo[i], src_hi[i]) for i in range(3))
    sl_dst = tuple(slice(dst_lo[i], dst_hi[i]) for i in range(3))
    out[sl_dst] = m[sl_src]
    return out


def mirror_reconstruct(
    defective_path: str | Path,
    healthy_path: str | Path,
    mirror_axis: int = 0,
    grid_size: int = 256,
    dilate_input: int = 3,
    cleanup_iters: int = 3,
    smooth_iters: int = 12,
    keep_largest: bool = True,
) -> MirrorResult:
    """Reconstruct a unilateral defect by mirroring the contralateral side.

    Args:
        defective_path: STL of the defective hip (the side with bone loss).
        healthy_path: STL of the healthy contralateral hip.
        mirror_axis: which world axis to flip across. 0 (X) is correct for the
            usual TotalSegmentator RAS orientation; try 1 or 2 if results are
            anatomically wrong.
        grid_size: longest-axis voxel count of the boolean grid (256 is plenty).
        dilate_input: voxels to dilate the defective input before subtraction,
            so the implant cleanly butts against existing bone instead of
            overlapping it.
        cleanup_iters: morphological-opening iterations applied to the implant
            volume. Higher values sever thin "fingers" that connect peripheral
            bits to the main implant body. 1 = minimal cleanup, 3 = clean,
            5 = aggressive (only the central reconstruction survives).
        smooth_iters: Taubin smoothing iterations on the final implant mesh.
        keep_largest: drop disconnected fragments (small floating noise specks
            from the boolean), keeping only the largest connected component.
    """
    defective = trimesh.load(str(defective_path), force="mesh", process=True)
    healthy = trimesh.load(str(healthy_path), force="mesh", process=True)
    if defective.is_empty or healthy.is_empty:
        raise ValueError("Failed to load one of the meshes")

    # 1. Mirror across requested axis
    mirrored = healthy.copy()
    M = np.eye(4)
    M[mirror_axis, mirror_axis] = -1
    mirrored.apply_transform(M)
    # Mirroring flips face winding — invert so normals point outward again
    mirrored.invert()

    # 2. Coarse alignment via centroid translation
    mirrored.apply_translation(defective.centroid - mirrored.centroid)

    # 3. Refine with multi-pass ICP from defective -> mirrored.
    #
    # Why this direction: every defective vertex lies on intact bone, so each
    # one has a real correspondence on the full healthy mirrored mesh. The
    # opposite direction (mirrored -> defective) sends ~40% of points (the
    # ones over the missing region) to wrong nearest neighbours, biasing the
    # transform — that's what caused the misalignment in the first version.
    #
    # We run two passes and reject outliers between them so any thin sliver
    # bone in the defective mesh that still doesn't have a clean match
    # (e.g. the narrow remaining bridge through the defect) doesn't drag
    # alignment.
    try:
        # Subsample for speed: 5k points are plenty for a hip
        defective_pts = np.asarray(defective.vertices, dtype=np.float64)
        mirrored_pts = np.asarray(mirrored.vertices, dtype=np.float64)
        if len(defective_pts) > 5000:
            sel = np.random.default_rng(0).choice(len(defective_pts), 5000, replace=False)
            defective_pts = defective_pts[sel]

        # Pass 1: full point set
        M1, _aligned, _ = trimesh.registration.icp(
            defective_pts, mirrored_pts, max_iterations=60, scale=False,
        )
        # Find which defective points matched well (within median distance)
        aligned1 = (M1 @ np.c_[defective_pts, np.ones(len(defective_pts))].T).T[:, :3]
        from scipy.spatial import cKDTree
        tree = cKDTree(mirrored_pts)
        dists, _ = tree.query(aligned1, k=1)
        keep = dists < np.percentile(dists, 80)  # drop worst 20% as outliers

        # Pass 2: refined alignment using only inliers
        if keep.sum() > 100:
            M2, _, _ = trimesh.registration.icp(
                defective_pts[keep], mirrored_pts,
                initial=M1, max_iterations=60, scale=False,
            )
            matrix = M2
        else:
            matrix = M1

        # We aligned defective -> mirrored. Apply the inverse to mirrored
        # so it lands on the defective bone in its original coordinates.
        mirrored.apply_transform(np.linalg.inv(matrix))
    except Exception as exc:
        print(f"[mirror] ICP failed ({exc}); using centroid-only alignment")

    # 4. Voxel-based boolean: implant = mirrored ∖ (dilated defective)
    bmin = np.minimum(defective.bounds[0], mirrored.bounds[0]) - 2.0
    bmax = np.maximum(defective.bounds[1], mirrored.bounds[1]) + 2.0
    pitch = float((bmax - bmin).max()) / float(grid_size)
    shape = np.ceil((bmax - bmin) / pitch).astype(int)

    def_vol = _voxelize_to_grid(defective, pitch, bmin, shape)
    mir_vol = _voxelize_to_grid(mirrored, pitch, bmin, shape)

    if dilate_input > 0:
        def_vol = ndi.binary_dilation(def_vol, iterations=dilate_input)
    implant_vol = mir_vol & ~def_vol
    # Erode + dilate (binary_opening) severs thin necks connecting peripheral
    # shards to the main implant, then restores volume of the main body.
    if cleanup_iters > 0:
        implant_vol = ndi.binary_opening(implant_vol, iterations=cleanup_iters)

    if implant_vol.sum() == 0:
        print("[mirror] empty implant — try a different mirror_axis or check alignment")
        return MirrorResult(defective, mirrored, trimesh.Trimesh(), pitch)

    # Drop disconnected fragments — they're voxel-level noise from the boolean,
    # not real anatomy. Keeps only the largest connected component.
    if keep_largest:
        labels, n = ndi.label(implant_vol)
        if n > 1:
            sizes = ndi.sum(implant_vol, labels, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            implant_vol = labels == largest

    # 5. Mesh the implant volume back to world coordinates
    padded = np.pad(implant_vol.astype(np.uint8), 1)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts -= 1.0
    verts = verts * pitch + bmin
    faces = faces[:, ::-1]  # match preprocessing.voxels_to_mesh winding fix
    implant_mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if smooth_iters > 0 and len(implant_mesh.vertices) > 0:
        try:
            trimesh.smoothing.filter_taubin(
                implant_mesh, lamb=0.5, nu=-0.53, iterations=smooth_iters
            )
        except Exception:
            pass

    return MirrorResult(defective, mirrored, implant_mesh, pitch)


def detect_acetabulum(
    hip_mesh: trimesh.Trimesh,
    deep_percentile: float = 85.0,
    min_points: int = 20,
) -> tuple[np.ndarray, float]:
    """Estimate acetabular centre and radius from a healthy hip mesh.

    The acetabulum is the cup that cradles the femoral head. It's a deep
    concavity that **opens laterally** (away from the body midline). We
    disambiguate it from the iliac fossa (smooth medial concavity) by
    requiring vertices to face away from the bone's main mass.

    Algorithm:
      1. Compute convex hull, measure each vertex's distance to the hull.
      2. Take the deepest 15% of vertices.
      3. Cluster those by spatial proximity; pick the cluster whose mean
         normal points furthest away from the bone centroid (= laterally
         outward), which selects the acetabulum over the iliac fossa.
      4. Centroid + median radius gives cup centre + size.

    Returns (centre_xyz, radius_mm). Radius clamped to physiological 20-28 mm.
    """
    hull = hip_mesh.convex_hull
    _closest, dists, _ = trimesh.proximity.closest_point(hull, hip_mesh.vertices)
    threshold = np.percentile(dists, deep_percentile)
    deep_idx = np.where(dists > threshold)[0]
    if len(deep_idx) < min_points:
        return hip_mesh.centroid, 24.0
    deep_pts = hip_mesh.vertices[deep_idx]
    deep_normals = hip_mesh.vertex_normals[deep_idx]

    # Cluster the deep vertices spatially. Use simple connectivity via DBSCAN-like
    # approach: KDTree + connected components on a radius graph.
    from scipy.spatial import cKDTree
    tree = cKDTree(deep_pts)
    cluster_radius = max(8.0, np.linalg.norm(hip_mesh.extents) * 0.05)
    pairs = tree.query_pairs(r=cluster_radius)
    n = len(deep_pts)
    # Union-find on cluster pairs
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    unique_roots, sizes = np.unique(roots, return_counts=True)
    # Score each cluster: prefer (a) large, (b) facing away from bone centroid
    bone_centroid = hip_mesh.centroid
    best_score = -np.inf
    best_centre = deep_pts.mean(axis=0)
    best_radius = 24.0
    for root, size in zip(unique_roots, sizes):
        if size < 5:
            continue
        mask = roots == root
        cluster_pts = deep_pts[mask]
        cluster_normals = deep_normals[mask]
        cluster_centre = cluster_pts.mean(axis=0)
        # Outward direction = away from bone centroid
        outward = cluster_centre - bone_centroid
        outward /= np.linalg.norm(outward) + 1e-8
        # Mean normal of the cluster should point in same direction (concavity opens outward)
        mean_normal = cluster_normals.mean(axis=0)
        mean_normal /= np.linalg.norm(mean_normal) + 1e-8
        # Note: surface normals point outward from the bone. For a concavity
        # that opens laterally, the surface normals point laterally too.
        outward_score = float(mean_normal @ outward)
        # Combine: prefer outward-facing AND reasonably large
        score = outward_score * np.log1p(size)
        if score > best_score:
            best_score = score
            best_centre = cluster_centre
            best_radius = float(np.median(np.linalg.norm(cluster_pts - cluster_centre, axis=1)))
    return best_centre, float(np.clip(best_radius, 20.0, 28.0))


def add_femoral_socket(
    implant_mesh: trimesh.Trimesh,
    centre: np.ndarray,
    radius: float = 24.0,
    pitch: float | None = None,
    smooth_iters: int = 6,
    keep_largest: bool = True,
) -> trimesh.Trimesh:
    """Carve a hemispherical socket into the implant for the femoral head.

    Args:
        implant_mesh: bone-shape implant from mirror_reconstruct.
        centre: world-space centre of the acetabular cup (3,).
        radius: femoral head radius in mm (typical adult: 22-26).
        pitch: voxel pitch for the boolean. Defaults to radius / 24.
        smooth_iters: Taubin smoothing on the modified implant.
        keep_largest: drop disconnected fragments left over from the boolean,
            keeping only the largest connected component.
    """
    if implant_mesh.is_empty:
        return implant_mesh
    if pitch is None:
        pitch = max(radius / 24.0, 0.4)

    sphere = trimesh.creation.icosphere(subdivisions=4, radius=radius)
    sphere.apply_translation(np.asarray(centre, dtype=float))

    bmin = np.minimum(implant_mesh.bounds[0], sphere.bounds[0]) - 2.0
    bmax = np.maximum(implant_mesh.bounds[1], sphere.bounds[1]) + 2.0
    shape = np.ceil((bmax - bmin) / pitch).astype(int)

    impl_vol = _voxelize_to_grid(implant_mesh, pitch, bmin, shape)
    sph_vol = _voxelize_to_grid(sphere, pitch, bmin, shape)

    result_vol = impl_vol & ~sph_vol
    result_vol = ndi.binary_opening(result_vol, iterations=1)
    if result_vol.sum() == 0:
        return implant_mesh

    if keep_largest:
        # Keep only the largest connected component to remove fragments
        labels, n = ndi.label(result_vol)
        if n > 1:
            sizes = ndi.sum(result_vol, labels, range(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            result_vol = labels == largest

    padded = np.pad(result_vol.astype(np.uint8), 1)
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts -= 1.0
    verts = verts * pitch + bmin
    faces = faces[:, ::-1]
    out = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    if smooth_iters > 0 and len(out.vertices) > 0:
        try:
            trimesh.smoothing.filter_taubin(
                out, lamb=0.5, nu=-0.53, iterations=smooth_iters
            )
        except Exception:
            pass
    return out
