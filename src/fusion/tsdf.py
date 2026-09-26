"""TSDF fusion of the depth contract into a mesh and a point cloud (v3).

Only observed space is carved, so there is no closed shell under the ground
and no Poisson hallucination around the edges. Per-vertex view counts come
from depth-map visibility and feed the coverage report.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict

import cv2
import numpy as np
import open3d as o3d

from src.depth.common import load_depth_contract, read_depth


def _o3d_intrinsic(cam: Dict[str, Any]) -> o3d.camera.PinholeCameraIntrinsic:
    K = np.asarray(cam["K"])
    # COLMAP puts the first pixel centre at 0.5; Open3D at 0.0.
    return o3d.camera.PinholeCameraIntrinsic(cam["width"], cam["height"], K[0, 0], K[1, 1],
                                             K[0, 2] - 0.5, K[1, 2] - 0.5)


def _extrinsic(cam: Dict[str, Any]) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(cam["R_cw"])
    T[:3, 3] = np.asarray(cam["t_cw"])
    return T


def vertex_view_counts(vertices: np.ndarray, depth_root: str, cams, rel_tol: float = 0.03,
                       abs_tol: float = 0.3) -> np.ndarray:
    counts = np.zeros(len(vertices), np.int32)
    for cam in cams:
        d, _ = read_depth(depth_root, cam["name"])
        K = np.asarray(cam["K"])
        R, t = np.asarray(cam["R_cw"]), np.asarray(cam["t_cw"])
        x = vertices @ R.T + t
        z = x[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.round(K[0, 0] * x[:, 0] / z + K[0, 2] - 0.5).astype(np.int64)
            v = np.round(K[1, 1] * x[:, 1] / z + K[1, 2] - 0.5).astype(np.int64)
        h, w = d.shape
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dz = np.zeros_like(z)
        dz[ok] = d[v[ok], u[ok]]
        seen = ok & (dz > 0) & (np.abs(dz - z) <= np.maximum(abs_tol, rel_tol * z))
        counts += seen
    return counts


def grazing_mask(depth: np.ndarray, K: np.ndarray, min_cos: float) -> np.ndarray:
    """True where the viewing ray meets the surface at a usable angle.

    Surfaces seen nearly edge-on (distant fields towards the horizon, walls at a
    glancing angle) get a few pixels spread over many metres; they fragment into
    slivers in the TSDF. Normals come from the unprojected depth map."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    P = np.stack([x, y, depth], -1)
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, 1:-1] = P[:, 2:] - P[:, :-2]
    dv[1:-1] = P[2:] - P[:-2]
    n = np.cross(du, dv)
    nn = np.linalg.norm(n, axis=-1)
    ray = P / np.maximum(np.linalg.norm(P, axis=-1, keepdims=True), 1e-9)
    cos = np.abs(np.sum(n * ray, axis=-1)) / np.maximum(nn, 1e-12)
    valid = (depth > 0)
    nb = valid.copy()
    nb[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    nb[1:-1] &= valid[2:] & valid[:-2]
    return nb & (cos >= min_cos)


def _device():
    try:
        if o3d.core.cuda.is_available():
            return o3d.core.Device("CUDA:0")
    except Exception:
        pass
    return o3d.core.Device("CPU:0")


def fuse(depth_root: str, out_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Tensor VoxelBlockGrid TSDF.

    The legacy ScalableTSDFVolume in Open3D 0.20 extracts nothing at aerial depths
    (verified on a clean plane at 60 m), so the tensor API is used.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta, _ = load_depth_contract(depth_root)
    cams = meta["cameras"]
    voxel = float(cfg["voxel_m"])
    trunc_mult = float(cfg["trunc_voxels"])
    depth_max = float(cfg["depth_max_m"])
    # far, grazing surfaces (towards the horizon) fragment into slivers; cap the fusion range
    # at a multiple of the typical scene depth
    factor = cfg.get("depth_cap_factor")
    if factor:
        samples = []
        for cam in cams[:: max(1, len(cams) // 8)]:
            d, c = read_depth(depth_root, cam["name"])
            v = d[(d > 0) & (c >= cfg["min_conf"])]
            if len(v):
                samples.append(np.median(v))
        if samples:
            depth_max = min(depth_max, float(factor) * float(np.median(samples)))
    dev = _device()
    T: Dict[str, float] = {}
    t0 = time.perf_counter()
    block_res = 8
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3d.core.float32, o3d.core.float32, o3d.core.float32),
        attr_channels=((1), (1), (3)), voxel_size=voxel, block_resolution=block_res,
        block_count=int(cfg.get("block_count", 60000)), device=dev)
    integrated = 0
    for cam in cams:
        depth, conf = read_depth(depth_root, cam["name"])
        depth = np.where((conf >= cfg["min_conf"]) & (depth < depth_max), depth, 0.0).astype(np.float32)
        # beyond z = voxel * f a pixel covers more than one voxel: samples scatter and fragment
        if cfg.get("footprint_factor"):
            zc = float(cfg["footprint_factor"]) * voxel * float(np.asarray(cam["K"])[0, 0])
            depth = np.where(depth <= zc, depth, 0.0).astype(np.float32)
        if cfg.get("min_view_cos"):
            depth = np.where(grazing_mask(depth, np.asarray(cam["K"]), float(cfg["min_view_cos"])), depth, 0.0
                             ).astype(np.float32)
        if not (depth > 0).any():
            continue
        bgr = cv2.imread(cam["image"])
        if bgr.shape[:2] != depth.shape:
            bgr = cv2.resize(bgr, (depth.shape[1], depth.shape[0]))
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).astype(np.float32) / 255.0
        K = np.asarray(cam["K"], dtype=np.float64).copy()
        K[0, 2] -= 0.5  # COLMAP pixel centres are at +0.5, Open3D at 0
        K[1, 2] -= 0.5
        Kt = o3d.core.Tensor(K, o3d.core.float64)
        Et = o3d.core.Tensor(_extrinsic(cam), o3d.core.float64)
        dt = o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(depth))).to(dev)
        ct = o3d.t.geometry.Image(o3d.core.Tensor(rgb)).to(dev)
        blocks = vbg.compute_unique_block_coordinates(dt, Kt, Et, 1.0, depth_max, trunc_mult)
        vbg.integrate(blocks, dt, ct, Kt, Et, 1.0, depth_max, trunc_mult)
        integrated += 1
    if integrated == 0:
        raise RuntimeError("no depth map passed the confidence filter; nothing to fuse")

    T["integrate"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    min_w = float(cfg.get("min_weight", 1.5))
    tmesh = vbg.extract_triangle_mesh(weight_threshold=min_w)
    mesh = tmesh.to_legacy()
    T["extract_mesh"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    n_raw = len(mesh.triangles)
    tri_c, tri_n, _ = mesh.cluster_connected_triangles()
    tri_c, tri_n = np.asarray(tri_c), np.asarray(tri_n)
    keep_clusters = np.where(tri_n >= cfg["min_component_frac"] * max(1, n_raw))[0]
    mesh.remove_triangles_by_mask(~np.isin(tri_c, keep_clusters))
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    kept = int(tri_n[keep_clusters].sum()) if len(keep_clusters) else 0
    largest_frac = float(tri_n.max() / max(1, kept)) if len(tri_n) else 0.0
    T["components"] = time.perf_counter() - t0
    t0 = time.perf_counter()

    verts = np.asarray(mesh.vertices)
    views = vertex_view_counts(verts, depth_root, cams)
    tris = np.asarray(mesh.triangles)
    face_views = views[tris].min(axis=1) if len(tris) else np.empty(0)
    v0, v1, v2 = (verts[tris[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    covered = float(area[face_views >= 2].sum() / max(area.sum(), 1e-9))

    T["view_counts"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    mesh_path = os.path.join(out_dir, "tsdf_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    np.save(os.path.join(out_dir, "vertex_view_count.npy"), views)

    T["write_mesh"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    pcd = vbg.extract_point_cloud(weight_threshold=min_w).to_legacy()
    pcd = pcd.voxel_down_sample(voxel)
    if len(pcd.points) > 100:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    cloud_path = os.path.join(out_dir, "dense_cloud.ply")
    o3d.io.write_point_cloud(cloud_path, pcd)
    T["cloud"] = time.perf_counter() - t0

    return {
        "timings_s": {k: round(v, 2) for k, v in T.items()},
        "faces_before_component_filter": int(n_raw),
        "kept_face_frac": float(len(tris) / max(1, n_raw)),
        "frames_integrated": integrated, "num_cameras": len(cams), "device": str(dev),
        "depth_max_used_m": depth_max,
        "mesh_vertices": int(len(verts)), "mesh_faces": int(len(tris)),
        "components_raw": int(len(tri_n)), "largest_component_frac": largest_frac,
        "cloud_points": int(len(pcd.points)),
        "area_m2": float(area.sum()), "coverage_area_frac_2views": covered,
        "mesh_path": mesh_path, "cloud_path": cloud_path,
        "bounds_min": verts.min(axis=0).tolist() if len(verts) else None,
        "bounds_max": verts.max(axis=0).tolist() if len(verts) else None,
    }
