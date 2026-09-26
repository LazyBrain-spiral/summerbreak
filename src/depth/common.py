"""Shared dense-depth contract and utilities (v3).

Contract written by both lanes into 05_depth/:
    cameras.json         per image: name, image path, width, height, K (pinhole),
                         R_cw, t_cw (ENU world -> metric camera), center_enu
    depth/<stem>.npy     float32 metres, 0 = invalid
    conf/<stem>.npy      float16 in [0, 1]
Images referenced by cameras.json are undistorted, so K is a plain pinhole.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.io import read_json, write_json
from src.georef.solver import poses_to_enu
from src.sfm.model_io import SparseModel, load_model


# ---------------------------------------------------------------------------
# undistortion
# ---------------------------------------------------------------------------

def undistort(model_dir: str, image_dir: str, out_dir: str, max_image_size: int) -> str:
    """Undistort with COLMAP (CLI if runnable, else pycolmap). Returns the workspace dir."""
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    exe = shutil.which("colmap")
    ok_cli = False
    if exe:
        probe = subprocess.run([exe, "help"], capture_output=True, text=True)
        ok_cli = probe.returncode == 0
    if ok_cli:
        subprocess.run([exe, "image_undistorter", "--image_path", image_dir, "--input_path", model_dir,
                        "--output_path", out_dir, "--output_type", "COLMAP",
                        "--max_image_size", str(max_image_size)], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        subprocess.run([exe, "model_converter", "--input_path", os.path.join(out_dir, "sparse"),
                        "--output_path", os.path.join(out_dir, "sparse"), "--output_type", "TXT"],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    else:
        import pycolmap

        opts = pycolmap.UndistortCameraOptions()
        opts.max_image_size = int(max_image_size)
        pycolmap.undistort_images(out_dir, model_dir, image_dir, undistort_options=opts)
    return out_dir


def build_cameras(undist_model: SparseModel, geo: Dict[str, Any], image_root: str) -> List[Dict[str, Any]]:
    cams = []
    for im in sorted(undist_model.images.values(), key=lambda i: i.name):
        cam = undist_model.cameras[im.camera_id]
        R, t = poses_to_enu(geo, im.R_cw, im.t_cw)
        cams.append({"name": im.name, "image": os.path.join(image_root, im.name),
                     "width": cam.width, "height": cam.height, "K": cam.K.tolist(),
                     "R_cw": R.tolist(), "t_cw": t.tolist(), "center_enu": (-R.T @ t).tolist()})
    return cams


def anchors_for_image(model: SparseModel, image_name: str, geo: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """SfM track observations in an undistorted image: pixel (N,2) and metric depth (N,)."""
    im = model.by_name()[image_name]
    sel = im.point3D_ids >= 0
    if not sel.any():
        return np.empty((0, 2)), np.empty(0)
    pidx = model.point_index()
    ids = im.point3D_ids[sel]
    ok = np.array([pid in pidx for pid in ids])
    X = model.xyz[[pidx[p] for p in ids[ok]]]
    z = (X @ im.R_cw.T + im.t_cw)[:, 2] * geo["scale"]
    uv = im.xys[sel][ok]
    good = z > 0
    return uv[good], z[good]


# ---------------------------------------------------------------------------
# scale / shift fitting of relative depth
# ---------------------------------------------------------------------------

def _sample(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Bilinear sample at COLMAP pixel coordinates (first pixel centre is 0.5, 0.5)."""
    from src.core.imgops import sample_points

    return sample_points(img.astype(np.float32), uv[:, 0] - 0.5, uv[:, 1] - 0.5)


def fit_affine_depth(pred: np.ndarray, uv: np.ndarray, z: np.ndarray, iters: int = 300,
                     rel_thresh: float = 0.05, seed: int = 0) -> Dict[str, Any]:
    """Fit metric depth to a relative prediction.

    Tries both parameterisations and keeps the better one:
      disparity:  1/z = a * pred + b     (Depth Anything style outputs)
      depth:        z = a * pred + b
    Returns dict with mode, a, b, median relative error, inlier count.
    """
    rng = np.random.default_rng(seed)
    p = _sample(pred, uv).astype(np.float64)
    best = None
    for mode in ("disparity", "depth"):
        target = 1.0 / z if mode == "disparity" else z
        n = len(p)
        best_inl = None
        for _ in range(iters):
            i, j = rng.choice(n, 2, replace=False)
            if abs(p[i] - p[j]) < 1e-9:
                continue
            a = (target[i] - target[j]) / (p[i] - p[j])
            b = target[i] - a * p[i]
            est = a * p + b
            with np.errstate(divide="ignore", invalid="ignore"):
                z_est = 1.0 / est if mode == "disparity" else est
            inl = np.isfinite(z_est) & (z_est > 0) & (np.abs(z_est - z) < rel_thresh * z)
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl = inl
        if best_inl is None or best_inl.sum() < 3:
            continue
        A = np.column_stack([p[best_inl], np.ones(best_inl.sum())])
        a, b = np.linalg.lstsq(A, target[best_inl], rcond=None)[0]
        est = a * p + b
        with np.errstate(divide="ignore", invalid="ignore"):
            z_est = 1.0 / est if mode == "disparity" else est
        rel = np.abs(z_est - z) / z
        rel = rel[np.isfinite(rel)]
        cand = {"mode": mode, "a": float(a), "b": float(b), "n_anchors": int(len(z)),
                "z_max": float(np.percentile(z, 99.5) * 2.0),
                "n_inliers": int(best_inl.sum()),
                "median_rel_err": float(np.median(rel)) if len(rel) else np.inf}
        if best is None or cand["n_inliers"] > best["n_inliers"] or (
                cand["n_inliers"] == best["n_inliers"] and cand["median_rel_err"] < best["median_rel_err"]):
            best = cand
    if best is None:
        return {"mode": None, "median_rel_err": np.inf, "n_anchors": int(len(z)), "n_inliers": 0}
    return best


def apply_affine(pred: np.ndarray, fit: Dict[str, Any]) -> np.ndarray:
    est = fit["a"] * pred.astype(np.float64) + fit["b"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = 1.0 / est if fit["mode"] == "disparity" else est
    z = np.where(np.isfinite(z) & (z > 0), z, 0.0)
    # sky / beyond-scene pixels: disparity near the fit's zero crossing explodes to huge depth
    if fit.get("z_max"):
        z[z > fit["z_max"]] = 0.0
    return z.astype(np.float32)


def local_correction(depth: np.ndarray, uv: np.ndarray, z: np.ndarray, cell_px: int = 24,
                     min_per_cell: int = 3, max_rel: float = 0.3) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Spatially varying correction of a globally fitted depth map using SfM anchors.

    A monocular network gets the global layout right but flattens local relief
    (buildings come out too low). With thousands of triangulated anchors per frame
    the residual in inverse depth can be mapped on a coarse grid, smoothed, and
    added back. Cells without anchors take the nearest cell's value; the map is
    clamped to +/- max_rel so a bad anchor cannot fold the surface.
    """
    from scipy import ndimage

    from src.core.imgops import sample_points

    h, w = depth.shape
    if len(z) < 30:
        return depth, {"applied": False, "reason": "few anchors"}
    d_at = sample_points(depth, uv[:, 0] - 0.5, uv[:, 1] - 0.5)
    ok = d_at > 0
    inv_res = np.where(ok, 1.0 / z - 1.0 / np.maximum(d_at, 1e-6), 0.0)
    gh, gw = -(-h // cell_px), -(-w // cell_px)
    ci = np.clip((uv[:, 1] - 0.5) // cell_px, 0, gh - 1).astype(int)
    cj = np.clip((uv[:, 0] - 0.5) // cell_px, 0, gw - 1).astype(int)
    flat = ci * gw + cj
    grid = np.full(gh * gw, np.nan)
    order = np.argsort(flat[ok])
    f_ok, r_ok = flat[ok][order], inv_res[ok][order]
    starts = np.r_[0, np.flatnonzero(np.diff(f_ok)) + 1]
    ends = np.r_[starts[1:], len(f_ok)]
    for a, b in zip(starts, ends):
        if b - a >= min_per_cell:
            grid[f_ok[a]] = np.median(r_ok[a:b])
    grid = grid.reshape(gh, gw)
    filled_frac = float(np.isfinite(grid).mean())
    if not np.isfinite(grid).any():
        return depth, {"applied": False, "reason": "no cell has enough anchors"}
    nan = ~np.isfinite(grid)
    if nan.any():
        _, (ri, cj2) = ndimage.distance_transform_edt(nan, return_indices=True)
        grid = grid[ri, cj2]
    grid = ndimage.gaussian_filter(grid, 1.0)
    corr = cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    inv = np.where(depth > 0, 1.0 / np.maximum(depth, 1e-6), 0.0)
    inv_new = inv + corr
    inv_new = np.clip(inv_new, inv * (1 - max_rel), inv * (1 + max_rel))
    out = np.where((depth > 0) & (inv_new > 0), 1.0 / np.maximum(inv_new, 1e-9), 0.0).astype(np.float32)
    after = sample_points(out, uv[:, 0] - 0.5, uv[:, 1] - 0.5)
    rel = np.abs(after[ok] - z[ok]) / z[ok]
    return out, {"applied": True, "cells_with_anchors": filled_frac,
                 "median_rel_err_after": float(np.median(rel)) if len(rel) else None}


# ---------------------------------------------------------------------------
# multi-view consistency
# ---------------------------------------------------------------------------

def reproject_check(depth_i: np.ndarray, cam_i: Dict[str, Any], depth_j: np.ndarray, cam_j: Dict[str, Any],
                    rel_tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """For every valid pixel of i: (visible_in_j, consistent_with_j) boolean maps."""
    h, w = depth_i.shape
    Ki, Kj = np.asarray(cam_i["K"]), np.asarray(cam_j["K"])
    Ri, ti = np.asarray(cam_i["R_cw"]), np.asarray(cam_i["t_cw"])
    Rj, tj = np.asarray(cam_j["R_cw"]), np.asarray(cam_j["t_cw"])
    v, u = np.nonzero(depth_i > 0)
    z = depth_i[v, u].astype(np.float64)
    x_cam = np.stack([(u + 0.5 - Ki[0, 2]) / Ki[0, 0] * z, (v + 0.5 - Ki[1, 2]) / Ki[1, 1] * z, z], 1)
    X = (x_cam - ti) @ Ri              # R^T (x - t)
    xj = X @ Rj.T + tj
    zj = xj[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uj = Kj[0, 0] * xj[:, 0] / zj + Kj[0, 2] - 0.5
        vj = Kj[1, 1] * xj[:, 1] / zj + Kj[1, 2] - 0.5
    hj, wj = depth_j.shape
    ui, vi = np.round(uj).astype(np.int64), np.round(vj).astype(np.int64)
    inside = (zj > 0) & (ui >= 0) & (ui < wj) & (vi >= 0) & (vi < hj)
    dj = np.zeros_like(zj)
    dj[inside] = depth_j[vi[inside], ui[inside]]
    # j sees something clearly closer: the point is occluded in j, which says nothing about it.
    occluded = dj < zj * (1.0 - rel_tol)
    visible = inside & (dj > 0) & ~occluded
    consistent = visible & (np.abs(dj - zj) <= rel_tol * zj)
    vis_map = np.zeros((h, w), bool)
    con_map = np.zeros((h, w), bool)
    vis_map[v, u] = visible
    con_map[v, u] = consistent
    return vis_map, con_map


def consistency_confidence(names: List[str], cams: Dict[str, Dict[str, Any]], depth_dir: str,
                           neighbors: int, rel_tol: float) -> Dict[str, float]:
    """conf = fraction of neighbouring views (that see the pixel) agreeing with it."""
    stats = {}
    load = lambda n: np.load(os.path.join(depth_dir, os.path.splitext(n)[0] + ".npy"))
    for i, name in enumerate(names):
        d_i = load(name)
        seen = np.zeros(d_i.shape, np.float32)
        agree = np.zeros(d_i.shape, np.float32)
        for j in range(max(0, i - neighbors), min(len(names), i + neighbors + 1)):
            if j == i:
                continue
            vis, con = reproject_check(d_i, cams[name], load(names[j]), cams[names[j]], rel_tol)
            seen += vis
            agree += con
        conf = np.where(seen > 0, agree / np.maximum(seen, 1), 0.0).astype(np.float16)
        conf[d_i <= 0] = 0
        np.save(os.path.join(os.path.dirname(depth_dir), "conf", os.path.splitext(name)[0] + ".npy"), conf)
        valid = d_i > 0
        stats[name] = float((conf[valid] >= 0.5).mean()) if valid.any() else 0.0
    return stats


def load_depth_contract(depth_root: str):
    cams = read_json(os.path.join(depth_root, "cameras.json"))
    by_name = {c["name"]: c for c in cams["cameras"]}
    return cams, by_name


def read_depth(depth_root: str, name: str) -> Tuple[np.ndarray, np.ndarray]:
    stem = os.path.splitext(name)[0]
    d = np.load(os.path.join(depth_root, "depth", stem + ".npy"))
    cp = os.path.join(depth_root, "conf", stem + ".npy")
    c = np.load(cp).astype(np.float32) if os.path.exists(cp) else (d > 0).astype(np.float32)
    return d, c
