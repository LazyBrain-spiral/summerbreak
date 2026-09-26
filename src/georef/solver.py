"""Georeferencing solver (v3): SfM frame -> local ENU similarity with honest accuracy.

Steps
  1. telemetry -> ENU about the first keyframe (pymap3d)
  2. collinearity check on SfM camera centres. A straight single pass makes the
     rotation about the flight line unobservable from positions alone.
  3. RANSAC Umeyama on camera centres, refit on inliers
  4. orientation prior, solved jointly with positions by robust least squares:
       gimbal  camera optical axis vs gimbal yaw/pitch from the captions
       ground  dominant plane of the sparse cloud vs ENU up
     Forced when the trajectory is degenerate.
  5. hold-out RMSE: every k-th frame is left out of the fit and scored
  6. up-error: angle between the fitted ground-plane normal and ENU up
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.georef.umeyama_aligner import umeyama_alignment


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def forward_from_gimbal(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Unit optical axis in ENU. yaw clockwise from north, pitch -90 = nadir."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])


def collinearity_ratio(points: np.ndarray) -> float:
    c = points - points.mean(axis=0)
    s = np.linalg.svd(c, compute_uv=False)
    return float(s[1] / max(s[0], 1e-12))


def fit_plane_ransac(pts: np.ndarray, thresh: float, iters: int = 300, seed: int = 0) -> Tuple[np.ndarray, float, np.ndarray]:
    """Return unit normal n, offset d (n.x + d = 0) and inlier mask."""
    rng = np.random.default_rng(seed)
    best_mask, best_n, best_d = None, None, None
    n_pts = len(pts)
    if n_pts < 3:
        raise ValueError("need at least 3 points for a plane")
    for _ in range(iters):
        a, b, c = pts[rng.choice(n_pts, 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        d = -n @ a
        mask = np.abs(pts @ n + d) < thresh
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask, best_n, best_d = mask, n, d
    inl = pts[best_mask]
    cen = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - cen, full_matrices=False)
    n = vt[2]
    if best_n is not None and n @ best_n < 0:
        n = -n
    return n, float(-n @ cen), best_mask


def ransac_umeyama(X: np.ndarray, Y: np.ndarray, thresh: float, iters: int, seed: int = 0):
    """LO-MSAC Umeyama.

    Each minimal-sample hypothesis is locally optimised (refit on its inliers,
    twice) before it is scored, and scoring uses truncated squared residuals.
    Without the local step a near-collinear single pass lets a lucky 3-point
    sample with a wrong scale win with a handful of inliers.
    """
    rng = np.random.default_rng(seed)
    n = len(X)
    t2 = thresh * thresh

    def score(s, R, t):
        r2 = np.sum((Y - _apply(s, R, t, X)) ** 2, axis=1)
        return float(np.sum(np.minimum(r2, t2))), r2 < t2

    best_cost, best_inl = np.inf, None
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        if collinearity_ratio(X[idx]) < 1e-3:
            continue
        try:
            s, R, t, _ = umeyama_alignment(X[idx], Y[idx])
        except Exception:
            continue
        cost, inl = score(s, R, t)
        for _lo in range(2):
            if inl.sum() < 3:
                break
            s, R, t, _ = umeyama_alignment(X[inl], Y[inl])
            cost, inl = score(s, R, t)
        if cost < best_cost:
            best_cost, best_inl = cost, inl
    if best_inl is None or best_inl.sum() < 3:
        best_inl = np.ones(n, bool)
    s, R, t, _ = umeyama_alignment(X[best_inl], Y[best_inl])
    return s, R, t, best_inl


# ---------------------------------------------------------------------------
# joint refinement
# ---------------------------------------------------------------------------

def _pack(s, R, t):
    return np.concatenate([[math.log(s)], Rotation.from_matrix(R).as_rotvec(), t])


def _unpack(x):
    return math.exp(x[0]), Rotation.from_rotvec(x[1:4]).as_matrix(), x[4:7]


def refine_with_orientation(X, Y, s0, R0, t0, sigma_pos: float,
                            cam_axes_sfm: Optional[np.ndarray] = None, gimbal_axes: Optional[np.ndarray] = None,
                            ground_normal_sfm: Optional[np.ndarray] = None, sigma_dir_rad: float = 0.05,
                            ground_weight: float = 5.0):
    """Minimise position residuals plus direction residuals over Sim(3)."""
    up = np.array([0.0, 0.0, 1.0])

    def residuals(x):
        s, R, t = _unpack(x)
        r = [((s * (R @ X.T)).T + t - Y).ravel() / sigma_pos]
        if cam_axes_sfm is not None and gimbal_axes is not None:
            r.append(((R @ cam_axes_sfm.T).T - gimbal_axes).ravel() / sigma_dir_rad)
        if ground_normal_sfm is not None:
            r.append(ground_weight * (R @ ground_normal_sfm - up) / sigma_dir_rad)
        return np.concatenate(r)

    sol = least_squares(residuals, _pack(s0, R0, t0), loss="soft_l1", f_scale=2.0)
    return _unpack(sol.x)


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def utm_epsg(lat: float, lon: float) -> int:
    zone = int((lon + 180.0) // 6.0) + 1
    return (32600 if lat >= 0 else 32700) + zone


def _apply(s, R, t, X):
    return (s * (R @ X.T)).T + t


def solve_georef(names: List[str], centers_sfm: np.ndarray, R_cw_sfm: np.ndarray,
                 telemetry: Dict[str, Dict[str, Any]], sparse_xyz: np.ndarray,
                 cfg: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
    """Return the georef dict. ``telemetry`` maps image name -> synchronised row."""
    import pymap3d

    keep = [i for i, n in enumerate(names)
            if n in telemetry and telemetry[n].get("lat") is not None and telemetry[n].get("alt") is not None
            and telemetry[n].get("in_span", True)]
    if len(keep) < 4:
        raise ValueError(f"only {len(keep)} registered frames have telemetry; need >= 4")
    rows = [telemetry[names[i]] for i in keep]
    lat0, lon0, h0 = rows[0]["lat"], rows[0]["lon"], rows[0]["alt"]
    Y = np.array([pymap3d.geodetic2enu(r["lat"], r["lon"], r["alt"], lat0, lon0, h0) for r in rows], float)
    X = centers_sfm[keep]
    Rcw = R_cw_sfm[keep]
    n = len(X)

    ratio = collinearity_ratio(Y)
    degenerate = ratio < cfg["collinear_ratio"]

    # orientation evidence
    cam_axes = np.stack([Rc.T[:, 2] for Rc in Rcw])  # optical axis in SfM frame
    gim = None
    if all(r.get("gimbal_pitch") is not None and r.get("gimbal_yaw") is not None for r in rows):
        gim = np.stack([forward_from_gimbal(r["gimbal_yaw"], r["gimbal_pitch"]) for r in rows])
    ground_n = None
    if len(sparse_xyz) >= 50:
        # plane threshold relative to scene extent in SfM units
        ext = float(np.linalg.norm(np.percentile(sparse_xyz, 95, axis=0) - np.percentile(sparse_xyz, 5, axis=0)))
        gn, _, gmask = fit_plane_ransac(sparse_xyz, thresh=0.01 * ext, seed=seed)
        # orient toward the cameras
        if np.mean((X - sparse_xyz[gmask].mean(axis=0)) @ gn) < 0:
            gn = -gn
        ground_n = gn
    mode = cfg["orientation_prior"]
    if mode == "auto":
        mode = "gimbal" if gim is not None else ("ground" if ground_n is not None else "none")
    if degenerate and mode == "none":
        raise ValueError("trajectory is collinear and no orientation evidence is available; up is unobservable")

    def fit(idx):
        s, R, t, inl = ransac_umeyama(X[idx], Y[idx], cfg["inlier_thresh_m"], cfg["ransac_iters"], seed)
        use = np.asarray(idx)[inl]
        if mode != "none":
            s, R, t = refine_with_orientation(
                X[use], Y[use], s, R, t, cfg["sigma_pos_m"],
                cam_axes_sfm=cam_axes[use] if mode == "gimbal" else None,
                gimbal_axes=gim[use] if mode == "gimbal" else None,
                ground_normal_sfm=ground_n if mode == "ground" else None,
                sigma_dir_rad=math.radians(cfg["sigma_dir_deg"]))
        return s, R, t, use

    all_idx = np.arange(n)
    k = max(2, int(cfg["holdout_every"]))
    hold = all_idx[(all_idx % k) == (k // 2)]
    train = np.setdiff1d(all_idx, hold)
    s_h, R_h, t_h, _ = fit(train)
    hold_res = np.linalg.norm(_apply(s_h, R_h, t_h, X[hold]) - Y[hold], axis=1) if len(hold) else np.array([])

    s, R, t, inliers = fit(all_idx)
    res = _apply(s, R, t, X) - Y
    rmse_fit = float(np.sqrt(np.mean(np.sum(res[inliers] ** 2, axis=1))))

    up_err = None
    if ground_n is not None:
        up_err = float(math.degrees(math.acos(np.clip((R @ ground_n)[2], -1.0, 1.0))))
    gimbal_err = None
    if gim is not None:
        ang = np.degrees(np.arccos(np.clip(np.sum((R @ cam_axes.T).T * gim, axis=1), -1, 1)))
        gimbal_err = float(np.median(ang))

    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return {
        "scale": float(s), "rotation": R.tolist(), "translation": t.tolist(), "matrix_sfm_to_enu": M.tolist(),
        "transformation_formula": "X_enu = scale * rotation @ X_sfm + translation",
        "enu_origin": {"lat": lat0, "lon": lon0, "alt": h0, "datum": "WGS84",
                       "alt_source": rows[0].get("alt_source")},
        "utm_epsg": utm_epsg(lat0, lon0),
        "num_frames": n, "num_inliers": int(len(inliers)),
        "rmse_fit_m": rmse_fit,
        "rmse_holdout_m": float(np.sqrt(np.mean(hold_res ** 2))) if len(hold_res) else None,
        "holdout_frames": int(len(hold)),
        "rmse_axis_m": np.sqrt(np.mean(res[inliers] ** 2, axis=0)).tolist(),
        "collinearity_ratio": ratio, "degenerate_trajectory": bool(degenerate),
        "orientation_prior": mode,
        "up_error_deg": up_err,
        "gimbal_axis_error_deg": gimbal_err,
        "per_frame_residual_m": {names[keep[i]]: float(np.linalg.norm(res[i])) for i in range(n)},
    }


def poses_to_enu(geo: Dict[str, Any], R_cw: np.ndarray, t_cw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """World(SfM)->camera pose to ENU->metric camera pose.

    x_cam_m = s * x_cam = R_cw R^T (X_enu - t) + s t_cw
    """
    s = geo["scale"]
    R = np.asarray(geo["rotation"])
    t = np.asarray(geo["translation"])
    R_new = R_cw @ R.T
    t_new = s * t_cw - R_new @ t
    return R_new, t_new


def solve_relative(names: List[str], centers_sfm: np.ndarray, R_cw_sfm: np.ndarray, sparse_xyz: np.ndarray,
                   assumed_altitude_m: float, seed: int = 0) -> Dict[str, Any]:
    """GPS-denied frame: z up from the dominant ground plane, y along the mean viewing
    direction, metric scale from an ASSUMED camera height above that plane.

    The output is not georeferenced (no CRS, no lat/lon). Relative shapes are right;
    absolute size is only as good as the assumed altitude.
    """
    if len(sparse_xyz) < 50:
        raise ValueError("too few sparse points to find the ground plane")
    ext = float(np.linalg.norm(np.percentile(sparse_xyz, 95, axis=0) - np.percentile(sparse_xyz, 5, axis=0)))
    n, d, mask = fit_plane_ransac(sparse_xyz, thresh=0.01 * ext, seed=seed)
    if np.mean(centers_sfm @ n + d) < 0:
        n, d = -n, -d
    heights = centers_sfm @ n + d
    h_med = float(np.median(heights))
    if h_med <= 0:
        raise ValueError("cameras are not above the fitted ground plane")
    fwd = np.stack([Rc.T[:, 2] for Rc in R_cw_sfm]).mean(axis=0)
    fwd = fwd - (fwd @ n) * n
    if np.linalg.norm(fwd) < 1e-9:
        fwd = np.cross(n, [1.0, 0.0, 0.0])
    y = fwd / np.linalg.norm(fwd)
    x = np.cross(y, n)
    R = np.stack([x, y, n])
    s = float(assumed_altitude_m) / h_med
    c0 = s * (R @ centers_sfm[0])
    ground0 = -s * d
    t = np.array([-c0[0], -c0[1], -ground0])
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return {
        "scale": s, "rotation": R.tolist(), "translation": t.tolist(), "matrix_sfm_to_enu": M.tolist(),
        "transformation_formula": "X_local = scale * rotation @ X_sfm + translation",
        "georeferenced": False, "scale_source": "assumed_altitude", "assumed_altitude_m": float(assumed_altitude_m),
        "enu_origin": {"lat": None, "lon": None, "alt": 0.0, "datum": None, "alt_source": "assumed"},
        "utm_epsg": None, "num_frames": int(len(names)), "num_inliers": int(len(names)),
        "rmse_fit_m": None, "rmse_holdout_m": None, "rmse_axis_m": None,
        "collinearity_ratio": collinearity_ratio(centers_sfm), "degenerate_trajectory": False,
        "orientation_prior": "ground_plane", "up_error_deg": 0.0, "gimbal_axis_error_deg": None,
        "ground_plane_inlier_frac": float(mask.mean()),
        "camera_height_spread_m": float(np.std(heights) * s),
    }
