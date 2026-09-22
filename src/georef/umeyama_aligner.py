"""
7-DoF Umeyama Sim(3) GPS Alignment Module (Workstream B - Stage 6)
Aligns SfM camera optical centers to GPS/IMU drone positions in a local East-North-Up (ENU)
metric Cartesian frame using the closed-form Umeyama SVD algorithm.
Calculates and logs the metric Root Mean Square Error (RMSE) and outputs georef.json.
"""

import os
import json
import csv
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

try:
    import pymap3d
    HAS_PYMAP3D = True
except ImportError:
    HAS_PYMAP3D = False


def geodetic_to_enu_fallback(
    lat: float, lon: float, alt: float,
    lat0: float, lon0: float, alt0: float
) -> Tuple[float, float, float]:
    """
    Fallback WGS84 Geodetic to Local ENU conversion using WGS84 ellipsoid parameters.
    Used when pymap3d is not installed.
    """
    # WGS84 semi-major axis and eccentricity squared
    a = 6378137.0
    e_sq = 0.00669437999014

    phi = np.radians(lat)
    lam = np.radians(lon)
    phi0 = np.radians(lat0)
    lam0 = np.radians(lon0)

    # ECEF conversion
    def to_ecef(p, l, h):
        sin_p = np.sin(p)
        cos_p = np.cos(p)
        sin_l = np.sin(l)
        cos_l = np.cos(l)
        N = a / np.sqrt(1 - e_sq * sin_p**2)
        x = (N + h) * cos_p * cos_l
        y = (N + h) * cos_p * sin_l
        z = (N * (1 - e_sq) + h) * sin_p
        return x, y, z

    x, y, z = to_ecef(phi, lam, alt)
    x0, y0, z0 = to_ecef(phi0, lam0, alt0)

    dx, dy, dz = x - x0, y - y0, z - z0

    # Rotate to ENU
    sin_p0 = np.sin(phi0)
    cos_p0 = np.cos(phi0)
    sin_l0 = np.sin(lam0)
    cos_l0 = np.cos(lam0)

    e = -sin_l0 * dx + cos_l0 * dy
    n = -sin_p0 * cos_l0 * dx - sin_p0 * sin_l0 * dy + cos_p0 * dz
    u = cos_p0 * cos_l0 * dx + cos_p0 * sin_l0 * dy + sin_p0 * dz

    return float(e), float(n), float(u)


def geodetic_to_enu(
    lat: float, lon: float, alt: float,
    lat0: float, lon0: float, alt0: float
) -> Tuple[float, float, float]:
    if HAS_PYMAP3D:
        return pymap3d.geodetic2enu(lat, lon, alt, lat0, lon0, alt0)
    return geodetic_to_enu_fallback(lat, lon, alt, lat0, lon0, alt0)


def umeyama_alignment(X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Computes the optimal similarity transform (scale s, rotation R, translation t)
    such that: Y ~ s * R @ X + t
    
    Args:
        X: Source points (e.g. SfM camera centers), shape (N, 3)
        Y: Target points (e.g. GPS coordinates in local ENU), shape (N, 3)
        
    Returns:
        s: Metric scale factor (float)
        R: 3x3 orthonormal rotation matrix
        t: 3D translation vector (3,)
        rmse: Root Mean Square Error in meters
    """
    N, m = X.shape
    assert Y.shape == (N, m), "Point sets X and Y must have the same shape"
    assert N >= 3, "At least 3 point correspondences are required for Sim(3) solve"

    # Compute centroids
    mu_X = np.mean(X, axis=0)
    mu_Y = np.mean(Y, axis=0)

    # Shift points to centroid
    X_zero = X - mu_X
    Y_zero = Y - mu_Y

    # Compute variances
    sigma_X_sq = np.sum(X_zero ** 2) / N

    # Cross-covariance matrix
    Sigma_XY = (Y_zero.T @ X_zero) / N

    # Singular Value Decomposition
    U, D, Vt = np.linalg.svd(Sigma_XY)
    V = Vt.T

    # Reflection check
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(V) < 0:
        S[m - 1, m - 1] = -1

    # Optimal rotation
    R = U @ S @ Vt

    # Optimal scale
    s = (1.0 / sigma_X_sq) * np.sum(D * np.diag(S))

    # Optimal translation
    t = mu_Y - s * (R @ mu_X)

    # Compute Residual RMSE in meters
    Y_pred = (s * (R @ X.T)).T + t
    residuals = np.linalg.norm(Y - Y_pred, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return float(s), R, t, rmse


def align_sfm_to_gps(
    sfm_camera_centers: Dict[str, np.ndarray],
    frame_gps_csv: str,
    output_georef_json: str
) -> Dict[str, Any]:
    """
    Matches camera centers with GPS coordinates, solves Umeyama Sim(3),
    and exports georef.json.
    """
    if not os.path.exists(frame_gps_csv):
        raise FileNotFoundError(f"GPS CSV file not found: {frame_gps_csv}")

    # Read GPS CSV
    gps_data: Dict[str, Tuple[float, float, float]] = {}
    with open(frame_gps_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row["filename"]
            lat = float(row["lat"])
            lon = float(row["lon"])
            alt = float(row["alt"])
            gps_data[fname] = (lat, lon, alt)

    # Match filenames
    common_keys = sorted(list(set(sfm_camera_centers.keys()) & set(gps_data.keys())))
    if len(common_keys) < 3:
        raise ValueError(f"Found only {len(common_keys)} common frames between SfM and GPS. Minimum 3 required.")

    # Select local origin (first frame)
    origin_fname = common_keys[0]
    lat0, lon0, alt0 = gps_data[origin_fname]

    X_list = []
    Y_list = []

    for k in common_keys:
        X_list.append(sfm_camera_centers[k])
        lat, lon, alt = gps_data[k]
        e, n, u = geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0)
        Y_list.append([e, n, u])

    X = np.array(X_list, dtype=np.float64)
    Y = np.array(Y_list, dtype=np.float64)

    # Solve Umeyama
    s, R, t, rmse = umeyama_alignment(X, Y)

    result = {
        "scale": s,
        "rotation": R.tolist(),
        "translation": t.tolist(),
        "rms_error_meters": round(rmse, 3),
        "num_aligned_views": len(common_keys),
        "local_origin": {
            "lat": lat0,
            "lon": lon0,
            "alt": alt0,
            "datum": "WGS84"
        },
        "transformation_formula": "Y_enu = scale * (Rotation @ X_sfm) + Translation"
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_georef_json)), exist_ok=True)
    with open(output_georef_json, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

    print(f"[Georeferencing] Umeyama Sim(3) Alignment complete:")
    print(f"  -> Metric Scale s : {s:.6f}")
    print(f"  -> RMS Error      : {rmse:.3f} meters")
    print(f"  -> Aligned Frames : {len(common_keys)}")
    print(f"  -> Saved to       : {output_georef_json}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Solve 7-DoF Umeyama Sim(3) alignment")
    parser.add_argument("--gps_csv", required=True, help="Path to frame_gps.csv")
    parser.add_argument("--output", default="data/georef.json", help="Output georef.json")
    args = parser.parse_args()
