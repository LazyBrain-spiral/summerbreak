import math

import numpy as np
import pytest

from src.georef.solver import collinearity_ratio, forward_from_gimbal, poses_to_enu, solve_georef
from synth3d import camera_to_world_rotation

pymap3d = pytest.importorskip("pymap3d")

CFG = {"inlier_thresh_m": 4.0, "ransac_iters": 300, "holdout_every": 5, "orientation_prior": "auto",
       "sigma_pos_m": 2.0, "sigma_dir_deg": 3.0, "collinear_ratio": 0.05,
       "max_holdout_rmse_m": 3.0, "max_up_error_deg": 3.0}
LAT0, LON0, H0 = 28.6, 77.2, 250.0


def _scene(n=40, curved=False, gps_sigma=1.0, outliers=0, seed=0, with_gimbal=True):
    """True ENU trajectory mapped into an arbitrary SfM frame by a known Sim(3)."""
    rng = np.random.default_rng(seed)
    s_true = 0.37
    R_true = camera_to_world_rotation(33.0, -20.0)  # any rotation
    t_true = np.array([4.0, -2.0, 7.0])
    xs = np.linspace(-80, 80, n)
    C_enu = np.stack([xs, (0.004 * xs ** 2 if curved else np.zeros(n)), np.full(n, 60.0)], 1)
    R_wc = [camera_to_world_rotation(90.0, -60.0) for _ in range(n)]
    # X_enu = s R X_sfm + t  =>  X_sfm = R^T (X_enu - t) / s
    C_sfm = ((C_enu - t_true) @ R_true) / s_true
    R_cw_sfm = np.stack([(R_true.T @ Rw).T for Rw in R_wc])
    gps = C_enu + rng.normal(0, gps_sigma, C_enu.shape)
    if outliers:
        gps[rng.choice(n, outliers, replace=False)] += rng.normal(0, 40, (outliers, 3))
    names = [f"f{i:03d}.jpg" for i in range(n)]
    tel = {}
    for i, nme in enumerate(names):
        lat, lon, h = pymap3d.enu2geodetic(*gps[i], LAT0, LON0, H0)
        tel[nme] = {"lat": lat, "lon": lon, "alt": h, "alt_source": "abs_alt", "in_span": True,
                    "gimbal_yaw": 90.0 if with_gimbal else None, "gimbal_pitch": -60.0 if with_gimbal else None}
    g = np.column_stack([rng.uniform(-100, 100, 400), rng.uniform(0, 80, 400), rng.normal(0, 0.1, 400)])
    g_sfm = ((g - t_true) @ R_true) / s_true
    return names, C_sfm, R_cw_sfm, tel, g_sfm, (s_true, R_true, t_true), C_enu


def test_collinearity_ratio():
    line = np.column_stack([np.linspace(0, 100, 50), np.zeros(50), np.zeros(50)])
    assert collinearity_ratio(line + np.random.default_rng(0).normal(0, 0.5, line.shape)) < 0.05


@pytest.mark.parametrize("prior", ["gimbal", "ground"])
def test_straight_pass_recovers_up(prior):
    names, C, Rcw, tel, g, truth, _ = _scene(with_gimbal=prior == "gimbal")
    geo = solve_georef(names, C, Rcw, tel, g, dict(CFG, orientation_prior=prior))
    assert geo["degenerate_trajectory"]
    assert abs(geo["scale"] / truth[0] - 1) < 0.02
    R_fit = np.asarray(geo["rotation"])
    up_sfm_true = truth[1].T @ np.array([0, 0, 1.0])
    ang = math.degrees(math.acos(np.clip((R_fit @ up_sfm_true)[2], -1, 1)))
    assert ang < 2.0, ang
    assert geo["rmse_holdout_m"] < 3.0


def test_straight_pass_without_prior_is_refused():
    names, C, Rcw, tel, g, _, _ = _scene(with_gimbal=False)
    with pytest.raises(ValueError, match="unobservable"):
        solve_georef(names, C, Rcw, tel, np.empty((0, 3)), dict(CFG, orientation_prior="auto"))


def test_ransac_rejects_gps_outliers():
    names, C, Rcw, tel, g, truth, _ = _scene(curved=True, outliers=4, seed=3)
    geo = solve_georef(names, C, Rcw, tel, g, CFG)
    assert geo["num_inliers"] <= len(names) - 3
    assert abs(geo["scale"] / truth[0] - 1) < 0.02


def test_poses_to_enu_relative_motion():
    names, C, Rcw, tel, g, _, C_enu = _scene(curved=True)
    geo = solve_georef(names, C, Rcw, tel, g, CFG)

    def center(i):
        R_new, t_new = poses_to_enu(geo, Rcw[i], -Rcw[i] @ C[i])
        return -R_new.T @ t_new

    assert np.linalg.norm((center(20) - center(7)) - (C_enu[20] - C_enu[7])) < 1.5


def test_forward_from_gimbal():
    np.testing.assert_allclose(forward_from_gimbal(0, -90), [0, 0, -1], atol=1e-12)
    np.testing.assert_allclose(forward_from_gimbal(90, 0), [1, 0, 0], atol=1e-12)


def test_ransac_keeps_inliers_on_noisy_straight_line():
    """Regression: plain RANSAC kept 4/19 frames on a straight pass because a lucky
    3-point sample with the wrong scale won. LO-MSAC must keep nearly all of them."""
    from src.georef.solver import ransac_umeyama

    for seed in range(5):
        names, C, Rcw, tel, g, truth, C_enu = _scene(n=19, gps_sigma=1.2, seed=seed)
        rows = [tel[n] for n in names]
        Y = np.array([pymap3d.geodetic2enu(r["lat"], r["lon"], r["alt"], LAT0, LON0, H0) for r in rows])
        s, R, t, inl = ransac_umeyama(C, Y, 4.0, 300, seed)
        assert inl.sum() >= 17, (seed, inl.sum())
        assert abs(s / truth[0] - 1) < 0.02
