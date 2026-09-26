import numpy as np

from src.depth.common import apply_affine, fit_affine_depth, reproject_check


def test_affine_disparity_recovery():
    rng = np.random.default_rng(0)
    h, w = 120, 160
    uu, vv = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    z = 50 + 20 * uu / w + 5 * vv / h
    pred = 37.0 / z + 0.12
    uv = np.column_stack([rng.uniform(1, w - 1, 400), rng.uniform(1, h - 1, 400)])
    zi = 50 + 20 * uv[:, 0] / w + 5 * uv[:, 1] / h
    zi[:20] *= 1.5  # outlier anchors
    fit = fit_affine_depth(pred.astype(np.float32), uv, zi)
    assert fit["mode"] == "disparity"
    assert fit["median_rel_err"] < 0.005
    d = apply_affine(pred, fit)
    assert np.median(np.abs(d - z) / z) < 0.005


def _cam(cx, K):
    return {"K": K.tolist(), "R_cw": np.eye(3).tolist(), "t_cw": (-np.array([cx, 0, 0])).tolist()}


def test_reproject_consistency_on_plane():
    K = np.array([[100.0, 0, 80], [0, 100.0, 60], [0, 0, 1]])
    depth = np.full((120, 160), 20.0, np.float32)
    ci, cj = _cam(0.0, K), _cam(1.0, K)
    vis, con = reproject_check(depth, ci, depth, cj, 0.03)
    assert vis.mean() > 0.9 and con[vis].mean() > 0.99
    # a floating surface in front of the true one: j sees through it, so it is inconsistent
    vis, con = reproject_check(depth * 0.8, ci, depth, cj, 0.03)
    assert vis.mean() > 0.9 and con.mean() < 0.05


def test_local_correction_restores_relief():
    """A 'network' that flattens a box by half is corrected by dense anchors."""
    from src.depth.common import local_correction

    h, w = 120, 160
    z_true = np.full((h, w), 60.0, np.float32)
    z_true[40:80, 60:110] = 48.0  # 12 m tall block
    flat = z_true.copy()
    flat[40:80, 60:110] = 54.0    # prediction halves the relief
    rng = np.random.default_rng(1)
    uv = np.column_stack([rng.uniform(0.5, w - 0.5, 3000), rng.uniform(0.5, h - 0.5, 3000)])
    z = z_true[(uv[:, 1] - 0.5).astype(int), (uv[:, 0] - 0.5).astype(int)]
    out, info = local_correction(flat, uv, z, cell_px=8)
    assert info["applied"]
    assert abs(float(np.median(out[48:72, 70:100])) - 48.0) < 1.0
    assert abs(float(np.median(out[:30, :])) - 60.0) < 0.5
