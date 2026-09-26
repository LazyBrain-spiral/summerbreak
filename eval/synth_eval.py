"""Score a pipeline run against tools/synth3d.py ground truth.

    python eval/synth_eval.py --run runs/<id> --gt test_data/synth3d_pass

Reports (written to <run>/eval.json):
  camera_position   absolute error of estimated camera centres in the run's ENU frame
  dsm               DSM height error on observed cells vs the true scene (ray cast straight down)
  buildings         footprint area, eave and ridge height error per matched building
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.core.io import read_json, write_json  # noqa: E402


def _gt_to_run_enu(gt: Dict[str, Any], georef: Dict[str, Any]):
    import pymap3d

    o = gt["origin"]
    r = georef["enu_origin"]

    def conv(p):
        lat, lon, h = pymap3d.enu2geodetic(p[0], p[1], p[2], o["lat"], o["lon"], o["alt"])
        return np.array(pymap3d.geodetic2enu(lat, lon, h, r["lat"], r["lon"], r["alt"]))

    return conv


def evaluate(run: str, gt_dir: str) -> Dict[str, Any]:
    import open3d as o3d

    gt = read_json(os.path.join(gt_dir, "gt.json"))
    georef = read_json(os.path.join(run, "04_georef", "georef.json"))
    conv = _gt_to_run_enu(gt, georef)
    out: Dict[str, Any] = {"run": run, "gt": gt_dir}

    # ---- camera centres
    kf = {k["name"]: k for k in read_json(os.path.join(run, "01_ingest", "keyframes.json"))["keyframes"]}
    cams = read_json(os.path.join(run, "05_depth", "cameras.json"))["cameras"]
    frames = {f["index"]: f for f in gt["frames"]}
    errs = []
    for c in cams:
        true_c = conv(frames[kf[c["name"]]["frame_index"]]["C_enu"])
        errs.append(np.asarray(c["center_enu"]) - true_c)
    errs = np.array(errs)
    out["camera_position"] = {"rmse_m": float(np.sqrt(np.mean(np.sum(errs ** 2, 1)))),
                              "rmse_axis_m": np.sqrt(np.mean(errs ** 2, 0)).tolist(),
                              "mean_axis_m": errs.mean(0).tolist(), "n": int(len(errs))}

    # ---- DSM vs truth (true scene moved into the run's ENU frame)
    g = np.load(os.path.join(run, "07_products", "grids.npz"))
    from src.products.geo import LocalGeo
    from src.products.rasters import Grid

    geo = LocalGeo(georef)
    dsm, observed = g["dsm_up"], g["observed"]
    grid = Grid(float(g["xmin"]), float(g["ymax"]), float(g["gsd"]), dsm.shape[1], dsm.shape[0])
    xy = geo.utm_to_enu(grid.centers().reshape(-1, 2))
    mesh = o3d.io.read_triangle_mesh(os.path.join(gt_dir, "scene.ply"))
    offset = conv(np.zeros(3))
    v = np.asarray(mesh.vertices) + offset  # translation dominates; rotation between tangent planes is tiny
    mesh.vertices = o3d.utility.Vector3dVector(v)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.column_stack([xy, np.full(len(xy), 500.0), np.zeros((len(xy), 2)), -np.ones(len(xy))]).astype(np.float32)
    t = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    true_z = (500.0 - t).reshape(dsm.shape)
    ok = observed & np.isfinite(true_z) & np.isfinite(dsm)
    dz = (dsm - true_z)[ok]
    out["dsm"] = {"median_abs_err_m": float(np.median(np.abs(dz))), "p90_abs_err_m": float(np.percentile(np.abs(dz), 90)),
                  "bias_m": float(np.median(dz)), "observed_cells": int(ok.sum())}

    # ---- buildings
    bpath = os.path.join(run, "08_completion", "gate.json")
    if os.path.exists(bpath):
        est = read_json(bpath)["metrics"].get("buildings", [])
        rows = []
        for b in gt["buildings"]:
            c = conv(np.array([*b["center_enu"], 0.0]))[:2]
            if not est:
                break
            d = [np.linalg.norm(np.asarray(e["centroid_enu"]) - c) for e in est]
            j = int(np.argmin(d))
            if d[j] > 10:
                rows.append({"gt_center": c.tolist(), "matched": False})
                continue
            e = est[j]
            gable = b["ridge_height_m"] > b["eave_height_m"]
            true_p20 = b["eave_height_m"] + (0.2 * (b["ridge_height_m"] - b["eave_height_m"]) if gable else 0.0)
            rows.append({
                "gt_center": c.tolist(), "matched": True, "centroid_err_m": float(d[j]),
                "area_gt": b["footprint_area_m2"], "area_est": e["footprint_area_m2"],
                "area_rel_err": (e["footprint_area_m2"] - b["footprint_area_m2"]) / b["footprint_area_m2"],
                "ridge_gt": b["ridge_height_m"], "ridge_est": e["ridge_height_m"],
                "ridge_err_m": e["ridge_height_m"] - b["ridge_height_m"],
                "eave_p20_gt": min(true_p20, b["ridge_height_m"]), "eave_p20_est": e["eave_height_m"],
                "roof_observed_frac": e["roof_observed_frac"],
            })
        m = [r for r in rows if r.get("matched")]
        out["buildings"] = {
            "gt": len(gt["buildings"]), "estimated": len(est), "matched": len(m),
            "median_abs_area_rel_err": float(np.median([abs(r["area_rel_err"]) for r in m])) if m else None,
            "median_abs_ridge_err_m": float(np.median([abs(r["ridge_err_m"]) for r in m])) if m else None,
            "rows": rows,
        }
    write_json(os.path.join(run, "eval.json"), out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gt", required=True)
    args = ap.parse_args()
    res = evaluate(args.run, args.gt)
    cp, dsm = res["camera_position"], res["dsm"]
    print(f"camera centres  rmse {cp['rmse_m']:.2f} m over {cp['n']} frames (axis {np.round(cp['rmse_axis_m'], 2)})")
    print(f"DSM             median |dz| {dsm['median_abs_err_m']:.2f} m, p90 {dsm['p90_abs_err_m']:.2f} m, bias {dsm['bias_m']:+.2f} m")
    if "buildings" in res:
        b = res["buildings"]
        print(f"buildings       matched {b['matched']}/{b['gt']} (estimated {b['estimated']}), "
              f"median area err {b['median_abs_area_rel_err']}, median ridge err {b['median_abs_ridge_err_m']} m")
        for r in b["rows"]:
            if r.get("matched"):
                print(f"   area {r['area_est']:8.1f} vs {r['area_gt']:6.1f}   ridge {r['ridge_est']:5.2f} vs {r['ridge_gt']:5.2f}"
                      f"   roof observed {r['roof_observed_frac']:.2f}")
            else:
                print(f"   unmatched building at {np.round(r['gt_center'], 1)}")


if __name__ == "__main__":
    main()
