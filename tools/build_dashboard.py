"""Build the v3 results dashboard: one self-contained HTML page from finished runs.

    python tools/build_dashboard.py --runs synth_survey synth_da2 synth_da synth_oracle \
        --mesh-runs synth_survey synth_da2 --out webapp/dashboard.html

Embeds per run: stage gates and timings, evaluation against ground truth (if
eval.json exists), orthomosaic / DSM hillshade / height-above-ground images,
building footprints in image pixels, and, for --mesh-runs, a decimated GLB of the
fused surface plus the completed LOD2 buildings.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from src.core.io import read_json  # noqa: E402

LABELS = {
    "testvideo2_pi3": ("Test video 2 · Pi3X", "Real 2.7K drone video, Pi3X multi-view depth conditioned on SfM; no GPS; scale from an assumed 60 m height"),
    "test4video_pi3": ("test4video · Pi3X", "Same video, Pi3X multi-view depth conditioned on SfM; no GPS; 83 m height from one car"),
    "synth_pi3": ("Pi3X", "Pi3X multi-view depth conditioned on SfM poses, intrinsics and sparse depth"),
    "upload_test5video_60e902": ("test5video", "Your test5video.mp4 (1080p, 15 s, oblique over forest), LIVE lane, no GPS; scale from an assumed 60 m height"),
    "upload_test4video_b2638a": ("test4video", "Your test4video.mp4 (832x464, 53 s, nadir), LIVE lane, no GPS; scale from an 83 m height estimated from one car"),
    "test4video_survey": ("test4video · SURVEY", "Same video, COLMAP PatchMatch stereo depth; no GPS; 83 m height from one car"),
    "upload_testv3_d9d9e9": ("testv3", "Your uploaded testv3.mp4 (848x480, 5 s), LIVE lane, no GPS"),
    "testvideo2_live": ("Test video 2", "Real 2.7K drone video, LIVE lane: Depth Anything V2 Small with anchor correction; no GPS"),
    "testvideo2_survey": ("Test video 2 · SURVEY", "Real 2.7K drone video, COLMAP PatchMatch stereo; no GPS"),
    "synth_survey": ("SURVEY", "COLMAP PatchMatch stereo (CUDA)"),
    "synth_da2": ("LIVE + correction", "Depth Anything V2 Small, per-frame fit and anchor correction map"),
    "synth_da": ("LIVE, fit only", "Depth Anything V2 Small, one scale and shift per frame"),
    "synth_oracle": ("Oracle", "True depth under an unknown scale and shift (validates everything but the network)"),
}


def b64_jpeg(img_bgr: np.ndarray, quality: int = 82, max_w: int = 900) -> str:
    h, w = img_bgr.shape[:2]
    if w > max_w:
        img_bgr = cv2.resize(img_bgr, (max_w, int(round(h * max_w / w))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def hillshade(z, gsd):
    from preview_run import hillshade as hs

    return hs(z, gsd)


def run_images(run: str):
    import rasterio

    g = np.load(os.path.join(run, "07_products", "grids.npz"))
    dsm, dtm, gsd = g["dsm_up"], g["dtm_up"], float(g["gsd"])
    with rasterio.open(os.path.join(run, "07_products", "ortho.tif")) as src:
        rgb = np.stack([src.read(i) for i in (1, 2, 3)], -1)
        alpha = src.read(4)
    ortho = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ortho[alpha == 0] = (28, 30, 29)
    hs = cv2.cvtColor(hillshade(dsm, gsd), cv2.COLOR_GRAY2BGR)
    nd = np.clip(np.nan_to_num(dsm - dtm, nan=0) / 20.0, 0, 1)
    ndc = cv2.applyColorMap((nd * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    ndc[np.isnan(dsm)] = (28, 30, 29)
    h, w = dsm.shape
    return {"ortho": b64_jpeg(ortho), "hillshade": b64_jpeg(hs), "ndsm": b64_jpeg(ndc),
            "size": [w, h], "gsd": gsd}, g


def footprints_px(run: str, g):
    from pyproj import Transformer

    from src.products.geo import LocalGeo

    gj_path = os.path.join(run, "08_completion", "buildings.geojson")
    if not os.path.exists(gj_path):
        return []
    geo = LocalGeo(read_json(os.path.join(run, "04_georef", "georef.json")))
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{geo.epsg}", always_xy=True) if geo.epsg else None
    xmin, ymax, gsd = float(g["xmin"]), float(g["ymax"]), float(g["gsd"])
    out = []
    for f in json.load(open(gj_path))["features"]:
        ring = np.array(f["geometry"]["coordinates"][0])
        x, y = tr.transform(ring[:, 0], ring[:, 1]) if tr else (ring[:, 0], ring[:, 1])
        px = np.column_stack([(np.asarray(x) - xmin) / gsd, (ymax - np.asarray(y)) / gsd])
        out.append({"id": f["properties"]["id"], "px": np.round(px, 1).tolist()})
    return out


def mesh_glb_b64(ply: str, georef, target_faces: int) -> str:
    import open3d as o3d
    import trimesh

    m = o3d.io.read_triangle_mesh(ply)
    if len(m.triangles) > target_faces:
        m = m.simplify_quadric_decimation(target_number_of_triangles=target_faces)
    m.remove_unreferenced_vertices()
    v = np.asarray(m.vertices)
    f = np.asarray(m.triangles)
    col = (np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).astype(np.uint8) if m.has_vertex_colors() else None
    v_gltf = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]).astype(np.float32)
    tm = trimesh.Trimesh(vertices=v_gltf, faces=f, vertex_colors=col, process=False)
    data = trimesh.Scene(tm).export(file_type="glb")
    return "data:model/gltf-binary;base64," + base64.b64encode(data).decode()


def stage_rows(report):
    rows = []
    for name, g in report["stages"].items():
        rows.append({"name": name, "passed": bool(g.get("passed")), "seconds": g.get("seconds"),
                     "cached": bool(g.get("cached", False)), "failures": g.get("failures", []),
                     "error": (g.get("error") or "")[:200]})
    return rows


def first_run_seconds(run_dirs, report):
    """Seconds per stage, taking cached stages from the run that actually executed them."""
    out = {}
    for name, g in report["stages"].items():
        out[name] = g.get("seconds")
    return out


def collect(run: str, with_mesh: bool, faces: int):
    rid = os.path.basename(run.rstrip("/"))
    rep = read_json(os.path.join(run, "report.json"))
    gates = {s: read_json(os.path.join(run, s, "gate.json")) for s in rep["stages"]}
    ev = read_json(os.path.join(run, "eval.json")) if os.path.exists(os.path.join(run, "eval.json")) else None
    imgs, g = run_images(run)
    comp = gates.get("08_completion", {}).get("metrics", {})
    geo = read_json(os.path.join(run, "04_georef", "georef.json"))
    d = {
        "id": rid, "label": LABELS.get(rid, (rid, ""))[0], "desc": LABELS.get(rid, (rid, ""))[1],
        "status": rep["status"], "stages": stage_rows(rep),
        "georef": {k: gates["04_georef"]["metrics"].get(k) for k in
                   ("rmse_fit_m", "rmse_holdout_m", "degenerate_trajectory", "orientation_prior",
                    "up_error_deg", "gimbal_axis_error_deg", "utm_epsg", "collinearity_ratio",
                    "georeferenced", "assumed_altitude_m", "ground_plane_inlier_frac")},
        "sfm": {k: gates["03_sfm"]["metrics"].get(k) for k in
                ("mapper", "num_registered", "num_images", "mean_reproj_px", "num_points")},
        "depth": {k: gates["05_depth"]["metrics"].get(k) for k in
                  ("lane", "predictor", "median_fit_rel_err", "median_consistency")},
        "fusion": {k: gates["06_fusion"]["metrics"].get(k) for k in
                   ("mesh_faces", "cloud_points", "coverage_area_frac_2views", "device")},
        "products": {k: gates["07_products"]["metrics"].get(k) for k in ("gsd_m", "epsg", "observed_cell_frac",
                                                                         "observed_in_hull_frac", "coverage_warning")},
        "buildings": comp.get("buildings", []),
        "prov": comp.get("face_provenance_counts"),
        "eval": ev, "images": imgs, "footprints": footprints_px(run, g),
        "origin": geo["enu_origin"],
        "ingest": gates["01_ingest"]["metrics"],
    }
    kf = read_json(os.path.join(run, "01_ingest", "keyframes.json"))
    d["video"] = os.path.basename(kf.get("video", ""))
    d["video_meta"] = {"resolution": "x".join(str(v) for v in kf.get("source_resolution", [])),
                       "duration_s": kf["video_frames"] / kf["video_fps"] if kf.get("video_fps") else None}
    if with_mesh:
        d["mesh"] = mesh_glb_b64(os.path.join(run, "06_fusion", "tsdf_mesh.ply"), geo, faces)
        lod = os.path.join(run, "08_completion", "buildings_lod2.ply")
        if os.path.exists(lod):
            d["lod2"] = mesh_glb_b64(lod, geo, 10 ** 9)
        cut = os.path.join(run, "08_completion", "surface_without_buildings.ply")
        if os.path.exists(cut):
            d["terrain"] = mesh_glb_b64(cut, geo, faces)
        reg = os.path.join(run, "08_completion", "buildings_regularized.glb")
        if os.path.exists(reg):
            d["reg"] = "data:model/gltf-binary;base64," + base64.b64encode(open(reg, "rb").read()).decode()
    meta_p = os.path.join(run, "08_completion", "buildings_meta.json")
    if os.path.exists(meta_p):
        meta = read_json(meta_p)
        d["meta"] = {k: {"thumb": (v.get("facade_thumb") or {}).get("jpg"),
                         "thumb_facing": (v.get("facade_thumb") or {}).get("facing"),
                         "doors": v.get("doors", [])} for k, v in meta.items()}
    d["roof_types"] = comp.get("roof_types")
    d["footprint_shapes"] = comp.get("footprint_shapes")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--mesh-runs", nargs="*", default=[])
    ap.add_argument("--faces", type=int, default=110000)
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--legacy-texture", default="webapp/models/textured_mesh.png")
    ap.add_argument("--out", default="webapp/dashboard.html", help="standalone page for the local web app")
    ap.add_argument("--fragment-out", default=None, help="body fragment for hosts that add their own skeleton")
    args = ap.parse_args()

    runs = []
    for rid in args.runs:
        print(f"[dashboard] collecting {rid}")
        runs.append(collect(os.path.join(args.run_root, rid), rid in args.mesh_runs, args.faces))
    gt_path = None
    for r in runs:
        if r["eval"]:
            gt_path = r["eval"].get("gt")
    gt = read_json(os.path.join(gt_path, "gt.json")) if gt_path and os.path.exists(os.path.join(gt_path, "gt.json")) else None
    legacy = None
    if os.path.exists(args.legacy_texture):
        legacy = b64_jpeg(cv2.imread(args.legacy_texture), 70, 520)
    synth = [r for r in runs if r["eval"]]
    ingest = synth[0]["ingest"] if synth else runs[0]["ingest"]
    data = {
        "runs": runs, "legacy_texture": legacy,
        "scene": {"frames": len(gt["frames"]) if gt else None, "fps": gt["fps"] if gt else None,
                  "gps_sigma_h": gt["gps_sigma_h"] if gt else None, "gps_sigma_v": gt["gps_sigma_v"] if gt else None,
                  "altitude_m": gt["altitude_m"] if gt else None, "pitch_deg": gt["pitch_deg"] if gt else None,
                  "buildings": len(gt["buildings"]) if gt else None},
        "ingest": ingest,
    }
    tpl = open(os.path.join(ROOT, "tools", "dashboard_template.html"), encoding="utf-8").read()
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    if args.fragment_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.fragment_out)), exist_ok=True)
        with open(args.fragment_out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"[dashboard] wrote {args.fragment_out} ({len(html) / 1e6:.1f} MB)")
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n')
    page = head + html + "\n</body>\n</html>\n"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"[dashboard] wrote {args.out} ({len(page) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
