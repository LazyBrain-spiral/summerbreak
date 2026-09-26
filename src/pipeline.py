"""v3 pipeline entry point.

    python -m src.pipeline --config configs/live.yaml --video flight.mp4 --telemetry flight.srt \
        --run-id demo [--until fusion] [--from georef] [--force] [--set dense.predictor=oracle]

Stages and their gates are described in ARCHITECTURE_V3.md section 3. Each
stage writes <run>/<stage>/gate.json; the run writes status.json while it
works and report.json at the end.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.core.config import load_config  # noqa: E402
from src.core.io import read_csv, read_json, write_csv, write_json  # noqa: E402
from src.core.stage import GateFailure, Pipeline, RunContext, Stage, gate_result  # noqa: E402

S_INGEST, S_MASKS, S_SFM, S_GEOREF = "01_ingest", "02_masks", "03_sfm", "04_georef"
S_DEPTH, S_FUSION, S_PRODUCTS, S_COMPLETION = "05_depth", "06_fusion", "07_products", "08_completion"


# ---------------------------------------------------------------------------
# stage functions
# ---------------------------------------------------------------------------

def stage_ingest(ctx: RunContext) -> Dict[str, Any]:
    from src.ingest.intrinsics import intrinsics_prior
    from src.ingest.keyframes import extract_keyframes
    from src.ingest.telemetry import parse_telemetry, synchronize

    cfg = ctx.cfg["ingest"]
    video = ctx.cfg["inputs"]["video"]
    if not video or not os.path.exists(video):
        raise GateFailure(f"input video not found: {video}")
    out = ctx.stage_dir(S_INGEST)
    info = extract_keyframes(video, out, cfg)
    write_json(os.path.join(out, "keyframes.json"), info)
    w, h = info["keyframe_resolution"]

    tel_path = ctx.cfg["inputs"].get("telemetry")
    coverage, focal, alt_source = 0.0, None, None
    if tel_path:
        records = parse_telemetry(tel_path)
        rows = synchronize(records, info["keyframes"])
        write_csv(os.path.join(out, "frame_telemetry.csv"), rows)
        coverage = float(np.mean([r["in_span"] and r["lat"] is not None for r in rows]))
        focals = [r["focal_len"] for r in rows if r["focal_len"] is not None]
        focal = float(np.median(focals)) if focals else None
        alt_source = rows[0]["alt_source"]
        n_records = len(records)
    else:
        ctx.decide("no telemetry supplied: georeferencing will be unavailable")
        n_records = 0
    intr = intrinsics_prior(w, h, focal)
    write_json(os.path.join(out, "intrinsics_prior.json"), intr)
    checks = {
        "enough_keyframes": info["num_keyframes"] >= cfg["min_keyframes"],
        "telemetry_coverage": (coverage >= cfg["min_telemetry_coverage"]) if tel_path else True,
    }
    return gate_result(checks, {
        "num_keyframes": info["num_keyframes"], "num_candidates": info["num_candidates"],
        "median_disp_frac": info["median_disp_frac"], "keyframe_resolution": [w, h],
        "telemetry_records": n_records, "telemetry_coverage": coverage,
        "alt_source": alt_source, "intrinsics_source": intr["source"], "focal_prior_px": intr["f"],
    })


def stage_masks(ctx: RunContext) -> Dict[str, Any]:
    cfg = ctx.cfg["masks"]
    out = ctx.stage_dir(S_MASKS)
    if not cfg["enabled"]:
        ctx.decide("dynamic masking disabled by config")
        return {"passed": True, "skipped": True}
    from src.semantics.masks import generate_masks

    stats = generate_masks(ctx.path(S_INGEST, "raw"), out, cfg, ctx.cfg["allow_cpu"])
    return gate_result({"all_frames_masked": stats["num_masks"] > 0}, stats)


def _masks_dir(ctx: RunContext):
    d = ctx.path(S_MASKS)
    return d if os.path.isdir(os.path.join(d, "dynamic")) else None


def stage_sfm(ctx: RunContext) -> Dict[str, Any]:
    from src.sfm.colmap_runner import run_sfm

    cfg = ctx.cfg["sfm"]
    ctx.gate(S_INGEST)
    intr = read_json(ctx.path(S_INGEST, "intrinsics_prior.json"))
    md = _masks_dir(ctx)
    colmap_masks = os.path.join(md, "colmap") if md else None
    stats = run_sfm(ctx.path(S_INGEST, "enh"), colmap_masks, ctx.stage_dir(S_SFM), intr, cfg)
    ctx.decide(f"SfM backend {stats['backend']}, mapper {stats['mapper']}")
    checks = {
        "registered_ratio": stats["registered_ratio"] >= cfg["min_registered_ratio"],
        "reprojection_error": stats["mean_reproj_px"] <= cfg["max_mean_reproj_px"],
    }
    return gate_result(checks, stats)


def stage_georef(ctx: RunContext) -> Dict[str, Any]:
    from src.georef.solver import solve_georef

    cfg = ctx.cfg["georef"]
    ctx.gate(S_SFM)
    tel_csv = ctx.path(S_INGEST, "frame_telemetry.csv")
    if not os.path.exists(tel_csv):
        from src.georef.solver import solve_relative

        poses = read_json(ctx.path(S_SFM, "poses.json"))
        names = [im["name"] for im in poses["images"]]
        alt = float(cfg.get("assumed_altitude_m") or 60.0)
        geo = solve_relative(names, np.array([im["center"] for im in poses["images"]]),
                             np.array([im["R_cw"] for im in poses["images"]]),
                             np.load(ctx.path(S_SFM, "tracks.npz"))["xyz"], alt)
        write_json(os.path.join(ctx.stage_dir(S_GEOREF), "georef.json"), geo)
        ctx.decide(f"no telemetry: NOT georeferenced; local frame with metric scale ASSUMED from a "
                   f"{alt:.0f} m camera height (set georef.assumed_altitude_m to correct)")
        keys = ("scale", "georeferenced", "scale_source", "assumed_altitude_m", "ground_plane_inlier_frac",
                "camera_height_spread_m", "orientation_prior", "num_frames", "utm_epsg", "up_error_deg",
                "rmse_holdout_m", "degenerate_trajectory", "collinearity_ratio", "gimbal_axis_error_deg", "rmse_fit_m")
        return gate_result({"ground_plane_found": geo["ground_plane_inlier_frac"] > 0.15},
                           {k: geo[k] for k in keys})
    rows = read_csv(tel_csv)

    def num(v):
        return None if v in (None, "", "None") else float(v)

    tel = {}
    for r in rows:
        tel[r["name"]] = {k: (num(v) if k not in ("name", "alt_source", "in_span") else v) for k, v in r.items()}
        tel[r["name"]]["in_span"] = r["in_span"] in ("True", "true", "1")
    poses = read_json(ctx.path(S_SFM, "poses.json"))
    names = [im["name"] for im in poses["images"]]
    C = np.array([im["center"] for im in poses["images"]])
    Rcw = np.array([im["R_cw"] for im in poses["images"]])
    tracks = np.load(ctx.path(S_SFM, "tracks.npz"))
    geo = solve_georef(names, C, Rcw, tel, tracks["xyz"], cfg)
    out = ctx.stage_dir(S_GEOREF)
    write_json(os.path.join(out, "georef.json"), geo)
    if geo["degenerate_trajectory"]:
        ctx.decide(f"straight-line trajectory (ratio {geo['collinearity_ratio']:.3f}); "
                   f"roll fixed by {geo['orientation_prior']} prior")
    checks = {
        "holdout_rmse": geo["rmse_holdout_m"] is not None and geo["rmse_holdout_m"] <= cfg["max_holdout_rmse_m"],
        "up_vector": geo["up_error_deg"] is None or geo["up_error_deg"] <= cfg["max_up_error_deg"],
    }
    metrics = {k: geo[k] for k in ("scale", "rmse_fit_m", "rmse_holdout_m", "rmse_axis_m", "num_frames",
                                   "num_inliers", "collinearity_ratio", "degenerate_trajectory",
                                   "orientation_prior", "up_error_deg", "gimbal_axis_error_deg", "utm_epsg")}
    return gate_result(checks, metrics)


def stage_depth(ctx: RunContext) -> Dict[str, Any]:
    from src.depth.lanes import run_live_lane, run_survey_lane

    cfg = ctx.cfg["dense"]
    ctx.gate(S_GEOREF)
    geo = read_json(ctx.path(S_GEOREF, "georef.json"))
    sfm_gate = read_json(ctx.path(S_SFM, "gate.json"))
    out = ctx.stage_dir(S_DEPTH)
    paths = {"model_dir": sfm_gate["metrics"]["model_dir"], "raw_dir": ctx.path(S_INGEST, "raw"),
             "undist_dir": os.path.join(out, "undistorted"), "out_dir": out, "masks_dir": _masks_dir(ctx)}
    keyframes = read_json(ctx.path(S_INGEST, "keyframes.json"))["keyframes"]
    if cfg["lane"] == "survey":
        stats = run_survey_lane(paths, geo, cfg)
    else:
        stats = run_live_lane(paths, geo, cfg, ctx.cfg["allow_cpu"], keyframes, ctx.cfg["inputs"].get("gt"))
    ctx.decide(f"dense lane {stats['lane']} with {stats['predictor']}")
    checks = {"valid_frames": stats["accepted_ratio"] >= cfg["min_valid_frame_ratio"],
              "consistency": stats["median_consistency"] >= 0.5}
    return gate_result(checks, stats)


def stage_fusion(ctx: RunContext) -> Dict[str, Any]:
    from src.fusion.tsdf import fuse

    cfg = ctx.cfg["fusion"]
    ctx.gate(S_DEPTH)
    stats = fuse(ctx.path(S_DEPTH), ctx.stage_dir(S_FUSION), cfg)
    try:
        from src.products.glb import ply_to_glb

        geo = read_json(ctx.path(S_GEOREF, "georef.json"))
        stats["glb"] = ply_to_glb(stats["mesh_path"], ctx.path(S_FUSION, "model.glb"), geo,
                                  {"provenance": "observed"})
    except ImportError as exc:
        ctx.decide(f"GLB export skipped: {exc}")
    checks = {"kept_after_fragment_filter": stats["kept_face_frac"] >= cfg["min_kept_face_frac"],
              "has_surface": stats["mesh_faces"] > 1000}
    return gate_result(checks, stats)


def stage_products(ctx: RunContext) -> Dict[str, Any]:
    from src.products.rasters import build_products

    ctx.gate(S_FUSION)
    geo = read_json(ctx.path(S_GEOREF, "georef.json"))
    stats = build_products(ctx.path(S_FUSION, "dense_cloud.ply"), ctx.path(S_DEPTH), geo,
                           ctx.stage_dir(S_PRODUCTS), ctx.cfg["products"], _masks_dir(ctx))
    # hard floor; between 8% and 30% the run continues with a sparse-coverage warning
    stats["coverage_warning"] = ("sparse coverage: mostly oblique or low-resolution footage, maps and "
                                 "buildings are incomplete") if stats["observed_in_hull_frac"] < 0.3 else None
    if stats["coverage_warning"]:
        ctx.decide(stats["coverage_warning"])
    checks = {"observed_cells": stats["observed_cells"] >= 1000 and stats["observed_in_hull_frac"] >= 0.08,
              "ortho_filled": stats.get("ortho_filled_frac", 1.0) > 0.5}
    return gate_result(checks, stats)


def stage_completion(ctx: RunContext) -> Dict[str, Any]:
    cfg = ctx.cfg["completion"]
    if not cfg["enabled"]:
        return {"passed": True, "skipped": True}
    from src.completion.extrusion import extrude_buildings

    ctx.gate(S_PRODUCTS)
    geo = read_json(ctx.path(S_GEOREF, "georef.json"))
    stats = extrude_buildings(ctx.path(S_PRODUCTS, "grids.npz"), geo, ctx.stage_dir(S_COMPLETION), cfg,
                              ortho_path=ctx.path(S_PRODUCTS, "ortho.tif"),
                              surface_path=ctx.path(S_FUSION, "tsdf_mesh.ply"))
    if stats.get("mesh"):
        try:
            from src.products.glb import ply_to_glb

            stats["glb"] = ply_to_glb(stats["mesh"], ctx.path(S_COMPLETION, "buildings_lod2.glb"), geo,
                                      {"face_colors": "orange=observed roof, blue=inferred wall"})
        except ImportError as exc:
            ctx.decide(f"GLB export skipped: {exc}")
    return gate_result({"ran": True}, stats)


STAGES = [
    Stage(S_INGEST, stage_ingest, ["ingest"]),
    Stage(S_MASKS, stage_masks, ["masks"], [S_INGEST], optional=True),
    Stage(S_SFM, stage_sfm, ["sfm"], [S_INGEST, S_MASKS]),
    Stage(S_GEOREF, stage_georef, ["georef"], [S_SFM]),
    Stage(S_DEPTH, stage_depth, ["dense"], [S_GEOREF]),
    Stage(S_FUSION, stage_fusion, ["fusion"], [S_DEPTH]),
    Stage(S_PRODUCTS, stage_products, ["products"], [S_FUSION]),
    Stage(S_COMPLETION, stage_completion, ["completion"], [S_PRODUCTS]),
]
ALIASES = {"ingest": S_INGEST, "masks": S_MASKS, "sfm": S_SFM, "georef": S_GEOREF, "depth": S_DEPTH,
           "fusion": S_FUSION, "products": S_PRODUCTS, "completion": S_COMPLETION}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--video")
    ap.add_argument("--telemetry")
    ap.add_argument("--gt", help="synth3d folder (only for predictor=oracle and evaluation)")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--until", choices=sorted(ALIASES))
    ap.add_argument("--from", dest="start", choices=sorted(ALIASES))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--skip-env-check", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    overrides = dict(kv.split("=", 1) for kv in args.set)
    for key, val in (("inputs.video", args.video), ("inputs.telemetry", args.telemetry), ("inputs.gt", args.gt)):
        if val:
            overrides[key] = os.path.abspath(val)
    if args.allow_cpu:
        overrides["allow_cpu"] = True
    cfg = load_config(args.config, overrides)

    if not args.skip_env_check:
        sys.path.insert(0, str(ROOT / "scripts"))
        from check_env import check_environment

        profile = "survey" if cfg["dense"]["lane"] == "survey" else "live"
        if cfg["dense"]["predictor"] == "oracle":
            profile = "core"
        env = check_environment(profile, cfg["allow_cpu"])
        if not env["ok"]:
            print("[pipeline] environment not ready:")
            for p in env["problems"]:
                print(f"  - {p}")
            print("[pipeline] fix it (scripts/setup_env.sh) or pass --skip-env-check")
            return 2

    run_dir = os.path.join(cfg["run_root"], args.run_id)
    ctx = RunContext(run_dir=os.path.abspath(run_dir), cfg=cfg)
    report = Pipeline(STAGES).run(ctx, until=ALIASES.get(args.until), start=ALIASES.get(args.start),
                                  force=args.force)
    return 0 if report["status"] in ("SUCCESS", "PARTIAL_SUCCESS") else 1


if __name__ == "__main__":
    sys.exit(main())
