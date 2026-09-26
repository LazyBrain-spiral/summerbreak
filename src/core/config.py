"""Run configuration: defaults, YAML overrides, dotted CLI overrides."""

from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import yaml

DEFAULTS: Dict[str, Any] = {
    "profile": "live",
    "run_root": "runs",
    "allow_cpu": False,
    "stop_on_failed_gate": True,
    "inputs": {"video": None, "telemetry": None, "gt": None},
    "ingest": {
        "candidate_hz": 5.0,
        "track_width": 640,
        "sharpness_reject_pct": 15.0,
        "min_disp_frac": 0.05,
        "max_gap_s": 3.0,
        "min_track_ratio": 0.6,
        "max_keyframes": 400,
        "min_keyframes": 20,
        "max_long_side": 1920,
        "jpeg_quality": 95,
        "clahe_clip": 2.0,
        "min_telemetry_coverage": 0.9,
    },
    "masks": {
        "enabled": True,
        "model": "yolov8n-seg.pt",
        "conf": 0.25,
        "dilate_frac": 0.01,
    },
    "sfm": {
        "backend": "auto",            # auto | cli | pycolmap
        "camera_model": "SIMPLE_RADIAL",
        "max_num_features": 8192,
        "overlap": 15,
        "mapper": "auto",             # auto | global | incremental
        "refine_focal": "auto",       # auto: fix focal when telemetry gave it | true | false
        "use_gpu": True,
        "min_registered_ratio": 0.8,
        "max_mean_reproj_px": 1.5,
    },
    "georef": {
        "inlier_thresh_m": 4.0,
        "ransac_iters": 400,
        "holdout_every": 5,
        "orientation_prior": "auto",  # auto | gimbal | ground | none
        "sigma_pos_m": 2.0,
        "sigma_dir_deg": 3.0,
        "collinear_ratio": 0.05,
        "max_holdout_rmse_m": 3.0,
        "max_up_error_deg": 3.0,
        "assumed_altitude_m": 60.0,   # GPS-denied runs only: camera height used to set scale
    },
    "dense": {
        "lane": "live",               # live | survey
        "max_image_size": 960,
        "predictor": "depth_anything",  # depth_anything | oracle
        "model": "depth-anything/Depth-Anything-V2-Small-hf",
        "min_anchors": 30,
        "local_correction": True,     # anchor-driven correction map on top of the per-frame affine fit
        "max_fit_rel_err": 0.08,
        "consistency_rel": 0.03,
        "neighbors": 2,
        "min_valid_frame_ratio": 0.9,
    },
    "fusion": {
        "voxel_m": 0.2,
        "trunc_voxels": 4.0,
        "depth_max_m": 250.0,
        "depth_cap_factor": 2.0,      # also cap fusion range at 2x the median scene depth
        "footprint_factor": 1.0,      # drop depth where one pixel spans more than one voxel
        "min_view_cos": 0.2,          # drop depth seen at > ~78 deg incidence (edge-on, fragments)
        "min_conf": 0.5,
        "min_weight": 1.5,            # views that must agree before a voxel is extracted
        "block_count": 60000,         # 8^3-voxel blocks; ~10 kB each on the device
        "min_component_frac": 0.01,
        "min_kept_face_frac": 0.6,    # faces left after dropping fragments < min_component_frac
    },
    "products": {
        "gsd_m": 0.25,
        "fill_holes_m": 1.5,
        "pmf_max_window_m": 40.0,     # must exceed the short side of the largest building
        "pmf_slope": 0.25,
        "pmf_dh0_m": 0.3,
        "pmf_dh_max_m": 2.5,
        "ortho": True,
    },
    "completion": {
        "enabled": True,
        "min_height_m": 2.5,
        "min_area_m2": 25.0,
        "max_roughness_m": 0.35,
        "max_curvature": 0.6,         # 1/m; roofs ~0.1, tree crowns > 1
        "vegetation_filter": True,    # reject green components (Excess Green index on the ortho)
        "max_exg": 0.1,
        "simplify_m": 0.5,
        "subdivide": 2,
        "regularize": True,           # replace buildings by rectangle/triangle/right-angled footprints + planar roofs
        "texture": True,              # texture regularised faces from the video frames (best view per texel)
        "texel_m": 0.06,
        "atlas_max_px": 4096,
        "metadata": True,             # dimensions, floors, colours, entrances per building
        "detect_facades": True,       # doors / windows with OWLv2 on rectified facades
        "facade_model": "google/owlv2-base-patch16-ensemble",
    },
}


def deep_merge(base: Dict[str, Any], over: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(value: str) -> Any:
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_dotted(cfg: Dict[str, Any], dotted: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    for key, value in dotted.items():
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(value) if isinstance(value, str) else value
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with open(path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    if overrides:
        cfg = apply_dotted(cfg, overrides)
    return cfg
