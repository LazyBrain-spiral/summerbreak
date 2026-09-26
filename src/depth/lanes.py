"""Dense depth lanes (v3). Both write the contract described in src/depth/common.py.

LIVE lane   relative depth from a learned monocular model, metric scale and shift
            fitted per frame to that frame's SfM track depths, then filtered by
            multi-view consistency.
SURVEY lane COLMAP PatchMatch stereo (CUDA) with geometric consistency.

Predictors for the LIVE lane:
  depth_anything  Depth Anything V2 via transformers (relative inverse depth)
  oracle          renders ground-truth depth from a synth3d scene and corrupts it
                  with an unknown affine map in disparity. For validating the
                  geometry downstream of the network, never for real footage.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.core.io import read_json, write_json
from src.depth.common import (anchors_for_image, apply_affine, build_cameras, consistency_confidence,
                              fit_affine_depth, local_correction, undistort)
from src.semantics.masks import load_dynamic_mask
from src.sfm.model_io import load_model


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------

class DepthAnythingPredictor:
    def __init__(self, model_id: str, allow_cpu: bool):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and not allow_cpu:
            raise RuntimeError("Depth Anything needs a CUDA device (set allow_cpu: true to accept CPU)")
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.name = f"depth_anything:{model_id}@{self.device}"

    def __call__(self, rgb: np.ndarray, name: str) -> np.ndarray:
        torch = self.torch
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: v.half() for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs).predicted_depth
        pred = torch.nn.functional.interpolate(out.unsqueeze(1).float(), size=rgb.shape[:2],
                                               mode="bicubic", align_corners=False)[0, 0]
        return pred.cpu().numpy().astype(np.float32)


class OraclePredictor:
    """Ground-truth disparity from a synth3d scene under an unknown affine map, plus noise."""

    def __init__(self, gt_dir: str, cameras: Dict[str, Dict[str, Any]], keyframes: Dict[str, Any], seed: int = 0,
                 noise: float = 0.01):
        import open3d as o3d

        gt = read_json(os.path.join(gt_dir, "gt.json"))
        mesh = o3d.io.read_triangle_mesh(os.path.join(gt_dir, "scene.ply"))
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.o3d = o3d
        self.frames = {f["index"]: f for f in gt["frames"]}
        self.cameras = cameras
        self.kf = {k["name"]: k for k in keyframes}
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.name = "oracle"

    def __call__(self, rgb: np.ndarray, name: str) -> np.ndarray:
        cam = self.cameras[name]
        f = self.frames[self.kf[name]["frame_index"]]
        R_wc = np.asarray(f["R_wc"])
        C = np.asarray(f["C_enu"])
        h, w = rgb.shape[:2]
        K = np.asarray(cam["K"])
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        d_world = d_cam @ R_wc.T
        norm = np.linalg.norm(d_world, axis=-1, keepdims=True)
        rays = np.concatenate([np.broadcast_to(C, d_world.shape), d_world / norm], -1).astype(np.float32)
        t = self.scene.cast_rays(self.o3d.core.Tensor(rays))["t_hit"].numpy()
        z = t / norm[..., 0]
        disp = np.where(np.isfinite(z), 1.0 / np.maximum(z, 1e-3), 0.0)
        a, b = self.rng.uniform(20, 80), self.rng.uniform(-0.2, 0.2)
        pred = a * disp + b
        pred *= 1.0 + self.noise * self.rng.standard_normal(pred.shape)
        return pred.astype(np.float32)


# ---------------------------------------------------------------------------
# lanes
# ---------------------------------------------------------------------------

def _prepare(ctx_paths: Dict[str, str], geo: Dict[str, Any], max_image_size: int):
    ws = undistort(ctx_paths["model_dir"], ctx_paths["raw_dir"], ctx_paths["undist_dir"], max_image_size)
    model = load_model(os.path.join(ws, "sparse"))
    cams = build_cameras(model, geo, os.path.join(ws, "images"))
    return ws, model, cams


def run_live_lane(paths: Dict[str, str], geo: Dict[str, Any], cfg: Dict[str, Any], allow_cpu: bool,
                  keyframes: List[Dict[str, Any]], gt_dir: Optional[str] = None) -> Dict[str, Any]:
    out = paths["out_dir"]
    for sub in ("depth", "conf"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    ws, model, cams = _prepare(paths, geo, cfg["max_image_size"])
    by_name = {c["name"]: c for c in cams}
    if cfg["predictor"] == "oracle":
        if not gt_dir:
            raise RuntimeError("predictor=oracle needs inputs.gt pointing at a synth3d folder")
        predictor = OraclePredictor(gt_dir, by_name, keyframes)
    else:
        predictor = DepthAnythingPredictor(cfg["model"], allow_cpu)

    fits, accepted = {}, []
    for cam in cams:
        name = cam["name"]
        bgr = cv2.imread(cam["image"])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = predictor(rgb, name)
        uv, z = anchors_for_image(model, name, geo)
        stem = os.path.splitext(name)[0]
        if len(z) < cfg["min_anchors"]:
            fits[name] = {"accepted": False, "reason": f"{len(z)} anchors"}
            np.save(os.path.join(out, "depth", stem + ".npy"), np.zeros(pred.shape, np.float32))
            continue
        fit = fit_affine_depth(pred, uv, z)
        ok = fit["mode"] is not None and fit["median_rel_err"] <= cfg["max_fit_rel_err"]
        depth = apply_affine(pred, fit) if ok else np.zeros(pred.shape, np.float32)
        if ok and cfg.get("local_correction", True):
            depth, fit["local_correction"] = local_correction(depth, uv, z)
        dyn = load_dynamic_mask(paths.get("masks_dir"), name, depth.shape)
        depth[dyn] = 0.0
        np.save(os.path.join(out, "depth", stem + ".npy"), depth)
        fit["accepted"] = bool(ok)
        fits[name] = fit
        if ok:
            accepted.append(name)
    names = [c["name"] for c in cams]
    agree = consistency_confidence(names, by_name, os.path.join(out, "depth"), cfg["neighbors"],
                                   cfg["consistency_rel"])
    write_json(os.path.join(out, "cameras.json"), {"lane": "live", "predictor": predictor.name,
                                                   "workspace": ws, "cameras": cams})
    write_json(os.path.join(out, "fits.json"), fits)
    errs = [f["median_rel_err"] for f in fits.values() if f.get("accepted")]
    return {
        "lane": "live", "predictor": predictor.name, "num_cameras": len(cams),
        "num_accepted": len(accepted), "accepted_ratio": len(accepted) / max(1, len(cams)),
        "median_fit_rel_err": float(np.median(errs)) if errs else None,
        "median_consistency": float(np.median([agree[n] for n in accepted])) if accepted else 0.0,
    }


def read_colmap_array(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = b""
        while header.count(b"&") < 3:
            header += f.read(1)
        w, h, c = (int(x) for x in header.split(b"&")[:3])
        data = np.fromfile(f, np.float32)
    return data.reshape((c, h, w)).transpose(1, 2, 0).squeeze()


def run_survey_lane(paths: Dict[str, str], geo: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    exe = shutil.which("colmap")
    if not exe:
        raise RuntimeError("SURVEY lane needs the colmap CLI")
    out = paths["out_dir"]
    for sub in ("depth", "conf"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    ws, model, cams = _prepare(paths, geo, cfg["max_image_size"])
    log = os.path.join(out, "patch_match.log")
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.run([exe, "patch_match_stereo", "--workspace_path", ws, "--workspace_format", "COLMAP",
                               "--PatchMatchStereo.geom_consistency", "true",
                               "--PatchMatchStereo.max_image_size", str(cfg["max_image_size"])],
                              stdout=lf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f"patch_match_stereo failed; see {log} (a CUDA build of COLMAP is required)")
    by_name = {c["name"]: c for c in cams}
    n_ok = 0
    for cam in cams:
        stem = os.path.splitext(cam["name"])[0]
        p = os.path.join(ws, "stereo", "depth_maps", cam["name"] + ".geometric.bin")
        if not os.path.exists(p):
            np.save(os.path.join(out, "depth", stem + ".npy"), np.zeros((cam["height"], cam["width"]), np.float32))
            continue
        d = read_colmap_array(p).astype(np.float32) * geo["scale"]
        d[~np.isfinite(d) | (d < 0)] = 0
        dyn = load_dynamic_mask(paths.get("masks_dir"), cam["name"], d.shape)
        d[dyn] = 0
        np.save(os.path.join(out, "depth", stem + ".npy"), d)
        n_ok += 1
    names = [c["name"] for c in cams]
    agree = consistency_confidence(names, by_name, os.path.join(out, "depth"), cfg["neighbors"],
                                   cfg["consistency_rel"])
    write_json(os.path.join(out, "cameras.json"), {"lane": "survey", "predictor": "colmap_patch_match",
                                                   "workspace": ws, "cameras": cams})
    return {"lane": "survey", "predictor": "colmap_patch_match", "num_cameras": len(cams),
            "num_accepted": n_ok, "accepted_ratio": n_ok / max(1, len(cams)),
            "median_consistency": float(np.median(list(agree.values()))) if agree else 0.0}
