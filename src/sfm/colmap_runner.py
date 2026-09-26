"""Pose backbone (v3): COLMAP features + sequential matching + global or incremental mapping.

There is no synthetic fallback. If COLMAP is missing or reconstruction fails,
the stage fails its gate with the reason.

Backends:
  cli       the ``colmap`` executable (conda-forge CUDA build); option names are
            discovered from ``colmap <cmd> -h`` so COLMAP 3.x and 4.x both work
  pycolmap  Python bindings (CPU wheel is fine for small clips)
Mapper preference (``mapper: auto``): colmap global_mapper, then glomap, then
colmap incremental mapper. The choice is recorded in the gate.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.core.io import list_images, write_json
from src.sfm.model_io import SparseModel, load_model, model_size


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

_HELP_CACHE: Dict[str, str] = {}


def _help(exe: str, cmd: Optional[str] = None) -> str:
    key = f"{exe}|{cmd}"
    if key not in _HELP_CACHE:
        args = [exe, cmd, "-h"] if cmd else [exe, "help"]
        try:
            out = subprocess.run(args, capture_output=True, text=True, timeout=60)
            _HELP_CACHE[key] = (out.stdout or "") + (out.stderr or "")
        except Exception as exc:
            _HELP_CACHE[key] = f"ERROR {exc}"
    return _HELP_CACHE[key]


def _options(exe: str, cmd: str) -> Set[str]:
    return set(re.findall(r"--([A-Za-z_]+\.[A-Za-z_]+|[a-z_]+)", _help(exe, cmd)))


def _pick(opts: Set[str], *candidates: str) -> Optional[str]:
    for c in candidates:
        if c in opts:
            return c
    return None


def _run(args: List[str], log_path: str) -> None:
    print("[sfm] $ " + " ".join(args))
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(args) + "\n")
        log.flush()
        proc = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        tail = open(log_path, encoding="utf-8", errors="ignore").read()[-2000:]
        raise RuntimeError(f"{args[1] if len(args) > 1 else args[0]} failed (exit {proc.returncode}):\n{tail}")


def resolve_backend(requested: str) -> str:
    if requested in ("cli", "pycolmap"):
        return requested
    exe = shutil.which("colmap")
    if exe:
        text = _help(exe)
        if ("COLMAP" in text or "Usage" in text) and "error" not in text.lower()[:200]:
            return "cli"
        print(f"[sfm] colmap CLI present but not runnable: {text[:200]!r}")
    try:
        import pycolmap  # noqa: F401

        return "pycolmap"
    except Exception:
        pass
    raise RuntimeError("no COLMAP backend: install conda-forge colmap (see scripts/setup_env.sh)")


# ---------------------------------------------------------------------------
# CLI pipeline
# ---------------------------------------------------------------------------

def _refine_focal(cfg: Dict[str, Any]) -> bool:
    v = cfg.get("refine_focal", "auto")
    if v == "auto":
        return cfg.get("_intrinsics_source") != "telemetry_focal_len"
    return bool(v)


def _cli_pipeline(image_dir: str, mask_dir: Optional[str], out: str, cam_params: str, cfg: Dict[str, Any]) -> str:
    exe = shutil.which("colmap")
    log = os.path.join(out, "colmap.log")
    db = os.path.join(out, "database.db")
    if os.path.exists(db):
        os.remove(db)
    gpu = "1" if cfg["use_gpu"] else "0"

    fe = _options(exe, "feature_extractor")
    args = [exe, "feature_extractor", "--database_path", db, "--image_path", image_dir,
            "--ImageReader.camera_model", cfg["camera_model"], "--ImageReader.single_camera", "1",
            "--ImageReader.camera_params", cam_params]
    if mask_dir:
        args += ["--ImageReader.mask_path", mask_dir]
    o = _pick(fe, "FeatureExtraction.use_gpu", "SiftExtraction.use_gpu")
    if o:
        args += [f"--{o}", gpu]
    o = _pick(fe, "SiftExtraction.max_num_features")
    if o:
        args += [f"--{o}", str(cfg["max_num_features"])]
    _run(args, log)

    sm = _options(exe, "sequential_matcher")
    args = [exe, "sequential_matcher", "--database_path", db,
            "--SequentialMatching.overlap", str(cfg["overlap"]),
            "--SequentialMatching.quadratic_overlap", "1"]
    o = _pick(sm, "FeatureMatching.use_gpu", "SiftMatching.use_gpu")
    if o:
        args += [f"--{o}", gpu]
    _run(args, log)

    sparse = os.path.join(out, "sparse")
    if os.path.isdir(sparse):
        shutil.rmtree(sparse)
    os.makedirs(sparse)
    mapper = cfg["mapper"]
    has_global = "global_mapper" in _help(exe)
    glomap = shutil.which("glomap")
    focal = "1" if _refine_focal(cfg) else "0"
    if mapper in ("auto", "global") and has_global:
        args = [exe, "global_mapper", "--database_path", db, "--image_path", image_dir, "--output_path", sparse]
        if "GlobalMapper.ba_refine_focal_length" in _options(exe, "global_mapper"):
            args += ["--GlobalMapper.ba_refine_focal_length", focal]
        _run(args, log)
        used = "colmap_global_mapper"
    elif mapper in ("auto", "global") and glomap:
        _run([glomap, "mapper", "--database_path", db, "--image_path", image_dir, "--output_path", sparse], log)
        used = "glomap"
    elif mapper == "global":
        raise RuntimeError("mapper=global requested but neither colmap global_mapper nor glomap is available")
    else:
        _run([exe, "mapper", "--database_path", db, "--image_path", image_dir, "--output_path", sparse,
              "--Mapper.ba_refine_focal_length", focal], log)
        used = "colmap_incremental"
    # TXT export so readers without pycolmap still work
    best = _best_submodel(sparse)
    if best:
        _run([exe, "model_converter", "--input_path", best, "--output_path", best, "--output_type", "TXT"], log)
    return used


# ---------------------------------------------------------------------------
# pycolmap pipeline
# ---------------------------------------------------------------------------

def _pycolmap_pipeline(image_dir: str, mask_dir: Optional[str], out: str, cam_params: str, cfg: Dict[str, Any]) -> str:
    import pycolmap

    db = os.path.join(out, "database.db")
    if os.path.exists(db):
        os.remove(db)
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = cfg["camera_model"]
    reader.camera_params = cam_params
    if mask_dir:
        reader.mask_path = mask_dir
    kwargs = dict(camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    try:
        sift = pycolmap.SiftExtractionOptions()
        sift.max_num_features = int(cfg["max_num_features"])
        pycolmap.extract_features(db, image_dir, sift_options=sift, **kwargs)
    except TypeError:
        ext = pycolmap.FeatureExtractionOptions()
        ext.sift.max_num_features = int(cfg["max_num_features"])
        pycolmap.extract_features(db, image_dir, extraction_options=ext, **kwargs)
    try:
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = int(cfg["overlap"])
        pairing.quadratic_overlap = True
        pycolmap.match_sequential(db, pairing_options=pairing)
    except (AttributeError, TypeError):
        seq = pycolmap.SequentialMatchingOptions()
        seq.overlap = int(cfg["overlap"])
        pycolmap.match_sequential(db, matching_options=seq)
    sparse = os.path.join(out, "sparse")
    if os.path.isdir(sparse):
        shutil.rmtree(sparse)
    os.makedirs(sparse)
    used = None
    if cfg["mapper"] in ("auto", "global") and hasattr(pycolmap, "global_mapping"):
        try:
            recs = pycolmap.global_mapping(db, image_dir, sparse)
            used = "pycolmap_global"
        except Exception as exc:
            if cfg["mapper"] == "global":
                raise
            print(f"[sfm] pycolmap global_mapping failed ({exc}); using incremental mapping")
    if used is None:
        recs = pycolmap.incremental_mapping(db, image_dir, sparse)
        used = "pycolmap_incremental"
    for idx, rec in (recs.items() if isinstance(recs, dict) else enumerate(recs)):
        d = os.path.join(sparse, str(idx))
        os.makedirs(d, exist_ok=True)
        rec.write(d)
        rec.write_text(d)
    return used


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------

def _best_submodel(sparse: str) -> Optional[str]:
    subs = [os.path.join(sparse, d) for d in sorted(os.listdir(sparse)) if os.path.isdir(os.path.join(sparse, d))]
    if not subs and os.path.exists(os.path.join(sparse, "cameras.bin")):
        return sparse
    if not subs:
        return None
    return max(subs, key=model_size)


def export_tracks(model: SparseModel, path: str) -> Dict[str, int]:
    """Flat arrays of every observation: which 3D point is seen where by which image."""
    pidx = model.point_index()
    obs_img, obs_pt, obs_uv = [], [], []
    names = []
    for k, im in enumerate(sorted(model.images.values(), key=lambda i: i.name)):
        names.append(im.name)
        sel = im.point3D_ids >= 0
        ids = im.point3D_ids[sel]
        keep = np.array([pid in pidx for pid in ids], bool)
        obs_img.append(np.full(int(keep.sum()), k, np.int32))
        obs_pt.append(np.array([pidx[pid] for pid in ids[keep]], np.int64))
        obs_uv.append(im.xys[sel][keep].astype(np.float32))
    np.savez_compressed(path, names=np.array(names), xyz=model.xyz, rgb=model.rgb, error=model.error,
                        obs_img=np.concatenate(obs_img) if obs_img else np.empty(0, np.int32),
                        obs_pt=np.concatenate(obs_pt) if obs_pt else np.empty(0, np.int64),
                        obs_uv=np.concatenate(obs_uv) if obs_uv else np.empty((0, 2), np.float32))
    return {"num_points": int(len(model.xyz)), "num_observations": int(sum(len(o) for o in obs_img))}


def reprojection_error(model: SparseModel) -> float:
    """Mean reprojection error recomputed from the model (mappers do not all store it)."""
    import cv2

    pidx = model.point_index()
    errs = []
    for im in model.images.values():
        cam = model.cameras[im.camera_id]
        sel = np.nonzero(im.point3D_ids >= 0)[0]
        ids = [pidx.get(int(p)) for p in im.point3D_ids[sel]]
        keep = [k for k, i in enumerate(ids) if i is not None]
        if not keep:
            continue
        X = model.xyz[[ids[k] for k in keep]]
        obs = im.xys[sel[keep]]
        rvec, _ = cv2.Rodrigues(im.R_cw)
        # COLMAP pixel centres are at +0.5; OpenCV at 0
        proj, _ = cv2.projectPoints(X.reshape(-1, 1, 3), rvec, im.t_cw, cam.K - np.array([[0, 0, 0.5], [0, 0, 0.5], [0, 0, 0]]),
                                    cam.distortion)
        errs.append(np.linalg.norm(proj.reshape(-1, 2) - (obs - 0.5), axis=1))
    return float(np.mean(np.concatenate(errs))) if errs else float("nan")


def poses_dict(model: SparseModel) -> Dict[str, Any]:
    cams = {str(c.camera_id): {"model": c.model, "width": c.width, "height": c.height,
                                "params": c.params.tolist()} for c in model.cameras.values()}
    ims = []
    for im in sorted(model.images.values(), key=lambda i: i.name):
        ims.append({"name": im.name, "camera_id": im.camera_id, "R_cw": im.R_cw.tolist(),
                    "t_cw": im.t_cw.tolist(), "center": im.center.tolist(),
                    "num_points3D": int((im.point3D_ids >= 0).sum())})
    return {"cameras": cams, "images": ims}


def run_sfm(image_dir: str, colmap_mask_dir: Optional[str], out_dir: str, intr: Dict[str, Any],
            cfg: Dict[str, Any]) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    names = list_images(image_dir)
    backend = resolve_backend(cfg["backend"])
    if cfg["camera_model"] in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
        cam_params = f"{intr['f']:.3f},{intr['cx']:.3f},{intr['cy']:.3f}"
        if cfg["camera_model"] == "SIMPLE_RADIAL":
            cam_params += ",0"
    elif cfg["camera_model"] == "OPENCV":
        cam_params = f"{intr['f']:.3f},{intr['f']:.3f},{intr['cx']:.3f},{intr['cy']:.3f},0,0,0,0"
    else:
        cam_params = f"{intr['f']:.3f},{intr['f']:.3f},{intr['cx']:.3f},{intr['cy']:.3f}"
    cfg = dict(cfg, _intrinsics_source=intr.get("source"))
    # COLMAP's SQLite database is very slow on network-style mounts (WSL /mnt/c, 9P).
    # Work on the local temp filesystem and copy the results back.
    import tempfile

    work = tempfile.mkdtemp(prefix="sfm_")
    try:
        if backend == "cli":
            mapper = _cli_pipeline(image_dir, colmap_mask_dir, work, cam_params, cfg)
        else:
            mapper = _pycolmap_pipeline(image_dir, colmap_mask_dir, work, cam_params, cfg)
        for name in ("database.db", "colmap.log"):
            if os.path.exists(os.path.join(work, name)):
                shutil.copy2(os.path.join(work, name), os.path.join(out_dir, name))
        if os.path.isdir(os.path.join(out_dir, "sparse")):
            shutil.rmtree(os.path.join(out_dir, "sparse"))
        shutil.copytree(os.path.join(work, "sparse"), os.path.join(out_dir, "sparse"))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    sparse = os.path.join(out_dir, "sparse")
    subs = [d for d in os.listdir(sparse) if os.path.isdir(os.path.join(sparse, d))]
    best = _best_submodel(sparse)
    if best is None:
        raise RuntimeError("mapper produced no model")
    model = load_model(best)
    write_json(os.path.join(out_dir, "poses.json"), poses_dict(model))
    track_stats = export_tracks(model, os.path.join(out_dir, "tracks.npz"))
    cam = next(iter(model.cameras.values()))
    return {
        "backend": backend, "mapper": mapper, "model_dir": best,
        "num_submodels": len(subs), "num_images": len(names),
        "num_registered": len(model.images),
        "registered_ratio": len(model.images) / max(1, len(names)),
        "mean_reproj_px": reprojection_error(model),
        "stored_mean_error_px": model.mean_reprojection_error(),
        "refine_focal": _refine_focal(cfg),
        "camera": {"model": cam.model, "params": cam.params.tolist()},
        "focal_prior_px": intr["f"],
        "focal_ratio_to_prior": float(cam.params[0] / intr["f"]),
        **track_stats,
    }
