"""Parallax-driven keyframe selection (v3).

Pass 1 decodes candidate frames at ``candidate_hz``, scores sharpness on a
downscaled grey image, and tracks sparse corners from the last accepted
keyframe with pyramidal Lucas-Kanade. A sharp candidate becomes a keyframe
when the median corner displacement since the last keyframe exceeds
``min_disp_frac`` of the image width, when too few tracks survive, or when
``max_gap_s`` has elapsed. Hovering produces few keyframes; fast flight
produces many, which is what matching needs.

Pass 2 re-decodes and writes the selected frames twice with identical names:
``raw/`` (original colour, for depth, texture and orthomosaic) and ``enh/``
(CLAHE on L, for feature extraction only).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _resize_long(img: np.ndarray, max_long: Optional[int]) -> np.ndarray:
    if not max_long:
        return img
    h, w = img.shape[:2]
    s = max_long / float(max(h, w))
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


class FrameSource:
    """Iterates (index, time_s, frame) over a video file or an image folder."""

    def __init__(self, path: str, candidate_hz: float):
        self.path = path
        self.is_dir = os.path.isdir(path)
        if self.is_dir:
            exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
            self.files = sorted(f for f in os.listdir(path) if f.lower().endswith(exts))
            if not self.files:
                raise RuntimeError(f"no images in {path}")
            self.fps = candidate_hz
            self.step = 1
            self.total = len(self.files)
        else:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"cannot open video {path}")
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.step = max(1, int(round(self.fps / candidate_hz)))

    def iterate(self, wanted: Optional[set] = None):
        if self.is_dir:
            for i, f in enumerate(self.files):
                if wanted is None or i in wanted:
                    img = cv2.imread(os.path.join(self.path, f), cv2.IMREAD_COLOR)
                    if img is not None:
                        yield i, i / self.fps, img
            return
        cap = cv2.VideoCapture(self.path)
        idx = 0
        last_wanted = max(wanted) if wanted else None
        while True:
            take = (idx % self.step == 0) if wanted is None else (idx in wanted)
            if take:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, idx / self.fps, frame
            else:
                if not cap.grab():
                    break
            idx += 1
            if last_wanted is not None and idx > last_wanted:
                break
        cap.release()


def select_keyframes(candidates: List[Dict[str, Any]], grays: List[np.ndarray], width: int,
                     min_disp_frac: float, min_track_ratio: float, max_gap_s: float,
                     sharp_thresh: float) -> List[int]:
    """Return indices into ``candidates`` chosen as keyframes."""
    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    min_disp = min_disp_frac * width
    chosen: List[int] = []
    p0 = pc = None
    prev_gray = None
    n0 = 0
    pending = False  # threshold crossed but candidate was blurry: take the next sharp one

    def detect(g):
        pts = cv2.goodFeaturesToTrack(g, maxCorners=800, qualityLevel=0.01, minDistance=8)
        return pts.reshape(-1, 2) if pts is not None else np.empty((0, 2), np.float32)

    for ci, (cand, gray) in enumerate(zip(candidates, grays)):
        sharp = cand["sharpness"] >= sharp_thresh
        if not chosen:
            if sharp:
                chosen.append(ci)
                p0 = detect(gray)
                pc, n0, prev_gray = p0.copy(), len(p0), gray
            continue
        # track from previous candidate to this one, forward-backward checked
        if len(pc) > 0:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pc.astype(np.float32), None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, prev_gray, nxt, None, **lk)
            fb = np.linalg.norm(back.reshape(-1, 2) - pc, axis=1)
            keep = (st.ravel() == 1) & (st2.ravel() == 1) & (fb < 1.0)
            p0, pc = p0[keep], nxt.reshape(-1, 2)[keep]
        prev_gray = gray
        disp = float(np.median(np.linalg.norm(pc - p0, axis=1))) if len(pc) else np.inf
        ratio = len(pc) / max(1, n0)
        gap = cand["t"] - candidates[chosen[-1]]["t"]
        cand["disp_px"] = disp
        if disp >= min_disp or ratio < min_track_ratio or gap >= max_gap_s:
            pending = True
        if pending and sharp:
            chosen.append(ci)
            p0 = detect(gray)
            pc, n0 = p0.copy(), len(p0)
            pending = False
    return chosen


def extract_keyframes(video: str, out_dir: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    raw_dir = os.path.join(out_dir, "raw")
    enh_dir = os.path.join(out_dir, "enh")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(enh_dir, exist_ok=True)
    src = FrameSource(video, cfg["candidate_hz"])

    # ---- pass 1: sharpness + small grey frames for tracking
    candidates, grays = [], []
    full_w = full_h = None
    for idx, t, frame in src.iterate():
        full_h, full_w = frame.shape[:2]
        s = cfg["track_width"] / float(full_w)
        small = cv2.resize(frame, (cfg["track_width"], int(round(full_h * s))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        candidates.append({"frame_index": idx, "t": t, "sharpness": laplacian_sharpness(gray)})
        grays.append(gray)
    if not candidates:
        raise RuntimeError(f"no frames decoded from {video}")

    sharp_vals = np.array([c["sharpness"] for c in candidates])
    sharp_thresh = float(np.percentile(sharp_vals, cfg["sharpness_reject_pct"]))
    disp, gap = cfg["min_disp_frac"], cfg["max_gap_s"]
    chosen = select_keyframes(candidates, grays, cfg["track_width"], disp, cfg["min_track_ratio"], gap, sharp_thresh)
    relaxed = 0
    # slow drift, hover or orbit: relax spacing until there are enough views for SfM
    while len(chosen) < cfg.get("min_keyframes", 20) and relaxed < 3 and len(candidates) > len(chosen):
        disp, gap, relaxed = disp * 0.5, gap * 0.5, relaxed + 1
        chosen = select_keyframes(candidates, grays, cfg["track_width"], disp, cfg["min_track_ratio"], gap,
                                  sharp_thresh)
    decimated = False
    if len(chosen) > cfg["max_keyframes"]:
        keep = np.linspace(0, len(chosen) - 1, cfg["max_keyframes"]).round().astype(int)
        chosen = [chosen[k] for k in keep]
        decimated = True
    del grays

    # ---- pass 2: write raw + enhanced keyframes
    wanted = {candidates[c]["frame_index"]: c for c in chosen}
    clahe = cv2.createCLAHE(clipLimit=cfg["clahe_clip"], tileGridSize=(8, 8))
    params = [cv2.IMWRITE_JPEG_QUALITY, int(cfg["jpeg_quality"])]
    keyframes = []
    for idx, t, frame in src.iterate(wanted=set(wanted)):
        frame = _resize_long(frame, cfg.get("max_long_side"))
        name = f"frame_{len(keyframes) + 1:05d}.jpg"
        cv2.imwrite(os.path.join(raw_dir, name), frame, params)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        cv2.imwrite(os.path.join(enh_dir, name), cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), params)
        c = candidates[wanted[idx]]
        keyframes.append({"name": name, "frame_index": idx, "t": round(t, 4),
                          "sharpness": round(c["sharpness"], 2), "disp_px": c.get("disp_px")})
    h, w = frame.shape[:2]
    disps = np.array([k["disp_px"] for k in keyframes[1:] if k["disp_px"] is not None and np.isfinite(k["disp_px"])])
    return {
        "video": video, "video_fps": src.fps, "video_frames": src.total,
        "source_resolution": [full_w, full_h], "keyframe_resolution": [w, h],
        "num_candidates": len(candidates), "num_keyframes": len(keyframes),
        "sharpness_threshold": sharp_thresh, "decimated": decimated,
        "auto_relaxed_steps": relaxed, "min_disp_frac_used": disp, "max_gap_s_used": gap,
        "median_disp_frac": float(np.median(disps) / cfg["track_width"]) if len(disps) else None,
        "keyframes": keyframes,
    }
