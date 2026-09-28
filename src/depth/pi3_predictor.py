"""Pi3X multi-view depth, conditioned on our SfM (poses, intrinsics, sparse depth).

Pi3X (Wang et al., 2025; BSD-licensed code, https://github.com/yyfz/Pi3) predicts
per-view point maps for a set of images jointly. Unlike a per-frame monocular
network it reasons across views, so small structures and repetitive texture
stay consistent between frames. We inject:

  poses       camera-to-world in our metric ENU frame (from SfM + georef)
  intrinsics  undistorted pinhole K, rescaled to the network resolution
  depths      sparse SfM track depths rasterised at their pixels (0 elsewhere)

so the dense output already lives in our frame. The lane still applies the
per-frame robust fit and the anchor correction map on top.

Frames are processed in overlapping windows (GPU memory); each frame takes
its depth from the window where it sits closest to the centre.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

PI3_DIRS = [os.environ.get("PI3_DIR", ""), os.path.expanduser("~/Pi3")]


def _import_pi3():
    for d in PI3_DIRS:
        if d and os.path.isdir(os.path.join(d, "pi3")) and d not in sys.path:
            sys.path.insert(0, d)
    from pi3.models.pi3x import Pi3X  # noqa: WPS433

    return Pi3X


class Pi3XPredictor:
    multiview = True

    def __init__(self, model_id: str = "yyfz233/Pi3X", pixel_limit: int = 255000, window: int = 12,
                 stride: int = 8, allow_cpu: bool = False):
        import torch

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu" and not allow_cpu:
            raise RuntimeError("Pi3X needs a CUDA device (set allow_cpu: true to accept CPU)")
        Pi3X = _import_pi3()
        self.model = Pi3X.from_pretrained(model_id).eval().to(self.device)
        self.pixel_limit, self.window, self.stride = pixel_limit, window, stride
        self.name = f"pi3x:{model_id}@{self.device} (conditioned on SfM poses, K, sparse depth)"

    def _size(self, w: int, h: int) -> Tuple[int, int]:
        s = (self.pixel_limit / float(w * h)) ** 0.5
        W, H = int(w * s) // 14 * 14, int(h * s) // 14 * 14
        return max(W, 14), max(H, 14)

    def predict_all(self, cams: List[Dict[str, Any]], anchors: Dict[str, Tuple[np.ndarray, np.ndarray]]
                    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Returns name -> (depth at the camera's working resolution, confidence 0..1)."""
        torch = self.torch
        w0, h0 = cams[0]["width"], cams[0]["height"]
        W, H = self._size(w0, h0)
        sx, sy = W / float(w0), H / float(h0)
        imgs, Ks, poses, deps = [], [], [], []
        for cam in cams:
            bgr = cv2.imread(cam["image"])
            rgb = cv2.cvtColor(cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            imgs.append(rgb.astype(np.float32) / 255.0)
            K = np.asarray(cam["K"], float).copy()
            K[0, 0] *= sx
            K[0, 2] = (K[0, 2]) * sx - 0.5  # COLMAP pixel centres at +0.5 -> array coordinates
            K[1, 1] *= sy
            K[1, 2] = (K[1, 2]) * sy - 0.5
            Ks.append(K)
            R, t = np.asarray(cam["R_cw"]), np.asarray(cam["t_cw"])
            c2w = np.eye(4)
            c2w[:3, :3] = R.T
            c2w[:3, 3] = -R.T @ t
            poses.append(c2w)
            d = np.zeros((H, W), np.float32)
            uv, z = anchors.get(cam["name"], (np.empty((0, 2)), np.empty(0)))
            if len(z):
                u = np.clip(((uv[:, 0] - 0.5) * sx).astype(int), 0, W - 1)
                v = np.clip(((uv[:, 1] - 0.5) * sy).astype(int), 0, H - 1)
                d[v, u] = z
            deps.append(d)
        imgs = np.stack(imgs).transpose(0, 3, 1, 2)
        n = len(cams)
        # windows covering all frames
        starts = list(range(0, max(1, n - self.window) + 1, self.stride))
        if starts[-1] + self.window < n:
            starts.append(n - self.window)
        starts = sorted(set(max(0, s) for s in starts))
        best: Dict[int, Tuple[float, np.ndarray, np.ndarray]] = {}
        dtype = torch.bfloat16 if self.device == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        for s in starts:
            idx = list(range(s, min(n, s + self.window)))
            T = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(self.device)[None]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=self.device == "cuda"):
                res = self.model(imgs=T(imgs[idx]), intrinsics=T(np.stack([Ks[i] for i in idx])),
                                 poses=T(np.stack([poses[i] for i in idx])), depths=T(np.stack([deps[i] for i in idx])))
            lp = res["local_points"][0].float().cpu().numpy()
            conf = torch.sigmoid(res["conf"][0, ..., 0].float()).cpu().numpy()
            centre = s + (len(idx) - 1) / 2.0
            for k, i in enumerate(idx):
                score = -abs(i - centre)
                if i not in best or score > best[i][0]:
                    best[i] = (score, lp[k, ..., 2], conf[k])
            del res
            if self.device == "cuda":
                torch.cuda.empty_cache()
        out = {}
        for i, cam in enumerate(cams):
            _, z, c = best[i]
            z = cv2.resize(z.astype(np.float32), (cam["width"], cam["height"]), interpolation=cv2.INTER_LINEAR)
            c = cv2.resize(c.astype(np.float32), (cam["width"], cam["height"]), interpolation=cv2.INTER_LINEAR)
            out[cam["name"]] = (z, c)
        return out
