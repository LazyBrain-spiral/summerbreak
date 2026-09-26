"""Backend-neutral sparse model container and readers.

``load_model`` prefers pycolmap. When pycolmap cannot be imported (for
example Windows with an application-control policy), it converts nothing and
instead reads COLMAP's TXT export, which ``colmap model_converter`` writes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray

    @property
    def K(self) -> np.ndarray:
        p = self.params
        if self.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"):
            f, cx, cy = p[0], p[1], p[2]
            return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    @property
    def distortion(self) -> np.ndarray:
        """OpenCV distortion vector (k1, k2, p1, p2[, k3...]) where representable."""
        p = self.params
        if self.model == "SIMPLE_RADIAL":
            return np.array([p[3], 0, 0, 0], float)
        if self.model == "RADIAL":
            return np.array([p[3], p[4], 0, 0], float)
        if self.model == "OPENCV":
            return np.array(p[4:8], float)
        if self.model == "FULL_OPENCV":
            return np.array(p[4:12], float)
        return np.zeros(4)


@dataclass
class Image:
    image_id: int
    name: str
    camera_id: int
    R_cw: np.ndarray            # world -> camera rotation
    t_cw: np.ndarray            # world -> camera translation
    xys: np.ndarray             # (N, 2) keypoint positions
    point3D_ids: np.ndarray     # (N,) -1 when not triangulated

    @property
    def center(self) -> np.ndarray:
        return -self.R_cw.T @ self.t_cw


@dataclass
class SparseModel:
    cameras: Dict[int, Camera]
    images: Dict[int, Image]
    point_ids: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    xyz: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    rgb: np.ndarray = field(default_factory=lambda: np.empty((0, 3), np.uint8))
    error: np.ndarray = field(default_factory=lambda: np.empty(0))
    track_len: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))

    def by_name(self) -> Dict[str, Image]:
        return {im.name: im for im in self.images.values()}

    def point_index(self) -> Dict[int, int]:
        return {int(pid): i for i, pid in enumerate(self.point_ids)}

    def mean_reprojection_error(self) -> float:
        return float(np.mean(self.error)) if len(self.error) else float("nan")


def _quat_to_R(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _load_pycolmap(path: str) -> SparseModel:
    import pycolmap

    rec = pycolmap.Reconstruction(path)
    cams = {}
    for cid, c in rec.cameras.items():
        model = c.model.name if hasattr(c.model, "name") else str(c.model)
        cams[cid] = Camera(cid, model, int(c.width), int(c.height), np.asarray(c.params, float))
    imgs = {}
    for iid, im in rec.images.items():
        has_pose = getattr(im, "has_pose", None)
        if has_pose is None:
            has_pose = getattr(im, "registered", True)
        if callable(has_pose):
            has_pose = has_pose()
        if not has_pose:
            continue
        cfw = im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world
        R = np.asarray(cfw.rotation.matrix(), float)
        t = np.asarray(cfw.translation, float)
        p2d = im.points2D
        xys = np.array([p.xy for p in p2d], float).reshape(-1, 2)
        ids = np.array([p.point3D_id if p.has_point3D() else -1 for p in p2d], np.int64)
        imgs[iid] = Image(iid, im.name, int(im.camera_id), R, t, xys, ids)
    pids, xyz, rgb, err, tl = [], [], [], [], []
    for pid, p in rec.points3D.items():
        pids.append(pid)
        xyz.append(p.xyz)
        rgb.append(p.color)
        err.append(p.error)
        tl.append(p.track.length())
    return SparseModel(cams, imgs, np.array(pids, np.int64), np.array(xyz, float).reshape(-1, 3),
                       np.array(rgb, np.uint8).reshape(-1, 3), np.array(err, float), np.array(tl, np.int64))


def _load_txt(path: str) -> SparseModel:
    cams = {}
    with open(os.path.join(path, "cameras.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            cams[int(p[0])] = Camera(int(p[0]), p[1], int(p[2]), int(p[3]), np.array(p[4:], float))
    imgs = {}
    with open(os.path.join(path, "images.txt"), encoding="utf-8") as f:
        lines = [l for l in f if not l.startswith("#")]
    for i in range(0, len(lines) - 1, 2):
        p = lines[i].split()
        if len(p) < 10:
            continue
        q = np.array(p[1:5], float)
        t = np.array(p[5:8], float)
        pts = lines[i + 1].split()
        arr = np.array(pts, float).reshape(-1, 3) if pts else np.empty((0, 3))
        imgs[int(p[0])] = Image(int(p[0]), p[9], int(p[8]), _quat_to_R(q), t,
                                arr[:, :2], arr[:, 2].astype(np.int64))
    pids, xyz, rgb, err, tl = [], [], [], [], []
    with open(os.path.join(path, "points3D.txt"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            pids.append(int(p[0]))
            xyz.append([float(v) for v in p[1:4]])
            rgb.append([int(v) for v in p[4:7]])
            err.append(float(p[7]))
            tl.append((len(p) - 8) // 2)
    return SparseModel(cams, imgs, np.array(pids, np.int64), np.array(xyz, float).reshape(-1, 3),
                       np.array(rgb, np.uint8).reshape(-1, 3), np.array(err, float), np.array(tl, np.int64))


def load_model(path: str) -> SparseModel:
    try:
        import pycolmap  # noqa: F401

        return _load_pycolmap(path)
    except ImportError:
        pass
    except RuntimeError:
        pass
    if os.path.exists(os.path.join(path, "images.txt")):
        return _load_txt(path)
    raise RuntimeError(f"cannot read sparse model at {path}: pycolmap unavailable and no TXT export")


def model_size(path: str) -> int:
    """Number of registered images, cheap enough to rank candidate sub-models."""
    try:
        return len(load_model(path).images)
    except Exception:
        return 0
