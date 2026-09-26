"""View-based texturing of regularised buildings from the source video frames.

Every planar face group of a building (each roof plane, each wall segment) gets
its own rectified texture patch in metres:

  1. candidate frames: face in front of the camera, projected inside the image,
     ranked by incidence angle and resolution (pixels per metre)
  2. per texel, the best candidate that actually SEES the texel: its depth map
     agrees with the texel's depth (trees, cars and other buildings in front
     are rejected) and it is not on a dynamic-object mask
  3. colour is sampled from the full-resolution raw keyframe through the SfM
     lens model (distortion included), not from the downscaled depth images
  4. gaps: small ones are inpainted; a wall never seen at all borrows the
     building's best-observed facade (marked "generated" in the metadata)
  5. patches are shelf-packed into one atlas; UVs are written per vertex

Also returns the rectified facade patches (metres known) for door and window
detection.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.imgops import sample_points
from src.core.io import read_json

NEUTRAL_WALL = np.array([206, 198, 184], np.uint8)
NEUTRAL_ROOF = np.array([150, 86, 62], np.uint8)


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------

class CameraSet:
    """Undistorted cameras (for visibility via depth maps) + SfM lens model (for colour)."""

    def __init__(self, run_dir: str):
        self.run = run_dir
        meta = read_json(os.path.join(run_dir, "05_depth", "cameras.json"))
        self.cams = meta["cameras"]
        poses = read_json(os.path.join(run_dir, "03_sfm", "poses.json"))
        self.sfm_cam_of = {im["name"]: poses["cameras"][str(im["camera_id"])] for im in poses["images"]}
        self._depth: Dict[str, np.ndarray] = {}
        self._img: Dict[str, np.ndarray] = {}
        self._mask: Dict[str, Optional[np.ndarray]] = {}
        masks_dir = os.path.join(run_dir, "02_masks", "dynamic")
        self.masks_dir = masks_dir if os.path.isdir(masks_dir) else None

    def depth(self, name: str) -> np.ndarray:
        if name not in self._depth:
            self._depth[name] = np.load(os.path.join(self.run, "05_depth", "depth", os.path.splitext(name)[0] + ".npy"))
        return self._depth[name]

    def image(self, name: str) -> np.ndarray:
        if name not in self._img:
            p = os.path.join(self.run, "01_ingest", "raw", name)
            img = cv2.imread(p)
            if img is None:  # fall back to the undistorted working image
                cam = next(c for c in self.cams if c["name"] == name)
                img = cv2.imread(cam["image"])
            self._img[name] = img
        return self._img[name]

    def dyn_mask(self, name: str) -> Optional[np.ndarray]:
        if name not in self._mask:
            m = None
            if self.masks_dir:
                p = os.path.join(self.masks_dir, os.path.splitext(name)[0] + ".png")
                if os.path.exists(p):
                    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 127
            self._mask[name] = m
        return self._mask[name]

    def project_raw(self, name: str, x_cam: np.ndarray) -> np.ndarray:
        """Camera-frame points -> raw-image array coordinates through the SfM lens model."""
        c = self.sfm_cam_of[name]
        model, p = c["model"], np.asarray(c["params"], float)
        z = np.maximum(x_cam[:, 2], 1e-9)
        xn, yn = x_cam[:, 0] / z, x_cam[:, 1] / z
        r2 = xn * xn + yn * yn
        if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            f, cx, cy = p[0], p[1], p[2]
            k1 = p[3] if model != "SIMPLE_PINHOLE" else 0.0
            k2 = p[4] if model == "RADIAL" else 0.0
            d = 1 + k1 * r2 + k2 * r2 * r2
            u, v = f * xn * d + cx, f * yn * d + cy
        else:  # PINHOLE / OPENCV
            fx, fy, cx, cy = p[0], p[1], p[2], p[3]
            if model == "OPENCV":
                k1, k2, p1, p2 = p[4:8]
                d = 1 + k1 * r2 + k2 * r2 * r2
                xd = xn * d + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
                yd = yn * d + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
            else:
                xd, yd = xn, yn
            u, v = fx * xd + cx, fy * yd + cy
        # raw keyframes may have been written larger/smaller than the SfM images
        img = self.image(name)
        sx, sy = img.shape[1] / float(c["width"]), img.shape[0] / float(c["height"])
        return np.column_stack([(u - 0.5) * sx + 0.5 * (sx - 1), (v - 0.5) * sy + 0.5 * (sy - 1)])


# ---------------------------------------------------------------------------
# face groups
# ---------------------------------------------------------------------------

def _plane_basis(verts: np.ndarray, kind: str, frame_angle: float):
    """origin, e1 (horizontal), e2 (in-plane 'up'), normal (outward / upward)."""
    c = verts.mean(axis=0)
    X = verts - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = vt[2]
    if kind == "roof" and n[2] < 0:
        n = -n
    up = np.array([0.0, 0.0, 1.0])
    if kind == "wall":
        e2 = up
        e1 = np.cross(e2, n)  # horizontal, along the wall
        e1 /= max(np.linalg.norm(e1), 1e-9)
    else:
        h = np.cross(up, n)
        if np.linalg.norm(h) < 1e-6:  # flat roof: align with the building frame
            h = np.array([math.cos(frame_angle), math.sin(frame_angle), 0.0])
        e1 = h / np.linalg.norm(h)
        e2 = np.cross(n, e1)
    return c, e1, e2, n


def _groups(reg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for kind, V, F, G in (("roof", reg["roof_v"], reg["roof_f"], reg["roof_groups"]),
                          ("wall", reg["wall_v"], reg["wall_f"], reg["wall_groups"])):
        if not len(F):
            continue
        for g in np.unique(G):
            tri = F[G == g]
            vid = np.unique(tri)
            local = {int(v): i for i, v in enumerate(vid)}
            f_local = np.vectorize(local.get)(tri)
            v = V[vid]
            if kind == "wall":
                # wall normal must point outward: the regulariser builds outward-facing triangles
                a, b, c = v[f_local[0]]
                nrm = np.cross(b - a, c - a)
                o, e1, e2, n = _plane_basis(v, kind, reg["frame_angle"])
                if n @ nrm < 0:
                    n, e1 = -n, -e1
            else:
                o, e1, e2, n = _plane_basis(v, kind, reg["frame_angle"])
            out.append({"kind": kind, "index": int(g), "v": v, "f": f_local, "o": o, "e1": e1, "e2": e2, "n": n})
    return out


# ---------------------------------------------------------------------------
# texture a group
# ---------------------------------------------------------------------------

def texture_group(grp: Dict[str, Any], cs: CameraSet, texel: float, max_cams: int = 5,
                  depth_tol_rel: float = 0.05, depth_tol_abs: float = 0.8) -> Dict[str, Any]:
    v, f, o, e1, e2, n = grp["v"], grp["f"], grp["o"], grp["e1"], grp["e2"], grp["n"]
    q = np.column_stack([(v - o) @ e1, (v - o) @ e2])
    qmin, qmax = q.min(axis=0), q.max(axis=0)
    pad = 2
    W = int(math.ceil((qmax[0] - qmin[0]) / texel)) + 2 * pad
    H = int(math.ceil((qmax[1] - qmin[1]) / texel)) + 2 * pad
    W, H = max(W, 4), max(H, 4)
    # texel centres: column along e1, row downwards along -e2 (image "up" = e2)
    cols = (np.arange(W) - pad + 0.5) * texel + qmin[0]
    rows = qmax[1] - (np.arange(H) - pad + 0.5) * texel
    S, T = np.meshgrid(cols, rows)
    P = o + S[..., None] * e1 + T[..., None] * e2
    P = P.reshape(-1, 3)
    # polygon mask in patch pixels
    px = np.column_stack([(q[:, 0] - qmin[0]) / texel + pad, (qmax[1] - q[:, 1]) / texel + pad])
    inside = np.zeros((H, W), np.uint8)
    for tri in f:
        cv2.fillConvexPoly(inside, np.round(px[tri] * 8).astype(np.int32), 1, shift=3)
    inside = cv2.dilate(inside, np.ones((3, 3), np.uint8)).astype(bool)
    # candidate cameras
    cen = v.mean(axis=0)
    cands = []
    for cam in cs.cams:
        R, t = np.asarray(cam["R_cw"]), np.asarray(cam["t_cw"])
        C = -R.T @ t
        ray = C - cen
        dist = np.linalg.norm(ray)
        cosang = float(n @ ray / max(dist, 1e-9))
        if cosang < 0.12:
            continue
        xc = R @ cen + t
        if xc[2] <= 0:
            continue
        K = np.asarray(cam["K"])
        u, w_ = K[0, 0] * xc[0] / xc[2] + K[0, 2], K[1, 1] * xc[1] / xc[2] + K[1, 2]
        if not (-0.2 * cam["width"] < u < 1.2 * cam["width"] and -0.2 * cam["height"] < w_ < 1.2 * cam["height"]):
            continue
        cands.append((cosang * K[0, 0] / dist, cam))
    cands.sort(key=lambda c: -c[0])
    cands = cands[:max_cams]

    out = np.zeros((H * W, 3), np.uint8)
    best = np.full(H * W, -np.inf)
    idx_in = np.flatnonzero(inside.ravel())
    Pin = P[idx_in]
    for _, cam in cands:
        name = cam["name"]
        R, t = np.asarray(cam["R_cw"]), np.asarray(cam["t_cw"])
        K = np.asarray(cam["K"])
        x = Pin @ R.T + t
        z = x[:, 2]
        okz = z > 0.1
        # visibility against the depth map (undistorted working image)
        d = cs.depth(name)
        dh, dw = d.shape
        with np.errstate(divide="ignore", invalid="ignore"):
            ud = np.round(K[0, 0] * x[:, 0] / z + K[0, 2] - 0.5).astype(np.int64)
            vd = np.round(K[1, 1] * x[:, 1] / z + K[1, 2] - 0.5).astype(np.int64)
        okp = okz & (ud >= 0) & (ud < dw) & (vd >= 0) & (vd < dh)
        dz = np.zeros(len(z))
        dz[okp] = d[vd[okp], ud[okp]]
        tol = np.maximum(depth_tol_abs, depth_tol_rel * z)
        # visible = the depth map has a surface there and nothing clearly in front of the texel.
        # Pixels without depth are sky or rejected geometry: never sample colour from them.
        vis = okp & (dz > 0) & (dz >= z - tol)
        # colour from the full-resolution raw frame
        img = cs.image(name)
        uv = cs.project_raw(name, x)
        ih, iw = img.shape[:2]
        inimg = (uv[:, 0] >= 0) & (uv[:, 0] < iw - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < ih - 1)
        vis &= inimg
        m = cs.dyn_mask(name)
        if m is not None and vis.any():
            mu = np.clip((uv[:, 0] * m.shape[1] / iw).astype(int), 0, m.shape[1] - 1)
            mv = np.clip((uv[:, 1] * m.shape[0] / ih).astype(int), 0, m.shape[0] - 1)
            vis &= ~m[mv, mu]
        C = -R.T @ t
        ray = C - Pin
        rn = np.linalg.norm(ray, axis=1)
        score = (ray @ n) / np.maximum(rn, 1e-9) * K[0, 0] / np.maximum(rn, 1e-9)
        take = vis & (score > best[idx_in])
        if not take.any():
            continue
        cols_ = sample_points(img, uv[take, 0], uv[take, 1])
        if grp["kind"] == "roof":
            # horizon sky can carry (network) depth; never paint a roof with bright sky blue
            b_, g_, r_ = (cols_[:, k].astype(np.int32) for k in range(3))
            sky = (b_ > r_ * 1.12) & (b_ >= g_) & (b_ > 120)
            ti = np.flatnonzero(take)
            take[ti[sky]] = False
            cols_ = cols_[~sky]
        out[idx_in[take]] = cols_
        best[idx_in[take]] = score[take]
    seen = np.isfinite(best).reshape(H, W) & inside
    patch = out.reshape(H, W, 3)
    return {"patch": patch, "seen": seen, "inside": inside, "W": W, "H": H, "qmin": qmin, "qmax": qmax,
            "pad": pad, "texel": texel, "observed_frac": float(seen.sum() / max(1, inside.sum())),
            "n_cams": len(cands)}


def _ortho_fill(tex: Dict[str, Any], grp: Dict[str, Any], ortho) -> None:
    """Fill unseen roof texels from the orthomosaic (every roof is seen from above)."""
    img, alpha, geo, xmin, ymax, gsd = ortho
    H, W, pad, texel = tex["H"], tex["W"], tex["pad"], tex["texel"]
    cols = (np.arange(W) - pad + 0.5) * texel + tex["qmin"][0]
    rows = tex["qmax"][1] - (np.arange(H) - pad + 0.5) * texel
    S, T = np.meshgrid(cols, rows)
    P = grp["o"] + S[..., None] * grp["e1"] + T[..., None] * grp["e2"]
    need = (tex["inside"] & ~tex["seen"]).ravel()
    if not need.any():
        return
    xy = geo.enu_to_utm(P.reshape(-1, 3)[need, :2])
    c = (xy[:, 0] - xmin) / gsd - 0.5
    r = (ymax - xy[:, 1]) / gsd - 0.5
    ok = (c >= 0) & (c < img.shape[1] - 1) & (r >= 0) & (r < img.shape[0] - 1)
    ci = np.clip(np.round(c).astype(int), 0, img.shape[1] - 1)
    ri = np.clip(np.round(r).astype(int), 0, img.shape[0] - 1)
    ok &= alpha[ri, ci] > 0
    if not ok.any():
        return
    idx = np.flatnonzero(need)[ok]
    flat = tex["patch"].reshape(-1, 3)
    flat[idx] = sample_points(img, c[ok], r[ok])
    seen = tex["seen"].reshape(-1)
    seen[idx] = True
    tex["ortho_frac"] = float(ok.mean())


def _fill(tex: Dict[str, Any], kind: str, donor: Optional[np.ndarray]) -> str:
    """Complete unseen texels. Returns provenance: observed | inpainted | generated | neutral."""
    patch, seen, inside = tex["patch"], tex["seen"], tex["inside"]
    frac = float(seen.sum() / max(1, inside.sum()))
    if frac >= 0.999:
        return "observed"
    if frac >= 0.25:
        hole = (inside & ~seen).astype(np.uint8)
        patch[:] = cv2.inpaint(patch, hole, 5, cv2.INPAINT_TELEA)
        return "observed" if frac > 0.9 else "inpainted"
    if donor is not None and kind == "wall":
        H, W = patch.shape[:2]
        dh, dw = donor.shape[:2]
        # tile the donor facade at its own scale (metres preserved), anchored at the ground line
        tiled = np.tile(donor, (-(-H // dh), -(-W // dw), 1))[-H:, :W]
        patch[:] = np.where(seen[..., None], patch, tiled)
        return "generated"
    base = NEUTRAL_WALL if kind == "wall" else NEUTRAL_ROOF
    if seen.any():
        base = np.median(patch[seen], axis=0).astype(np.uint8)
    patch[~seen] = base
    return "neutral"


# ---------------------------------------------------------------------------
# atlas
# ---------------------------------------------------------------------------

def pack(sizes: List[Tuple[int, int]], max_side: int = 4096) -> Tuple[List[Tuple[int, int]], int, int, float]:
    """Shelf packing. Returns positions, atlas W, H and a scale (<1 if it had to shrink)."""
    total = sum(w * h for w, h in sizes)
    scale = 1.0
    side = int(math.ceil(math.sqrt(total * 1.25)))
    if side > max_side:
        scale = max_side / float(side)
        side = max_side
    order = sorted(range(len(sizes)), key=lambda i: -sizes[i][1])
    pos = [None] * len(sizes)
    x = y = shelf = 0
    for i in order:
        w, h = int(math.ceil(sizes[i][0] * scale)), int(math.ceil(sizes[i][1] * scale))
        if x + w > side:
            x, y, shelf = 0, y + shelf, 0
        pos[i] = (x, y)
        x += w
        shelf = max(shelf, h)
    H = y + shelf
    return pos, side, max(H, 1), scale


def texture_buildings(regs: List[Dict[str, Any]], run_dir: str, texel: float = 0.06,
                      max_atlas: int = 4096, ortho=None) -> Dict[str, Any]:
    cs = CameraSet(run_dir)
    items = []  # (building, group, texture)
    for r in regs:
        groups = _groups(r)
        texs = [texture_group(g, cs, texel) for g in groups]
        walls = [(t, g) for t, g in zip(texs, groups) if g["kind"] == "wall"]
        donor = None
        if walls:
            tb, gb = max(walls, key=lambda tg: tg[0]["observed_frac"] * tg[0]["seen"].sum())
            if tb["observed_frac"] > 0.4:
                ys, xs = np.nonzero(tb["seen"])
                donor = tb["patch"][ys.min():ys.max() + 1, xs.min():xs.max() + 1].copy()
        prov = {}
        for t, g in zip(texs, groups):
            if ortho is not None and g["kind"] == "roof" and t["observed_frac"] < 0.9:
                _ortho_fill(t, g, ortho)
            p = _fill(t, g["kind"], donor)
            t["provenance"] = p
            prov[p] = prov.get(p, 0) + 1
            items.append((r, g, t))
        seen_area = sum(t["seen"].sum() for t, g in zip(texs, groups))
        all_area = sum(t["inside"].sum() for t, g in zip(texs, groups))
        r["texture_observed_frac"] = round(float(seen_area / max(1, all_area)), 3)
        r["walls_observed"] = int(sum(1 for t, g in zip(texs, groups) if g["kind"] == "wall" and t["observed_frac"] > 0.4))
        r["walls_total"] = int(sum(1 for g in groups if g["kind"] == "wall"))
        r["face_provenance"] = prov
        r["_groups"] = [(g, t) for t, g in zip(texs, groups)]
    if not items:
        return {"atlas": None}
    sizes = [(t["W"], t["H"]) for _, _, t in items]
    pos, AW, AH, scale = pack(sizes, max_atlas)
    atlas = np.zeros((AH, AW, 3), np.uint8)
    for (r, g, t), (x0, y0) in zip(items, pos):
        patch = t["patch"]
        if scale < 1.0:
            patch = cv2.resize(patch, (max(1, int(math.ceil(t["W"] * scale))), max(1, int(math.ceil(t["H"] * scale)))),
                               interpolation=cv2.INTER_AREA)
        h, w = patch.shape[:2]
        atlas[y0:y0 + h, x0:x0 + w] = patch
        # UVs of this group's vertices (patch pixel coords -> atlas)
        q = np.column_stack([(g["v"] - g["o"]) @ g["e1"], (g["v"] - g["o"]) @ g["e2"]])
        pxl = np.column_stack([(q[:, 0] - t["qmin"][0]) / texel + t["pad"], (t["qmax"][1] - q[:, 1]) / texel + t["pad"]])
        pxl *= scale
        g["uv"] = np.column_stack([(x0 + pxl[:, 0]) / AW, 1.0 - (y0 + pxl[:, 1]) / AH])
    return {"atlas": atlas, "scale": scale, "size": [AW, AH], "cameras": len(cs.cams)}
