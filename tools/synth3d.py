"""Synthetic single-pass drone flight over a real 3D scene, with ground truth.

Replaces the flat 2D-canvas generators (src/utils/generate_*_flight.py), whose
frames are pure translations of a plane and therefore carry no parallax.

The scene is a triangle mesh (terrain heightfield, gabled and flat-roofed
buildings, trees, a road). Frames are rendered by CPU ray casting with
Open3D's RaycastingScene, so there is no OpenGL dependency. Surfaces carry a
procedural 3D noise texture so SIFT has features everywhere.

Outputs in --out:
    flight.mp4          rendered video
    flight.srt          DJI-style captions (lat, lon, rel/abs alt, gimbal, focal_len)
    gt.json             intrinsics, per-frame camera-to-world poses in ENU, origin,
                        building ground truth (footprint, eave/ridge height)
    scene.ply           the scene mesh in ENU (for oracle depth and evaluation)
    scene_labels.npy    per-triangle semantic label (see LABELS)

Example:
    python tools/synth3d.py --out test_data/synth3d_pass --seconds 16 --fps 10
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d

LABELS = {"ground": 1, "road": 2, "roof": 3, "wall": 4, "tree": 5}
BASE_COLORS = {  # RGB 0..1
    1: (0.36, 0.45, 0.25),
    2: (0.30, 0.30, 0.31),
    3: (0.62, 0.35, 0.28),
    4: (0.78, 0.74, 0.66),
    5: (0.16, 0.32, 0.14),
}


# ----------------------------------------------------------------------------
# Procedural texture
# ----------------------------------------------------------------------------

class ValueNoise3D:
    """Lattice value noise with trilinear interpolation; deterministic and vectorised.

    Uses a periodic 128^3 random lattice (period 128 cells per octave). Octaves
    use non-integer frequency ratios, so the summed texture does not repeat at
    any scale that matters for feature matching.
    """

    N = 128

    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.table = rng.random(self.N ** 3).astype(np.float32)

    def __call__(self, p: np.ndarray, freq: float) -> np.ndarray:
        q = (p * freq).astype(np.float32)
        fl = np.floor(q)
        f = q - fl
        f = f * f * (3.0 - 2.0 * f)
        i = fl.astype(np.int32) & (self.N - 1)
        j = (i + 1) & (self.N - 1)
        n = self.N
        xs = (i[:, 0] * n * n, j[:, 0] * n * n)
        ys = (i[:, 1] * n, j[:, 1] * n)
        zs = (i[:, 2], j[:, 2])
        fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
        t = self.table
        c00 = t[xs[0] + ys[0] + zs[0]] * (1 - fz) + t[xs[0] + ys[0] + zs[1]] * fz
        c01 = t[xs[0] + ys[1] + zs[0]] * (1 - fz) + t[xs[0] + ys[1] + zs[1]] * fz
        c10 = t[xs[1] + ys[0] + zs[0]] * (1 - fz) + t[xs[1] + ys[0] + zs[1]] * fz
        c11 = t[xs[1] + ys[1] + zs[0]] * (1 - fz) + t[xs[1] + ys[1] + zs[1]] * fz
        c0 = c00 * (1 - fy) + c01 * fy
        c1 = c10 * (1 - fy) + c11 * fy
        return c0 * (1 - fx) + c1 * fx

    def fbm(self, p: np.ndarray, base_freq: float, octaves: int = 5) -> np.ndarray:
        total = np.zeros(len(p), dtype=np.float32)
        amp, norm, freq = 1.0, 0.0, base_freq
        for _ in range(octaves):
            total += amp * self(p, freq)
            norm += amp
            amp *= 0.55
            freq *= 2.1
        return total / norm


def make_fbm_texture(size: int, seed: int, octaves: int = 10) -> np.ndarray:
    """Tileable-enough 2D fBm texture in [0, 1], built from resized random grids."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), dtype=np.float32)
    amp, norm, cells = 1.0, 0.0, 6
    for _ in range(octaves):
        grid = rng.random((cells, cells)).astype(np.float32)
        acc += amp * cv2.resize(grid, (size, size), interpolation=cv2.INTER_CUBIC)
        norm += amp
        amp *= 0.78
        cells = int(cells * 2.3)
        if cells >= size:
            break
    acc /= norm
    acc -= acc.min()
    return acc / max(float(acc.max()), 1e-6)


def make_blob_texture(size: int, seed: int, count: int = 60000, rmin: int = 3, rmax: int = 22) -> np.ndarray:
    """High-contrast random blobs (bushes, stones, clutter): what SIFT locks on to."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 0.5, dtype=np.float32)
    xs = rng.integers(0, size, count)
    ys = rng.integers(0, size, count)
    rs = (rmin + (rmax - rmin) * rng.random(count) ** 2).astype(int)
    vs = rng.random(count).astype(np.float32)
    for x, y, r, v in zip(xs, ys, rs, vs):
        cv2.circle(img, (int(x), int(y)), int(r), float(v), -1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(img, (0, 0), 1.2)


# ----------------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------------

@dataclass
class Building:
    cx: float
    cy: float
    w: float           # extent along local x
    d: float           # extent along local y
    eave: float        # wall height above local ground
    ridge: float       # roof peak above local ground (== eave for flat roof)
    yaw_deg: float = 0.0

    def footprint(self) -> np.ndarray:
        hw, hd = self.w / 2.0, self.d / 2.0
        local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]])
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        rot = np.array([[c, -s], [s, c]])
        return local @ rot.T + np.array([self.cx, self.cy])


@dataclass
class Scene:
    size: float = 340.0
    seed: int = 7
    buildings: List[Building] = field(default_factory=list)
    trees: List[Tuple[float, float, float, float]] = field(default_factory=list)  # x, y, radius, height
    road_halfwidth: float = 4.0
    road_y: float = -22.0

    def terrain_z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Gentle hills, amplitude a few metres."""
        return (1.8 * np.sin(x / 37.0) * np.cos(y / 29.0)
                + 0.9 * np.sin((x + 2 * y) / 17.0)
                + 0.012 * x)


def default_scene(seed: int = 7) -> Scene:
    sc = Scene(seed=seed)
    sc.buildings = [
        Building(-45.0, 18.0, 30.0, 16.0, eave=7.0, ridge=11.0, yaw_deg=0.0),
        Building(-5.0, 22.0, 18.0, 18.0, eave=12.0, ridge=12.0, yaw_deg=15.0),
        Building(32.0, 14.0, 24.0, 12.0, eave=5.0, ridge=8.0, yaw_deg=-10.0),
        Building(62.0, 28.0, 14.0, 22.0, eave=18.0, ridge=18.0, yaw_deg=0.0),
        Building(10.0, -48.0, 36.0, 14.0, eave=6.0, ridge=6.0, yaw_deg=5.0),
    ]
    rng = np.random.default_rng(seed)
    trees = []
    while len(trees) < 26:
        x, y = rng.uniform(-95, 95), rng.uniform(-70, 70)
        if abs(y - sc.road_y) < sc.road_halfwidth + 4:
            continue
        if any(np.hypot(x - b.cx, y - b.cy) < max(b.w, b.d) * 0.75 + 4 for b in sc.buildings):
            continue
        trees.append((x, y, rng.uniform(2.0, 3.8), rng.uniform(6.0, 11.0)))
    sc.trees = trees
    return sc


class MeshBuilder:
    def __init__(self):
        self.v: List[np.ndarray] = []
        self.f: List[np.ndarray] = []
        self.lab: List[np.ndarray] = []
        self.n = 0

    def add(self, verts: np.ndarray, faces: np.ndarray, label: int):
        self.v.append(np.asarray(verts, dtype=np.float64))
        self.f.append(np.asarray(faces, dtype=np.int64) + self.n)
        self.lab.append(np.full(len(faces), label, dtype=np.uint8))
        self.n += len(verts)

    def build(self) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray]:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(np.vstack(self.v))
        mesh.triangles = o3d.utility.Vector3iVector(np.vstack(self.f).astype(np.int32))
        mesh.compute_triangle_normals()
        return mesh, np.concatenate(self.lab)


def _grid_faces(nx: int, ny: int) -> np.ndarray:
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])


def build_scene_mesh(sc: Scene, terrain_step: float = 1.0) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, List[Dict]]:
    mb = MeshBuilder()
    half = sc.size / 2.0
    xs = np.arange(-half, half + 1e-6, terrain_step)
    gx, gy = np.meshgrid(xs, xs)
    gz = sc.terrain_z(gx, gy)
    verts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    faces = _grid_faces(len(xs), len(xs))
    # split terrain faces into road / ground by centroid
    cen_y = verts[faces].mean(axis=1)[:, 1]
    on_road = np.abs(cen_y - sc.road_y) < sc.road_halfwidth
    mb.add(verts, faces[~on_road], LABELS["ground"])
    mb.add(verts + np.array([0, 0, 0.02]), faces[on_road], LABELS["road"])

    gt_buildings = []
    for b in sc.buildings:
        fp = b.footprint()
        base = float(sc.terrain_z(fp[:, 0], fp[:, 1]).min()) - 0.5  # sink below terrain
        ground_ref = float(sc.terrain_z(np.array([b.cx]), np.array([b.cy]))[0])
        eave_z = ground_ref + b.eave
        ridge_z = ground_ref + b.ridge
        bottom = np.column_stack([fp, np.full(4, base)])
        top = np.column_stack([fp, np.full(4, eave_z)])
        # walls
        for i in range(4):
            j = (i + 1) % 4
            quad = np.array([bottom[i], bottom[j], top[j], top[i]])
            mb.add(quad, np.array([[0, 1, 2], [0, 2, 3]]), LABELS["wall"])
        if b.ridge > b.eave + 0.05:
            # gable along the long local x axis: ridge between midpoints of short edges
            m_left = (top[0] + top[3]) / 2.0
            m_right = (top[1] + top[2]) / 2.0
            m_left[2] = ridge_z
            m_right[2] = ridge_z
            roof_v = np.array([top[0], top[1], m_right, m_left, top[3], top[2]])
            roof_f = np.array([[0, 1, 2], [0, 2, 3], [3, 2, 5], [3, 5, 4]])
            mb.add(roof_v, roof_f, LABELS["roof"])
            # gable end triangles (walls)
            mb.add(np.array([top[0], m_left, top[3]]), np.array([[0, 2, 1]]), LABELS["wall"])
            mb.add(np.array([top[1], top[2], m_right]), np.array([[0, 1, 2]]), LABELS["wall"])
        else:
            mb.add(top, np.array([[0, 1, 2], [0, 2, 3]]), LABELS["roof"])
        gt_buildings.append({
            "center_enu": [b.cx, b.cy],
            "footprint_enu": fp.tolist(),
            "footprint_area_m2": b.w * b.d,
            "ground_z": ground_ref,
            "eave_height_m": b.eave,
            "ridge_height_m": b.ridge,
            "yaw_deg": b.yaw_deg,
        })

    for (x, y, r, h) in sc.trees:
        gz0 = float(sc.terrain_z(np.array([x]), np.array([y]))[0])
        crown = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
        cv = np.asarray(crown.vertices) * np.array([r, r, h * 0.35])
        cv += np.array([x, y, gz0 + h * 0.62])
        mb.add(cv, np.asarray(crown.triangles), LABELS["tree"])
        trunk = o3d.geometry.TriangleMesh.create_cylinder(radius=0.35, height=h * 0.5, resolution=8)
        tv = np.asarray(trunk.vertices) + np.array([x, y, gz0 + h * 0.25])
        mb.add(tv, np.asarray(trunk.triangles), LABELS["tree"])

    mesh, labels = mb.build()
    return mesh, labels, gt_buildings


# ----------------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------------

def camera_to_world_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """OpenCV camera axes (x right, y down, z forward) expressed in ENU.

    yaw: heading clockwise from north. pitch: gimbal pitch, -90 = nadir.
    """
    yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(pitch), math.sin(pitch)])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    down = np.cross(fwd, right)
    return np.column_stack([right, down, fwd])


def intrinsics_from_hfov(width: int, height: int, hfov_deg: float) -> np.ndarray:
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[fx, 0, width / 2.0], [0, fx, height / 2.0], [0, 0, 1.0]])


class Renderer:
    def __init__(self, mesh: o3d.geometry.TriangleMesh, labels: np.ndarray, seed: int = 3,
                 sun_dir: Tuple[float, float, float] = (0.45, -0.35, 0.82)):
        self.scene = o3d.t.geometry.RaycastingScene()
        self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        self.normals = np.asarray(mesh.triangle_normals, dtype=np.float32)
        self.labels = labels
        self.tex_fine = 0.45 * make_fbm_texture(4096, seed) + 0.55 * make_blob_texture(4096, seed + 5)  # 5 cm texels
        self.tex_coarse = make_fbm_texture(1024, seed + 11)  # 17 cm texels, 174 m period
        s = np.asarray(sun_dir, dtype=np.float32)
        self.sun = s / np.linalg.norm(s)
        base = np.zeros((max(BASE_COLORS) + 1, 3), dtype=np.float32)
        for k, c in BASE_COLORS.items():
            base[k] = c
        self.base = base

    def rays(self, K: np.ndarray, R_wc: np.ndarray, C: np.ndarray, w: int, h: int) -> np.ndarray:
        u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
        d_cam = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1)
        d_world = d_cam @ R_wc.T
        d_world /= np.linalg.norm(d_world, axis=-1, keepdims=True)
        origin = np.broadcast_to(C, d_world.shape)
        return np.concatenate([origin, d_world], -1).astype(np.float32), d_cam

    def cast(self, K, R_wc, C, w, h):
        rays, d_cam = self.rays(K, R_wc, C, w, h)
        ans = self.scene.cast_rays(o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy()
        prim = ans["primitive_ids"].numpy().astype(np.int64)
        hit = np.isfinite(t)
        # t is along a unit world ray; z-depth = t * (unit camera dir).z
        dz = d_cam[..., 2] / np.linalg.norm(d_cam, axis=-1)
        depth = np.where(hit, t * dz, 0.0).astype(np.float32)
        return rays, t, prim, hit, depth

    def _sample_textures(self, p: np.ndarray, n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Planar projection on the dominant normal axis, bilinear sampling via cv2.remap."""
        ax = np.argmax(np.abs(n), axis=1)
        uv = np.where((ax == 2)[:, None], p[:, [0, 1]],
                      np.where((ax == 0)[:, None], p[:, [1, 2]], p[:, [0, 2]]))
        uv = uv + ax[:, None].astype(np.float32) * 37.0  # decorrelate the three projections
        out = []
        n_pts = len(uv)
        cols = 2048
        rows = -(-n_pts // cols)
        pad = rows * cols - n_pts
        for tex, texel in ((self.tex_fine, 0.05), (self.tex_coarse, 0.17)):
            m = np.mod(uv / texel, tex.shape[0] - 1).astype(np.float32)
            m = np.concatenate([m, np.zeros((pad, 2), np.float32)]).reshape(rows, cols, 2)
            val = cv2.remap(tex, m[..., 0], m[..., 1], interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_WRAP)
            out.append(val.reshape(-1)[:n_pts])
        return out[0], out[1]

    def render(self, K, R_wc, C, w, h) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rays, t, prim, hit, depth = self.cast(K, R_wc, C, w, h)
        img = np.zeros((h, w, 3), dtype=np.float32)
        # sky gradient
        up = rays[..., 5]
        sky = np.stack([0.55 + 0.25 * up, 0.68 + 0.2 * up, 0.9 + 0.08 * up], -1)
        img[~hit] = np.clip(sky[~hit], 0, 1)
        if hit.any():
            p = rays[..., :3][hit] + rays[..., 3:][hit] * t[hit][:, None]
            pid = prim[hit]
            lab = self.labels[pid]
            n = self.normals[pid]
            base = self.base[lab]
            tex, fine = self._sample_textures(p, n)
            shade = 0.35 + 0.65 * np.clip(n @ self.sun, 0.0, 1.0)
            col = base * (0.35 + 1.1 * tex[:, None]) * (0.7 + 0.6 * fine[:, None]) * shade[:, None]
            # road markings
            road = lab == LABELS["road"]
            if road.any():
                dash = (np.abs(p[road, 1] - p[road, 1].mean()) < 0.15) & ((np.floor(p[road, 0] / 3.0) % 2) == 0)
                col[np.where(road)[0][dash]] = 0.9
            img[hit] = np.clip(col, 0, 1)
        label_img = np.zeros((h, w), dtype=np.uint8)
        label_img[hit] = self.labels[prim[hit]]
        return (img * 255).astype(np.uint8), depth, label_img


# ----------------------------------------------------------------------------
# Flight and telemetry
# ----------------------------------------------------------------------------

def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    hh, rem = divmod(ms, 3600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def fly(out_dir: str, seconds: float = 16.0, fps: float = 10.0, width: int = 960, height: int = 540,
        altitude: float = 60.0, speed: float = 8.0, pitch: float = -60.0, heading: float = 90.0,
        lateral_offset: float = -5.0, hfov: float = 78.0, origin=(28.6139, 77.2090, 216.0),
        gps_sigma_h: float = 0.0, gps_sigma_v: float = 0.0, seed: int = 7, curve: float = 0.0,
        max_frames: Optional[int] = None) -> Dict:
    """Render a single straight (optionally slightly curved) pass.

    heading 90 = flying east. lateral_offset shifts the track south (negative)
    so the buildings (north of the road) fall inside the oblique forward view.
    """
    import pymap3d

    os.makedirs(out_dir, exist_ok=True)
    sc = default_scene(seed)
    mesh, labels, gt_buildings = build_scene_mesh(sc)
    renderer = Renderer(mesh, labels)
    K = intrinsics_from_hfov(width, height, hfov)
    n_frames = int(round(seconds * fps))
    if max_frames:
        n_frames = min(n_frames, max_frames)
    rng = np.random.default_rng(seed + 1)

    video_path = os.path.join(out_dir, "flight.mp4")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("cv2.VideoWriter could not open mp4v output")

    hd = math.radians(heading)
    along = np.array([math.sin(hd), math.cos(hd), 0.0])
    left = np.array([-along[1], along[0], 0.0])
    start = -along * (speed * seconds / 2.0) + left * lateral_offset
    lat0, lon0, h0 = origin
    diag = math.hypot(width, height)
    focal_eq = K[0, 0] * 43.2666 / diag  # 35 mm equivalent focal length

    frames, srt_blocks = [], []
    for i in range(n_frames):
        t = i / fps
        s = speed * t
        C = start + along * s + left * (curve * (s ** 2) / 200.0)
        C = C.copy()
        C[2] = altitude  # constant altitude above the ENU origin plane
        yaw = heading + math.degrees(math.atan2(curve * s / 100.0, 1.0)) if curve else heading
        R_wc = camera_to_world_rotation(yaw, pitch)
        img, _, _ = renderer.render(K, R_wc, C, width, height)
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        noisy = C + np.array([rng.normal(0, gps_sigma_h), rng.normal(0, gps_sigma_h), rng.normal(0, gps_sigma_v)])
        lat, lon, alt = pymap3d.enu2geodetic(noisy[0], noisy[1], noisy[2], lat0, lon0, h0)
        rel_alt = noisy[2]
        frames.append({"index": i, "t": t, "R_wc": R_wc.tolist(), "C_enu": C.tolist(),
                       "yaw_deg": yaw, "pitch_deg": pitch})
        srt_blocks.append(
            f"{i + 1}\n{_srt_time(t)} --> {_srt_time(t + 1.0 / fps)}\n"
            f"<font size=\"28\">FrameCnt: {i + 1}, DiffTime: {int(1000 / fps)}ms\n"
            f"[iso: 100] [shutter: 1/1000.0] [fnum: 2.8] [ev: 0] [focal_len: {focal_eq:.2f}] "
            f"[latitude: {lat:.7f}] [longitude: {lon:.7f}] [rel_alt: {rel_alt:.3f} abs_alt: {alt:.3f}] "
            f"[gb_yaw: {((yaw + 180) % 360) - 180:.1f} gb_pitch: {pitch:.1f} gb_roll: 0.0]</font>\n"
        )
        if (i + 1) % 20 == 0 or i == n_frames - 1:
            print(f"[synth3d] rendered {i + 1}/{n_frames}")
    writer.release()

    with open(os.path.join(out_dir, "flight.srt"), "w", encoding="utf-8") as f:
        f.write("\n".join(srt_blocks))

    o3d.io.write_triangle_mesh(os.path.join(out_dir, "scene.ply"), mesh)
    np.save(os.path.join(out_dir, "scene_labels.npy"), labels)
    gt = {
        "description": "synth3d single-pass flight; poses are exact, SRT GPS has the listed noise",
        "origin": {"lat": lat0, "lon": lon0, "alt": h0},
        "width": width, "height": height, "fps": fps,
        "K": K.tolist(), "hfov_deg": hfov, "focal_len_35mm": focal_eq,
        "gps_sigma_h": gps_sigma_h, "gps_sigma_v": gps_sigma_v,
        "altitude_m": altitude, "pitch_deg": pitch, "heading_deg": heading,
        "frames": frames,
        "buildings": gt_buildings,
        "labels": LABELS,
    }
    with open(os.path.join(out_dir, "gt.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=1)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(
            "# synth3d flight\n\nRendered by `tools/synth3d.py`. Real 3D geometry (terrain, buildings, trees), "
            "exact poses in `gt.json`, GPS noise sigma "
            f"{gps_sigma_h} m horizontal / {gps_sigma_v} m vertical in `flight.srt`.\n"
        )
    return gt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=16.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--altitude", type=float, default=60.0)
    ap.add_argument("--speed", type=float, default=8.0)
    ap.add_argument("--pitch", type=float, default=-60.0)
    ap.add_argument("--heading", type=float, default=90.0)
    ap.add_argument("--lateral-offset", type=float, default=-5.0)
    ap.add_argument("--gps-sigma-h", type=float, default=1.0)
    ap.add_argument("--gps-sigma-v", type=float, default=1.5)
    ap.add_argument("--curve", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    fly(args.out, seconds=args.seconds, fps=args.fps, width=args.width, height=args.height,
        altitude=args.altitude, speed=args.speed, pitch=args.pitch, heading=args.heading,
        lateral_offset=args.lateral_offset, gps_sigma_h=args.gps_sigma_h,
        gps_sigma_v=args.gps_sigma_v, curve=args.curve, seed=args.seed)


if __name__ == "__main__":
    main()
