"""Shape-regularised building models (v3, completion tier 1b).

Point maps from a single pass bend walls and round corners. Buildings are
man-made: footprints are rectangles, triangles or right-angled polygons, walls
are vertical, roofs are a few planes. This module replaces each detected
building with the simplest such primitive that explains the reconstruction:

  footprint  candidates: rotated rectangle, triangle, right-angled (rectilinear)
             polygon on the dominant orientation, simplified polygon. The one
             with the fewest vertices whose IoU with the detected mask is high
             enough wins.
  roof       candidates: flat, shed (one plane), gable along either footprint
             axis (two planes meeting at a ridge). Robust least squares on the
             DSM inside the footprint; a small penalty per parameter picks the
             simplest adequate model.
  walls      vertical, from the roof edge down to one base height per building.
  mesh       roof triangles take UVs into the orthomosaic (real roof texture);
             wall vertices take colours from the nearest observed surface point,
             or a neutral facade colour where the wall was never seen.

The fused TSDF surface is kept for terrain, trees and everything else: faces
inside a building footprint and above its base are cut out and replaced by the
primitive, giving a hybrid model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

FACADE_COLOR = np.array([214, 206, 192], np.uint8)


# ---------------------------------------------------------------------------
# footprint regularisation (pixel coordinates of the product grid)
# ---------------------------------------------------------------------------

def _poly_mask(poly_px: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(poly_px).astype(np.int32)], 1)
    return m.astype(bool)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def dominant_angle(contour: np.ndarray) -> float:
    """Length-weighted dominant edge direction modulo 90 degrees (radians)."""
    pts = contour.reshape(-1, 2).astype(float)
    approx = cv2.approxPolyDP(pts.astype(np.float32), 1.5, True).reshape(-1, 2).astype(float)
    if len(approx) < 3:
        approx = pts
    d = np.roll(approx, -1, axis=0) - approx
    L = np.hypot(d[:, 0], d[:, 1])
    ang = np.arctan2(d[:, 1], d[:, 0])
    z = np.sum(L * np.exp(4j * ang))
    return float(np.angle(z) / 4.0)


def rectilinear_polygon(mask: np.ndarray, angle: float, step_px: float,
                        min_edge_px: Optional[float] = None) -> Optional[np.ndarray]:
    """Right-angled outline on the dominant orientation with steps of ~step_px."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 10:
        return None
    cx, cy = xs.mean(), ys.mean()
    pad = int(max(mask.shape) * 0.5) + 4
    big = np.zeros((mask.shape[0] + 2 * pad, mask.shape[1] + 2 * pad), np.uint8)
    big[pad:pad + mask.shape[0], pad:pad + mask.shape[1]] = mask
    c = (cx + pad, cy + pad)
    M = cv2.getRotationMatrix2D(c, math.degrees(angle), 1.0)
    rot = cv2.warpAffine(big, M, big.shape[::-1], flags=cv2.INTER_NEAREST)
    s = max(1, int(round(step_px)))
    h, w = rot.shape
    hc, wc = -(-h // s), -(-w // s)
    padded = np.zeros((hc * s, wc * s), np.uint8)
    padded[:h, :w] = rot
    coarse = padded.reshape(hc, s, wc, s).mean(axis=(1, 3)) >= 0.5
    sq = np.ones((3, 3), bool)  # a square kernel keeps right-angle corners (a cross chamfers them)
    coarse = ndimage.binary_opening(ndimage.binary_closing(coarse, structure=sq), structure=sq)
    lab, n = ndimage.label(coarse)
    if n == 0:
        return None
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = (lab == 1 + int(np.argmax(sizes))).astype(np.uint8)
    # trace on the upsampled coarse grid: edges land exactly on cell boundaries,
    # concave corners included
    up = np.kron(keep, np.ones((s, s), np.uint8))
    cnts, _ = cv2.findContours(up, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    poly_rot = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float) + 0.5
    # snap to the cell lattice: removes the 1-px diagonal chamfers the tracer leaves at
    # concave corners, so every edge is exactly horizontal or vertical
    poly_rot = np.round(poly_rot / s) * s
    Q = []
    for p_ in poly_rot:
        if not Q or np.abs(p_ - Q[-1]).max() > 1e-9:
            Q.append(p_)
    if len(Q) > 1 and np.abs(Q[0] - Q[-1]).max() < 1e-9:
        Q.pop()
    poly_rot = np.array(Q)
    if min_edge_px:
        poly_rot = simplify_orthogonal(poly_rot, min_edge_px)
    if len(poly_rot) < 4:
        return None
    Minv = cv2.invertAffineTransform(M)
    pts = np.column_stack([poly_rot, np.ones(len(poly_rot))]) @ Minv.T
    return pts - pad


def simplify_orthogonal(poly: np.ndarray, min_edge_px: float, max_iter: int = 200) -> np.ndarray:
    """Remove short steps from a right-angled outline.

    A short edge between two edges running the same way is a step (Z shape): drop
    it by moving the shorter neighbour so the two neighbours line up. Collinear
    vertices are merged afterwards."""
    P = [np.asarray(p, float) for p in poly]

    def merge_collinear(P):
        out = []
        n = len(P)
        for i in range(n):
            a, b, c = P[i - 1], P[i], P[(i + 1) % n]
            cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if abs(cr) > 1e-6 * max(1.0, np.linalg.norm(b - a) * np.linalg.norm(c - b)):
                out.append(b)
        return out

    P = merge_collinear(P)
    for _ in range(max_iter):
        n = len(P)
        if n <= 4:
            break
        L = [np.linalg.norm(P[(i + 1) % n] - P[i]) for i in range(n)]
        i = int(np.argmin(L))
        if L[i] >= min_edge_px:
            break
        # edge i runs P[i] -> P[i+1]; neighbours are edges i-1 and i+1
        a, b = P[i], P[(i + 1) % n]
        d = b - a
        if L[(i - 1) % n] <= L[(i + 1) % n]:
            P[(i - 1) % n] = P[(i - 1) % n] + d   # shift the previous edge onto the next one
            P[i] = P[i] + d
        else:
            P[(i + 1) % n] = P[(i + 1) % n] - d
            P[(i + 2) % n] = P[(i + 2) % n] - d
        P = merge_collinear([p for k, p in enumerate(P)])
        # drop duplicate points
        Q = []
        for p in P:
            if not Q or np.linalg.norm(p - Q[-1]) > 1e-6:
                Q.append(p)
        if len(Q) > 1 and np.linalg.norm(Q[0] - Q[-1]) < 1e-6:
            Q.pop()
        P = merge_collinear(Q)
    return np.array(P)


def snap_polygon(poly: np.ndarray, angle: float, tol_deg: float = 15.0, min_edge_px: float = 2.0) -> np.ndarray:
    """Turn edges within tol of the dominant axis (or its perpendicular) onto it and
    rebuild corners as line intersections: right angles where the building has them,
    true diagonals kept where it does not."""
    P = np.asarray(poly, float)
    n = len(P)
    if n < 3:
        return P
    tol = math.radians(tol_deg)
    lines = []
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        d = b - a
        th = math.atan2(d[1], d[0])
        rel = (th - angle + math.pi / 4) % (math.pi / 2) - math.pi / 4
        if abs(rel) < tol:
            th -= rel
        u = np.array([math.cos(th), math.sin(th)])
        lines.append(((a + b) / 2.0, u))
    out = []
    for i in range(n):
        (m0, u0), (m1, u1) = lines[i - 1], lines[i]
        A = np.column_stack([u0, -u1])
        if abs(np.linalg.det(A)) < 1e-3:  # parallel: project the original corner onto the current line
            out.append(m1 + ((P[i] - m1) @ u1) * u1)
        else:
            tt = np.linalg.solve(A, m1 - m0)
            out.append(m0 + tt[0] * u0)
    Q = np.array(out)
    # drop degenerate short edges created by snapping
    changed = True
    while changed and len(Q) > 3:
        changed = False
        L = np.linalg.norm(np.roll(Q, -1, axis=0) - Q, axis=1)
        i = int(np.argmin(L))
        if L[i] < min_edge_px:
            Q = np.delete(Q, (i + 1) % len(Q), axis=0)
            changed = True
    return Q


def regularize_footprint(mask: np.ndarray, gsd: float, min_iou: float = 0.82) -> Dict[str, Any]:
    """Pick the simplest footprint shape that explains a building mask (pixel coords)."""
    m8 = mask.astype(np.uint8)
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea)
    angle = dominant_angle(cnt)
    cands = []
    rect = cv2.boxPoints(cv2.minAreaRect(cnt.astype(np.float32))).astype(float)
    cands.append(("rectangle", rect, 4))
    ok, tri = cv2.minEnclosingTriangle(cnt.astype(np.float32))
    if ok and tri is not None:
        cands.append(("triangle", tri.reshape(-1, 2).astype(float), 3))
    rl = rectilinear_polygon(mask, angle, step_px=max(2.0, 1.0 / gsd), min_edge_px=1.2 / gsd)
    if rl is not None and len(rl) >= 4:
        cands.append(("rectilinear", rl, len(rl)))
    simp = cv2.approxPolyDP(cnt.astype(np.float32), max(1.0, 0.75 / gsd), True).reshape(-1, 2).astype(float)
    if len(simp) >= 3:
        snapped = snap_polygon(simp, angle, min_edge_px=max(2.0, 0.8 / gsd))
        from shapely.geometry import Polygon as _P

        if len(snapped) >= 3 and _P(snapped).is_valid:
            cands.append(("snapped polygon", snapped, len(snapped) + 1))
        cands.append(("polygon", simp, len(simp) + 3))  # least regular: last resort
    scored = []
    for name, poly, nv in cands:
        iou = _iou(_poly_mask(poly, mask.shape), mask)
        scored.append({"shape": name, "poly": poly, "iou": iou, "nv": nv,
                       "score": iou - 0.006 * max(0, nv - 4)})
    good = [c for c in scored if c["iou"] >= min_iou]
    best = max(good or scored, key=lambda c: c["score"])
    # regular shapes beat a free polygon unless the polygon is clearly better
    if best["shape"] == "polygon":
        for pref in ("rectilinear", "snapped polygon"):
            c = next((x for x in scored if x["shape"] == pref), None)
            if c and c["iou"] >= min_iou and c["iou"] >= best["iou"] - 0.06:
                best = c
                break
    # prefer a plain rectangle / triangle when it is nearly as good as anything else
    for simple in ("rectangle", "triangle"):
        c = next((x for x in scored if x["shape"] == simple), None)
        if c and c["iou"] >= min_iou and c["iou"] >= best["iou"] - 0.06:
            best = c
            break
    best["angle"] = angle
    best["candidates"] = {c["shape"]: round(c["iou"], 3) for c in scored}
    return best


# ---------------------------------------------------------------------------
# roof fitting (local metric frame of the building)
# ---------------------------------------------------------------------------

@dataclass
class Roof:
    kind: str
    params: Dict[str, float]
    median_abs_residual: float
    frame_origin: np.ndarray = field(default_factory=lambda: np.zeros(2))
    frame_angle: float = 0.0

    def to_local(self, xy: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.frame_angle), math.sin(self.frame_angle)
        d = xy - self.frame_origin
        return np.column_stack([d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c])

    def height(self, xy: np.ndarray) -> np.ndarray:
        uv = self.to_local(xy)
        u, v = uv[:, 0], uv[:, 1]
        p = self.params
        if self.kind == "flat":
            return np.full(len(xy), p["c"])
        if self.kind == "shed":
            return p["a"] * u + p["b"] * v + p["c"]
        if self.kind == "gable_u":  # ridge parallel to u, at v = v0
            return p["zr"] - p["s"] * np.abs(v - p["v0"])
        if self.kind == "gable_v":
            return p["zr"] - p["s"] * np.abs(u - p["u0"])
        raise ValueError(self.kind)

    def ridge_line(self) -> Optional[Tuple[str, float]]:
        if self.kind == "gable_u":
            return ("v", self.params["v0"])
        if self.kind == "gable_v":
            return ("u", self.params["u0"])
        return None


def _robust_lstsq(A: np.ndarray, z: np.ndarray, iters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    w = np.ones(len(z))
    x = np.linalg.lstsq(A, z, rcond=None)[0]
    for _ in range(iters):
        r = z - A @ x
        s = 1.4826 * np.median(np.abs(r)) + 1e-6
        w = 1.0 / np.maximum(1.0, np.abs(r) / (1.5 * s))
        sw = np.sqrt(w)
        x = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)[0]
    return x, z - A @ x


def fit_roof(xy: np.ndarray, z: np.ndarray, origin: np.ndarray, angle: float, allow_gable: bool = True,
             penalty: float = 0.04) -> Roof:
    base = Roof("flat", {}, 0.0, origin, angle)
    uv = base.to_local(xy)
    u, v = uv[:, 0], uv[:, 1]
    fits = []
    x, r = _robust_lstsq(np.ones((len(z), 1)), z)
    fits.append(Roof("flat", {"c": float(x[0])}, float(np.median(np.abs(r))), origin, angle))
    x, r = _robust_lstsq(np.column_stack([u, v, np.ones(len(z))]), z)
    fits.append(Roof("shed", {"a": float(x[0]), "b": float(x[1]), "c": float(x[2])},
                     float(np.median(np.abs(r))), origin, angle))
    if allow_gable:
        for kind, coord in (("gable_u", v), ("gable_v", u)):
            lo, hi = np.percentile(coord, 10), np.percentile(coord, 90)
            best = None
            for c0 in np.linspace(lo, hi, 41):
                A = np.column_stack([np.ones(len(z)), -np.abs(coord - c0)])
                x, r = _robust_lstsq(A, z, iters=1)
                if x[1] <= 0.05:  # needs a real slope down from the ridge
                    continue
                mr = float(np.median(np.abs(r)))
                if best is None or mr < best[0]:
                    best = (mr, c0, x)
            if best is not None:
                mr, c0, x = best
                A = np.column_stack([np.ones(len(z)), -np.abs(coord - c0)])
                x, r = _robust_lstsq(A, z)
                key = "v0" if kind == "gable_u" else "u0"
                fits.append(Roof(kind, {"zr": float(x[0]), "s": float(x[1]), key: float(c0)},
                                 float(np.median(np.abs(r))), origin, angle))
    nparams = {"flat": 1, "shed": 3, "gable_u": 3, "gable_v": 3}
    return min(fits, key=lambda f: f.median_abs_residual + penalty * nparams[f.kind])


# ---------------------------------------------------------------------------
# primitive mesh
# ---------------------------------------------------------------------------

def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    return poly if area > 0 else poly[::-1]


def _insert_ridge_points(ring: np.ndarray, roof: Roof) -> np.ndarray:
    """Add vertices where footprint edges cross the ridge line, so walls follow the roof."""
    rl = roof.ridge_line()
    if rl is None:
        return ring
    axis, c0 = rl
    uv = roof.to_local(ring)
    k = 1 if axis == "v" else 0
    out = []
    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        out.append(ring[i])
        a, b = uv[i, k] - c0, uv[j, k] - c0
        if a * b < 0:
            t = a / (a - b)
            out.append(ring[i] + t * (ring[j] - ring[i]))
    return np.array(out)


def _triangulate(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import shapely
    from shapely.geometry import Polygon

    P = Polygon(poly)
    if not P.is_valid:
        P = P.buffer(0)
    tri = shapely.constrained_delaunay_triangles(P)
    verts, faces, index = [], [], {}
    for t in tri.geoms:
        ids = []
        for c in list(t.exterior.coords)[:3]:
            key = (round(c[0], 5), round(c[1], 5))
            if key not in index:
                index[key] = len(verts)
                verts.append(key)
            ids.append(index[key])
        faces.append(ids)
    v = np.array(verts, float)
    f = np.array(faces, np.int64).reshape(-1, 3)
    if len(f):
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        cr = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        f[cr < 0] = f[cr < 0][:, [0, 2, 1]]
    return v, f


def roof_mesh(ring: np.ndarray, roof: Roof) -> Tuple[np.ndarray, np.ndarray]:
    """Roof triangles; gable roofs are split along the ridge so each face is planar."""
    from shapely.geometry import LineString, Polygon
    from shapely.ops import split

    rl = roof.ridge_line()
    pieces = [ring]
    if rl is not None:
        axis, c0 = rl
        uv = roof.to_local(ring)
        span = float(np.ptp(uv, axis=0).max()) * 4 + 10
        c, s = math.cos(roof.frame_angle), math.sin(roof.frame_angle)
        if axis == "v":
            ends_uv = np.array([[-span, c0], [span, c0]])
        else:
            ends_uv = np.array([[c0, -span], [c0, span]])
        ends = np.column_stack([ends_uv[:, 0] * c - ends_uv[:, 1] * s, ends_uv[:, 0] * s + ends_uv[:, 1] * c]) + roof.frame_origin
        try:
            parts = split(Polygon(ring), LineString(ends))
            pieces = [np.array(g.exterior.coords)[:-1] for g in parts.geoms if g.area > 1e-6]
        except Exception:
            pieces = [ring]
    V, F, G = [], [], []
    for gi, piece in enumerate(pieces):
        v2, f = _triangulate(piece)
        if not len(f):
            continue
        off = sum(len(x) for x in V)
        V.append(v2)
        F.append(f + off)
        G.append(np.full(len(f), gi, np.int32))
    if not V:
        roof_mesh.last_groups = np.empty(0, np.int32)
        return np.empty((0, 3)), np.empty((0, 3), np.int64)
    v2 = np.vstack(V)
    roof_mesh.last_groups = np.concatenate(G)
    return np.column_stack([v2, roof.height(v2)]), np.vstack(F)


def wall_mesh(ring: np.ndarray, roof: Roof, base_z: float, cell_m: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:
    """Vertical walls, subdivided into ~cell_m quads so they can carry observed colours."""
    V, F, G = [], [], []
    n = len(ring)
    tops = roof.height(ring)
    for i in range(n):
        j = (i + 1) % n
        a, b = ring[i], ring[j]
        L = float(np.linalg.norm(b - a))
        if L < 1e-6:
            continue
        nu = max(1, int(math.ceil(L / cell_m)))
        ts = np.linspace(0, 1, nu + 1)
        pts = a + ts[:, None] * (b - a)
        top = tops[i] + ts * (tops[j] - tops[i])
        hmax = float(max(top.max() - base_z, 0.1))
        nv = max(1, int(math.ceil(hmax / cell_m)))
        fr = np.linspace(0, 1, nv + 1)
        grid = []
        for k in range(nu + 1):
            zs = base_z + fr * (top[k] - base_z)
            grid.append(np.column_stack([np.repeat(pts[k][None], nv + 1, 0), zs]))
        grid = np.stack(grid)  # (nu+1, nv+1, 3)
        off = sum(len(x) for x in V)
        V.append(grid.reshape(-1, 3))
        idx = np.arange((nu + 1) * (nv + 1)).reshape(nu + 1, nv + 1) + off
        # ring is CCW from above, so the outward normal is to the right of a->b
        a0, a1 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
        b0, b1 = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
        F.append(np.stack([a0, b0, b1], 1))
        F.append(np.stack([a0, b1, a1], 1))
        G.append(np.full(2 * len(a0), i, np.int32))
    if not V:
        return np.empty((0, 3)), np.empty((0, 3), np.int64)
    wall_mesh.last_groups = np.concatenate(G) if G else np.empty(0, np.int32)
    return np.vstack(V), np.vstack(F)


# ---------------------------------------------------------------------------
# per-building driver
# ---------------------------------------------------------------------------

def regularize_building(comp: np.ndarray, dsm: np.ndarray, dtm: np.ndarray, geo, xmin: float, ymax: float,
                        gsd: float) -> Optional[Dict[str, Any]]:
    fp = regularize_footprint(comp, gsd)
    poly_px = fp["poly"]
    # pixel (col,row) -> product grid (UTM or local) -> ENU
    poly_grid = np.column_stack([xmin + (poly_px[:, 0] + 0.5) * gsd, ymax - (poly_px[:, 1] + 0.5) * gsd])
    ring = _ensure_ccw(geo.utm_to_enu(poly_grid))
    # roof samples: DSM cells inside the footprint, eroded to stay off wall-top spill
    inside = _poly_mask(poly_px, comp.shape)
    core = ndimage.binary_erosion(inside, iterations=max(1, int(round(0.6 / gsd))))
    if core.sum() < 20:
        core = inside
    rows, cols = np.nonzero(core & np.isfinite(dsm))
    if len(rows) < 20:
        return None
    step = max(1, len(rows) // 6000)
    rows, cols = rows[::step], cols[::step]
    xy_grid = np.column_stack([xmin + (cols + 0.5) * gsd, ymax - (rows + 0.5) * gsd])
    xy = geo.utm_to_enu(xy_grid)
    z = dsm[rows, cols].astype(float)
    # frame: origin at the footprint centroid, u along the dominant (long) axis
    origin = ring.mean(axis=0)
    ang = fp["angle"]
    if fp["shape"] == "rectangle":
        e = ring[1] - ring[0]
        e2 = ring[2] - ring[1]
        long_e = e if np.linalg.norm(e) >= np.linalg.norm(e2) else e2
        ang = math.atan2(long_e[1], long_e[0])
    roof = fit_roof(xy, z, origin, ang, allow_gable=fp["shape"] in ("rectangle", "rectilinear", "polygon"))
    # base: low percentile of the terrain along and just outside the outline
    ring_mask = ndimage.binary_dilation(inside, iterations=max(1, int(round(1.0 / gsd)))) & ~inside
    tz = dtm[ring_mask & np.isfinite(dtm)]
    if not len(tz):
        tz = dtm[inside & np.isfinite(dtm)]
    # walls reach the low side of sloped ground (no floating gaps); heights are reported
    # from the median surrounding ground, the usual definition for building height
    base_z = float(np.percentile(tz, 10)) if len(tz) else float(np.nanmin(dsm))
    ground_z = float(np.median(tz)) if len(tz) else base_z
    ring2 = _insert_ridge_points(ring, roof)
    tops = roof.height(ring2)
    if float(np.min(tops)) <= base_z + 0.5:
        # roof model dips to the ground (bad fit); fall back to a flat roof at the median height
        roof = Roof("flat", {"c": float(np.median(z))}, float(np.median(np.abs(z - np.median(z)))), origin, ang)
        ring2 = ring
    rv, rf = roof_mesh(ring2, roof)
    rg = roof_mesh.last_groups
    wv, wf = wall_mesh(ring2, roof, base_z, cell_m=2.0)
    wg = wall_mesh.last_groups
    from shapely.geometry import Polygon

    P = Polygon(ring)
    ridge = float(roof.height(ring2).max() if roof.kind == "flat" else max(roof.height(ring2).max(),
                  roof.params.get("zr", -np.inf)))
    return {
        "shape": fp["shape"], "footprint_iou": round(fp["iou"], 3), "candidates": fp["candidates"],
        "roof_type": roof.kind, "roof_residual_m": round(roof.median_abs_residual, 3),
        "ring_enu": ring, "ring_px": poly_px, "base_z": base_z,
        "eave_height_m": round(float(roof.height(ring2).min() - ground_z), 2),
        "ridge_height_m": round(ridge - ground_z, 2), "ground_z": ground_z,
        "footprint_area_m2": round(float(P.area), 2), "perimeter_m": round(float(P.length), 2),
        "roof_v": rv, "roof_f": rf, "wall_v": wv, "wall_f": wf, "roof_groups": rg, "wall_groups": wg,
        "ring2": ring2, "roof_model": roof,
        "roof_pitch_deg": round(float(math.degrees(math.atan(roof.params.get("s", 0.0)))) if roof.kind.startswith("gable")
                                else float(math.degrees(math.atan(math.hypot(roof.params.get("a", 0.0), roof.params.get("b", 0.0))))), 1),
        "frame_angle": float(ang),
    }


def wall_colors(wall_v: np.ndarray, tsdf_v: np.ndarray, tsdf_c: np.ndarray, tree, radius: float = 0.9):
    """Colour wall vertices from the nearest observed surface point; neutral where unseen."""
    if tree is None or not len(wall_v):
        return np.tile(FACADE_COLOR, (len(wall_v), 1)), np.zeros(len(wall_v), bool)
    d, idx = tree.query(wall_v, k=1, distance_upper_bound=radius)
    seen = np.isfinite(d)
    col = np.tile(FACADE_COLOR, (len(wall_v), 1)).astype(np.uint8)
    if seen.any():
        col[seen] = (np.clip(tsdf_c[idx[seen]], 0, 1) * 255).astype(np.uint8)
        # unseen parts of a wall take that building's median observed wall colour
        med = np.median(col[seen], axis=0).astype(np.uint8)
        col[~seen] = (0.6 * med + 0.4 * FACADE_COLOR).astype(np.uint8)
    return col, seen


def cut_buildings_from_surface(v: np.ndarray, f: np.ndarray, rings: List[np.ndarray], bases: List[float],
                               buffer_m: float = 0.8, above_m: float = 0.8) -> np.ndarray:
    """Boolean mask of surface faces to KEEP (drops faces that belong to a replaced building)."""
    import shapely
    from shapely.geometry import Polygon

    cen = v[f].mean(axis=1)
    drop = np.zeros(len(f), bool)
    for ring, base in zip(rings, bases):
        P = Polygon(ring).buffer(buffer_m, join_style=2)
        minx, miny, maxx, maxy = P.bounds
        cand = np.nonzero((cen[:, 0] >= minx) & (cen[:, 0] <= maxx) & (cen[:, 1] >= miny) & (cen[:, 1] <= maxy)
                          & (cen[:, 2] > base + above_m))[0]
        if len(cand):
            inside = shapely.contains_xy(P, cen[cand, 0], cen[cand, 1])
            drop[cand[inside]] = True
    return ~drop
