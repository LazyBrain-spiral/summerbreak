"""Tier-1 completion of unobserved building surfaces by footprint extrusion (v3).

A single pass sees roofs and at most one or two facades. Buildings are,
to first order, roof surfaces over vertical walls standing on the terrain,
so the unseen walls can be inferred deterministically and metrically:

  1. nDSM = DSM - DTM; candidate cells have nDSM >= min_height
  2. connected components; reject small ones and rough ones (trees)
  3. footprint polygon = simplified outer contour, in UTM
  4. roof = constrained triangulation of the footprint, midpoint-subdivided,
     heights sampled from a roof-only DSM (keeps gables and slopes)
  5. walls = one vertical quad per roof boundary edge down to the DTM
     (shares roof boundary vertices, so roof and walls are watertight)

Every face carries a provenance label:
  observed  roof faces (from the reconstructed DSM)
  inferred  walls and any roof face whose cell was hole-filled rather than observed
The measurement report uses the same footprint and DSM/DTM statistics.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage

from src.products.geo import LocalGeo

PROVENANCE = {"observed": 0, "inferred": 1, "generated": 2}
PROVENANCE_COLOR = {0: (0.85, 0.55, 0.30), 1: (0.35, 0.55, 0.90), 2: (0.80, 0.30, 0.80)}


def roughness(dsm: np.ndarray, gsd: float) -> np.ndarray:
    """Deviation from a median-filtered surface, lightly smoothed.

    A median filter reproduces planes (flat or sloped roofs) and straight step
    edges (walls), so buildings score ~0 right up to their outline. Tree crowns
    are bumpy and curved, so they score high. A Gaussian residual would also
    flag a ~3 m band along every roof edge and erode footprints.
    """
    k = max(3, int(round(1.5 / gsd)) | 1)
    med = ndimage.median_filter(dsm, size=k)
    return ndimage.gaussian_filter(np.abs(dsm - med), max(1.0, 0.25 / gsd))


def curvature(dsm: np.ndarray, gsd: float) -> np.ndarray:
    """|Laplacian| (1/m) of the median-filtered DSM at a 0.5 m scale.

    Roof planes, flat or sloped, have ~0 curvature away from ridges; tree crowns
    are domes with high curvature everywhere (measured on the synthetic flight:
    buildings 0.07-0.19, trees 1.4-3.0)."""
    k = max(3, int(round(1.5 / gsd)) | 1)
    med = ndimage.median_filter(dsm, size=k)
    return np.abs(ndimage.gaussian_laplace(med, sigma=max(1.0, 0.5 / gsd))) / (gsd * gsd)


def _bilinear(grid: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    from src.core.imgops import sample_points

    return sample_points(grid.astype(np.float32), cols, rows)


def _triangulate_polygon(poly_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    import shapely
    from shapely.geometry import Polygon

    tri = shapely.constrained_delaunay_triangles(Polygon(poly_xy))
    verts: List[Tuple[float, float]] = []
    index: Dict[Tuple[float, float], int] = {}
    faces = []
    for t in tri.geoms:
        coords = list(t.exterior.coords)[:3]
        ids = []
        for c in coords:
            key = (round(c[0], 6), round(c[1], 6))
            if key not in index:
                index[key] = len(verts)
                verts.append(key)
            ids.append(index[key])
        faces.append(ids)
    v = np.array(verts, float)
    f = np.array(faces, np.int64)
    # consistent counter-clockwise orientation (roof normals up)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    f[cross < 0] = f[cross < 0][:, [0, 2, 1]]
    return v, f


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    key = np.sort(edges, axis=1)
    _, inv, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return edges[counts[inv.ravel()] == 1]  # directed as in their (CCW) triangle


def vegetation_index(ortho_path: str, shape) -> Optional[np.ndarray]:
    """Excess Green (2G - R - B) / (R + G + B) of the orthomosaic on the product grid; NaN where unseen."""
    if not ortho_path or not os.path.exists(ortho_path):
        return None
    import rasterio

    with rasterio.open(ortho_path) as src:
        rgb = np.stack([src.read(i) for i in (1, 2, 3)], -1).astype(np.float32)
        alpha = src.read(4) if src.count >= 4 else np.full(rgb.shape[:2], 255, np.uint8)
    if rgb.shape[:2] != tuple(shape):
        return None
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    exg = (2 * g - r - b) / (r + g + b + 1e-6)
    exg[alpha == 0] = np.nan
    return exg


def extrude_buildings(grids_path: str, georef: Dict[str, Any], out_dir: str, cfg: Dict[str, Any],
                      ortho_path: str = None, surface_path: str = None) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    g = np.load(grids_path)
    dsm, dtm, observed = g["dsm_up"], g["dtm_up"], g["observed"]
    xmin, ymax, gsd = float(g["xmin"]), float(g["ymax"]), float(g["gsd"])
    geo = LocalGeo(georef)

    ndsm = dsm - dtm
    # median-filtered nDSM for measurement: removes the edge spikes a fused surface
    # leaves along roof outlines, which otherwise bias high percentiles upward
    k = max(3, int(round(1.25 / gsd)) | 1)
    ndsm_s = ndimage.median_filter(np.nan_to_num(ndsm, nan=0.0), size=k)
    ndsm_s[np.isnan(ndsm)] = np.nan
    rough = roughness(np.nan_to_num(dsm, nan=np.nanmin(dsm)), gsd)
    tall = np.nan_to_num(ndsm, nan=0.0) >= cfg["min_height_m"]
    # Split trees from buildings before labelling: a tree touching a building would
    # otherwise join its component and inflate footprint and height. Smooth cores are
    # labelled, then grown back (edges are rough on every building) inside `tall`.
    curv = curvature(np.nan_to_num(dsm, nan=np.nanmin(dsm)), gsd)
    max_curv = float(cfg.get("max_curvature", 0.6))
    exg_early = vegetation_index(ortho_path, dsm.shape) if cfg.get("vegetation_filter", True) else None
    if exg_early is not None:
        # colour is the stronger tree test; curvature (which small, noisy roofs also fail) only where colour is missing
        green = np.nan_to_num(exg_early, nan=0.0) > cfg.get("max_exg", 0.1)
        flat_ok = np.where(np.isfinite(exg_early), True, curv <= max_curv)
        core = tall & (rough <= cfg["max_roughness_m"]) & ~green & flat_ok
    else:
        green = None
        core = tall & (rough <= cfg["max_roughness_m"]) & (curv <= max_curv)
    core = ndimage.binary_opening(core, iterations=max(1, int(round(0.5 / gsd))))
    labels, n = ndimage.label(core)
    # Roof edges have high curvature too, but only in a thin band; tree crowns are
    # blobs >= 2 m in radius. Opening with a 1 m disk keeps crowns and drops edge bands.
    # Building cores then grow back 1.5 m over everything tall that is not a crown.
    grow = max(1, int(np.ceil(1.5 / gsd)))
    rad = max(1, int(round(1.0 / gsd)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disk = (xx * xx + yy * yy) <= rad * rad
    treeish = (green if green is not None else (curv > 2.0 * max_curv)) | (rough > 3.0 * cfg["max_roughness_m"])
    tree_core = ndimage.binary_opening(tall & treeish, structure=disk)
    if n:
        grown = ndimage.grey_dilation(labels, size=(2 * grow + 1, 2 * grow + 1))
        labels = np.where(labels > 0, labels, np.where(tall & ~tree_core, grown, 0))
    cell_area = gsd * gsd

    all_v, all_f, all_p = [], [], []
    regs = []
    features, rows_out = [], []
    rejected = {"small": 0, "rough": 0, "degenerate": 0, "vegetation": 0}
    exg = vegetation_index(ortho_path, dsm.shape) if cfg.get("vegetation_filter", True) else None
    for lab in range(1, n + 1):
        comp = labels == lab
        area_cells = int(comp.sum())
        if area_cells * cell_area < cfg["min_area_m2"]:
            rejected["small"] += 1
            continue
        if exg is not None:
            v = exg[comp]
            v = v[np.isfinite(v)]
            # tree canopies are green; roofs, even green-painted ones, rarely reach this greenness
            if len(v) and float(np.median(v)) > cfg.get("max_exg", 0.1):
                rejected["vegetation"] += 1
                continue
        med_rough = float(np.median(rough[comp]))
        if med_rough > cfg["max_roughness_m"] or (exg is None and float(np.median(curv[comp])) > max_curv):
            rejected["rough"] += 1
            continue
        # Fused surfaces slope down at roof edges instead of dropping vertically.
        # Cut each building at half its own eave height: the middle of that skirt.
        eave0 = float(np.nanpercentile(ndsm_s[comp], 20))
        cut = comp & (np.nan_to_num(ndsm_s, nan=0.0) >= max(cfg["min_height_m"], 0.5 * eave0))
        sub, n_sub = ndimage.label(cut)
        if n_sub == 0:
            rejected["degenerate"] += 1
            continue
        sizes = ndimage.sum(np.ones_like(sub), sub, index=np.arange(1, n_sub + 1))
        comp = sub == (1 + int(np.argmax(sizes)))
        if comp.sum() * cell_area < cfg["min_area_m2"]:
            rejected["small"] += 1
            continue
        mask8 = comp.astype(np.uint8)
        contours, _ = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnt = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(cnt, cfg["simplify_m"] / gsd, True).reshape(-1, 2).astype(float)
        if len(approx) < 3 or cv2.contourArea(approx.astype(np.float32)) < 4:
            rejected["degenerate"] += 1
            continue
        # contour runs along pixel centres; push outward half a cell to the cell boundary
        cen = approx.mean(axis=0)
        d = approx - cen
        approx = approx + 0.5 * d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        # pixel (col,row) -> UTM
        poly_utm = np.column_stack([xmin + (approx[:, 0] + 0.5) * gsd, ymax - (approx[:, 1] + 0.5) * gsd])
        poly_enu = geo.utm_to_enu(poly_utm)
        from shapely.geometry import Polygon as _Poly

        if not _Poly(poly_enu).is_valid or _Poly(poly_enu).area < 4 * cell_area:
            rejected["degenerate"] += 1
            continue

        v2, f = _triangulate_polygon(poly_enu)
        roof = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.column_stack([v2, np.zeros(len(v2))])),
                                         o3d.utility.Vector3iVector(f.astype(np.int32)))
        if cfg["subdivide"] > 0:
            roof = roof.subdivide_midpoint(number_of_iterations=int(cfg["subdivide"]))
        rv = np.asarray(roof.vertices).copy()
        rf = np.asarray(roof.triangles).astype(np.int64)

        # roof-only DSM so boundary samples do not fall off the eave
        roof_dsm = np.where(comp, dsm, np.nan)
        roof_dsm = _fill(roof_dsm)
        roof_dsm = ndimage.median_filter(roof_dsm, size=3)
        utm = geo.enu_to_utm(rv[:, :2])
        cols = (utm[:, 0] - xmin) / gsd - 0.5
        rows = (ymax - utm[:, 1]) / gsd - 0.5
        rv[:, 2] = _bilinear(roof_dsm, rows, cols)
        ground_z = _bilinear(np.nan_to_num(dtm, nan=float(np.nanmedian(dtm))), rows, cols)

        obs_face = _face_observed(rv, rf, observed, geo, xmin, ymax, gsd)
        be = _boundary_edges(rf)
        nv = len(rv)
        base = np.column_stack([rv[:, :2], ground_z])
        verts = np.vstack([rv, base])
        wall_faces = []
        for a, b in be:
            # roof edge a->b is CCW seen from above, so the outward wall is (a, b, b', a') reversed
            wall_faces.append([b, a, a + nv])
            wall_faces.append([b, a + nv, b + nv])
        wall_faces = np.array(wall_faces, np.int64)
        faces = np.vstack([rf, wall_faces])
        prov = np.concatenate([np.where(obs_face, PROVENANCE["observed"], PROVENANCE["inferred"]),
                               np.full(len(wall_faces), PROVENANCE["inferred"])])
        off = sum(len(x) for x in all_v)
        all_v.append(verts)
        all_f.append(faces + off)
        all_p.append(prov)

        h = ndsm_s[comp]
        h = h[np.isfinite(h)]
        from shapely.geometry import Polygon

        poly = Polygon(poly_utm)
        rec = {
            "id": len(features) + 1,
            "footprint_area_m2": round(float(poly.area), 2),
            "perimeter_m": round(float(poly.length), 2),
            "eave_height_m": round(float(np.percentile(h, 20)), 2),
            "ridge_height_m": round(float(np.percentile(h, 95)), 2),
            "mean_height_m": round(float(np.mean(h)), 2),
            "volume_m3": round(float(np.sum(np.clip(h, 0, None)) * cell_area), 1),
            "ground_height_m": round(float(np.nanmedian(geo.enu_up_to_height(dtm[comp]))), 2),
            "roof_observed_frac": round(float(np.mean(observed[comp])), 3),
            "roughness_m": round(med_rough, 3),
            "centroid_enu": [round(float(x), 2) for x in poly_enu.mean(axis=0)],
            "num_vertices": int(len(poly_utm)),
        }
        ring_for_gis = poly_utm
        if cfg.get("regularize", True):
            from src.completion.regularize import regularize_building

            try:
                reg = regularize_building(comp, dsm, dtm, geo, xmin, ymax, gsd)
            except Exception as exc:  # keep the raw model for this building
                reg = None
                rec["regularize_error"] = str(exc)[:200]
            if reg is not None:
                rec.update({
                    "raw_footprint_area_m2": rec["footprint_area_m2"], "raw_ridge_height_m": rec["ridge_height_m"],
                    "raw_eave_height_m": rec["eave_height_m"],
                    "footprint_area_m2": reg["footprint_area_m2"], "perimeter_m": reg["perimeter_m"],
                    "eave_height_m": reg["eave_height_m"], "ridge_height_m": reg["ridge_height_m"],
                    "shape": reg["shape"], "footprint_iou": reg["footprint_iou"], "roof_type": reg["roof_type"],
                    "roof_residual_m": reg["roof_residual_m"],
                    "centroid_enu": [round(float(x), 2) for x in reg["ring_enu"].mean(axis=0)],
                })
                reg["id"] = rec["id"]
                regs.append(reg)
                ring_for_gis = geo.enu_to_utm(reg["ring_enu"])
        rows_out.append(rec)
        lonlat = geo.utm_to_lonlat(np.vstack([ring_for_gis, ring_for_gis[:1]]))
        features.append({"type": "Feature", "properties": rec,
                         "geometry": {"type": "Polygon", "coordinates": [lonlat.tolist()]}})

    result: Dict[str, Any] = {"num_buildings": len(rows_out), "rejected_components": rejected,
                              "buildings": rows_out}
    gj = {"type": "FeatureCollection", "name": "buildings", "features": features}
    if geo.epsg:
        gj["crs"] = {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}}
    else:
        gj["note"] = "NOT georeferenced: coordinates are local metres, scale from an assumed camera height"
    with open(os.path.join(out_dir, "buildings.geojson"), "w", encoding="utf-8") as fh:
        json.dump(gj, fh, indent=1)
    if rows_out:
        V = np.vstack(all_v)
        F = np.vstack(all_f)
        P = np.concatenate(all_p)
        mesh = _face_colored_mesh(V, F, P)
        path = os.path.join(out_dir, "buildings_lod2.ply")
        o3d.io.write_triangle_mesh(path, mesh)
        np.save(os.path.join(out_dir, "buildings_face_provenance.npy"), P.astype(np.uint8))
        result["mesh"] = path
        result["faces"] = int(len(F))
        result["face_provenance_counts"] = {k: int((P == v).sum()) for k, v in PROVENANCE.items()}
    if regs:
        from src.completion.regularized_export import export_regularized

        result["regularized"] = export_regularized(regs, geo, xmin, ymax, gsd, dsm.shape, out_dir,
                                                   ortho_path, surface_path, cfg)
        meta = result["regularized"].pop("meta", {})
        for rec in rows_out:
            m = meta.get(str(rec["id"]))
            if m:
                rec.update({k: v for k, v in m.items() if k not in ("facade_thumb", "doors")})
                rec["doors_count"] = len(m.get("doors", []))
        result["roof_types"] = {k: sum(1 for r in regs if r["roof_type"] == k) for k in
                                sorted({r["roof_type"] for r in regs})}
        result["footprint_shapes"] = {k: sum(1 for r in regs if r["shape"] == k) for k in
                                      sorted({r["shape"] for r in regs})}
    return result


def _fill(a: np.ndarray) -> np.ndarray:
    nan = np.isnan(a)
    if not nan.any():
        return a
    _, (ri, ci) = ndimage.distance_transform_edt(nan, return_indices=True)
    return a[ri, ci]


def _face_observed(v, f, observed, geo, xmin, ymax, gsd) -> np.ndarray:
    cen = v[f].mean(axis=1)
    utm = geo.enu_to_utm(cen[:, :2])
    c = np.clip(((utm[:, 0] - xmin) / gsd).astype(int), 0, observed.shape[1] - 1)
    r = np.clip(((ymax - utm[:, 1]) / gsd).astype(int), 0, observed.shape[0] - 1)
    return observed[r, c]


def _face_colored_mesh(V: np.ndarray, F: np.ndarray, P: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Duplicate vertices per face so provenance can be shown as flat vertex colours."""
    tri_v = V[F].reshape(-1, 3)
    tri_f = np.arange(len(tri_v)).reshape(-1, 3)
    col = np.repeat(np.array([PROVENANCE_COLOR[int(p)] for p in P]), 3, axis=0)
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tri_v), o3d.utility.Vector3iVector(tri_f.astype(np.int32)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(col)
    mesh.compute_vertex_normals()
    return mesh
