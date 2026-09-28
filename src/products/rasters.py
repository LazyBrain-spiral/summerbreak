"""DSM, DTM, true orthomosaic and LAS export (v3).

DSM   max height per cell of the fused cloud in the UTM grid; small holes filled
DTM   progressive morphological filter (Zhang et al. 2003) on the DSM, ground
      cells interpolated by nearest fill and smoothed
Ortho per cell, the camera with the steepest view whose depth map agrees
      with the DSM point (visibility test) and that has no dynamic-mask hit
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

from src.core.imgops import sample_points
from src.depth.common import load_depth_contract, read_depth
from src.products.geo import LocalGeo
from src.semantics.masks import load_dynamic_mask

NODATA = -9999.0


class Grid:
    def __init__(self, xmin: float, ymax: float, gsd: float, width: int, height: int):
        self.xmin, self.ymax, self.gsd, self.width, self.height = xmin, ymax, gsd, width, height

    @classmethod
    def around(cls, xy: np.ndarray, gsd: float, pad: float = 2.0) -> "Grid":
        lo, hi = xy.min(axis=0) - pad, xy.max(axis=0) + pad
        xmin = np.floor(lo[0] / gsd) * gsd
        ymax = np.ceil(hi[1] / gsd) * gsd
        w = int(np.ceil((hi[0] - xmin) / gsd))
        h = int(np.ceil((ymax - lo[1]) / gsd))
        return cls(xmin, ymax, gsd, w, h)

    def cell(self, xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        col = np.floor((xy[:, 0] - self.xmin) / self.gsd).astype(np.int64)
        row = np.floor((self.ymax - xy[:, 1]) / self.gsd).astype(np.int64)
        return row, col

    def centers(self) -> np.ndarray:
        c = self.xmin + (np.arange(self.width) + 0.5) * self.gsd
        r = self.ymax - (np.arange(self.height) + 0.5) * self.gsd
        cc, rr = np.meshgrid(c, r)
        return np.stack([cc, rr], -1)

    def transform(self):
        from rasterio.transform import from_origin

        return from_origin(self.xmin, self.ymax, self.gsd, self.gsd)


def rasterize_max(xy: np.ndarray, z: np.ndarray, grid: Grid) -> np.ndarray:
    row, col = grid.cell(xy)
    ok = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    flat = row[ok] * grid.width + col[ok]
    out = np.full(grid.height * grid.width, -np.inf)
    np.maximum.at(out, flat, z[ok])
    out = out.reshape(grid.height, grid.width)
    out[~np.isfinite(out)] = np.nan
    return out


def fill_nearest(a: np.ndarray, max_dist_cells: Optional[float] = None) -> np.ndarray:
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return a.copy()
    dist, (ri, ci) = ndimage.distance_transform_edt(nan, return_indices=True)
    out = a[ri, ci]
    if max_dist_cells is not None:
        out[dist > max_dist_cells] = np.nan
    return out


def fill_holes_low(a: np.ndarray, max_dist_cells: float) -> np.ndarray:
    """Grow valid cells into small holes using the LOWEST neighbouring value.

    Holes next to buildings are almost always occlusion shadows on the ground
    behind them, so nearest-value filling would smear roof height outward and
    inflate footprints. Larger holes stay NaN (unobserved)."""
    out = a.copy()
    for _ in range(int(np.ceil(max_dist_cells))):
        nan = np.isnan(out)
        if not nan.any():
            break
        low = ndimage.minimum_filter(np.where(nan, np.inf, out), size=3)
        grow = nan & np.isfinite(low)
        if not grow.any():
            break
        out[grow] = low[grow]
    return out


def progressive_morphological_filter(dsm: np.ndarray, gsd: float, max_window_m: float, slope: float,
                                     dh0: float, dh_max: float) -> np.ndarray:
    """Boolean ground mask."""
    surface = fill_nearest(dsm)
    valid = ~np.isnan(dsm)
    ground = valid.copy()
    k, prev_w = 0, 1
    while True:
        w = 2 * (2 ** k) + 1
        if w * gsd > max_window_m:
            break
        opened = ndimage.grey_opening(surface, size=(w, w))
        dh = min(dh_max, dh0 + slope * (w - prev_w) * gsd)
        ground &= (surface - opened) <= dh
        surface = opened
        prev_w = w
        k += 1
    return ground


def build_dtm(dsm: np.ndarray, ground: np.ndarray, gsd: float) -> np.ndarray:
    g = np.where(ground, dsm, np.nan)
    dtm = fill_nearest(g)
    return ndimage.gaussian_filter(dtm, sigma=max(1.0, 1.0 / gsd))


def true_ortho(dsm_enu_up: np.ndarray, grid: Grid, geo: LocalGeo, depth_root: str,
               masks_dir: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    meta, _ = load_depth_contract(depth_root)
    h, w = dsm_enu_up.shape
    valid = ~np.isnan(dsm_enu_up)
    xy_enu = geo.utm_to_enu(grid.centers()[valid])
    P = np.column_stack([xy_enu, dsm_enu_up[valid]])
    best = np.full(len(P), -np.inf)
    rgb = np.zeros((len(P), 3), np.uint8)
    for cam in meta["cameras"]:
        K = np.asarray(cam["K"])
        R, t = np.asarray(cam["R_cw"]), np.asarray(cam["t_cw"])
        C = np.asarray(cam["center_enu"])
        x = P @ R.T + t
        z = x[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = K[0, 0] * x[:, 0] / z + K[0, 2] - 0.5
            v = K[1, 1] * x[:, 1] / z + K[1, 2] - 0.5
        d, _ = read_depth(depth_root, cam["name"])
        ih, iw = d.shape
        ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)
        ok = (z > 0) & (ui >= 0) & (ui < iw) & (vi >= 0) & (vi < ih)
        idx = np.nonzero(ok)[0]
        dz = d[vi[idx], ui[idx]]
        vis = (dz > 0) & (np.abs(dz - z[idx]) <= np.maximum(0.5, 0.03 * z[idx]))
        dyn = load_dynamic_mask(masks_dir, cam["name"], (ih, iw)) if masks_dir else None
        if dyn is not None and dyn.any():
            vis &= ~dyn[vi[idx], ui[idx]]
        idx = idx[vis]
        ray = C - P[idx]
        score = (ray[:, 2] / np.linalg.norm(ray, axis=1)) / (1.0 + 0.01 * np.linalg.norm(ray, axis=1))
        better = score > best[idx]
        idx, score = idx[better], score[better]
        if not len(idx):
            continue
        img = cv2.imread(cam["image"])
        col = sample_points(img, u[idx], v[idx])
        best[idx] = score
        rgb[idx] = col[:, ::-1]
    out = np.zeros((h, w, 3), np.uint8)
    alpha = np.zeros((h, w), np.uint8)
    flat_rgb = out.reshape(-1, 3)
    vidx = np.flatnonzero(valid.ravel())
    has = np.isfinite(best)
    flat_rgb[vidx[has]] = rgb[has]
    alpha.ravel()[vidx[has]] = 255
    return out, alpha


def write_geotiff(path: str, data: np.ndarray, grid: Grid, geo: LocalGeo, nodata: Optional[float] = None,
                  alpha: Optional[np.ndarray] = None, tags: Optional[Dict[str, str]] = None) -> str:
    import rasterio

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if data.ndim == 2:
        bands = [np.where(np.isnan(data), nodata, data).astype(np.float32)]
        dtype = "float32"
    else:
        bands = [data[..., i] for i in range(data.shape[2])]
        if alpha is not None:
            bands.append(alpha)
        dtype = "uint8"
    profile = dict(driver="GTiff", width=grid.width, height=grid.height, count=len(bands), dtype=dtype,
                   transform=grid.transform(), compress="deflate", tiled=True)
    if geo.epsg:
        profile["crs"] = f"EPSG:{geo.epsg}"
    if nodata is not None and dtype == "float32":
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        for i, b in enumerate(bands, start=1):
            dst.write(b, i)
        if alpha is not None and dtype == "uint8":
            from rasterio.enums import ColorInterp

            dst.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
        dst.update_tags(**(tags or {}))
    return path


def write_las(path: str, xyz_utm_h: np.ndarray, rgb: Optional[np.ndarray], geo: LocalGeo,
              classification: Optional[np.ndarray] = None) -> str:
    import laspy
    from pyproj import CRS

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.offsets = np.floor(xyz_utm_h.min(axis=0))
    header.scales = np.array([0.001, 0.001, 0.001])
    if geo.epsg:
        header.add_crs(CRS.from_epsg(geo.epsg))
    las = laspy.LasData(header)
    las.x, las.y, las.z = xyz_utm_h[:, 0], xyz_utm_h[:, 1], xyz_utm_h[:, 2]
    if rgb is not None:
        las.red, las.green, las.blue = (rgb[:, i].astype(np.uint16) * 257 for i in range(3))
    if classification is not None:
        las.classification = classification.astype(np.uint8)
    las.write(path)
    return path


def _hull_frac(observed: np.ndarray) -> float:
    """Observed cells as a fraction of the convex hull of the observed area.

    Oblique footage spreads a few far points over a large bounding box; the hull
    measures how complete the mapped area itself is."""
    if observed.sum() < 3:
        return 0.0
    # hull of the largest dense region: scattered far points must not inflate the area
    closed = ndimage.binary_closing(observed, structure=np.ones((5, 5), bool))
    lab, n = ndimage.label(closed)
    if n == 0:
        return 0.0
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    main = lab == 1 + int(np.argmax(sizes))
    ys, xs = np.nonzero(main)
    if len(xs) < 3:
        return 0.0
    hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
    m = np.zeros(observed.shape, np.uint8)
    cv2.fillPoly(m, [hull], 1)
    inside = m.astype(bool)
    return float((observed & inside).sum() / max(1, inside.sum()))


def build_products(cloud_path: str, depth_root: str, georef: Dict[str, Any], out_dir: str, cfg: Dict[str, Any],
                   masks_dir: Optional[str]) -> Dict[str, Any]:
    import open3d as o3d

    os.makedirs(out_dir, exist_ok=True)
    geo = LocalGeo(georef)
    pcd = o3d.io.read_point_cloud(cloud_path)
    pts = np.asarray(pcd.points)
    cols = (np.asarray(pcd.colors) * 255).astype(np.uint8) if pcd.has_colors() else None
    if len(pts) < 100:
        raise RuntimeError(f"dense cloud has only {len(pts)} points")
    xy_utm = geo.enu_to_utm(pts[:, :2])
    gsd = float(cfg["gsd_m"])
    grid = Grid.around(xy_utm, gsd)
    dsm_up = rasterize_max(xy_utm, pts[:, 2], grid)
    fill_cells = cfg["fill_holes_m"] / gsd
    observed = ~np.isnan(dsm_up)
    dsm_up = fill_holes_low(dsm_up, fill_cells)
    ground = progressive_morphological_filter(dsm_up, gsd, cfg["pmf_max_window_m"], cfg["pmf_slope"],
                                              cfg["pmf_dh0_m"], cfg["pmf_dh_max_m"])
    dtm_up = build_dtm(dsm_up, ground, gsd)
    dtm_up[np.isnan(dsm_up)] = np.nan

    if geo.epsg:
        tags = {"vertical_reference": f"telemetry altitude ({geo.alt_source}) + ENU up; not geoid corrected",
                "enu_origin": f"{geo.lat0},{geo.lon0},{geo.h0}", "producer": "SIH26158 v3"}
    else:
        tags = {"georeferenced": "NO - local metric frame, scale from assumed camera height",
                "producer": "SIH26158 v3"}
    dsm_path = write_geotiff(os.path.join(out_dir, "dsm.tif"), geo.enu_up_to_height(dsm_up), grid, geo, NODATA, tags=tags)
    dtm_path = write_geotiff(os.path.join(out_dir, "dtm.tif"), geo.enu_up_to_height(dtm_up), grid, geo, NODATA, tags=tags)
    np.savez_compressed(os.path.join(out_dir, "grids.npz"), dsm_up=dsm_up.astype(np.float32),
                        dtm_up=dtm_up.astype(np.float32), ground=ground, observed=observed,
                        xmin=grid.xmin, ymax=grid.ymax, gsd=gsd)
    result = {"gsd_m": gsd, "grid_size": [grid.width, grid.height], "epsg": geo.epsg, "georeferenced": bool(geo.epsg),
              "dsm": dsm_path, "dtm": dtm_path,
              "observed_cell_frac": float(observed.mean()),
              "observed_cells": int(observed.sum()),
              "observed_in_hull_frac": _hull_frac(observed),
              "ground_cell_frac": float(np.nanmean(ground[~np.isnan(dsm_up)])) if (~np.isnan(dsm_up)).any() else 0.0}
    if cfg.get("ortho", True):
        rgb, alpha = true_ortho(dsm_up, grid, geo, depth_root, masks_dir)
        result["ortho"] = write_geotiff(os.path.join(out_dir, "ortho.tif"), rgb, grid, geo, alpha=alpha, tags=tags)
        result["ortho_filled_frac"] = float((alpha > 0).sum() / max(1, (~np.isnan(dsm_up)).sum()))
    xyz = np.column_stack([xy_utm, geo.enu_up_to_height(pts[:, 2])])
    result["las"] = write_las(os.path.join(out_dir, "dense_cloud.las"), xyz, cols, geo)
    return result
