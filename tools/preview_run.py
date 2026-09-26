"""Quick-look PNGs for a finished run: orthomosaic, DSM hillshade, height above ground,
all with building footprints drawn on top.

    python tools/preview_run.py --run runs/<id>        # writes runs/<id>/previews/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def hillshade(z: np.ndarray, gsd: float, az: float = 315.0, alt: float = 45.0) -> np.ndarray:
    zf = np.nan_to_num(z, nan=np.nanmin(z))
    gy, gx = np.gradient(zf, gsd)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, e = np.radians(az), np.radians(alt)
    hs = np.sin(e) * np.cos(slope) + np.cos(e) * np.sin(slope) * np.cos(a - aspect)
    out = (np.clip(hs, 0, 1) * 255).astype(np.uint8)
    out[np.isnan(z)] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = args.run
    out = os.path.join(run, "previews")
    os.makedirs(out, exist_ok=True)
    g = np.load(os.path.join(run, "07_products", "grids.npz"))
    dsm, dtm, gsd = g["dsm_up"], g["dtm_up"], float(g["gsd"])
    xmin, ymax = float(g["xmin"]), float(g["ymax"])

    polys = []
    gj_path = os.path.join(run, "08_completion", "buildings.geojson")
    if os.path.exists(gj_path):
        from src.core.io import read_json
        from src.products.geo import LocalGeo

        geo = LocalGeo(read_json(os.path.join(run, "04_georef", "georef.json")))
        from pyproj import Transformer

        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{geo.epsg}", always_xy=True) if geo.epsg else None
        for f in json.load(open(gj_path))["features"]:
            ring = np.array(f["geometry"]["coordinates"][0])
            x, y = tr.transform(ring[:, 0], ring[:, 1]) if tr else (ring[:, 0], ring[:, 1])
            px = np.column_stack([(np.asarray(x) - xmin) / gsd, (ymax - np.asarray(y)) / gsd])
            polys.append((px.astype(np.int32), f["properties"]))

    def draw(img):
        img = img.copy()
        for px, p in polys:
            cv2.polylines(img, [px], True, (0, 0, 255), 2, cv2.LINE_AA)
            c = px.mean(axis=0).astype(int)
            cv2.putText(img, f"{p['footprint_area_m2']:.0f}m2 {p['ridge_height_m']:.1f}m", tuple(c - [40, 0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, f"{p['footprint_area_m2']:.0f}m2 {p['ridge_height_m']:.1f}m", tuple(c - [40, 0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 180), 1, cv2.LINE_AA)
        return img

    written = []
    ortho_path = os.path.join(run, "07_products", "ortho.tif")
    if os.path.exists(ortho_path):
        import rasterio

        with rasterio.open(ortho_path) as src:
            rgb = np.stack([src.read(i) for i in (1, 2, 3)], -1)
        p = os.path.join(out, "ortho_buildings.png")
        cv2.imwrite(p, draw(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)))
        written.append(p)
    hs = cv2.cvtColor(hillshade(dsm, gsd), cv2.COLOR_GRAY2BGR)
    p = os.path.join(out, "dsm_hillshade_buildings.png")
    cv2.imwrite(p, draw(hs))
    written.append(p)
    nd = np.clip(np.nan_to_num(dsm - dtm, nan=0) / 20.0, 0, 1)
    ndc = cv2.applyColorMap((nd * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    ndc[np.isnan(dsm)] = 0
    p = os.path.join(out, "height_above_ground.png")
    cv2.imwrite(p, draw(ndc))
    written.append(p)
    print("\n".join(written))


if __name__ == "__main__":
    main()
