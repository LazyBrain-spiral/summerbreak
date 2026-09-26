"""Local ENU <-> UTM / WGS84 conversion for products (v3).

Rasters and GIS layers are written in the UTM zone of the ENU origin. Over a
survey area of a few kilometres the ENU tangent plane maps to UTM by a 2D
affine (grid convergence rotation + point scale factor) to millimetre level,
which is what ``LocalGeo`` fits from three points. Heights are ENU up plus
the origin's telemetry altitude (ellipsoidal or barometric, as reported).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


class LocalGeo:
    def __init__(self, georef: Dict[str, Any]):
        o = georef["enu_origin"]
        if georef.get("utm_epsg") is None:
            # GPS-denied: local metric frame, no CRS
            self.local = True
            self.lat0 = self.lon0 = None
            self.h0 = 0.0
            self.epsg = None
            self.alt_source = o.get("alt_source")
            self.A = np.eye(2)
            self.A_inv = np.eye(2)
            self.b = np.zeros(2)
            return
        self.local = False
        import pymap3d
        from pyproj import Transformer

        self.lat0, self.lon0, self.h0 = o["lat"], o["lon"], o["alt"]
        self.epsg = int(georef["utm_epsg"])
        self.alt_source = o.get("alt_source")
        self._to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        self._to_ll = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)
        pts_enu = np.array([[0.0, 0.0], [500.0, 0.0], [0.0, 500.0]])
        ll = [pymap3d.enu2geodetic(e, n, 0.0, self.lat0, self.lon0, self.h0) for e, n in pts_enu]
        utm = np.array([self._to_utm.transform(lon, lat) for lat, lon, _ in ll])
        self.b = utm[0]
        self.A = np.column_stack([(utm[1] - utm[0]) / 500.0, (utm[2] - utm[0]) / 500.0])
        self.A_inv = np.linalg.inv(self.A)

    def enu_to_utm(self, xy: np.ndarray) -> np.ndarray:
        return xy @ self.A.T + self.b

    def utm_to_enu(self, xy: np.ndarray) -> np.ndarray:
        return (xy - self.b) @ self.A_inv.T

    def enu_up_to_height(self, z: np.ndarray) -> np.ndarray:
        return z + self.h0

    def height_to_enu_up(self, h: np.ndarray) -> np.ndarray:
        return h - self.h0

    def utm_to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        if self.local:
            return xy.copy()  # local metres; GeoJSON written without a geographic CRS
        lon, lat = self._to_ll.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])

    def crs_wkt(self) -> str:
        from pyproj import CRS

        return CRS.from_epsg(self.epsg).to_wkt()
