"""Camera intrinsics prior from telemetry or defaults.

DJI captions report a 35 mm-equivalent focal length (sometimes x10, e.g. 240
for 24.0 mm). The equivalent focal length is defined on the 43.27 mm diagonal
of a full-frame sensor, so f_px = f_eq * diag_px / 43.27.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

FULL_FRAME_DIAG_MM = 43.2666
DEFAULT_DFOV_DEG = 82.0  # typical DJI video diagonal field of view


def normalise_focal_35mm(value: Optional[float]) -> Optional[float]:
    if value is None or not math.isfinite(value) or value <= 0:
        return None
    if value > 100.0:  # 240 -> 24.0
        value /= 10.0
    if not 8.0 <= value <= 300.0:
        return None
    return value


def intrinsics_prior(width: int, height: int, focal_len_35mm: Optional[float] = None,
                     dfov_deg: Optional[float] = None) -> Dict[str, Any]:
    diag = math.hypot(width, height)
    f_eq = normalise_focal_35mm(focal_len_35mm)
    if f_eq is not None:
        f = f_eq * diag / FULL_FRAME_DIAG_MM
        source = "telemetry_focal_len"
    else:
        fov = dfov_deg or DEFAULT_DFOV_DEG
        f = (diag / 2.0) / math.tan(math.radians(fov) / 2.0)
        source = "dfov_default" if dfov_deg is None else "dfov"
    return {"width": width, "height": height, "f": f, "cx": width / 2.0, "cy": height / 2.0,
            "focal_len_35mm": f_eq, "source": source,
            "hfov_deg": math.degrees(2 * math.atan(width / (2 * f)))}
