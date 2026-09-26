"""Small I/O helpers shared by v3 stages."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from typing import Any, Dict, Iterable, List

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):  # noqa: D401 - json hook
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def write_json(path: str, data: Any) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)
    os.replace(tmp, path)
    return path


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV {path}")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def stable_hash(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, cls=_NumpyEncoder, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def list_images(folder: str, exts: Iterable[str] = (".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> List[str]:
    exts = tuple(e.lower() for e in exts)
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
