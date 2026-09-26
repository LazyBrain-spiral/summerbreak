"""Telemetry parsing and keyframe synchronisation (v3).

Supported inputs:
  * DJI SRT, bracket style:  [latitude: 28.61] [longitude: 77.20] [rel_alt: 50.2 abs_alt: 266.1]
    [gb_yaw: 12.0 gb_pitch: -60.0 gb_roll: 0.0] [focal_len: 24.00], with or without FrameCnt/DiffTime
  * DJI SRT, older bracket style with gimbal_pitch / gimbal_yaw / altitude
  * DJI SRT, Phantom/Mavic style:  GPS(28.61,77.20,19) BAROMETER:55.2  or  GPS (lon, lat, n) ... H 55.0m
  * CSV flight logs with recognisable column names (lat/latitude, lon/longitude/lng, time/offset, ...)

Output rows carry the altitude source used so georeferencing can report it.
"""

from __future__ import annotations

import csv
import math
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np

TIME_RANGE = re.compile(r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})")
KV = re.compile(r"([A-Za-z_\.]+)\s*[:=]\s*\(?\s*([-+]?\d+(?:\.\d+)?)")
GPS_TUPLE = re.compile(r"GPS\s*\(\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)")
H_FIELD = re.compile(r"(?<![A-Za-z\.])H\s+([-+]?\d+(?:\.\d+)?)\s*m")

ALIASES = {
    "latitude": "lat", "lat": "lat",
    "longitude": "lon", "lon": "lon", "lng": "lon", "long": "lon",
    "abs_alt": "abs_alt", "absolute_altitude": "abs_alt", "altitude_above_sealevel": "abs_alt",
    "rel_alt": "rel_alt", "altitude": "rel_alt", "alt": "rel_alt", "barometer": "rel_alt",
    "height": "rel_alt", "relative_altitude": "rel_alt",
    "gb_yaw": "gimbal_yaw", "gimbal_yaw": "gimbal_yaw", "yaw": "gimbal_yaw",
    "gb_pitch": "gimbal_pitch", "gimbal_pitch": "gimbal_pitch", "pitch": "gimbal_pitch",
    "gb_roll": "gimbal_roll", "gimbal_roll": "gimbal_roll", "roll": "gimbal_roll",
    "focal_len": "focal_len", "focal_length": "focal_len",
    "framecnt": "frame_cnt", "frame": "frame_cnt",
}
FIELDS = ["lat", "lon", "abs_alt", "rel_alt", "gimbal_yaw", "gimbal_pitch", "gimbal_roll", "focal_len"]


def parse_srt_time(s: str) -> float:
    s = s.replace(",", ".")
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(sec)


def _parse_srt_block(text: str) -> Dict[str, Any]:
    rec: Dict[str, Any] = {}
    m = TIME_RANGE.search(text)
    if m:
        rec["t"] = parse_srt_time(m.group(1))
    for key, val in KV.findall(text):
        canon = ALIASES.get(key.lower().strip("."))
        if canon and canon not in rec:
            rec[canon] = float(val)
    g = GPS_TUPLE.search(text)
    if g and ("lat" not in rec or "lon" not in rec):
        a, b, c = (float(x) for x in g.groups())
        # "GPS (lon, lat, sats)" on Mavic Air style vs "GPS(lat, lon, alt)" on Phantom style.
        if abs(a) > 90 or "GPS (" in text:
            rec["lon"], rec["lat"] = a, b
        else:
            rec["lat"], rec["lon"] = a, b
    h = H_FIELD.search(text)
    if h and "rel_alt" not in rec:
        rec["rel_alt"] = float(h.group(1))
    return rec


def parse_srt(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().replace("\r\n", "\n")
    content = re.sub(r"<[^>]+>", " ", content)  # strip <font> tags
    records = []
    for block in re.split(r"\n\s*\n", content.strip()):
        rec = _parse_srt_block(block)
        if "t" in rec and "lat" in rec and "lon" in rec:
            records.append(rec)
    return records


CSV_TIME_KEYS = ["time_s", "time(s)", "time", "timestamp", "offset", "offset_time", "seconds",
                 "time(millisecond)", "time_ms"]


def parse_csv(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c: c.lower().strip() for c in (reader.fieldnames or [])}
        time_col, ms = None, False
        for want in CSV_TIME_KEYS:
            for c, lc in cols.items():
                if lc == want:
                    time_col, ms = c, "milli" in lc or lc.endswith("_ms")
                    break
            if time_col:
                break
        for i, row in enumerate(reader):
            rec: Dict[str, Any] = {}
            for c, lc in cols.items():
                key = re.sub(r"\(.*?\)", "", lc).strip().replace(" ", "_")
                canon = ALIASES.get(key)
                if canon and row.get(c) not in (None, ""):
                    try:
                        rec[canon] = float(row[c])
                    except ValueError:
                        pass
            if time_col and row.get(time_col):
                try:
                    rec["t"] = float(row[time_col]) / (1000.0 if ms else 1.0)
                except ValueError:
                    continue
            else:
                rec["t"] = float(i)
            if "lat" in rec and "lon" in rec:
                records.append(rec)
    return records


def parse_telemetry(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    recs = parse_csv(path) if path.lower().endswith(".csv") else parse_srt(path)
    recs = [r for r in recs if not (r["lat"] == 0.0 and r["lon"] == 0.0)]  # no-fix placeholders
    recs.sort(key=lambda r: r["t"])
    return recs


def _interp(t: np.ndarray, tt: np.ndarray, values: List[Optional[float]], angle: bool = False):
    v = np.array([np.nan if x is None else x for x in values], dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.full(len(t), np.nan)
    if angle:
        v = v.copy()
        v[ok] = np.degrees(np.unwrap(np.radians(v[ok])))
    out = np.interp(t, tt[ok], v[ok])
    if angle:
        out = (out + 180.0) % 360.0 - 180.0
    return out


def synchronize(records: List[Dict[str, Any]], keyframes: List[Dict[str, Any]],
                time_offset_s: float = 0.0) -> List[Dict[str, Any]]:
    """Interpolate telemetry at each keyframe timestamp (video time + offset)."""
    if not records:
        raise ValueError("telemetry has no usable records")
    tt = np.array([r["t"] for r in records], dtype=np.float64)
    kt = np.array([k["t"] for k in keyframes], dtype=np.float64) + time_offset_s
    if tt[0] > 1000.0 and kt.max() < 1000.0:  # time-of-day captions
        tt = tt - tt[0]
    cols = {}
    for fld in FIELDS:
        cols[fld] = _interp(kt, tt, [r.get(fld) for r in records], angle=fld in ("gimbal_yaw", "gimbal_roll"))
    span = (tt[0] - 0.5, tt[-1] + 0.5)
    rows = []
    for i, k in enumerate(keyframes):
        row = {"name": k["name"], "t": round(float(kt[i]), 4),
               "in_span": bool(span[0] <= kt[i] <= span[1])}
        for fld in FIELDS:
            val = cols[fld][i]
            row[fld] = None if not math.isfinite(val) else round(float(val), 8 if fld in ("lat", "lon") else 4)
        if row["abs_alt"] is not None:
            row["alt"], row["alt_source"] = row["abs_alt"], "abs_alt"
        elif row["rel_alt"] is not None:
            row["alt"], row["alt_source"] = row["rel_alt"], "rel_alt"
        else:
            row["alt"], row["alt_source"] = None, "none"
        rows.append(row)
    return rows
