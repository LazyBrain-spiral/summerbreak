"""Turn a geotagged image set (e.g. an OpenDroneMap sample dataset) into video + DJI-style SRT.

Public drone video with synchronised telemetry is rare; geotagged survey
photos are common. Ordering the photos by capture time and encoding them as a
low-frame-rate video exercises the full pipeline on real imagery.

    python tools/images_to_video_srt.py --images path/to/images --out test_data/odm_clip --fps 2

Reads EXIF GPS (lat, lon, altitude), DateTimeOriginal, focal length in 35 mm,
and DJI XMP gimbal / flight angles when present. Frames are letterboxed to a
constant size so the video encoder accepts them.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def _ratio(v) -> float:
    try:
        return float(v.numerator) / float(v.denominator)
    except AttributeError:
        if isinstance(v, tuple):
            return float(v[0]) / float(v[1])
        return float(v)


def _dms(vals, ref) -> float:
    d, m, s = (_ratio(x) for x in vals)
    out = d + m / 60.0 + s / 3600.0
    return -out if ref in ("S", "W", b"S", b"W") else out


def read_meta(path: str) -> Optional[Dict[str, Any]]:
    from PIL import ExifTags, Image

    img = Image.open(path)
    exif = img._getexif() or {}
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    gps_raw = tags.get("GPSInfo")
    if not gps_raw:
        return None
    gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_raw.items()}
    try:
        lat = _dms(gps["GPSLatitude"], gps.get("GPSLatitudeRef", "N"))
        lon = _dms(gps["GPSLongitude"], gps.get("GPSLongitudeRef", "E"))
    except KeyError:
        return None
    alt = _ratio(gps["GPSAltitude"]) if "GPSAltitude" in gps else None
    t = None
    if "DateTimeOriginal" in tags:
        try:
            t = datetime.strptime(tags["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S").timestamp()
        except ValueError:
            pass
    meta: Dict[str, Any] = {"path": path, "lat": lat, "lon": lon, "abs_alt": alt, "time": t,
                            "focal_35mm": tags.get("FocalLengthIn35mmFilm")}
    # DJI XMP block
    with open(path, "rb") as f:
        head = f.read(200_000).decode("latin1", errors="ignore")
    for key, name in (("RelativeAltitude", "rel_alt"), ("GimbalPitchDegree", "gimbal_pitch"),
                      ("GimbalYawDegree", "gimbal_yaw"), ("GimbalRollDegree", "gimbal_roll")):
        m = re.search(rf'drone-dji:{key}="([-+\d\.]+)"', head)
        if m:
            meta[name] = float(m.group(1))
    return meta


def _fmt(t: float) -> str:
    ms = int(round(t * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def convert(images: str, out: str, fps: float, width: int) -> Dict[str, Any]:
    os.makedirs(out, exist_ok=True)
    files = sorted(os.path.join(images, f) for f in os.listdir(images)
                   if f.lower().endswith((".jpg", ".jpeg", ".tif", ".tiff", ".png")))
    metas: List[Dict[str, Any]] = [m for m in (read_meta(f) for f in files) if m]
    if len(metas) < 3:
        raise RuntimeError(f"only {len(metas)} geotagged images in {images}")
    if all(m["time"] is not None for m in metas):
        metas.sort(key=lambda m: (m["time"], m["path"]))
    first = cv2.imread(metas[0]["path"])
    h0, w0 = first.shape[:2]
    height = int(round(width * h0 / w0 / 2) * 2)
    writer = cv2.VideoWriter(os.path.join(out, "flight.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    blocks = []
    for i, m in enumerate(metas):
        img = cv2.imread(m["path"])
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        writer.write(img)
        t = i / fps
        fields = [f"[latitude: {m['lat']:.7f}]", f"[longitude: {m['lon']:.7f}]"]
        alt = []
        if m.get("rel_alt") is not None:
            alt.append(f"rel_alt: {m['rel_alt']:.3f}")
        if m.get("abs_alt") is not None:
            alt.append(f"abs_alt: {m['abs_alt']:.3f}")
        if alt:
            fields.append("[" + " ".join(alt) + "]")
        if m.get("gimbal_pitch") is not None:
            fields.append(f"[gb_yaw: {m.get('gimbal_yaw', 0.0):.1f} gb_pitch: {m['gimbal_pitch']:.1f} "
                          f"gb_roll: {m.get('gimbal_roll', 0.0):.1f}]")
        if m.get("focal_35mm"):
            fields.append(f"[focal_len: {float(m['focal_35mm']):.2f}]")
        blocks.append(f"{i + 1}\n{_fmt(t)} --> {_fmt(t + 1.0 / fps)}\nFrameCnt: {i + 1}\n" + " ".join(fields) + "\n")
    writer.release()
    with open(os.path.join(out, "flight.srt"), "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write(f"# {os.path.basename(os.path.abspath(images))} as video\n\nMade by tools/images_to_video_srt.py from "
                f"{len(metas)} geotagged images at {fps} fps, {width}x{height}.\n")
    return {"frames": len(metas), "size": [width, height]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--width", type=int, default=1920)
    args = ap.parse_args()
    print(convert(args.images, args.out, args.fps, args.width))


if __name__ == "__main__":
    main()
