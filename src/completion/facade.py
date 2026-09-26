"""Building metadata from rectified facades: doors, windows, floors, colours.

Wall texture patches are rectified and metric (texel size known, bottom row at
the base of the wall), so detections convert straight into metres. Doors and
windows come from OWLv2 open-vocabulary detection (no training data needed)
and are then filtered by physical size and position:

  door    0.6-3.5 m wide, 1.6-3.6 m tall, bottom within 0.9 m of the ground
  window  0.3-3.0 m wide, 0.4-3.0 m tall, not touching the ground

Floors: window centre heights clustered into rows (a new row starts after a
1.2 m gap); the row count is the estimate when at least two windows are found,
otherwise eave height / 3 m. Both are reported.
"""

from __future__ import annotations

import base64
import colorsys
import math
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_DETECTOR = None


def _detector(model_id: str):
    global _DETECTOR
    if _DETECTOR is None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        proc = Owlv2Processor.from_pretrained(model_id)
        model = Owlv2ForObjectDetection.from_pretrained(model_id).to(dev).eval()
        _DETECTOR = (proc, model, dev, torch)
    return _DETECTOR


QUERIES = ["a door", "a window", "a garage door", "an entrance"]


def detect(patch_rgb: np.ndarray, model_id: str, thresh: float = 0.18) -> List[Dict[str, Any]]:
    proc, model, dev, torch = _detector(model_id)
    h, w = patch_rgb.shape[:2]
    # OWLv2 pads to a square internally; give it a reasonably sized input
    s = 960.0 / max(h, w)
    img = cv2.resize(patch_rgb, (max(8, int(w * s)), max(8, int(h * s)))) if s < 1 else patch_rgb
    inputs = proc(text=[QUERIES], images=img, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**inputs)
    ih, iw = img.shape[:2]
    side = max(ih, iw)  # boxes are relative to the padded square
    ts = torch.tensor([[side, side]], device=dev)
    pp = getattr(proc, "post_process_object_detection", None) or getattr(proc.image_processor, "post_process_object_detection", None)
    if pp is not None:
        res = pp(out, threshold=thresh, target_sizes=ts)[0]
    else:  # transformers >= 5 renamed it
        res = proc.post_process_grounded_object_detection(out, threshold=thresh, target_sizes=ts,
                                                           text_labels=[QUERIES])[0]
    dets = []
    for box, score, lab in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy()):
        x0, y0, x1, y1 = box / (s if s < 1 else 1.0)
        dets.append({"label": QUERIES[int(lab)], "score": float(score),
                     "box": [float(max(0, x0)), float(max(0, y0)), float(min(w, x1)), float(min(h, y1))]})
    return dets


def _nms(dets: List[Dict[str, Any]], iou_thr: float = 0.4) -> List[Dict[str, Any]]:
    dets = sorted(dets, key=lambda d: -d["score"])
    keep = []
    for d in dets:
        x0, y0, x1, y1 = d["box"]
        ok = True
        for k in keep:
            a0, b0, a1, b1 = k["box"]
            iw_, ih_ = max(0, min(x1, a1) - max(x0, a0)), max(0, min(y1, b1) - max(y0, b0))
            inter = iw_ * ih_
            union = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter
            if union > 0 and inter / union > iou_thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def colour_name(rgb: np.ndarray) -> str:
    r, g, b = (float(x) / 255.0 for x in rgb)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    if l > 0.82:
        return "white"
    if l < 0.18:
        return "black"
    if s < 0.18:
        return "grey" if l < 0.6 else "light grey"
    deg = h * 360
    if deg < 18 or deg >= 340:
        return "red"
    if deg < 40:
        return "terracotta" if l < 0.55 else "orange"
    if deg < 65:
        return "yellow" if s > 0.35 else "beige"
    if deg < 170:
        return "green"
    if deg < 260:
        return "blue"
    return "purple"


def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(x) for x in rgb)


def _compass(az_deg: float) -> str:
    names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    return names[int(((az_deg % 360) + 22.5) // 45) % 8]


def building_metadata(reg: Dict[str, Any], georeferenced: bool, detector_model: Optional[str],
                      texel: float) -> Dict[str, Any]:
    """Doors, windows, floors, colours, dimensions and a facade thumbnail for one building."""
    ring = reg["ring_enu"]
    rect = cv2.minAreaRect(ring.astype(np.float32))
    (cx, cy), (w, h), ang = rect
    length, width = max(w, h), min(w, h)
    # azimuth of the long axis (clockwise from north, 0-180)
    box = cv2.boxPoints(rect)
    e = [box[1] - box[0], box[2] - box[1]]
    le = max(e, key=lambda v: np.hypot(*v))
    az = (math.degrees(math.atan2(le[0], le[1])) + 180) % 180
    ground = reg.get("ground_z", reg["base_z"])

    roof_px, wall_px = [], []
    doors, windows = [], []
    best_wall = None
    for g, t in reg.get("_groups", []):
        seen = t["seen"]
        if seen.any():
            (roof_px if g["kind"] == "roof" else wall_px).append(t["patch"][seen])
        if g["kind"] != "wall" or t["observed_frac"] < 0.4 or t.get("provenance") in ("generated", "neutral"):
            continue
        facing = (math.degrees(math.atan2(g["n"][0], g["n"][1])) + 360) % 360
        if best_wall is None or t["seen"].sum() > best_wall[1]["seen"].sum():
            best_wall = (g, t, facing)
        if not detector_model:
            continue
        dets = _nms(detect(t["patch"][..., ::-1].copy(), detector_model))
        o_z = float(g["o"][2])
        for d in dets:
            x0, y0, x1, y1 = d["box"]
            wm, hm = (x1 - x0) * texel, (y1 - y0) * texel
            # row -> plane coordinate t -> height above ground
            top_h = o_z + (t["qmax"][1] - (y0 - t["pad"]) * texel) - ground
            bot_h = o_z + (t["qmax"][1] - (y1 - t["pad"]) * texel) - ground
            rec = {"wall": int(g["index"]), "facing_deg": round(facing, 0), "facing": _compass(facing),
                   "width_m": round(wm, 2), "height_m": round(hm, 2), "bottom_m": round(bot_h, 2),
                   "along_wall_m": round(((x0 + x1) / 2 - t["pad"]) * texel, 2), "score": round(d["score"], 2),
                   "box_px": [round(v, 1) for v in d["box"]], "group": id(t)}
            is_door_label = d["label"] in ("a door", "a garage door", "an entrance")
            if is_door_label and 0.6 <= wm <= 3.5 and 1.6 <= hm <= 3.6 and bot_h <= 0.9:
                rec["type"] = "garage door" if d["label"] == "a garage door" or wm > 2.4 else "door"
                doors.append(rec)
            elif d["label"] == "a window" and 0.3 <= wm <= 3.0 and 0.4 <= hm <= 3.0 and bot_h > 0.3:
                rec["type"] = "window"
                windows.append(rec)
    # floors
    eave = float(reg.get("eave_height_m", 0.0))
    floors_h = max(1, int(round(eave / 3.0))) if eave > 0 else None
    rows = 0
    if len(windows) >= 2:
        cz = sorted((w_["bottom_m"] + w_["height_m"] / 2) for w_ in windows)
        rows = 1
        for a, b in zip(cz, cz[1:]):
            if b - a > 1.2:
                rows += 1
        # a ground floor with a door but windows only above counts too
        if doors and min(cz) > 2.5:
            rows += 1
    floors = rows if rows else floors_h
    method = "window rows" if rows else "eave height / 3 m"
    # colours
    roof_rgb = np.median(np.vstack(roof_px), axis=0)[::-1] if roof_px else None
    wall_rgb = np.median(np.vstack(wall_px), axis=0)[::-1] if wall_px else None
    # facade thumbnail with detections
    thumb = None
    if best_wall is not None:
        g, t, facing = best_wall
        img = t["patch"].copy()
        for d in doors + windows:
            if d["group"] != id(t):
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in d["box_px"])
            col = (230, 110, 40) if d["type"] != "window" else (40, 150, 240)  # BGR: doors blue, windows orange
            cv2.rectangle(img, (x0, y0), (x1, y1), col, max(2, img.shape[1] // 200))
        s = 360.0 / max(img.shape[1], 1)
        if s < 1:
            img = cv2.resize(img, (360, max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
        thumb = {"jpg": "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode(),
                 "facing": _compass(facing), "facing_deg": round(facing, 0)}
    for d in doors + windows:
        d.pop("group", None)
        d.pop("box_px", None)
    return {
        "length_m": round(float(length), 2), "width_m": round(float(width), 2),
        "orientation_deg": round(float(az), 0),
        "orientation_ref": "true north" if georeferenced else "local frame (no GPS)",
        "floors_estimate": floors, "floors_method": method, "floors_from_height": floors_h,
        "window_rows": rows, "doors": doors, "windows_count": len(windows),
        "entrances": [f"{d['type']} on {d['facing']} wall, {d['width_m']} m wide" for d in doors],
        "roof_colour": colour_name(roof_rgb) if roof_rgb is not None else None,
        "roof_colour_hex": _hex(roof_rgb) if roof_rgb is not None else None,
        "facade_colour": colour_name(wall_rgb) if wall_rgb is not None else None,
        "facade_colour_hex": _hex(wall_rgb) if wall_rgb is not None else None,
        "facade_thumb": thumb,
    }
