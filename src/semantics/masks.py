"""Dynamic-object masks (v3).

Writes two conventions with the keyframe basename:
  dynamic/<stem>.png         255 = dynamic object, 0 = static  (used by depth, texture, ortho)
  colmap/<image_name>.png    0 = ignore, 255 = use             (COLMAP ImageReader.mask_path)

Masks are dilated to cover motion-blur halos around vehicles and people.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import cv2
import numpy as np

from src.core.io import list_images

# COCO ids: person, bicycle, car, motorcycle, bus, truck, boat, bird, cat, dog, horse, sheep, cow
DYNAMIC_COCO = {0, 1, 2, 3, 5, 7, 8, 14, 15, 16, 17, 18, 19}


def _load_yolo(model_name: str, allow_cpu: bool):
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not allow_cpu:
        raise RuntimeError("YOLO masking needs a CUDA device (set allow_cpu: true to accept CPU)")
    return YOLO(model_name), device


def generate_masks(image_dir: str, out_dir: str, cfg: Dict[str, Any], allow_cpu: bool) -> Dict[str, Any]:
    dyn_dir = os.path.join(out_dir, "dynamic")
    col_dir = os.path.join(out_dir, "colmap")
    os.makedirs(dyn_dir, exist_ok=True)
    os.makedirs(col_dir, exist_ok=True)
    model, device = _load_yolo(cfg["model"], allow_cpu)
    names = list_images(image_dir)
    frac = []
    for name in names:
        img = cv2.imread(os.path.join(image_dir, name))
        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)
        res = model.predict(source=img, conf=cfg["conf"], device=device, verbose=False)
        if res and res[0].masks is not None:
            classes = res[0].boxes.cls.cpu().numpy().astype(int)
            for cls_id, poly in zip(classes, res[0].masks.xy):
                if cls_id in DYNAMIC_COCO and len(poly) >= 3:
                    cv2.fillPoly(mask, [np.asarray(poly, np.int32)], 255)
        k = max(1, int(round(cfg["dilate_frac"] * w)))
        if mask.any():
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
        stem = os.path.splitext(name)[0]
        cv2.imwrite(os.path.join(dyn_dir, stem + ".png"), mask)
        cv2.imwrite(os.path.join(col_dir, name + ".png"), 255 - mask)
        frac.append(float((mask > 0).mean()))
    return {"device": device, "num_masks": len(names),
            "mean_masked_fraction": float(np.mean(frac)) if frac else 0.0,
            "max_masked_fraction": float(np.max(frac)) if frac else 0.0}


def load_dynamic_mask(masks_dir: str, image_name: str, shape_hw) -> np.ndarray:
    """Boolean mask (True = dynamic) resized to shape_hw, or all-False if masks are absent."""
    if not masks_dir:
        return np.zeros(shape_hw, bool)
    p = os.path.join(masks_dir, "dynamic", os.path.splitext(image_name)[0] + ".png")
    if not os.path.exists(p):
        return np.zeros(shape_hw, bool)
    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if m.shape != tuple(shape_hw):
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return m > 127
