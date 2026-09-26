"""Image sampling helpers."""

from __future__ import annotations

import cv2
import numpy as np


def sample_points(img: np.ndarray, x: np.ndarray, y: np.ndarray, interp: int = cv2.INTER_LINEAR,
                  border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
    """Sample ``img`` at array coordinates (x = column, y = row) for any number of points.

    cv2.remap refuses maps with a side >= 32767, so points are laid out as a
    (rows, 4096) map. Returns (N,) for 2-D images and (N, C) for multi-channel.
    """
    n = len(x)
    if n == 0:
        return np.empty((0,) + img.shape[2:], img.dtype)
    cols = 4096
    rows = -(-n // cols)
    pad = rows * cols - n
    mx = np.concatenate([np.asarray(x, np.float32), np.zeros(pad, np.float32)]).reshape(rows, cols)
    my = np.concatenate([np.asarray(y, np.float32), np.zeros(pad, np.float32)]).reshape(rows, cols)
    src = img if img.dtype in (np.uint8, np.float32) else img.astype(np.float32)
    out = cv2.remap(src, mx, my, interp, borderMode=border)
    return out.reshape((rows * cols,) + img.shape[2:])[:n]
