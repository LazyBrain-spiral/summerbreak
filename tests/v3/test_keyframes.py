import os

import cv2
import numpy as np

from src.core.config import DEFAULTS
from src.ingest.keyframes import extract_keyframes


def _make_video(path, n=60, fps=10, w=320, h=180, speed_px=6, hover=(20, 35)):
    """Textured strip scrolling past the camera, with a hover segment and one blurred frame."""
    rng = np.random.default_rng(0)
    canvas = (rng.random((h, w * 6)) * 255).astype(np.uint8)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 1.5)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    x = 0
    for i in range(n):
        if not (hover[0] <= i < hover[1]):
            x += speed_px
        frame = canvas[:, x:x + w].copy()
        if i == 45:
            frame = cv2.GaussianBlur(frame, (0, 0), 6)
        vw.write(frame)
    vw.release()


def test_parallax_selection_skips_hover(tmp_path):
    video = str(tmp_path / "v.mp4")
    _make_video(video)
    cfg = dict(DEFAULTS["ingest"], candidate_hz=10.0, track_width=320, min_disp_frac=0.08, max_gap_s=10.0,
               min_keyframes=3)
    info = extract_keyframes(video, str(tmp_path / "kf"), cfg)
    idx = [k["frame_index"] for k in info["keyframes"]]
    assert 4 <= len(idx) <= 25, idx
    # nothing new is selected while hovering (frames 20..34), apart from possibly the frame that starts it
    assert sum(21 <= i < 35 for i in idx) == 0, idx
    assert 45 not in idx  # blurred frame
    assert sorted(os.listdir(tmp_path / "kf" / "raw")) == sorted(os.listdir(tmp_path / "kf" / "enh"))


def test_slow_clip_relaxes_spacing(tmp_path):
    """A slowly drifting clip gets enough keyframes for SfM without manual tuning."""
    video = str(tmp_path / "slow.mp4")
    _make_video(video, n=80, speed_px=1, hover=(0, 0))
    cfg = dict(DEFAULTS["ingest"], candidate_hz=10.0, track_width=320, min_disp_frac=0.08, max_gap_s=10.0,
               min_keyframes=12)
    info = extract_keyframes(video, str(tmp_path / "kf"), cfg)
    assert info["auto_relaxed_steps"] >= 1
    assert info["num_keyframes"] >= 12
