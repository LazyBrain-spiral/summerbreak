"""
Keyframe Extractor Module (Workstream A - Stage 1)
Extracts keyframes from input drone video, scores sharpness using Laplacian variance,
and saves the top sharpest frames with minimal motion blur.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any


def compute_laplacian_variance(image: np.ndarray) -> float:
    """Compute the variance of the Laplacian as a metric for image sharpness."""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def extract_keyframes(
    video_path: str,
    output_dir: str,
    target_fps: float = 2.0,
    max_frames: int = 500,
    min_sharpness: float = 20.0,
    apply_clahe: bool = True,
) -> Dict[str, Any]:
    """
    Extract keyframes from drone video stream.
    
    Args:
        video_path: Path to input video (mp4, mov, etc.)
        output_dir: Output directory to write extracted keyframes
        target_fps: Sampling rate in Hz (default 2 Hz, i.e., 1 frame every 0.5s)
        max_frames: Maximum number of keyframes to retain (400-600 recommended)
        min_sharpness: Laplacian variance cutoff to reject blurry frames
        apply_clahe: Whether to apply CLAHE enhancement to kept keyframes
        
    Returns:
        Summary dict containing extraction statistics
    """
    os.makedirs(output_dir, exist_ok=True)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    candidates: List[Tuple[int, float, float, np.ndarray]] = []

    if os.path.isdir(video_path):
        # Input is already an image directory
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.lower().endswith(img_exts)])
        if not image_files:
            raise RuntimeError(f"No image files found in directory: {video_path}")

        first_img = cv2.imread(image_files[0])
        height, width = first_img.shape[:2]
        video_fps = target_fps
        total_raw_frames = len(image_files)
        duration_sec = total_raw_frames / target_fps

        for idx, fpath in enumerate(image_files):
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            timestamp = idx / target_fps
            sharpness = compute_laplacian_variance(frame)
            candidates.append((idx, timestamp, sharpness, frame))
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_raw_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_raw_frames / video_fps

        sample_step = max(1, int(round(video_fps / target_fps)))
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_step == 0:
                timestamp = frame_idx / video_fps
                sharpness = compute_laplacian_variance(frame)
                if sharpness >= min_sharpness:
                    candidates.append((frame_idx, timestamp, sharpness, frame))

            frame_idx += 1

        cap.release()

    # If candidates exceed max_frames, pick the highest sharpness frames while maintaining temporal spread
    if len(candidates) > max_frames:
        # Uniformly bucket and pick best sharpness per bucket
        bucket_size = len(candidates) / max_frames
        selected_candidates = []
        for i in range(max_frames):
            start = int(i * bucket_size)
            end = int((i + 1) * bucket_size)
            bucket = candidates[start:end]
            if bucket:
                best_in_bucket = max(bucket, key=lambda x: x[2])
                selected_candidates.append(best_in_bucket)
        candidates = selected_candidates

    # Sort back chronologically
    candidates.sort(key=lambda x: x[0])

    saved_frames = []
    for out_idx, (f_idx, timestamp, sharpness, frame) in enumerate(candidates, start=1):
        processed_frame = frame
        if apply_clahe:
            # Convert to LAB, apply CLAHE on L channel, convert back
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_enhanced = clahe.apply(l)
            enhanced_lab = cv2.merge((l_enhanced, a, b))
            processed_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        filename = f"frame_{out_idx:05d}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, processed_frame)
        saved_frames.append({
            "frame_id": out_idx,
            "original_frame_idx": f_idx,
            "timestamp_sec": timestamp,
            "sharpness": sharpness,
            "filename": filename,
            "filepath": filepath,
        })

    return {
        "video_fps": video_fps,
        "total_raw_frames": total_raw_frames,
        "duration_sec": duration_sec,
        "resolution": f"{width}x{height}",
        "keyframes_extracted": len(saved_frames),
        "frames": saved_frames,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract sharp keyframes from drone video")
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output_dir", default="data/keyframes", help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Sampling frequency (Hz)")
    parser.add_argument("--max_frames", type=int, default=500, help="Max frames to retain")
    args = parser.parse_args()

    stats = extract_keyframes(args.video, args.output_dir, target_fps=args.fps, max_frames=args.max_frames)
    print(f"[Keyframe Extractor] Processed {stats['total_raw_frames']} frames -> Extracted {stats['keyframes_extracted']} keyframes to {args.output_dir}")
