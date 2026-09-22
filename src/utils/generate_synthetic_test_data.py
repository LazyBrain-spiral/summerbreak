"""
Synthetic Drone Flight Data Generator
Generates a realistic short video clip (10 seconds @ 30fps) with aerial ground textures,
moving objects (simulated vehicles), and a matching DJI SRT subtitle telemetry track.
Enables immediate testing of Phase 1 pipeline without requiring external drone video.
"""

import os
import cv2
import numpy as np
from datetime import datetime, timedelta


def generate_flight_data(output_dir: str, duration_sec: int = 6, fps: int = 30):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "flight.mp4")
    srt_path = os.path.join(output_dir, "flight.srt")

    width, height = 1280, 720
    total_frames = duration_sec * fps

    # OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Base GPS coordinates (e.g. New Delhi campus)
    base_lat = 28.613900
    base_lon = 77.209000
    base_alt = 55.0  # 55 meters above ground

    srt_entries = []
    base_time = datetime(2026, 9, 20, 0, 0, 0)

    print(f"[Synthetic Generator] Generating {total_frames} frames of synthetic drone footage...")

    # Static ground features (roads, buildings, trees)
    ground_canvas = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    # Ground color (dirt/grass)
    ground_canvas[:] = (45, 95, 45)
    # Draw gray road
    cv2.line(ground_canvas, (0, height), (width * 2, height), (70, 70, 70), 120)
    # Road dash lines
    for x in range(0, width * 2, 80):
        cv2.line(ground_canvas, (x, height), (x + 40, height), (220, 220, 220), 4)

    # Draw some buildings (polygons)
    cv2.rectangle(ground_canvas, (300, 200), (600, 500), (140, 130, 120), -1)
    cv2.rectangle(ground_canvas, (320, 220), (580, 480), (110, 100, 90), -1)
    cv2.rectangle(ground_canvas, (900, 800), (1300, 1100), (160, 150, 140), -1)

    # Moving vehicle positions
    car_x = 100
    car_speed = 8  # pixels per frame

    for f_idx in range(total_frames):
        t_sec = f_idx / fps
        t_start = base_time + timedelta(seconds=t_sec)
        t_end = base_time + timedelta(seconds=(f_idx + 1) / fps)

        # Drone camera moves along ground
        offset_x = int(f_idx * 4)
        offset_y = int(np.sin(f_idx * 0.05) * 20)

        # Crop view
        x1 = offset_x
        y1 = height // 4 + offset_y
        frame = ground_canvas[y1:y1 + height, x1:x1 + width].copy()

        # Draw moving dynamic car on the road
        car_draw_x = (car_x + f_idx * car_speed) - offset_x
        car_draw_y = height // 2 - 20
        if 0 <= car_draw_x <= width - 80:
            # Draw blue vehicle
            cv2.rectangle(frame, (car_draw_x, car_draw_y), (car_draw_x + 70, car_draw_y + 35), (200, 40, 30), -1)
            cv2.rectangle(frame, (car_draw_x + 10, car_draw_y + 5), (car_draw_x + 50, car_draw_y + 30), (240, 180, 50), -1)

        # Add light camera jitter / blur variation
        if f_idx % 25 == 0:
            frame = cv2.GaussianBlur(frame, (5, 5), 0)

        out.write(frame)

        # Drone coordinates (advancing along latitude/longitude)
        curr_lat = base_lat + (f_idx * 0.000015)
        curr_lon = base_lon + (f_idx * 0.000008)
        curr_alt = base_alt + np.sin(f_idx * 0.08) * 1.5

        # Format SRT entry
        time_str_start = t_start.strftime("%H:%M:%S,%f")[:12]
        time_str_end = t_end.strftime("%H:%M:%S,%f")[:12]

        srt_block = (
            f"{f_idx + 1}\n"
            f"{time_str_start} --> {time_str_end}\n"
            f"<font size=\"28\">Srt:{f_idx + 1}/{total_frames}\n"
            f"[latitude: {curr_lat:.6f}] [longitude: {curr_lon:.6f}] [altitude: {curr_alt:.2f}] [rel_alt: {curr_alt:.2f}]\n"
            f"[gimbal_pitch: -60.0] [gimbal_roll: 0.0] [gimbal_yaw: 45.0]\n"
            f"</font>\n"
        )
        srt_entries.append(srt_block)

    out.release()

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries))

    print(f"[Synthetic Generator] Successfully created:")
    print(f"  -> Video: {video_path}")
    print(f"  -> Telemetry: {srt_path}")
    return video_path, srt_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic drone video and SRT telemetry")
    parser.add_argument("--output_dir", default="test_data", help="Output directory")
    parser.add_argument("--duration", type=int, default=6, help="Duration in seconds")
    args = parser.parse_args()
    generate_flight_data(args.output_dir, duration_sec=args.duration)
