"""
Industrial Warehouse Drone Flight Generator
Generates realistic aerial survey footage over a logistics warehouse facility:
- Large distribution warehouse building (gable roof, rooftop HVAC units, loading docks)
- Office annex and asphalt truck apron
- Dynamic moving semi-trucks / delivery vans on access roads
- Synchronized DJI SRT telemetry flight path (lawnmower survey at 50m AGL)
"""

import os
import cv2
import numpy as np
from datetime import datetime, timedelta


def draw_warehouse_site(canvas_w: int = 2400, canvas_h: int = 1800) -> np.ndarray:
    """Renders high-detail 2D aerial site map of a warehouse logistics hub."""
    site = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    
    # Ground base: grass / perimeter soil
    site[:] = (55, 90, 50)

    # Asphalt access roads & truck maneuvering apron
    cv2.rectangle(site, (200, 200), (2200, 1600), (60, 60, 62), -1)
    # Concrete apron around warehouse
    cv2.rectangle(site, (450, 400), (1950, 1400), (135, 135, 140), -1)

    # Road marking lines
    cv2.line(site, (200, 900), (450, 900), (220, 220, 220), 4)
    cv2.line(site, (1950, 900), (2200, 900), (220, 220, 220), 4)

    # Main Distribution Warehouse Building (800m x 450m in canvas coords)
    wh_x1, wh_y1 = 650, 550
    wh_x2, wh_y2 = 1750, 1250

    # Warehouse shadow (cast to bottom-right)
    cv2.rectangle(site, (wh_x1 + 35, wh_y1 + 35), (wh_x2 + 35, wh_y2 + 35), (25, 30, 25), -1)

    # Warehouse metallic roof base (corrugated industrial light grey)
    cv2.rectangle(site, (wh_x1, wh_y1), (wh_x2, wh_y2), (190, 195, 200), -1)

    # Gabled roof ridge line along length
    mid_y = (wh_y1 + wh_y2) // 2
    cv2.line(site, (wh_x1, mid_y), (wh_x2, mid_y), (140, 145, 150), 6)

    # Roof panels / seams across width
    for x in range(wh_x1 + 40, wh_x2, 60):
        cv2.line(site, (x, wh_y1), (x, wh_y2), (170, 175, 180), 2)

    # Industrial HVAC cooling units & skylights on roof
    for x in range(wh_x1 + 100, wh_x2 - 100, 180):
        # North side units
        cv2.rectangle(site, (x, wh_y1 + 80), (x + 70, wh_y1 + 140), (80, 85, 90), -1)
        cv2.rectangle(site, (x + 10, wh_y1 + 90), (x + 60, wh_y1 + 130), (120, 125, 130), -1)
        # South side units
        cv2.rectangle(site, (x, wh_y2 - 140), (x + 70, wh_y2 - 80), (80, 85, 90), -1)
        cv2.rectangle(site, (x + 10, wh_y2 - 130), (x + 60, wh_y2 - 90), (120, 125, 130), -1)

    # Skylight strips along roof
    for x in range(wh_x1 + 60, wh_x2 - 60, 120):
        cv2.rectangle(site, (x, mid_y - 25), (x + 45, mid_y + 25), (210, 230, 245), -1)

    # Loading Docks along the South face (trailers backed into bays)
    dock_y = wh_y2
    for dock_x in range(wh_x1 + 80, wh_x2 - 120, 100):
        # Dock bay opening
        cv2.rectangle(site, (dock_x, dock_y - 8), (dock_x + 50, dock_y + 8), (40, 40, 40), -1)
        # Parked trailer in alternating docks
        if (dock_x // 100) % 2 == 0:
            cv2.rectangle(site, (dock_x + 5, dock_y + 8), (dock_x + 45, dock_y + 110), (225, 230, 235), -1)
            # Trailer tractor head
            cv2.rectangle(site, (dock_x + 7, dock_y + 110), (dock_x + 43, dock_y + 145), (190, 40, 30), -1)

    # Office Annex building on West side
    cv2.rectangle(site, (480, 750), (650, 1050), (160, 150, 140), -1)
    cv2.rectangle(site, (500, 770), (630, 1030), (130, 120, 110), -1)

    # Employee car parking spaces
    for py in range(720, 1080, 30):
        cv2.rectangle(site, (360, py), (420, py + 22), (220, 220, 220), 2)
        if py % 60 == 0:
            # Parked car
            cv2.rectangle(site, (365, py + 2), (415, py + 20), (30, 70, 160), -1)

    return site


def generate_warehouse_dataset(
    output_dir: str = "data_warehouse",
    duration_sec: int = 8,
    fps: int = 30
):
    """Generates warehouse inspection video clip and DJI telemetry track."""
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "warehouse_flight.mp4")
    srt_path = os.path.join(output_dir, "warehouse_flight.srt")

    width, height = 1280, 720
    total_frames = duration_sec * fps
    site_map = draw_warehouse_site()
    site_h, site_w = site_map.shape[:2]

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Real-world coordinates: Industrial Logistics Hub (Manesar, Haryana)
    base_lat = 28.368500
    base_lon = 76.938200
    base_alt = 52.0  # 52 meters AGL survey altitude

    srt_entries = []
    base_time = datetime(2026, 9, 20, 0, 0, 0)

    # Dynamic delivery truck traveling along the South perimeter road
    truck_start_x = 350
    truck_speed = 7.0  # pixels/frame

    print(f"[WarehouseGenerator] Generating {total_frames} frames of warehouse aerial survey...")

    # Survey flight path: Sweeping from West to East diagonally across the warehouse
    sweep_start_x = 250
    sweep_start_y = 350

    for f_idx in range(total_frames):
        t_sec = f_idx / fps
        t_start = base_time + timedelta(seconds=t_sec)
        t_end = base_time + timedelta(seconds=(f_idx + 1) / fps)

        # Camera viewport position
        cam_x = int(sweep_start_x + f_idx * 5.2)
        cam_y = int(sweep_start_y + np.sin(f_idx * 0.04) * 35)

        # Copy background viewport
        cam_x = max(0, min(site_w - width, cam_x))
        cam_y = max(0, min(site_h - height, cam_y))
        frame = site_map[cam_y:cam_y + height, cam_x:cam_x + width].copy()

        # Render moving semi-truck on access road (dynamic entity)
        truck_curr_x = int(truck_start_x + f_idx * truck_speed)
        truck_curr_y = 1480  # along south apron road
        truck_screen_x = truck_curr_x - cam_x
        truck_screen_y = truck_curr_y - cam_y

        if 0 <= truck_screen_x <= width - 110 and 0 <= truck_screen_y <= height - 40:
            # Trailer body (white)
            cv2.rectangle(frame, (truck_screen_x, truck_screen_y), (truck_screen_x + 90, truck_screen_y + 35), (245, 245, 250), -1)
            # Tractor cab (red)
            cv2.rectangle(frame, (truck_screen_x + 90, truck_screen_y + 3), (truck_screen_x + 120, truck_screen_y + 32), (30, 40, 210), -1)

        # Light camera turbulence / exposure variation
        if f_idx % 20 == 0:
            frame = cv2.GaussianBlur(frame, (3, 3), 0)

        out.write(frame)

        # GPS progression
        curr_lat = base_lat + (f_idx * 0.000012)
        curr_lon = base_lon + (f_idx * 0.000018)
        curr_alt = base_alt + np.sin(f_idx * 0.05) * 1.2

        # Format DJI SRT block
        time_str_start = t_start.strftime("%H:%M:%S,%f")[:12]
        time_str_end = t_end.strftime("%H:%M:%S,%f")[:12]

        srt_block = (
            f"{f_idx + 1}\n"
            f"{time_str_start} --> {time_str_end}\n"
            f"<font size=\"28\">Srt:{f_idx + 1}/{total_frames}\n"
            f"[latitude: {curr_lat:.6f}] [longitude: {curr_lon:.6f}] [altitude: {curr_alt:.2f}] [rel_alt: {curr_alt:.2f}]\n"
            f"[gimbal_pitch: -65.0] [gimbal_roll: 0.0] [gimbal_yaw: 85.0]\n"
            f"</font>\n"
        )
        srt_entries.append(srt_block)

    out.release()

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(srt_entries))

    print(f"[WarehouseGenerator] Complete:")
    print(f"  -> Video: {video_path} ({total_frames} frames)")
    print(f"  -> Telemetry: {srt_path}")
    return video_path, srt_path


if __name__ == "__main__":
    generate_warehouse_dataset()
