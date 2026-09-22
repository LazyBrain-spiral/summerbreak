"""
DJI SRT Telemetry Parser Module (Workstream A - Stage 1 & 2)
Parses DJI video SRT subtitles or flight-log CSV, extracts per-frame GPS/IMU coordinates,
and synchronizes/interpolates them to keyframe timestamps into frame_gps.csv.
"""

import os
import re
import csv
from typing import List, Dict, Any, Optional
import numpy as np


def parse_timestamp_ms(time_str: str) -> float:
    """Convert SRT time format HH:MM:SS,mmm or HH:MM:SS.mmm to seconds."""
    time_str = time_str.replace(',', '.')
    parts = time_str.strip().split(':')
    if len(parts) == 3:
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return h * 3600.0 + m * 60.0 + s
    return 0.0


def parse_dji_srt(srt_path: str) -> List[Dict[str, Any]]:
    """
    Parse DJI SRT file into a list of telemetry points.
    Matches standard DJI subtitle blocks containing latitude, longitude, altitude, gimbal angles.
    """
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    # Direct CSV flight log support
    if srt_path.lower().endswith('.csv'):
        telemetry_records = []
        with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                lat, lon, alt = None, None, 50.0
                pitch, yaw, roll = -60.0, 0.0, 0.0
                t_sec = idx * 0.5
                for k, v in row.items():
                    kl = k.lower().strip()
                    try:
                        if 'time' in kl: t_sec = float(v)
                        elif 'lat' in kl: lat = float(v)
                        elif 'lon' in kl or 'lng' in kl: lon = float(v)
                        elif 'alt' in kl: alt = float(v)
                        elif 'pitch' in kl: pitch = float(v)
                        elif 'yaw' in kl: yaw = float(v)
                        elif 'roll' in kl: roll = float(v)
                    except (ValueError, TypeError):
                        pass
                if lat is not None and lon is not None:
                    telemetry_records.append({
                        "timestamp_sec": t_sec,
                        "lat": lat, "lon": lon, "alt": alt,
                        "pitch": pitch, "yaw": yaw, "roll": roll
                    })
        return telemetry_records

    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split into SRT subtitle blocks (separated by blank lines)
    blocks = re.split(r'\n\s*\n', content.strip())
    telemetry_records = []

    # Regex patterns for DJI SRT telemetry
    # Common format 1: [latitude: 28.6139] [longitude: 77.2090] [rel_alt: 50.2 abs_alt: 250.2]
    # Common format 2: LATITUDE: 28.6139 LONGITUDE: 77.2090 ALTITUDE: 50.2
    lat_pat = re.compile(r'(?:\[?latitude\s*[:=]\s*|\blat\s*[:=]\s*)([-+]?\d*\.?\d+)', re.IGNORECASE)
    lon_pat = re.compile(r'(?:\[?longitude\s*[:=]\s*|\blon\s*[:=]\s*)([-+]?\d*\.?\d+)', re.IGNORECASE)
    alt_pat = re.compile(r'(?:\[?rel_alt\s*[:=]\s*|\[?altitude\s*[:=]\s*|\balt\s*[:=]\s*)([-+]?\d*\.?\d+)', re.IGNORECASE)
    pitch_pat = re.compile(r'(?:\[?gimbal_pitch\s*[:=]\s*|\bpitch\s*[:=]\s*)([-+]?\d*\.?\d+)', re.IGNORECASE)
    yaw_pat = re.compile(r'(?:\[?gimbal_yaw\s*[:=]\s*|\byaw\s*[:=]\s*)([-+]?\d*\.?\d+)', re.IGNORECASE)
    roll_pat = re.compile(r'(?:\[?gimbal_roll\s*[:=]\s*|\broll\s*[:=]\s*)([-+]?\d*\.?\d+)', re.IGNORECASE)
    time_range_pat = re.compile(r'(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})')

    for block in blocks:
        lines = block.strip().split('\n')
        if not lines:
            continue

        timestamp = 0.0
        # Find timestamp line
        for line in lines:
            m = time_range_pat.search(line)
            if m:
                t_start = parse_timestamp_ms(m.group(1))
                t_end = parse_timestamp_ms(m.group(2))
                timestamp = (t_start + t_end) / 2.0
                break

        full_text = " ".join(lines)
        lat_m = lat_pat.search(full_text)
        lon_m = lon_pat.search(full_text)
        alt_m = alt_pat.search(full_text)

        if lat_m and lon_m:
            lat = float(lat_m.group(1))
            lon = float(lon_m.group(1))
            alt = float(alt_m.group(1)) if alt_m else 50.0
            
            pitch_m = pitch_pat.search(full_text)
            yaw_m = yaw_pat.search(full_text)
            roll_m = roll_pat.search(full_text)

            pitch = float(pitch_m.group(1)) if pitch_m else -90.0
            yaw = float(yaw_m.group(1)) if yaw_m else 0.0
            roll = float(roll_m.group(1)) if roll_m else 0.0

            telemetry_records.append({
                "timestamp_sec": timestamp,
                "lat": lat,
                "lon": lon,
                "alt": alt,
                "roll": roll,
                "pitch": pitch,
                "yaw": yaw
            })

    telemetry_records.sort(key=lambda x: x["timestamp_sec"])
    return telemetry_records


def synchronize_telemetry_to_keyframes(
    telemetry_records: List[Dict[str, Any]],
    keyframes_metadata: List[Dict[str, Any]],
    output_csv_path: str,
) -> str:
    """
    Interpolate telemetry records to match each extracted keyframe's exact timestamp,
    exporting the final frame_gps.csv file.
    """
    if not telemetry_records:
        raise ValueError("Telemetry records list is empty. Cannot synchronize.")

    t_tel = np.array([r["timestamp_sec"] for r in telemetry_records])
    lats = np.array([r["lat"] for r in telemetry_records])
    lons = np.array([r["lon"] for r in telemetry_records])
    alts = np.array([r["alt"] for r in telemetry_records])
    rolls = np.array([r["roll"] for r in telemetry_records])
    pitches = np.array([r["pitch"] for r in telemetry_records])
    yaws = np.array([r["yaw"] for r in telemetry_records])

    # Check if telemetry is in absolute time-of-day while keyframes start near zero
    kf_times = [kf["timestamp_sec"] for kf in keyframes_metadata]
    if len(t_tel) > 0 and len(kf_times) > 0:
        if t_tel[0] > 1000.0 and kf_times[0] < 100.0:
            # Telemetry is using time-of-day (e.g. 10:00:00); normalize to start at 0
            t_tel = t_tel - t_tel[0]

    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["frame_id", "filename", "timestamp_sec", "lat", "lon", "alt", "roll", "pitch", "yaw"])

        for kf in keyframes_metadata:
            t = kf["timestamp_sec"]
            # 1D linear interpolation with edge extrapolation
            lat_interp = float(np.interp(t, t_tel, lats))
            lon_interp = float(np.interp(t, t_tel, lons))
            alt_interp = float(np.interp(t, t_tel, alts))
            roll_interp = float(np.interp(t, t_tel, rolls))
            pitch_interp = float(np.interp(t, t_tel, pitches))
            yaw_interp = float(np.interp(t, t_tel, yaws))

            writer.writerow([
                kf["frame_id"],
                kf["filename"],
                f"{t:.4f}",
                f"{lat_interp:.8f}",
                f"{lon_interp:.8f}",
                f"{alt_interp:.3f}",
                f"{roll_interp:.2f}",
                f"{pitch_interp:.2f}",
                f"{yaw_interp:.2f}"
            ])

    return output_csv_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Parse DJI SRT telemetry")
    parser.add_argument("--srt", required=True, help="Input SRT file")
    parser.add_argument("--output", default="data/frame_gps.csv", help="Output CSV path")
    args = parser.parse_args()

    recs = parse_dji_srt(args.srt)
    print(f"[SRT Parser] Parsed {len(recs)} telemetry points from {args.srt}")
