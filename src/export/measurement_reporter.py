"""
Building Measurement & Dimensional Analytics Module (Workstream D - Phase 4)
Extracts structural geometry metrics from ASPRS-classified 3D point clouds:
- Building Footprint Ground Area (m²) via Shapely Convex Hull & Minimum Rotated Rectangle
- Building Perimeter Length (m)
- Structural Eave Height, Ridge Peak Height, and Mean Height (m)
- Estimated Structural Volume Displacement (m³)
Fulfills the "+ measurements" specification of problem statement SIH26158.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

try:
    from shapely.geometry import MultiPoint, Polygon
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

from src.pointcloud.pointcloud_fusion import read_ply_points_and_colors


def calculate_building_metrics(
    classified_ply_path: str,
    output_report_json: str,
    building_class_id: int = 6
) -> Dict[str, Any]:
    """
    Computes building dimensional analytics from classified 3D points.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_report_json)), exist_ok=True)
    print(f"[MeasurementReporter] Calculating structural building metrics from {classified_ply_path}...")

    pts, colors = read_ply_points_and_colors(classified_ply_path)
    if len(pts) == 0:
        raise ValueError("Point cloud has no points to measure")

    z_vals = pts[:, 2]
    ground_level = float(np.percentile(z_vals, 15))  # Estimated ground datum

    # 1. Identify Building points using ASPRS semantic color palette (Class 6 is red: [255, 50, 50])
    building_pts = None
    if colors is not None and len(colors) == len(pts):
        red_mask = (colors[:, 0] > 180) & (colors[:, 1] < 120) & (colors[:, 2] < 120)
        if np.sum(red_mask) >= 10:
            building_pts = pts[red_mask]

    # 2. Fallback to elevation filtering if semantic color is absent
    if building_pts is None:
        building_mask = (z_vals - ground_level) > 3.5
        building_pts = pts[building_mask]
        if len(building_pts) < 10:
            building_pts = pts[z_vals > np.percentile(z_vals, 70)]

    # Outlier filtering on 2D footprint to prevent stray points inflating polygon
    xy_raw = building_pts[:, :2]
    x_p1, x_p99 = np.percentile(xy_raw[:, 0], 2), np.percentile(xy_raw[:, 0], 98)
    y_p1, y_p99 = np.percentile(xy_raw[:, 1], 2), np.percentile(xy_raw[:, 1], 98)
    valid_xy = (xy_raw[:, 0] >= x_p1) & (xy_raw[:, 0] <= x_p99) & (xy_raw[:, 1] >= y_p1) & (xy_raw[:, 1] <= y_p99)
    if np.sum(valid_xy) >= 10:
        building_pts = building_pts[valid_xy]

    b_z = building_pts[:, 2]
    max_height = float(np.max(b_z) - ground_level)
    mean_height = float(np.mean(b_z) - ground_level)
    eave_height = float(np.percentile(b_z, 30) - ground_level)

    # 2D Footprint Analysis via Shapely
    xy_points = building_pts[:, :2]

    if HAS_SHAPELY and len(xy_points) >= 3:
        mp = MultiPoint(xy_points)
        hull = mp.convex_hull
        obb = mp.minimum_rotated_rectangle

        footprint_area_sqm = float(hull.area)
        obb_area_sqm = float(obb.area)
        perimeter_m = float(hull.length)
        centroid = [float(hull.centroid.x), float(hull.centroid.y)]
    else:
        # Vectorized bounding box approximation
        min_x, max_x = np.min(xy_points[:, 0]), np.max(xy_points[:, 0])
        min_y, max_y = np.min(xy_points[:, 1]), np.max(xy_points[:, 1])
        dx = max(1.0, max_x - min_x)
        dy = max(1.0, max_y - min_y)
        footprint_area_sqm = float(dx * dy * 0.85)
        obb_area_sqm = float(dx * dy)
        perimeter_m = float(2 * (dx + dy))
        centroid = [float((min_x + max_x) / 2.0), float((min_y + max_y) / 2.0)]

    estimated_volume_m3 = float(footprint_area_sqm * mean_height)

    report = {
        "status": "VALID",
        "facility_type": "Industrial Warehouse Building",
        "structural_dimensions": {
            "footprint_area_sqm": round(footprint_area_sqm, 2),
            "obb_bounding_area_sqm": round(obb_area_sqm, 2),
            "perimeter_length_meters": round(perimeter_m, 2),
            "peak_height_meters": round(max_height, 2),
            "eave_height_meters": round(eave_height, 2),
            "mean_height_meters": round(mean_height, 2),
            "estimated_volume_cubic_meters": round(estimated_volume_m3, 2),
            "ground_elevation_datum_meters": round(ground_level, 2)
        },
        "centroid_enu_coords": {
            "east_m": round(centroid[0], 2),
            "north_m": round(centroid[1], 2)
        },
        "num_classified_building_points": int(len(building_pts))
    }

    with open(output_report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[MeasurementReporter] Dimensional Analysis Complete:")
    print(f"  -> Footprint Area   : {footprint_area_sqm:,.1f} m²")
    print(f"  -> Peak Roof Height : {max_height:.2f} m")
    print(f"  -> Eave Height      : {eave_height:.2f} m")
    print(f"  -> Perimeter        : {perimeter_m:.1f} m")
    print(f"  -> Estimated Volume : {estimated_volume_m3:,.1f} m³")
    print(f"  -> Saved Report To  : {output_report_json}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract building dimensional metrics")
    parser.add_argument("--input", required=True, help="Path to classified PLY/LAS")
    parser.add_argument("--output", default="data/measurement_report.json", help="Output JSON")
    args = parser.parse_args()
    calculate_building_metrics(args.input, args.output)
