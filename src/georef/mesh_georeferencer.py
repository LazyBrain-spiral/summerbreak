"""
Mesh & Point Cloud Georeferencer Module (Workstream B - Stage 9 / Sim(3) Transform)
Applies the 7-DoF Umeyama Sim(3) transformation matrix (s * R * x + t) from georef.json
directly to mesh vertices and point clouds, producing metric-scaled, georeferenced outputs.
"""

import os
import json
import numpy as np
from typing import Dict, Any, Tuple

from src.pointcloud.pointcloud_fusion import read_ply_points_and_colors
from src.depth.depth_interface import write_ply


def load_sim3_transform(georef_json_path: str) -> Tuple[float, np.ndarray, np.ndarray]:
    """Loads scale s, rotation R (3x3), and translation t (3,) from georef.json."""
    if not os.path.exists(georef_json_path):
        raise FileNotFoundError(f"georef.json not found: {georef_json_path}")

    with open(georef_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    s = float(data["scale"])
    R = np.array(data["rotation"], dtype=np.float64)
    t = np.array(data["translation"], dtype=np.float64)

    return s, R, t


def transform_points(points: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Applies Y = s * R @ X + t to array of 3D coordinates (N, 3)."""
    return (s * (R @ points.T)).T + t


def georeference_obj_mesh(
    input_obj_path: str,
    output_obj_path: str,
    georef_json_path: str
) -> str:
    """
    Transforms all vertex coordinates in an OBJ mesh using Sim(3) parameters.
    Preserves all normals, texture coordinates (vt), materials, and face definitions.
    """
    s, R, t = load_sim3_transform(georef_json_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_obj_path)), exist_ok=True)

    print(f"[Georeferencer] Applying Sim(3) to mesh {input_obj_path} (s={s:.4f})...")

    with open(input_obj_path, 'r', encoding='utf-8') as fin, open(output_obj_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    pt = np.array([[x, y, z]], dtype=np.float64)
                    pt_georef = transform_points(pt, s, R, t)[0]
                    fout.write(f"v {pt_georef[0]:.4f} {pt_georef[1]:.4f} {pt_georef[2]:.4f}\n")
                else:
                    fout.write(line)
            else:
                fout.write(line)

    print(f"[Georeferencer] Georeferenced mesh saved to: {output_obj_path}")
    return output_obj_path


def georeference_ply_pointcloud(
    input_ply_path: str,
    output_ply_path: str,
    georef_json_path: str
) -> str:
    """
    Transforms point cloud to metric georeferenced coordinates.
    """
    s, R, t = load_sim3_transform(georef_json_path)
    pts, clrs = read_ply_points_and_colors(input_ply_path)

    pts_georef = transform_points(pts, s, R, t)
    write_ply(output_ply_path, pts_georef, clrs)

    print(f"[Georeferencer] Georeferenced point cloud saved to: {output_ply_path}")
    return output_ply_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Georeference 3D mesh via Sim(3)")
    parser.add_argument("--input_obj", required=True, help="Input OBJ mesh")
    parser.add_argument("--georef_json", required=True, help="Path to georef.json")
    parser.add_argument("--output_obj", default="data/model_georeferenced.obj", help="Output OBJ")
    args = parser.parse_args()
    georeference_obj_mesh(args.input_obj, args.output_obj, args.georef_json)
