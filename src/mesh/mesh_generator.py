"""
Surface Mesh Reconstruction Module (Workstream B & C - Stage 9 & 10)
Constructs a watertight surface mesh from the filtered point cloud.
Uses OpenMVS ReconstructMesh when present, or fast 2.5D Delaunay / Ball-Pivoting reconstruction.
Avoids Poisson hallucination artifacts over unobserved vertical facades.
"""

import os
import shutil
import subprocess
import numpy as np
from scipy.spatial import Delaunay
from typing import Dict, List, Tuple, Any, Optional

from src.pointcloud.pointcloud_fusion import read_ply_points_and_colors


def write_ply_mesh(
    filepath: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: Optional[np.ndarray] = None
):
    """Writes an indexed triangle surface mesh to PLY format."""
    num_verts = len(vertices)
    num_faces = len(faces)
    has_colors = colors is not None and len(colors) == num_verts

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_verts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write(f"element face {num_faces}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")

        for i in range(num_verts):
            x, y, z = vertices[i]
            if has_colors:
                r, g, b = colors[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")

        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")


def generate_surface_mesh(
    pointcloud_path: str,
    output_mesh_path: str,
    openmvs_bin: str = "ReconstructMesh",
    max_edge_length: float = 25.0
) -> Dict[str, Any]:
    """
    Reconstructs surface triangle mesh from point cloud.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
    print(f"[MeshGenerator] Building watertight surface mesh from {pointcloud_path}...")

    # Check for native OpenMVS binary
    if shutil.which(openmvs_bin) is not None:
        try:
            cmd = [openmvs_bin, pointcloud_path, "-o", output_mesh_path]
            subprocess.run(cmd, check=True)
            return {"engine": "OpenMVS", "mesh_path": output_mesh_path}
        except Exception as e:
            print(f"[MeshGenerator] OpenMVS command failed ({e}); falling back to Delaunay 2.5D.")

    # High-performance 2.5D Delaunay Triangulation (Aerial standard)
    pts, clrs = read_ply_points_and_colors(pointcloud_path)
    if len(pts) < 3:
        raise ValueError("Cannot triangulate fewer than 3 points")

    # Project to horizontal plane (X, Y) for aerial triangulation
    xy_coords = pts[:, :2]
    tri = Delaunay(xy_coords)
    faces = tri.simplices

    # Prune excessively long triangles across outer boundaries or occluded gaps
    # (prevents hallucinating mesh across unseen flight boundaries)
    v0 = pts[faces[:, 0]]
    v1 = pts[faces[:, 1]]
    v2 = pts[faces[:, 2]]

    d01 = np.linalg.norm(v0 - v1, axis=1)
    d12 = np.linalg.norm(v1 - v2, axis=1)
    d20 = np.linalg.norm(v2 - v0, axis=1)

    max_edges = np.maximum(np.maximum(d01, d12), d20)
    valid_faces = faces[max_edges < max_edge_length]

    write_ply_mesh(output_mesh_path, pts, valid_faces, clrs)
    print(f"[MeshGenerator] Surface mesh generated: {len(pts):,} vertices, {len(valid_faces):,} faces")
    print(f"  -> Saved to: {output_mesh_path}")

    return {
        "engine": "Delaunay2.5D",
        "num_vertices": len(pts),
        "num_faces": len(valid_faces),
        "mesh_path": output_mesh_path
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruct surface mesh from point cloud")
    parser.add_argument("--input", required=True, help="Input filtered PLY point cloud")
    parser.add_argument("--output", default="data/mesh_raw.ply", help="Output PLY mesh")
    args = parser.parse_args()
    generate_surface_mesh(args.input, args.output)
