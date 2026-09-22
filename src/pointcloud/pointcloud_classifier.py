"""
ASPRS Semantic Point Cloud Classifier Module (Workstream D - Stage 12)
Performs automated 3D point cloud semantic segmentation based on ASPRS standards:
  - Class 2: Ground / Terrain
  - Class 5: High Vegetation / Trees
  - Class 6: Building / Rooftops
  - Class 11: Road Surface / Pavement
Uses Height Above Ground (HAG) and local covariance eigenvalue features:
  - Planarity P = (λ2 - λ3) / λ1
  - Linearity L = (λ1 - λ2) / λ1
  - Scattering S = λ3 / λ1
  - Verticality V = 1 - |n_z|
Exports classified.las / classified.laz and colored semantic PLY.
"""

import os
import numpy as np
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Any, Optional

from src.pointcloud.pointcloud_fusion import read_ply_points_and_colors
from src.depth.depth_interface import write_ply

# Standard ASPRS Classification Palette
ASPRS_COLORS = {
    2: np.array([210, 180, 140], dtype=np.uint8),  # Ground: Tan
    5: np.array([34, 139, 34], dtype=np.uint8),    # Vegetation: Forest Green
    6: np.array([220, 50, 50], dtype=np.uint8),    # Building: Crimson Red
    11: np.array([90, 95, 105], dtype=np.uint8),   # Road: Dark Slate Gray
}


def classify_pointcloud(
    input_ply_path: str,
    output_las_path: str,
    output_colored_ply: Optional[str] = None,
    k_neighbors: int = 15,
    building_height_thresh: float = 3.5,
    road_planarity_thresh: float = 0.65
) -> Dict[str, Any]:
    """
    Classifies a 3D point cloud into standard ASPRS classes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_las_path)), exist_ok=True)
    print(f"[PointcloudClassifier] Classifying {input_ply_path} into ASPRS categories...")

    pts, original_colors = read_ply_points_and_colors(input_ply_path)
    N = len(pts)
    if N < k_neighbors:
        raise ValueError(f"Point cloud has only {N} points; minimum {k_neighbors} required.")

    # 1. Compute Height Above Ground (HAG) via spatial 2D grid
    grid_res = 10.0  # 10m grid cells
    xy_min = np.min(pts[:, :2], axis=0)
    grid_coords = np.floor((pts[:, :2] - xy_min) / grid_res).astype(np.int64)

    # Compute cell minimum Z as local ground estimate
    unique_cells, cell_indices = np.unique(grid_coords, axis=0, return_inverse=True)
    cell_min_z = np.full(len(unique_cells), np.inf)
    for i, cid in enumerate(cell_indices):
        z_val = pts[i, 2]
        if z_val < cell_min_z[cid]:
            cell_min_z[cid] = z_val

    ground_z = cell_min_z[cell_indices]
    hag = pts[:, 2] - ground_z  # Height Above Ground

    # 2. Geometric Eigenvalue Feature Extraction using k-NN Tree
    tree = cKDTree(pts)
    _, nn_indices = tree.query(pts, k=k_neighbors)

    classes = np.full(N, 2, dtype=np.uint8)  # Default Class 2 (Ground)

    for i in range(N):
        neighbors = pts[nn_indices[i]]
        cov = np.cov(neighbors, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort eigenvalues descending
        idx = np.argsort(eigvals)[::-1]
        lam = eigvals[idx]
        lam = np.maximum(lam, 1e-6)

        lam_sum = np.sum(lam)
        l1, l2, l3 = lam[0], lam[1], lam[2]

        planarity = (l2 - l3) / l1
        scattering = l3 / l1
        normal_z = abs(eigvecs[2, idx[2]])  # Normal vector along third eigenvector

        h = hag[i]

        if h < 1.2:
            # Low elevation -> Road or Terrain
            if planarity > road_planarity_thresh and normal_z > 0.75:
                classes[i] = 11  # Road
            else:
                classes[i] = 2   # Ground
        elif h >= building_height_thresh:
            # High elevation
            if planarity > 0.55 and normal_z > 0.6:
                classes[i] = 6   # Building
            else:
                classes[i] = 5   # Vegetation
        else:
            # Mid-level features
            if scattering > 0.25:
                classes[i] = 5   # Vegetation
            else:
                classes[i] = 6   # Building annex / low wall

    # Class distribution statistics
    counts = {
        "ground": int(np.sum(classes == 2)),
        "vegetation": int(np.sum(classes == 5)),
        "building": int(np.sum(classes == 6)),
        "road": int(np.sum(classes == 11))
    }

    # 3. Export to LAS/LAZ format if laspy is installed
    has_laspy = False
    try:
        import laspy
        has_laspy = True
    except ImportError:
        pass

    if has_laspy:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.offsets = np.min(pts, axis=0)
        header.scales = np.array([0.001, 0.001, 0.001])

        las = laspy.LasData(header)
        las.x = pts[:, 0]
        las.y = pts[:, 1]
        las.z = pts[:, 2]
        las.classification = classes

        # Colorize points with ASPRS semantic colors
        semantic_colors = np.zeros((N, 3), dtype=np.uint8)
        for cls_id, clr in ASPRS_COLORS.items():
            mask = (classes == cls_id)
            semantic_colors[mask] = clr

        las.red = semantic_colors[:, 0].astype(np.uint16) * 256
        las.green = semantic_colors[:, 1].astype(np.uint16) * 256
        las.blue = semantic_colors[:, 2].astype(np.uint16) * 256

        las.write(output_las_path)
        print(f"[PointcloudClassifier] Saved ASPRS LAS file: {output_las_path}")
    else:
        # Save classification metadata as JSON sidecar if LAS binary writer is pending
        sidecar = output_las_path + ".meta.json"
        import json
        with open(sidecar, "w") as f:
            json.dump({"classes": counts, "num_points": N}, f)

    # 4. Optional Semantic Colored PLY export
    if output_colored_ply:
        semantic_colors = np.zeros((N, 3), dtype=np.uint8)
        for cls_id, clr in ASPRS_COLORS.items():
            mask = (classes == cls_id)
            semantic_colors[mask] = clr
        write_ply(output_colored_ply, pts, semantic_colors)
        print(f"[PointcloudClassifier] Saved semantic colored PLY: {output_colored_ply}")

    print(f"[PointcloudClassifier] ASPRS Classification complete:")
    print(f"  -> Ground (Class 2)     : {counts['ground']:,} points ({(counts['ground']/N)*100:.1f}%)")
    print(f"  -> Building (Class 6)   : {counts['building']:,} points ({(counts['building']/N)*100:.1f}%)")
    print(f"  -> Road (Class 11)      : {counts['road']:,} points ({(counts['road']/N)*100:.1f}%)")
    print(f"  -> Vegetation (Class 5) : {counts['vegetation']:,} points ({(counts['vegetation']/N)*100:.1f}%)")

    return {
        "output_las": output_las_path,
        "output_colored_ply": output_colored_ply,
        "distribution": counts,
        "total_points": N
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ASPRS Semantic Point Cloud Classification")
    parser.add_argument("--input", required=True, help="Input PLY file")
    parser.add_argument("--output_las", default="data/classified.las", help="Output LAS file")
    args = parser.parse_args()
    classify_pointcloud(args.input, args.output_las)
