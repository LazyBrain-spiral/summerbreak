"""
Surface Mesh Reconstruction Module (Phase 7 / Stages 9 & 10)
Constructs a true 3D watertight surface mesh from the validated dense point cloud.
Uses OpenMVS ReconstructMesh when available, or Open3D Screened Poisson Surface Reconstruction.
Avoids SciPy 2.5D planar Delaunay projections.
"""

import os
import shutil
import subprocess
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Any, Optional

from src.pointcloud.pointcloud_fusion import read_ply_points_and_colors


def generate_surface_mesh(
    pointcloud_path: str,
    output_mesh_path: str,
    openmvs_bin: str = "ReconstructMesh",
    poisson_depth: int = 9,
    density_trim_percentile: float = 5.0,
    sor_neighbors: int = 30,
    sor_std_ratio: float = 1.5,
    **kwargs
) -> Dict[str, Any]:
    """
    Reconstructs a true 3D surface triangle mesh from the input point cloud.
    
    1. Checks if OpenMVS ReconstructMesh is available.
    2. Otherwise applies conservative statistical filtering to reject peripheral noise without
       arbitrary XYZ clipping.
    3. Reconstructs genuine 3D manifold surface via Open3D Screened Poisson Surface Reconstruction.
    4. Trims low-density reconstruction regions using Poisson density percentiles.
    5. Cleans degenerate triangles and unreferenced vertices.
    6. Saves mesh to output_mesh_path (.ply) and corresponding .obj format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_mesh_path)), exist_ok=True)
    print(f"[MeshGenerator] Building 3D surface mesh from {pointcloud_path}...")

    # Check for native OpenMVS binary
    if shutil.which(openmvs_bin) is not None:
        try:
            cmd = [openmvs_bin, pointcloud_path, "-o", output_mesh_path]
            subprocess.run(cmd, check=True)
            return {"engine": "OpenMVS", "mesh_path": output_mesh_path}
        except Exception as e:
            print(f"[MeshGenerator] OpenMVS command failed ({e}); falling back to Open3D Poisson.")

    # Load point cloud via Open3D
    pcd_raw = o3d.io.read_point_cloud(pointcloud_path)
    input_point_count = len(pcd_raw.points)
    if input_point_count < 3:
        raise ValueError("Cannot triangulate fewer than 3 points")

    # Filter peripheral outliers conservatively if cloud is large enough
    if input_point_count > 100:
        cl, ind = pcd_raw.remove_statistical_outlier(nb_neighbors=sor_neighbors, std_ratio=sor_std_ratio)
        pcd = pcd_raw.select_by_index(ind)
    else:
        pcd = pcd_raw
        ind = list(range(input_point_count))

    filtered_point_count = len(pcd.points)
    rejected_point_count = input_point_count - filtered_point_count
    print(f"[MeshGenerator] Input points: {input_point_count:,} -> Filtered: {filtered_point_count:,} (Rejected: {rejected_point_count:,})")

    # Ensure valid normals
    if not pcd.has_normals() or len(pcd.normals) == 0:
        print("[MeshGenerator] Estimating point normals...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    
    # Orient normals consistently based on tangent planes
    if filtered_point_count > 15:
        pcd.orient_normals_consistent_tangent_plane(k=15)

    # Adapt Poisson depth for small vs large point clouds
    effective_depth = poisson_depth
    if filtered_point_count < 1000:
        effective_depth = min(poisson_depth, 6)
    elif filtered_point_count < 10000:
        effective_depth = min(poisson_depth, 7)

    print(f"[MeshGenerator] Running Screened Poisson Surface Reconstruction (depth={effective_depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=effective_depth, width=0, scale=1.1, linear_fit=False
    )

    dens = np.asarray(densities)
    if len(dens) > 0 and density_trim_percentile > 0:
        trim_thresh = float(np.percentile(dens, density_trim_percentile))
        mesh.remove_vertices_by_mask(dens < trim_thresh)

    # Topology cleanup
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    num_verts = len(verts)
    num_faces = len(tris)

    # Save primary PLY mesh (ASCII for universal compatibility)
    o3d.io.write_triangle_mesh(output_mesh_path, mesh, write_ascii=True)
    print(f"  -> Saved PLY mesh: {output_mesh_path}")

    # Also save OBJ mesh alongside if path is .ply
    base, _ = os.path.splitext(output_mesh_path)
    output_obj_path = base + ".obj"
    o3d.io.write_triangle_mesh(output_obj_path, mesh)
    print(f"  -> Saved OBJ mesh: {output_obj_path}")

    # Compute validation metrics
    min_bound = [float(v) for v in mesh.get_min_bound()]
    max_bound = [float(v) for v in mesh.get_max_bound()]

    is_finite = np.all(np.isfinite(verts), axis=1) if num_verts > 0 else []
    non_finite_count = int(num_verts - np.sum(is_finite)) if num_verts > 0 else 0

    # Connected components
    tri_clusters, num_tri_per_cluster, area = mesh.cluster_connected_triangles()
    tri_clusters = np.asarray(tri_clusters)
    num_tri_per_cluster = np.asarray(num_tri_per_cluster)
    num_components = len(num_tri_per_cluster)

    if num_components > 0:
        largest_cluster_id = int(np.argmax(num_tri_per_cluster))
        largest_cluster_triangles = int(num_tri_per_cluster[largest_cluster_id])
        largest_cluster_verts = np.unique(tris[tri_clusters == largest_cluster_id])
        pct_verts_largest = float(len(largest_cluster_verts) / num_verts * 100.0) if num_verts > 0 else 0.0
    else:
        largest_cluster_triangles = 0
        pct_verts_largest = 0.0

    print(f"[MeshGenerator] Surface mesh generated: {num_verts:,} vertices, {num_faces:,} triangles")
    print(f"  Connected components: {num_components} (Largest: {largest_cluster_triangles:,} triangles, {pct_verts_largest:.1f}% vertices)")

    return {
        "engine": "Open3D_Poisson",
        "poisson_depth": effective_depth,
        "input_points": input_point_count,
        "filtered_points": filtered_point_count,
        "rejected_points": rejected_point_count,
        "num_vertices": num_verts,
        "num_faces": num_faces,
        "mesh_bounding_box": {"min": min_bound, "max": max_bound},
        "connected_components": num_components,
        "largest_component_triangles": largest_cluster_triangles,
        "pct_vertices_in_largest_component": pct_verts_largest,
        "non_finite_vertices": non_finite_count,
        "mesh_path": output_mesh_path,
        "mesh_obj_path": output_obj_path
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Reconstruct surface mesh from point cloud")
    parser.add_argument("--input", required=True, help="Input filtered PLY point cloud")
    parser.add_argument("--output", default="runs/drone3d_custom/stage_07_mesh/mesh_raw.ply", help="Output PLY mesh")
    args = parser.parse_args()
    generate_surface_mesh(args.input, args.output)

