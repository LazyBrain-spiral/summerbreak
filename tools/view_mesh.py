"""
Interactive 3D Mesh Viewer using Open3D.
Allows interactive rotation, panning, zooming, wireframe inspection, and normal rendering.
"""

import sys
import os
import open3d as o3d

def main():
    default_ply = "runs/drone3d_custom/stage_07_mesh/mesh_raw.ply"
    default_obj = "runs/drone3d_custom/stage_07_mesh/mesh_raw.obj"

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    elif os.path.exists(default_ply):
        target_path = default_ply
    elif os.path.exists(default_obj):
        target_path = default_obj
    else:
        print(f"Error: Neither {default_ply} nor {default_obj} found.")
        sys.exit(1)

    print("=" * 70)
    print(f"Loading 3D Mesh: {target_path}")
    mesh = o3d.io.read_triangle_mesh(target_path)
    if mesh.is_empty():
        print(f"Error: Mesh at {target_path} is empty or could not be loaded.")
        sys.exit(1)

    print(f"Vertices:  {len(mesh.vertices):,}")
    print(f"Triangles: {len(mesh.triangles):,}")
    print("=" * 70)
    print("Controls:")
    print("  - Left Mouse Button + Drag:  Rotate view")
    print("  - Right Mouse Button + Drag: Pan / translate view")
    print("  - Mouse Scroll Wheel:        Zoom in / out")
    print("  - Key 'W':                   Toggle wireframe mode")
    print("  - Key 'N':                   Toggle surface normals")
    print("  - Key 'R':                   Reset viewpoint")
    print("  - Key 'H':                   Print help controls")
    print("  - Key 'Q' or 'Esc':          Close viewer")
    print("=" * 70)

    # Compute vertex normals for smooth Phong shading if not present
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [mesh],
        window_name=f"Interactive 3D Mesh Inspection - {os.path.basename(target_path)}",
        width=1280,
        height=800,
        left=50,
        top=50
    )

if __name__ == "__main__":
    main()
