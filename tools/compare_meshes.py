"""
Interactive 3D Mesh Comparison Tool (Open3D)
Allows instantaneous 1-to-1 visual comparison between:
- Key [1]: 35m Filtered Mesh (runs/drone3d_custom/stage_08_mesh_filtered_35m/mesh_raw.ply)
- Key [2]: 40m Filtered Mesh (runs/drone3d_custom/stage_08_mesh_filtered_40m/mesh_raw.ply)
- Key [3]: Stage 07 Reference Mesh (runs/drone3d_custom/stage_07_mesh/mesh_raw.ply)

Camera viewpoint, orientation, and zoom are completely preserved across transitions.
"""

import os
import sys
import open3d as o3d
import numpy as np

def main():
    p_35 = "runs/drone3d_custom/stage_08_mesh_filtered_35m/mesh_raw.ply"
    p_40 = "runs/drone3d_custom/stage_08_mesh_filtered_40m/mesh_raw.ply"
    p_07 = "runs/drone3d_custom/stage_07_mesh/mesh_raw.ply"

    print("=" * 75)
    print("Loading meshes for comparative inspection...")
    print(f"  [1] 35m Mesh: {p_35}")
    mesh_35 = o3d.io.read_triangle_mesh(p_35)
    mesh_35.compute_vertex_normals()
    mesh_35.paint_uniform_color([0.2, 0.7, 0.4]) # Soft terrain green

    print(f"  [2] 40m Mesh: {p_40}")
    mesh_40 = o3d.io.read_triangle_mesh(p_40)
    mesh_40.compute_vertex_normals()
    mesh_40.paint_uniform_color([0.2, 0.6, 0.85]) # Soft cyan/blue

    mesh_07 = None
    if os.path.exists(p_07):
        print(f"  [3] Stage 07 Reference Mesh: {p_07}")
        mesh_07 = o3d.io.read_triangle_mesh(p_07)
        mesh_07.compute_vertex_normals()
        mesh_07.paint_uniform_color([0.85, 0.3, 0.3]) # Red alert color

    print("=" * 75)
    print("INTERACTIVE COMPARISON CONTROLS:")
    print("  - Press '1': Display 35m Cutoff Mesh (Core Terrain, Roads, Buildings)")
    print("  - Press '2': Display 40m Cutoff Mesh (Extended Ground Margin)")
    print("  - Press '3': Display Stage 07 Original Mesh (Reference with Spire)")
    print("  - Press 'W': Toggle Wireframe Mode")
    print("  - Press 'N': Toggle Normals Display")
    print("  - Mouse Left-Click + Drag:  Rotate")
    print("  - Mouse Right-Click + Drag: Pan")
    print("  - Mouse Scroll Wheel:       Zoom")
    print("  - Press 'Q' or 'Esc':       Quit")
    print("=" * 75)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        window_name="3D Mesh Inspection: Press [1] for 35m, [2] for 40m, [3] for Stage 07",
        width=1360,
        height=850,
        left=40,
        top=40
    )

    current = {"id": 1, "mesh": mesh_35}
    vis.add_geometry(mesh_35)

    def switch_to_mesh(target_id, target_mesh, label):
        if current["id"] == target_id:
            return False
        
        # Save camera state
        ctr = vis.get_view_control()
        cam_params = ctr.convert_to_pinhole_camera_parameters()
        
        vis.remove_geometry(current["mesh"], reset_bounding_box=False)
        vis.add_geometry(target_mesh, reset_bounding_box=False)
        current["id"] = target_id
        current["mesh"] = target_mesh
        
        # Restore camera state so viewpoint and zoom remain 100% identical
        ctr.convert_from_pinhole_camera_parameters(cam_params)
        print(f"--> Switched to [{target_id}]: {label} ({len(target_mesh.vertices):,} verts, {len(target_mesh.triangles):,} faces)")
        return False

    vis.register_key_callback(ord('1'), lambda v: switch_to_mesh(1, mesh_35, "35m Mesh (Clean Core Terrain)"))
    vis.register_key_callback(ord('2'), lambda v: switch_to_mesh(2, mesh_40, "40m Mesh (Extended Ground Margin)"))
    if mesh_07:
        vis.register_key_callback(ord('3'), lambda v: switch_to_mesh(3, mesh_07, "Stage 07 Reference Mesh (With Spire)"))

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
