"""GLB export for the web viewer (v3). Coordinates stay in local ENU (metres, z up);
the ENU origin and UTM EPSG go into the scene extras so a viewer can place the model."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def ply_to_glb(ply_path: str, glb_path: str, georef: Dict[str, Any], extras: Optional[Dict[str, Any]] = None) -> str:
    import open3d as o3d
    import trimesh

    m = o3d.io.read_triangle_mesh(ply_path)
    v = np.asarray(m.vertices)
    f = np.asarray(m.triangles)
    colors = (np.asarray(m.vertex_colors) * 255).astype(np.uint8) if m.has_vertex_colors() else None
    # glTF is y-up: rotate ENU (x east, y north, z up) to (x east, y up, z south)
    v_gltf = np.column_stack([v[:, 0], v[:, 2], -v[:, 1]])
    mesh = trimesh.Trimesh(vertices=v_gltf, faces=f, vertex_colors=colors, process=False)
    scene = trimesh.Scene(mesh)
    scene.metadata["extras"] = {"enu_origin": georef.get("enu_origin"), "utm_epsg": georef.get("utm_epsg"),
                                "axes": "glTF y-up; x=east, y=up, z=-north", **(extras or {})}
    scene.export(glb_path)
    return glb_path
