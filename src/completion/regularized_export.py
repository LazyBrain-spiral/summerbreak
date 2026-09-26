"""Export regularised buildings: view-textured GLB (one node per building), hybrid model,
the cut surface, and per-building metadata (doors, windows, floors, colours)."""

from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.completion.regularize import cut_buildings_from_surface, wall_colors


def _to_gltf(v: np.ndarray) -> np.ndarray:
    """ENU (x east, y north, z up) -> glTF (x east, y up, z south)."""
    return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]).astype(np.float32)


def _jpeg_image(rgb: np.ndarray, quality: int = 88):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf)


def _textured_nodes(regs, atlas_bgr: np.ndarray):
    """One trimesh per building (roof + walls merged) sharing the atlas material."""
    import trimesh

    tex = _jpeg_image(np.ascontiguousarray(atlas_bgr[..., ::-1]))
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, metallicFactor=0.0, roughnessFactor=0.9,
                                              doubleSided=True)
    nodes = []
    for r in regs:
        V, F, UV = [], [], []
        for g, t in r.get("_groups", []):
            if "uv" not in g:
                continue
            F.append(g["f"] + sum(len(x) for x in V))
            V.append(g["v"])
            UV.append(g["uv"])
        if not V:
            continue
        m = trimesh.Trimesh(_to_gltf(np.vstack(V)), np.vstack(F),
                            visual=trimesh.visual.TextureVisuals(uv=np.vstack(UV), material=mat), process=False)
        nodes.append((f"b{r['id']}", m))
    return nodes


def _vertex_colour_nodes(regs, surf_v, surf_c, tree):
    """Fallback when texturing is off: walls from nearest surface colours, roofs flat colour."""
    import trimesh

    nodes = []
    for r in regs:
        parts = []
        if len(r["roof_f"]):
            parts.append(trimesh.Trimesh(_to_gltf(r["roof_v"]), r["roof_f"],
                                         vertex_colors=np.tile([170, 90, 60, 255], (len(r["roof_v"]), 1)), process=False))
        if len(r["wall_f"]):
            wc, _ = wall_colors(r["wall_v"], surf_v, surf_c, tree)
            parts.append(trimesh.Trimesh(_to_gltf(r["wall_v"]), r["wall_f"],
                                         vertex_colors=np.column_stack([wc, np.full(len(wc), 255)]), process=False))
        if parts:
            nodes.append((f"b{r['id']}", trimesh.util.concatenate(parts)))
    return nodes


def export_regularized(regs: List[Dict[str, Any]], geo, xmin: float, ymax: float, gsd: float, grid_shape,
                       out_dir: str, ortho_path: Optional[str], surface_path: Optional[str],
                       cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    import open3d as o3d
    import trimesh
    from scipy.spatial import cKDTree

    cfg = cfg or {}
    run_dir = os.path.dirname(os.path.abspath(out_dir))
    out: Dict[str, Any] = {}

    surf = None
    surf_v = surf_c = tree = None
    if surface_path and os.path.exists(surface_path):
        surf = o3d.io.read_triangle_mesh(surface_path)
        surf_v = np.asarray(surf.vertices)
        surf_c = np.asarray(surf.vertex_colors) if surf.has_vertex_colors() else None
        if surf_c is not None and len(surf_v):
            tree = cKDTree(surf_v)

    nodes = None
    texel = float(cfg.get("texel_m", 0.06))
    if cfg.get("texture", True) and os.path.exists(os.path.join(run_dir, "05_depth", "cameras.json")):
        try:
            from src.completion.texturing import texture_buildings

            ortho = None
            if ortho_path and os.path.exists(ortho_path):
                import rasterio

                with rasterio.open(ortho_path) as src:
                    rgb = np.stack([src.read(i) for i in (1, 2, 3)], -1)
                    alpha = src.read(4) if src.count >= 4 else np.full(rgb.shape[:2], 255, np.uint8)
                ortho = (np.ascontiguousarray(rgb[..., ::-1]), alpha, geo, xmin, ymax, gsd)
            tx = texture_buildings(regs, run_dir, texel=texel, max_atlas=int(cfg.get("atlas_max_px", 4096)),
                                   ortho=ortho)
            if tx.get("atlas") is not None:
                import cv2

                cv2.imwrite(os.path.join(out_dir, "buildings_atlas.jpg"), tx["atlas"], [cv2.IMWRITE_JPEG_QUALITY, 90])
                nodes = _textured_nodes(regs, tx["atlas"])
                out["texture"] = {"atlas_px": tx["size"], "atlas_scale": round(tx["scale"], 3), "texel_m": texel,
                                  "cameras": tx["cameras"]}
        except Exception as exc:  # texturing is an enhancement; never lose the geometry over it
            out["texture_error"] = f"{type(exc).__name__}: {exc}"[:300]
    if nodes is None:
        nodes = _vertex_colour_nodes(regs, surf_v, surf_c, tree)

    # ---- metadata (doors, windows, floors, colours)
    meta: Dict[str, Any] = {}
    if cfg.get("metadata", True):
        from src.completion.facade import building_metadata

        model = cfg.get("facade_model") if cfg.get("detect_facades", True) else None
        for r in regs:
            try:
                meta[str(r["id"])] = building_metadata(r, bool(geo.epsg), model, texel)
            except Exception as exc:
                meta[str(r["id"])] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                if model and "facade_error" not in out:
                    out["facade_error"] = meta[str(r["id"])]["error"]
                    model = None  # do not retry a broken detector for every building
            m = meta[str(r["id"])]
            m.update({k: r.get(k) for k in ("texture_observed_frac", "walls_observed", "walls_total",
                                            "face_provenance", "roof_pitch_deg")})
        with open(os.path.join(out_dir, "buildings_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        out["metadata_json"] = os.path.join(out_dir, "buildings_meta.json")
    out["meta"] = meta

    scene = trimesh.Scene()
    for name, m in nodes:
        scene.add_geometry(m, node_name=name, geom_name=name)
    reg_path = os.path.join(out_dir, "buildings_regularized.glb")
    scene.export(reg_path)
    out["buildings_glb"] = reg_path

    # ---- hybrid: fused surface with replaced buildings cut out
    if surf is not None and len(surf_v):
        f = np.asarray(surf.triangles)
        keep = cut_buildings_from_surface(surf_v, f, [r["ring_enu"] for r in regs], [r["base_z"] for r in regs])
        cut = o3d.geometry.TriangleMesh(surf)
        cut.remove_triangles_by_mask(~keep)
        cut.remove_unreferenced_vertices()
        cut_path = os.path.join(out_dir, "surface_without_buildings.ply")
        o3d.io.write_triangle_mesh(cut_path, cut)
        out["surface_cut_ply"] = cut_path
        out["surface_faces_removed"] = int((~keep).sum())
        cv = np.asarray(cut.vertices)
        cc = (np.clip(np.asarray(cut.vertex_colors), 0, 1) * 255).astype(np.uint8) if cut.has_vertex_colors() else None
        terrain = trimesh.Trimesh(_to_gltf(cv), np.asarray(cut.triangles),
                                  vertex_colors=None if cc is None else np.column_stack([cc, np.full(len(cc), 255)]),
                                  process=False)
        hyb = trimesh.Scene()
        hyb.add_geometry(terrain, node_name="terrain", geom_name="terrain")
        for name, m in nodes:
            hyb.add_geometry(m, node_name=name, geom_name=name)
        hyb_path = os.path.join(out_dir, "model_hybrid.glb")
        hyb.export(hyb_path)
        out["hybrid_glb"] = hyb_path
    return out
