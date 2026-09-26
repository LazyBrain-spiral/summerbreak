import json
import os

import numpy as np
import pytest

pytest.importorskip("pyproj")
pytest.importorskip("pymap3d")

from src.completion.extrusion import PROVENANCE, extrude_buildings  # noqa: E402
from src.products.geo import LocalGeo  # noqa: E402
from src.products.rasters import Grid, build_dtm, progressive_morphological_filter  # noqa: E402

GEOREF = {"enu_origin": {"lat": 28.6139, "lon": 77.2090, "alt": 216.0, "alt_source": "abs_alt"},
          "utm_epsg": 32643}
GSD = 0.25


def _synthetic_grids(tmp_path):
    """Sloped terrain + one 30x16 m gabled building (eave 7, ridge 11) + one tree."""
    geo = LocalGeo(GEOREF)
    enu_extent = np.array([[-60.0, -40.0], [60.0, 40.0]])
    utm = geo.enu_to_utm(enu_extent)
    grid = Grid.around(utm, GSD, pad=0.0)
    centers = grid.centers().reshape(-1, 2)
    enu = geo.utm_to_enu(centers).reshape(grid.height, grid.width, 2)
    x, y = enu[..., 0], enu[..., 1]
    terrain = 0.03 * x + 0.5 * np.sin(y / 15.0)
    dsm = terrain.copy()
    inside = (np.abs(x + 10) <= 15) & (np.abs(y - 5) <= 8)
    ground_ref = 0.03 * -10 + 0.5 * np.sin(5 / 15.0)
    roof = ground_ref + 7.0 + 4.0 * (1 - np.abs(y - 5) / 8.0)  # ridge along x at y = 5
    dsm[inside] = roof[inside]
    tree = np.hypot(x - 35, y + 20)
    dsm = np.where(tree < 3.5, terrain + 8.0 * np.sqrt(np.clip(1 - (tree / 3.5) ** 2, 0, 1))
                   + 0.6 * np.random.default_rng(0).standard_normal(tree.shape), dsm)
    ground = progressive_morphological_filter(dsm, GSD, 40.0, 0.25, 0.3, 2.5)
    dtm = build_dtm(dsm, ground, GSD)
    path = os.path.join(tmp_path, "grids.npz")
    np.savez_compressed(path, dsm_up=dsm.astype(np.float32), dtm_up=dtm.astype(np.float32), ground=ground,
                        observed=np.ones_like(ground), xmin=grid.xmin, ymax=grid.ymax, gsd=GSD)
    return path, terrain, dtm, inside


def test_local_geo_roundtrip():
    geo = LocalGeo(GEOREF)
    pts = np.array([[0.0, 0.0], [350.0, -120.0], [-800.0, 900.0]])
    back = geo.utm_to_enu(geo.enu_to_utm(pts))
    np.testing.assert_allclose(back, pts, atol=1e-6)
    import pymap3d

    lat, lon, _ = pymap3d.enu2geodetic(800.0, -600.0, 0.0, 28.6139, 77.2090, 216.0)
    utm = geo._to_utm.transform(lon, lat)
    np.testing.assert_allclose(geo.enu_to_utm(np.array([[800.0, -600.0]]))[0], utm, atol=0.02)


def test_pmf_dtm_follows_terrain(tmp_path):
    _, terrain, dtm, inside = _synthetic_grids(tmp_path)
    err = np.abs(dtm - terrain)
    assert np.median(err) < 0.1
    assert np.percentile(err[inside], 90) < 0.8  # under the building, interpolated


def test_extrusion_measures_building(tmp_path):
    path, _, _, _ = _synthetic_grids(tmp_path)
    cfg = {"min_height_m": 2.5, "min_area_m2": 25.0, "max_roughness_m": 0.35, "max_curvature": 0.6, "simplify_m": 0.5, "subdivide": 2}
    res = extrude_buildings(path, GEOREF, str(tmp_path / "out"), cfg)
    assert res["num_buildings"] == 1, res["rejected_components"]
    # the tree at ENU (35, -20) must not become a building
    assert all(np.hypot(b["centroid_enu"][0] - 35, b["centroid_enu"][1] + 20) > 10 for b in res["buildings"])
    b = res["buildings"][0]
    assert abs(b["footprint_area_m2"] - 480.0) / 480.0 < 0.05, b
    assert abs(b["eave_height_m"] - 7.8) < 0.8, b  # 20th percentile of a gable is above the eave
    assert abs(b["ridge_height_m"] - 11.0) < 0.4, b
    assert res["face_provenance_counts"]["inferred"] > 0
    gj = json.load(open(tmp_path / "out" / "buildings.geojson"))
    ring = np.array(gj["features"][0]["geometry"]["coordinates"][0])
    assert 77.1 < ring[:, 0].mean() < 77.3 and 28.5 < ring[:, 1].mean() < 28.7


def test_extruded_mesh_is_closed_above_ground(tmp_path):
    import open3d as o3d

    path, _, _, _ = _synthetic_grids(tmp_path)
    cfg = {"min_height_m": 2.5, "min_area_m2": 25.0, "max_roughness_m": 0.35, "max_curvature": 0.6, "simplify_m": 0.5, "subdivide": 1}
    res = extrude_buildings(path, GEOREF, str(tmp_path / "out"), cfg)
    mesh = o3d.io.read_triangle_mesh(res["mesh"])
    mesh.merge_close_vertices(1e-6)
    mesh.remove_duplicated_vertices()
    tris = np.asarray(mesh.triangles)
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    # the only open boundary is the bottom ring (walls meet the roof with no gaps)
    verts = np.asarray(mesh.vertices)
    uniq, cnt = np.unique(edges, axis=0, return_counts=True)
    open_edges = uniq[cnt == 1]
    z = verts[open_edges].mean(axis=1)[:, 2]
    assert len(open_edges) > 0
    assert np.all(z < 3.0), f"open edges above the ground ring (eave is at ~7 m): {z.max():.2f}"
    assert PROVENANCE["observed"] == 0


def test_regularized_gable_is_rectangle_with_gable_roof(tmp_path):
    path, _, _, _ = _synthetic_grids(tmp_path)
    cfg = {"min_height_m": 2.5, "min_area_m2": 25.0, "max_roughness_m": 0.35, "max_curvature": 0.6,
           "simplify_m": 0.5, "subdivide": 1, "regularize": True}
    res = extrude_buildings(path, GEOREF, str(tmp_path / "out"), cfg)
    b = res["buildings"][0]
    assert b["shape"] == "rectangle", b
    assert b["roof_type"] == "gable_u", b
    assert abs(b["footprint_area_m2"] - 480.0) / 480.0 < 0.04, b
    assert abs(b["ridge_height_m"] - 11.0) < 0.4, b
    assert abs(b["eave_height_m"] - 7.0) < 0.5, b
    assert os.path.exists(res["regularized"]["buildings_glb"])


def test_regularized_l_shape_is_rectilinear():
    from src.completion.regularize import regularize_footprint

    m = np.zeros((200, 200), bool)
    m[40:160, 40:90] = True
    m[110:160, 40:170] = True
    fp = regularize_footprint(m, 0.25)
    assert fp["shape"] in ("rectilinear", "snapped polygon"), fp["candidates"]
    assert fp["iou"] > 0.9
    assert 6 <= len(fp["poly"]) <= 8
    # every corner is a right angle
    P = np.asarray(fp["poly"])
    d = np.roll(P, -1, axis=0) - P
    cosines = np.abs(np.sum(d * np.roll(d, 1, axis=0), axis=1)) / (np.linalg.norm(d, axis=1) * np.linalg.norm(np.roll(d, 1, axis=0), axis=1))
    assert cosines.max() < 0.05


def test_simplify_orthogonal_removes_small_steps():
    from src.completion.regularize import simplify_orthogonal

    # 10 x 6 rectangle with a 0.5-unit notch on the top edge
    poly = np.array([[0, 0], [10, 0], [10, 6], [6, 6], [6, 5.5], [4, 5.5], [4, 6], [0, 6]], float)
    out = simplify_orthogonal(poly, min_edge_px=1.0)
    assert len(out) == 4, out
