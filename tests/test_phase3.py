"""
Unit Tests for Phase 3 Semantic Classification & GIS Deliverables
Tests:
- ASPRS Semantic Point Cloud Classifier
- LAS file writing with classification bytes
- GeoTIFF Orthomosaic rasterization
"""

import os
import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth.depth_interface import write_ply
from src.pointcloud.pointcloud_classifier import classify_pointcloud
from src.export.gis_exporter import generate_orthomosaic_raster


class TestPhase3Classification(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/scratch_phase3"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_pointcloud_classification(self):
        """Test ASPRS classification distinguishing ground, buildings, and vegetation."""
        ply_path = os.path.join(self.test_dir, "test_scene.ply")
        las_path = os.path.join(self.test_dir, "classified.las")
        colored_ply = os.path.join(self.test_dir, "classified_colored.ply")

        # 1. Ground points at Z = 0
        ground_x, ground_y = np.meshgrid(np.linspace(0, 30, 20), np.linspace(0, 30, 20))
        ground_pts = np.stack([ground_x.flatten(), ground_y.flatten(), np.zeros(400)], axis=1)

        # 2. Building roof points at Z = 12 (flat, high planarity)
        roof_x, roof_y = np.meshgrid(np.linspace(10, 20, 10), np.linspace(10, 20, 10))
        roof_pts = np.stack([roof_x.flatten(), roof_y.flatten(), np.full(100, 12.0)], axis=1)

        # 3. Tree/vegetation points with random scatter at Z = 6
        tree_pts = np.random.uniform(5, 8, (60, 3))
        tree_pts[:, 0] += 22
        tree_pts[:, 1] += 5

        all_pts = np.vstack([ground_pts, roof_pts, tree_pts])
        write_ply(ply_path, all_pts)

        stats = classify_pointcloud(ply_path, las_path, colored_ply, k_neighbors=10, building_height_thresh=3.0)

        dist = stats["distribution"]
        self.assertGreater(dist["ground"] + dist["road"], 200)
        self.assertGreater(dist["building"], 50)
        self.assertTrue(os.path.exists(las_path))

    def test_orthomosaic_generation(self):
        """Test GeoTIFF orthomosaic export."""
        dummy_tex = os.path.join(self.test_dir, "dummy_tex.png")
        out_tif = os.path.join(self.test_dir, "ortho.tif")

        import cv2
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        img[:] = (50, 100, 50)
        cv2.imwrite(dummy_tex, img)

        res_path = generate_orthomosaic_raster(
            mesh_obj_path="dummy.obj",
            texture_png_path=dummy_tex,
            output_geotiff_path=out_tif,
            resolution_pixels=256
        )
        self.assertTrue(os.path.exists(res_path))


if __name__ == "__main__":
    unittest.main()
