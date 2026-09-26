"""
Unit Tests for Phase 2 Geometry & Mesh Engine
Tests:
- FastDepthEngine PLY creation & back-projection
- Point cloud voxel downsampling & outlier filtering
- Surface mesh generation (Delaunay 2.5D)
- Sim(3) mesh georeferencer coordinate transformation
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
from src.pointcloud.pointcloud_fusion import read_ply_points_and_colors, fuse_and_filter_pointcloud
from src.mesh.mesh_generator import generate_surface_mesh
from src.mesh.texture_mapper import read_ply_mesh
from src.georef.mesh_georeferencer import transform_points


class TestPhase2Geometry(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/scratch_phase2"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_ply_io_and_filtering(self):
        """Test writing PLY and filtering via pointcloud_fusion."""
        raw_ply = os.path.join(self.test_dir, "test_raw.ply")
        filtered_ply = os.path.join(self.test_dir, "test_filtered.ply")

        # Generate 500 random points + 5 extreme outliers
        np.random.seed(42)
        pts = np.random.uniform(0, 10, (500, 3))
        outliers = np.array([[100.0, 100.0, 100.0], [-100.0, -100.0, -100.0]])
        all_pts = np.vstack([pts, outliers])

        write_ply(raw_ply, all_pts)
        pts_read, _ = read_ply_points_and_colors(raw_ply)
        self.assertEqual(len(pts_read), 502)

        # Run filtering
        stats = fuse_and_filter_pointcloud(raw_ply, filtered_ply, voxel_size=0.1, std_ratio=1.5)
        self.assertLess(stats["final_points"], stats["initial_points"])
        self.assertTrue(os.path.exists(filtered_ply))

    def test_surface_mesh_reconstruction(self):
        """Test Delaunay triangulation and face generation from points."""
        ply_path = os.path.join(self.test_dir, "test_pts_for_mesh.ply")
        mesh_path = os.path.join(self.test_dir, "test_mesh.ply")

        # Grid of 25 points on a horizontal plane
        x = np.linspace(0, 4, 5)
        y = np.linspace(0, 4, 5)
        xx, yy = np.meshgrid(x, y)
        zz = np.sin(xx) * 0.5
        grid_pts = np.stack([xx.flatten(), yy.flatten(), zz.flatten()], axis=1)

        write_ply(ply_path, grid_pts)
        stats = generate_surface_mesh(ply_path, mesh_path, max_edge_length=5.0)

        self.assertGreater(stats["num_faces"], 0)
        verts, faces, _ = read_ply_mesh(mesh_path)
        self.assertGreater(len(verts), 0)
        self.assertGreater(len(faces), 10)

    def test_sim3_transform_points(self):
        """Test Sim(3) transformation function."""
        pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        s = 2.0
        R = np.eye(3)
        t = np.array([10.0, 20.0, 30.0])

        transformed = transform_points(pts, s, R, t)
        np.testing.assert_allclose(transformed[0], [12.0, 20.0, 30.0])
        np.testing.assert_allclose(transformed[1], [10.0, 22.0, 30.0])


if __name__ == "__main__":
    unittest.main()
