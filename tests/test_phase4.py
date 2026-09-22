"""
Unit Tests for Phase 4 Dimensional Analytics, Occlusion Coverage & Degraded Fallbacks
Tests:
- Building Footprint Area, Perimeter, and Heights (Shapely)
- Ray-intersection Honest Coverage Metric
- GPS-Denied Fallback Mode Georeferencing
"""

import os
import sys
import json
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.depth.depth_interface import write_ply
from src.export.measurement_reporter import calculate_building_metrics
from src.utils.fallback_modes import compute_honest_coverage, generate_gps_denied_georef


class TestPhase4Analytics(unittest.TestCase):

    def setUp(self):
        self.test_dir = "tests/scratch_phase4"
        os.makedirs(self.test_dir, exist_ok=True)

    def test_building_dimensional_measurements(self):
        """Test Shapely-based footprint area, perimeter, and peak/eave heights."""
        ply_path = os.path.join(self.test_dir, "test_warehouse_pts.ply")
        json_path = os.path.join(self.test_dir, "measurement_report.json")

        # Ground points at z = 0 (100m x 100m)
        gx, gy = np.meshgrid(np.linspace(0, 100, 20), np.linspace(0, 100, 20))
        ground_pts = np.stack([gx.flatten(), gy.flatten(), np.zeros(400)], axis=1)

        # Warehouse roof points: 40m wide (x: 20 to 60) by 25m long (y: 20 to 45) -> Area = 1000 m²
        # Roof height: 10m
        rx, ry = np.meshgrid(np.linspace(20, 60, 20), np.linspace(20, 45, 15))
        roof_pts = np.stack([rx.flatten(), ry.flatten(), np.full(300, 10.0)], axis=1)

        all_pts = np.vstack([ground_pts, roof_pts])
        write_ply(ply_path, all_pts)

        metrics = calculate_building_metrics(ply_path, json_path)
        self.assertTrue(os.path.exists(json_path))

        dims = metrics["structural_dimensions"]
        # Expected footprint: 40 * 25 = 1000 m²
        self.assertAlmostEqual(dims["footprint_area_sqm"], 1000.0, delta=50.0)
        # Expected perimeter: 2 * (40 + 25) = 130 m
        self.assertAlmostEqual(dims["perimeter_length_meters"], 130.0, delta=10.0)
        # Expected peak height: ~10.0 m
        self.assertAlmostEqual(dims["peak_height_meters"], 10.0, delta=1.0)
        # Estimated volume: ~10,000 m³
        self.assertGreater(dims["estimated_volume_cubic_meters"], 8000.0)

    def test_honest_coverage_metric(self):
        """Test camera viewing ray coverage calculation over a mesh."""
        mesh_ply = os.path.join(self.test_dir, "test_mesh.ply")
        cov_json = os.path.join(self.test_dir, "coverage_report.json")

        # Create a simple flat 2-triangle quad facing up (Z=0)
        verts = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [10.0, 10.0, 0.0],
            [0.0, 10.0, 0.0]
        ], dtype=np.float32)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3]
        ], dtype=np.int32)

        # Write PLY with vertices and faces
        with open(mesh_ply, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            for v in verts:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

        # Camera looking straight down from Z=50
        camera_centers = {"cam_001": np.array([5.0, 5.0, 50.0])}
        cov_stats = compute_honest_coverage(mesh_ply, camera_centers, cov_json)

        self.assertTrue(os.path.exists(cov_json))
        self.assertEqual(cov_stats["total_mesh_triangles"], 2)
        self.assertEqual(cov_stats["observed_triangles"], 2)
        self.assertAlmostEqual(cov_stats["coverage_metric"], 1.0)

    def test_gps_denied_fallback(self):
        """Test graceful degraded mode fallback when GPS is missing."""
        georef_fallback_json = os.path.join(self.test_dir, "georef_fallback.json")
        res = generate_gps_denied_georef(georef_fallback_json, default_scale=1.0)

        self.assertTrue(os.path.exists(georef_fallback_json))
        self.assertEqual(res["mode"], "GPS_DENIED_FALLBACK")
        self.assertEqual(res["scale"], 1.0)
        self.assertIn("visual_scale_prior", res)


if __name__ == "__main__":
    unittest.main()
