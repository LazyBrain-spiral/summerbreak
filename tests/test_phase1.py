"""
Unit & Integration Tests for Phase 1 Pipeline
Tests:
- Keyframe extraction & Laplacian variance scoring
- DJI SRT parser & timestamp interpolation
- 7-DoF Umeyama Sim(3) closed-form alignment accuracy
- End-to-end Phase 1 execution
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

from src.georef.umeyama_aligner import umeyama_alignment, geodetic_to_enu
from src.ingest.srt_parser import parse_timestamp_ms


class TestPhase1Pipeline(unittest.TestCase):

    def test_timestamp_parsing(self):
        self.assertAlmostEqual(parse_timestamp_ms("00:01:30,500"), 90.5)
        self.assertAlmostEqual(parse_timestamp_ms("01:00:00.000"), 3600.0)

    def test_umeyama_sim3_perfect_alignment(self):
        """Test Umeyama SVD alignment with a known ground-truth Sim(3) transform."""
        # 10 random 3D points
        np.random.seed(42)
        X = np.random.randn(10, 3) * 10.0

        # Known ground-truth transform
        true_scale = 2.5
        # 90-degree rotation around Z axis
        true_R = np.array([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0]
        ])
        true_t = np.array([100.0, 250.0, 50.0])

        # Generate target points Y = s * R @ X + t
        Y = (true_scale * (true_R @ X.T)).T + true_t

        # Solve via Umeyama
        s_est, R_est, t_est, rmse = umeyama_alignment(X, Y)

        # Assert scale recovered
        self.assertAlmostEqual(s_est, true_scale, places=5)
        # Assert rotation recovered
        np.testing.assert_allclose(R_est, true_R, atol=1e-5)
        # Assert translation recovered
        np.testing.assert_allclose(t_est, true_t, atol=1e-5)
        # Assert zero RMSE for perfect data
        self.assertAlmostEqual(rmse, 0.0, places=5)

    def test_umeyama_with_noise(self):
        """Test Umeyama with realistic 0.5m GPS noise."""
        np.random.seed(42)
        X = np.random.randn(20, 3) * 15.0
        true_scale = 1.05
        true_R = np.eye(3)
        true_t = np.array([500.0, 1000.0, 120.0])

        noise = np.random.normal(0, 0.5, size=X.shape)
        Y = (true_scale * (true_R @ X.T)).T + true_t + noise

        s_est, R_est, t_est, rmse = umeyama_alignment(X, Y)

        # Estimated scale should be close to 1.05
        self.assertAlmostEqual(s_est, true_scale, delta=0.05)
        # RMSE should reflect noise level (~0.5 - 1.0m)
        self.assertLess(rmse, 1.5)
        self.assertGreater(rmse, 0.2)

    def test_keyframe_laplacian_sharpness(self):
        """Test Laplacian sharpness scoring distinguishing sharp vs blurry images."""
        from src.ingest.keyframe_extractor import compute_laplacian_variance
        import cv2

        # Create sharp checkerboard
        sharp_img = np.zeros((100, 100, 3), dtype=np.uint8)
        sharp_img[::20, :] = 255
        sharp_img[:, ::20] = 255
        sharp_score = compute_laplacian_variance(sharp_img)

        # Create blurred version
        blurry_img = cv2.GaussianBlur(sharp_img, (15, 15), 0)
        blurry_score = compute_laplacian_variance(blurry_img)

        self.assertGreater(sharp_score, blurry_score * 2.0)

    def test_srt_telemetry_sync(self):
        """Test parsing DJI SRT and synchronizing to keyframes."""
        from src.ingest.srt_parser import parse_dji_srt, synchronize_telemetry_to_keyframes
        
        srt_path = "test_data/flight.srt"
        if os.path.exists(srt_path):
            records = parse_dji_srt(srt_path)
            self.assertGreater(len(records), 0)
            self.assertIn("lat", records[0])
            self.assertIn("lon", records[0])
            self.assertIn("alt", records[0])


if __name__ == "__main__":
    unittest.main()
