"""
Global Structure-from-Motion (GLOMAP) Runner Module (Workstream B - Stage 4 & 5)
Orchestrates COLMAP feature extraction (with dynamic masking), sequential matching,
and GLOMAP global bundle adjustment to recover camera poses and sparse 3D geometry.
Includes fallback simulation for environments where GLOMAP binary is being compiled.
"""

import os
import shutil
import subprocess
import struct
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


def check_binary_available(binary_name: str) -> bool:
    """Check if a CLI binary is in PATH."""
    return shutil.which(binary_name) is not None


class GlomapRunner:
    def __init__(
        self,
        colmap_bin: str = "colmap",
        glomap_bin: str = "glomap",
        use_gpu: bool = True
    ):
        self.colmap_bin = colmap_bin
        self.glomap_bin = glomap_bin
        self.use_gpu = use_gpu

    def run_sfm_pipeline(
        self,
        image_dir: str,
        mask_dir: Optional[str],
        output_dir: str,
        overlap_window: int = 10,
    ) -> Dict[str, Any]:
        """
        Execute full SfM pipeline:
        1. Feature extraction with dynamic masks
        2. Sequential feature matching
        3. GLOMAP global reconstruction
        """
        os.makedirs(output_dir, exist_ok=True)
        database_path = os.path.join(output_dir, "database.db")
        sparse_dir = os.path.join(output_dir, "sparse", "0")
        os.makedirs(sparse_dir, exist_ok=True)

        has_colmap = check_binary_available(self.colmap_bin)
        has_glomap = check_binary_available(self.glomap_bin)

        if has_colmap:
            print(f"[SfM] Running COLMAP Feature Extraction on {image_dir}...")
            feat_cmd = [
                self.colmap_bin, "feature_extractor",
                "--database_path", database_path,
                "--image_path", image_dir,
                "--ImageReader.camera_model", "OPENCV",
                "--ImageReader.single_camera", "1",
                "--SiftExtraction.use_gpu", "1" if self.use_gpu else "0",
            ]
            if mask_dir and os.path.exists(mask_dir):
                feat_cmd.extend(["--ImageReader.mask_path", mask_dir])
            
            subprocess.run(feat_cmd, check=True)

            print(f"[SfM] Running COLMAP Sequential Matching (window={overlap_window})...")
            match_cmd = [
                self.colmap_bin, "sequential_matcher",
                "--database_path", database_path,
                "--SequentialMatching.overlap", str(overlap_window),
                "--SequentialMatching.quadratic_overlap", "1",
                "--SiftMatching.use_gpu", "1" if self.use_gpu else "0",
            ]
            subprocess.run(match_cmd, check=True)

            if has_glomap:
                print(f"[SfM] Running GLOMAP Global Mapper...")
                glomap_cmd = [
                    self.glomap_bin, "mapper",
                    "--database_path", database_path,
                    "--image_path", image_dir,
                    "--output_path", sparse_dir
                ]
                subprocess.run(glomap_cmd, check=True)
            else:
                print(f"[SfM] GLOMAP not found in PATH; falling back to COLMAP mapper...")
                colmap_map_cmd = [
                    self.colmap_bin, "mapper",
                    "--database_path", database_path,
                    "--image_path", image_dir,
                    "--output_path", os.path.join(output_dir, "sparse")
                ]
                subprocess.run(colmap_map_cmd, check=True)
        else:
            print(f"[SfM] Notice: Neither COLMAP nor GLOMAP found in PATH.")
            print(f"[SfM] Generating development synthetic camera poses for downstream integration testing...")
            self._generate_synthetic_sparse_model(image_dir, sparse_dir)

        # Parse recovered camera centers
        camera_centers = self.read_camera_centers(sparse_dir)
        return {
            "database_path": database_path,
            "sparse_dir": sparse_dir,
            "num_cameras_reconstructed": len(camera_centers),
            "camera_centers": camera_centers,
        }

    def _generate_synthetic_sparse_model(self, image_dir: str, sparse_dir: str):
        """Generates synthetic cameras.txt, images.txt, and points3D.txt for dev testing."""
        images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg'))])
        images_txt_path = os.path.join(sparse_dir, "images.txt")
        cameras_txt_path = os.path.join(sparse_dir, "cameras.txt")
        points_txt_path = os.path.join(sparse_dir, "points3D.txt")

        with open(cameras_txt_path, "w") as f:
            f.write("# Camera list\n1 PINHOLE 1920 1080 1500 1500 960 540\n")

        with open(images_txt_path, "w") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            # Simulating a forward lawnmower sweep in SfM coordinates proportional to flight
            for idx, img_name in enumerate(images, start=1):
                tx = idx * 2.5
                ty = idx * 2.0 + np.sin(idx * 0.5) * 0.2
                tz = 3.5 + np.cos(idx * 0.2) * 0.05
                qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
                f.write(f"{idx} {qw} {qx} {qy} {qz} {tx:.4f} {ty:.4f} {tz:.4f} 1 {img_name}\n\n")

        with open(points_txt_path, "w") as f:
            f.write("# 3D point list\n")
            for pid in range(1, 201):
                px = np.random.uniform(0, 100)
                py = np.random.uniform(-10, 10)
                pz = np.random.uniform(0, 5)
                f.write(f"{pid} {px:.3f} {py:.3f} {pz:.3f} 128 128 128 0.1 1 1\n")

    def read_camera_centers(self, sparse_dir: str) -> Dict[str, np.ndarray]:
        """
        Reads camera optical centers from COLMAP/GLOMAP sparse model (txt or bin).
        Returns a dict mapping image filename to 3D position vector in SfM frame.
        """
        images_txt = os.path.join(sparse_dir, "images.txt")
        images_bin = os.path.join(sparse_dir, "images.bin")
        camera_centers = {}

        if os.path.exists(images_txt):
            with open(images_txt, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    image_id = int(parts[0])
                    qw, qx, qy, qz = map(float, parts[1:5])
                    tx, ty, tz = map(float, parts[5:8])
                    name = parts[9]

                    # Convert quaternion to rotation matrix R_cw (camera from world)
                    # and compute optical center in world frame: C = -R^T * t
                    R = quaternion_to_rotation_matrix(np.array([qw, qx, qy, qz]))
                    t = np.array([tx, ty, tz])
                    center = -R.T @ t
                    camera_centers[name] = center
        elif os.path.exists(images_bin):
            # Parse COLMAP binary images format
            with open(images_bin, "rb") as f:
                num_reg_images = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num_reg_images):
                    image_id = struct.unpack("<I", f.read(4))[0]
                    qvec = struct.unpack("<4d", f.read(32))
                    tvec = struct.unpack("<3d", f.read(24))
                    camera_id = struct.unpack("<I", f.read(4))[0]
                    name_chars = []
                    while True:
                        ch = f.read(1)
                        if ch == b"\x00":
                            break
                        name_chars.append(ch.decode("latin1"))
                    name = "".join(name_chars)
                    num_points2D = struct.unpack("<Q", f.read(8))[0]
                    f.seek(num_points2D * 24, os.SEEK_CUR)

                    R = quaternion_to_rotation_matrix(np.array(qvec))
                    t = np.array(tvec)
                    center = -R.T @ t
                    camera_centers[name] = center

        return camera_centers


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert [qw, qx, qy, qz] quaternion to 3x3 rotation matrix."""
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,     1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw,     1 - 2*qx**2 - 2*qy**2]
    ])
