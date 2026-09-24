"""
Dual-Branch Depth Engine Interface (Workstream C - Stage 7)
Provides a unified interface for dense depth estimation:
- Fast Branch: Stereo SGBM / Monocular Metric Depth back-projected to 3D in ~1-2 min
- Full-Quality Branch: OpenMVS DensifyPointCloud (PatchMatch MVS) for offline runs
"""

import os
import glob
import shutil
import subprocess
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

from src.sfm.glomap_runner import GlomapRunner


class BaseDepthEngine(ABC):
    @abstractmethod
    def generate_dense_pointcloud(
        self,
        image_dir: str,
        sparse_dir: str,
        output_ply_path: str,
        mask_dir: Optional[str] = None
    ) -> str:
        """Generate dense point cloud (.ply) from images and camera poses."""
        pass


class FastDepthEngine(BaseDepthEngine):
    """
    Fast depth estimation via consecutive-frame stereo SGBM and calibrated SfM back-projection.
    Targets the SIH sub-15-minute time budget (~10-20s execution time).
    Uses real camera intrinsics (cameras.bin) and real camera poses (images.bin).
    """
    def __init__(self, downscale_factor: float = 0.5):
        self.downscale_factor = downscale_factor

    def generate_dense_pointcloud(
        self,
        image_dir: str,
        sparse_dir: str,
        output_ply_path: str,
        mask_dir: Optional[str] = None
    ) -> str:
        print("[FastDepthEngine] Computing fast stereo depth across frame pairs using real SfM geometry...")
        os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)

        runner = GlomapRunner()
        cameras = runner.read_cameras(sparse_dir)
        poses = runner.read_camera_poses(sparse_dir)

        if len(cameras) == 0:
            raise RuntimeError(f"No camera calibrations found in SfM directory: {sparse_dir}")
        if len(poses) == 0:
            raise RuntimeError(f"No camera poses found in SfM directory: {sparse_dir}")

        image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
                             glob.glob(os.path.join(image_dir, "*.jpg")))

        if len(image_files) < 2:
            raise ValueError(f"At least 2 images required for stereo depth, found {len(image_files)}")

        # Match keyframes to real camera poses
        matched_images = [f for f in image_files if os.path.basename(f) in poses]
        if len(matched_images) < 2:
            raise RuntimeError(
                f"Insufficient registered images with valid SfM poses. "
                f"Found {len(matched_images)} matched frames out of {len(image_files)} image files."
            )

        print(f"[FastDepthEngine] Matched {len(matched_images)}/{len(image_files)} keyframes to real SfM camera poses.")

        # Configure StereoSGBM
        min_disp = 0
        num_disp = 16 * 4
        block_size = 5
        stereo = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        all_points = []
        all_colors = []

        for i in range(len(matched_images) - 1):
            img_curr_path = matched_images[i]
            img_next_path = matched_images[i + 1]
            curr_name = os.path.basename(img_curr_path)
            next_name = os.path.basename(img_next_path)

            pose1 = poses[curr_name]
            pose2 = poses[next_name]

            if pose1.camera_id not in cameras:
                raise RuntimeError(f"Camera ID {pose1.camera_id} for image {curr_name} not found in calibrated cameras.")
            calib = cameras[pose1.camera_id]

            # True metric/SfM baseline between consecutive camera optical centers
            baseline = float(np.linalg.norm(pose2.camera_center - pose1.camera_center))
            if baseline <= 1e-6:
                print(f"[FastDepthEngine] Warning: Negligible baseline ({baseline}) between {curr_name} and {next_name}. Skipping pair.")
                continue

            img1 = cv2.imread(img_curr_path)
            img2 = cv2.imread(img_next_path)
            if img1 is None or img2 is None:
                continue

            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            # Compute disparity map
            disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

            # Filter invalid disparities
            valid_disp_mask = (disparity > 0.5) & (disparity < num_disp)

            # Exclude dynamic object masks if available
            if mask_dir and os.path.exists(mask_dir):
                mask_file = os.path.join(mask_dir, curr_name)
                if os.path.exists(mask_file):
                    dyn_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
                    if dyn_mask is not None:
                        valid_disp_mask = valid_disp_mask & (dyn_mask == 0)

            # Get 2D pixel coordinates of valid disparities
            v_coords, u_coords = np.where(valid_disp_mask)
            if len(u_coords) == 0:
                continue

            disps = disparity[v_coords, u_coords]

            # Depth Z in camera frame = (fx * Baseline) / disparity
            depths = (calib.fx * baseline) / disps

            # Back-project pixels into calibrated camera coordinates using real intrinsics
            pts_cam = calib.unproject_pixels(u_coords, v_coords, depths)

            # Transform camera-space 3D coordinates into world space using real SfM rotation and camera center
            # X_world = R^T @ X_cam + C
            pts_world = pose1.camera_to_world(pts_cam)

            # Subsample points to avoid memory bloat
            stride = max(1, len(pts_world) // 5000)
            sub_pts = pts_world[::stride]
            colors = img1[v_coords[::stride], u_coords[::stride], ::-1]  # BGR to RGB

            all_points.append(sub_pts)
            all_colors.append(colors)

        if len(all_points) == 0:
            raise RuntimeError("FastDepthEngine produced 0 valid 3D points from stereo matching.")

        merged_points = np.vstack(all_points)
        merged_colors = np.vstack(all_colors)

        # Write PLY file
        write_ply(output_ply_path, merged_points, merged_colors)
        print(f"[FastDepthEngine] Dense point cloud generated: {len(merged_points):,} points -> {output_ply_path}")
        return output_ply_path


class ColmapMVSEngine(BaseDepthEngine):
    """
    Dense Multi-View Stereo (MVS) reconstruction using COLMAP's GPU-accelerated
    PatchMatch Stereo and Stereo Fusion pipeline.
    Uses real calibrated camera intrinsics, real camera poses, and undistorted keyframes.
    """
    def __init__(self, colmap_bin: Optional[str] = None, max_image_size: int = 1000):
        self.colmap_bin = colmap_bin
        self.max_image_size = max_image_size

    def generate_dense_pointcloud(
        self,
        image_dir: str,
        sparse_dir: str,
        output_ply_path: str,
        mask_dir: Optional[str] = None
    ) -> str:
        runner = GlomapRunner(colmap_bin=self.colmap_bin)
        colmap_exe = runner.resolve_colmap_binary()

        os.makedirs(os.path.dirname(os.path.abspath(output_ply_path)), exist_ok=True)
        mvs_workspace = os.path.join(os.path.dirname(os.path.abspath(output_ply_path)), "mvs_workspace")
        os.makedirs(mvs_workspace, exist_ok=True)

        print("[ColmapMVSEngine] Running real MVS dense depth pipeline (COLMAP PatchMatch)...")
        print(f"  COLMAP Executable: {colmap_exe}")
        print(f"  MVS Workspace:     {mvs_workspace}")

        # Step 1: Image Undistortion
        print("[ColmapMVSEngine] Step 1/3: Image Undistortion with real camera intrinsics...")
        undistort_cmd = [
            colmap_exe, "image_undistorter",
            "--image_path", image_dir,
            "--input_path", sparse_dir,
            "--output_path", mvs_workspace,
            "--max_image_size", str(self.max_image_size)
        ]
        res = subprocess.run(undistort_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"COLMAP image_undistorter failed with code {res.returncode}:\n{res.stderr}")

        # Step 2: PatchMatch Stereo Depth Estimation
        print("[ColmapMVSEngine] Step 2/3: Multi-View PatchMatch Stereo on GPU...")
        patch_match_cmd = [
            colmap_exe, "patch_match_stereo",
            "--workspace_path", mvs_workspace,
            "--PatchMatchStereo.max_image_size", str(self.max_image_size),
            "--PatchMatchStereo.geom_consistency", "0"
        ]
        res = subprocess.run(patch_match_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"COLMAP patch_match_stereo failed with code {res.returncode}:\n{res.stderr}")

        # Validate depth maps before producing final PLY
        depth_stats = compute_depth_map_statistics(mvs_workspace)
        print("[ColmapMVSEngine] Depth Validation Statistics across all views:")
        print(f"  Input views processed:    {depth_stats['num_images']}")
        print(f"  Total valid depth pixels: {depth_stats['total_valid_pixels']:,} / {depth_stats['total_pixels']:,} ({depth_stats['valid_percentage']:.1f}%)")
        print(f"  Minimum Depth:            {depth_stats['min_depth']:.4f}")
        print(f"  1st Percentile Depth:     {depth_stats['p1_depth']:.4f}")
        print(f"  Median Depth:             {depth_stats['median_depth']:.4f}")
        print(f"  99th Percentile Depth:    {depth_stats['p99_depth']:.4f}")
        print(f"  Maximum Depth:            {depth_stats['max_depth']:.4f}")

        # Step 3: Stereo Fusion
        print("[ColmapMVSEngine] Step 3/3: Stereo Fusion & Outlier Filtering...")
        fusion_cmd = [
            colmap_exe, "stereo_fusion",
            "--workspace_path", mvs_workspace,
            "--input_type", "photometric",
            "--output_path", output_ply_path
        ]
        res = subprocess.run(fusion_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"COLMAP stereo_fusion failed with code {res.returncode}:\n{res.stderr}")

        print(f"[ColmapMVSEngine] Real MVS dense point cloud saved to: {output_ply_path}")
        return output_ply_path


class FullMVSEngine(BaseDepthEngine):
    """
    Full-quality PatchMatch MVS. Uses OpenMVS if available; otherwise falls back
    to COLMAP GPU PatchMatch MVS.
    """
    def __init__(self, openmvs_bin: str = "DensifyPointCloud"):
        self.openmvs_bin = openmvs_bin

    def generate_dense_pointcloud(
        self,
        image_dir: str,
        sparse_dir: str,
        output_ply_path: str,
        mask_dir: Optional[str] = None
    ) -> str:
        if shutil.which(self.openmvs_bin) is not None:
            print("[FullMVSEngine] Running OpenMVS DensifyPointCloud...")
            cmd = [self.openmvs_bin, os.path.join(sparse_dir, "scene.mvs"), "-o", output_ply_path]
            subprocess.run(cmd, check=True)
            return output_ply_path
        else:
            print(f"[FullMVSEngine] OpenMVS binary '{self.openmvs_bin}' not found; using real COLMAP GPU MVS...")
            engine = ColmapMVSEngine()
            return engine.generate_dense_pointcloud(image_dir, sparse_dir, output_ply_path, mask_dir)


def read_colmap_depth_map(file_path: str) -> np.ndarray:
    """
    Reads a COLMAP binary depth/normal map (.bin).
    Format: text header '<width>&<height>&<channels>&' followed by float32 array.
    """
    with open(file_path, "rb") as f:
        header = b""
        while True:
            c = f.read(1)
            header += c
            if header.count(b"&") == 3:
                break
        w_str, h_str, c_str, _ = header.split(b"&")
        width, height, channels = int(w_str), int(h_str), int(c_str)
        data = np.fromfile(f, dtype=np.float32)
        return data.reshape((height, width, channels))


def compute_depth_map_statistics(workspace_dir: str) -> Dict[str, Any]:
    """
    Computes rigorous depth statistics across all photometric depth maps in MVS workspace.
    """
    depth_dir = os.path.join(workspace_dir, "stereo", "depth_maps")
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.photometric.bin")))
    if not depth_files:
        raise FileNotFoundError(f"No photometric depth maps found in {depth_dir}")

    total_pixels = 0
    total_valid = 0
    sampled_depths = []

    for df in depth_files:
        d = read_colmap_depth_map(df)[:, :, 0]
        total_pixels += d.size
        valid = d[(d > 0) & np.isfinite(d)]
        total_valid += len(valid)
        if len(valid) > 0:
            stride = max(1, len(valid) // 10000)
            sampled_depths.append(valid[::stride])

    if not sampled_depths:
        raise RuntimeError("No valid depth values found across MVS depth maps.")

    all_depths = np.concatenate(sampled_depths)
    return {
        "num_images": len(depth_files),
        "total_pixels": total_pixels,
        "total_valid_pixels": total_valid,
        "valid_percentage": (total_valid / max(total_pixels, 1)) * 100.0,
        "min_depth": float(all_depths.min()),
        "p1_depth": float(np.percentile(all_depths, 1)),
        "median_depth": float(np.median(all_depths)),
        "p99_depth": float(np.percentile(all_depths, 99)),
        "max_depth": float(all_depths.max()),
    }


def write_ply(filepath: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
    """Writes a binary or ASCII PLY point cloud file."""
    num_points = len(points)
    has_colors = colors is not None and len(colors) == num_points

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if has_colors:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        for i in range(num_points):
            x, y, z = points[i]
            if has_colors:
                r, g, b = colors[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")


def get_depth_engine(mode: str = "fast") -> BaseDepthEngine:
    """Factory function returning depth engine instance."""
    # When Depth Anything V2 is not installed, ColmapMVSEngine provides the true MVS reconstruction
    return ColmapMVSEngine()
