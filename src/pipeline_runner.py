"""
Master Pipeline Orchestrator (Phase 1 & Phase 2 Execution Engine)
Coordinates Stages 1 through 11 of the SIH26158 3D Drone Reconstruction Architecture:
  Stage 1: Keyframe Extraction & Laplacian Blur Scoring
  Stage 2: DJI SRT Telemetry Parsing & Keyframe Synchronization
  Stage 3: Dynamic Object Detection & Mask Generation (YOLOv8-seg)
  Stage 4-5: Camera Pose Recovery & Global SfM (COLMAP + GLOMAP)
  Stage 6: 7-DoF Umeyama Sim(3) Alignment (WGS84 -> ENU -> Georeferencing)
  Stage 7: Dense Depth Generation (Adaptive Dual-Branch: Fast SGBM vs Full MVS)
  Stage 8: Point Cloud Fusion & Statistical Outlier Removal
  Stage 9-10: Surface Mesh Reconstruction (Delaunay 2.5D / OpenMVS)
  Stage 11: Texture Mapping with Dynamic Vehicle Mask Exclusion
  Final: Metric Georeferenced OBJ & Point Cloud Transformation
"""

import os
import sys
import time
import json
import argparse
from typing import Optional, Dict, Any, List
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ingest.keyframe_extractor import extract_keyframes
from src.ingest.srt_parser import parse_dji_srt, synchronize_telemetry_to_keyframes
from src.masking.mask_generator import DynamicMaskGenerator
from src.sfm.glomap_runner import GlomapRunner
from src.georef.umeyama_aligner import align_sfm_to_gps
from src.depth.depth_interface import get_depth_engine
from src.pointcloud.pointcloud_fusion import fuse_and_filter_pointcloud
from src.mesh.mesh_generator import generate_surface_mesh
from src.mesh.texture_mapper import generate_textured_mesh
from src.georef.mesh_georeferencer import georeference_obj_mesh, georeference_ply_pointcloud
from src.pointcloud.pointcloud_classifier import classify_pointcloud
from src.export.gis_exporter import export_gis_deliverables
from src.export.measurement_reporter import calculate_building_metrics
from src.utils.fallback_modes import compute_honest_coverage, generate_gps_denied_georef


def run_full_pipeline(
    video_path: str,
    telemetry_path: Optional[str],
    output_dir: str,
    target_fps: float = 2.0,
    max_frames: int = 500,
    mode: str = "fast",
    skip_masking: bool = False,
    no_gps: bool = False
):
    print("=" * 80)
    print(" SIH26158: Drone 3D Reconstruction Pipeline (Phase 1 & 2 Runner)")
    print("=" * 80)
    print(f" Video Input     : {video_path}")
    print(f" Telemetry Input : {telemetry_path}")
    print(f" Workspace Dir   : {output_dir}")
    print(f" Pipeline Mode   : {mode.upper()} {'(Targets <15m budget)' if mode=='fast' else '(Offline Quality)'}")
    print("=" * 80)

    start_total_time = time.perf_counter()
    timings = {}

    # Define stage artifact subdirectories according to ARCHITECTURE.md
    stage1_dir = os.path.join(output_dir, "stage_01_ingest")
    keyframes_dir = os.path.join(stage1_dir, "keyframes")
    frame_gps_csv = os.path.join(stage1_dir, "frame_gps.csv")

    stage2_dir = os.path.join(output_dir, "stage_02_masking")
    masks_dir = os.path.join(stage2_dir, "masks")

    stage3_dir = os.path.join(output_dir, "stage_03_sfm")
    stage4_dir = os.path.join(output_dir, "stage_04_georef")
    georef_json = os.path.join(stage4_dir, "georef.json")

    stage5_dir = os.path.join(output_dir, "stage_05_depth")
    dense_raw_ply = os.path.join(stage5_dir, "dense_raw.ply")

    stage6_dir = os.path.join(output_dir, "stage_06_pointcloud")
    dense_filtered_ply = os.path.join(stage6_dir, "dense_filtered.ply")

    stage7_dir = os.path.join(output_dir, "stage_07_mesh")
    mesh_raw_ply = os.path.join(stage7_dir, "mesh_raw.ply")
    textured_obj = os.path.join(stage7_dir, "textured_mesh.obj")
    georef_obj = os.path.join(stage7_dir, "model_georeferenced.obj")
    georef_ply = os.path.join(stage6_dir, "pointcloud_georeferenced.ply")

    # -------------------------------------------------------------
    # STAGE 1: Keyframe Extraction & Quality Filtering
    # -------------------------------------------------------------
    print("\n[STAGE 1] Keyframe Extraction & Laplacian Sharpness Scoring...")
    t0 = time.perf_counter()
    extract_stats = extract_keyframes(
        video_path=video_path,
        output_dir=keyframes_dir,
        target_fps=target_fps,
        max_frames=max_frames
    )
    timings["stage_01_keyframe_extraction_sec"] = round(time.perf_counter() - t0, 3)
    print(f"  -> Extracted {extract_stats['keyframes_extracted']} keyframes in {timings['stage_01_keyframe_extraction_sec']}s")

    # -------------------------------------------------------------
    # STAGE 2: DJI SRT Parsing & Synchronization
    # -------------------------------------------------------------
    print("\n[STAGE 2] Parsing DJI SRT Telemetry & Synchronizing to Keyframes...")
    t0 = time.perf_counter()
    has_gps = False
    if not no_gps and telemetry_path and os.path.exists(telemetry_path):
        try:
            telemetry_records = parse_dji_srt(telemetry_path)
            synchronize_telemetry_to_keyframes(
                telemetry_records=telemetry_records,
                keyframes_metadata=extract_stats["frames"],
                output_csv_path=frame_gps_csv
            )
            has_gps = True
            timings["stage_02_telemetry_sync_sec"] = round(time.perf_counter() - t0, 3)
            print(f"  -> Synchronized {len(extract_stats['frames'])} coordinates -> {frame_gps_csv} ({timings['stage_02_telemetry_sync_sec']}s)")
        except Exception as e:
            print(f"  -> Telemetry parsing error ({e}); switching to GPS-denied fallback.")
            has_gps = False
            timings["stage_02_telemetry_sync_sec"] = 0.0
    else:
        print("  -> GPS telemetry absent or --no_gps flag set. Operating in GPS-denied mode.")
        has_gps = False
        timings["stage_02_telemetry_sync_sec"] = 0.0

    # -------------------------------------------------------------
    # STAGE 3: Dynamic Object Detection & Masking (YOLOv8-seg)
    # -------------------------------------------------------------
    print("\n[STAGE 3] Generating Dynamic Object Masks (YOLOv8-seg)...")
    t0 = time.perf_counter()
    if not skip_masking:
        mask_gen = DynamicMaskGenerator(model_size="yolov8n-seg.pt")
        mask_files = mask_gen.process_directory(keyframes_dir, masks_dir)
        timings["stage_03_dynamic_masking_sec"] = round(time.perf_counter() - t0, 3)
        print(f"  -> Generated {len(mask_files)} dynamic masks ({timings['stage_03_dynamic_masking_sec']}s)")
    else:
        print("  -> Masking skipped by user flag.")
        masks_dir = None
        timings["stage_03_dynamic_masking_sec"] = 0.0

    # -------------------------------------------------------------
    # STAGE 4 & 5: Feature Extraction, Matching & GLOMAP SfM
    # -------------------------------------------------------------
    print("\n[STAGE 4 & 5] Global SfM Pose Recovery (GLOMAP / COLMAP)...")
    t0 = time.perf_counter()
    glomap_runner = GlomapRunner()
    sfm_results = glomap_runner.run_sfm_pipeline(
        image_dir=keyframes_dir,
        mask_dir=masks_dir,
        output_dir=stage3_dir
    )
    timings["stage_04_05_global_sfm_sec"] = round(time.perf_counter() - t0, 3)
    print(f"  -> Reconstructed {sfm_results['num_cameras_reconstructed']} camera poses ({timings['stage_04_05_global_sfm_sec']}s)")

    # -------------------------------------------------------------
    # STAGE 6: Sim(3) Umeyama GPS Alignment & Metric Scaling
    # -------------------------------------------------------------
    print("\n[STAGE 6] Solving 7-DoF Umeyama Sim(3) Alignment to GPS...")
    t0 = time.perf_counter()
    if has_gps and os.path.exists(frame_gps_csv):
        georef_results = align_sfm_to_gps(
            sfm_camera_centers=sfm_results["camera_centers"],
            frame_gps_csv=frame_gps_csv,
            output_georef_json=georef_json
        )
    else:
        print("  -> Activating designed GPS-Denied Fallback (Visual Relative Scale)...")
        georef_results = generate_gps_denied_georef(georef_json, default_scale=1.0)
    timings["stage_06_sim3_georef_sec"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------
    # STAGE 7: Dense Depth Estimation (Dual-Branch Engine)
    # -------------------------------------------------------------
    print(f"\n[STAGE 7] Dense Depth Estimation ({mode.upper()} branch)...")
    t0 = time.perf_counter()
    depth_engine = get_depth_engine(mode=mode)
    depth_engine.generate_dense_pointcloud(
        image_dir=keyframes_dir,
        sparse_dir=sfm_results["sparse_dir"],
        output_ply_path=dense_raw_ply,
        mask_dir=masks_dir
    )
    timings["stage_07_dense_depth_sec"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------
    # STAGE 8: Point Cloud Fusion & Outlier Removal
    # -------------------------------------------------------------
    print("\n[STAGE 8] Point Cloud Fusion & Outlier Removal...")
    t0 = time.perf_counter()
    fusion_stats = fuse_and_filter_pointcloud(
        input_ply_path=dense_raw_ply,
        output_ply_path=dense_filtered_ply,
        voxel_size=0.08,
        mask_dir=masks_dir
    )
    timings["stage_08_pointcloud_fusion_sec"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------
    # STAGE 9 & 10: Surface Mesh Reconstruction
    # -------------------------------------------------------------
    print("\n[STAGE 9 & 10] Watertight Surface Mesh Reconstruction...")
    t0 = time.perf_counter()
    mesh_stats = generate_surface_mesh(
        pointcloud_path=dense_filtered_ply,
        output_mesh_path=mesh_raw_ply
    )
    timings["stage_09_10_mesh_reconstruction_sec"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------
    # STAGE 11: Texture Mapping & Seam Blending
    # -------------------------------------------------------------
    print("\n[STAGE 11] Texture Mapping & Seam Blending...")
    t0 = time.perf_counter()
    tex_stats = generate_textured_mesh(
        mesh_ply_path=mesh_raw_ply,
        keyframes_dir=keyframes_dir,
        output_obj_dir=stage7_dir,
        mask_dir=masks_dir
    )
    timings["stage_11_texture_mapping_sec"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------
    # GEOREFERENCING: Apply Sim(3) to Mesh & Point Cloud
    # -------------------------------------------------------------
    print("\n[STAGE 11b] Applying 7-DoF Sim(3) Transformation...")
    t0 = time.perf_counter()
    georeference_obj_mesh(textured_obj, georef_obj, georef_json)
    georeference_ply_pointcloud(dense_filtered_ply, georef_ply, georef_json)
    timings["georeference_application_sec"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------
    # STAGE 12: ASPRS Point Cloud Semantic Classification
    # -------------------------------------------------------------
    print("\n[STAGE 12] ASPRS Semantic Point Cloud Classification...")
    t0 = time.perf_counter()
    stage8_dir = os.path.join(output_dir, "stage_08_classify")
    classified_las = os.path.join(stage8_dir, "classified.las")
    classified_ply = os.path.join(stage8_dir, "classified_semantic.ply")
    
    cls_stats = classify_pointcloud(
        input_ply_path=georef_ply,
        output_las_path=classified_las,
        output_colored_ply=classified_ply
    )
    timings["stage_12_semantic_classification_sec"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------
    # STAGE 13: GIS Deliverables Packaging & GeoTIFF Export
    # -------------------------------------------------------------
    print("\n[STAGE 13] GIS Deliverables Packaging & GeoTIFF Export...")
    t0 = time.perf_counter()
    deliverables_dir = os.path.join(output_dir, "deliverables")
    gis_files = export_gis_deliverables(
        georef_obj_path=georef_obj,
        mtl_path=os.path.join(stage7_dir, "textured_mesh.mtl"),
        texture_png_path=os.path.join(stage7_dir, "textured_mesh.png"),
        classified_las_path=classified_las,
        georef_json_path=georef_json,
        output_deliverables_dir=deliverables_dir
    )
    timings["stage_13_gis_export_sec"] = round(time.perf_counter() - t0, 3)

    # -------------------------------------------------------------
    # STAGE 14: Dimensional Analytics & Honest Occlusion Coverage
    # -------------------------------------------------------------
    print("\n[STAGE 14] Dimensional Measurement Analytics & Occlusion Analysis...")
    t0 = time.perf_counter()
    measurement_report_path = os.path.join(deliverables_dir, "measurement_report.json")
    coverage_report_path = os.path.join(deliverables_dir, "coverage_report.json")

    measurement_stats = calculate_building_metrics(
        classified_ply_path=classified_ply,
        output_report_json=measurement_report_path
    )
    coverage_stats = compute_honest_coverage(
        mesh_ply_path=mesh_raw_ply,
        camera_centers=sfm_results["camera_centers"],
        output_coverage_json=coverage_report_path
    )
    timings["stage_14_measurements_and_coverage_sec"] = round(time.perf_counter() - t0, 3)

    total_duration = round(time.perf_counter() - start_total_time, 3)
    timings["total_wall_clock_sec"] = total_duration

    # -------------------------------------------------------------
    # Audit & Verification Report
    # -------------------------------------------------------------
    audit_report = {
        "status": "SUCCESS",
        "phase": "Phase 1, 2, 3 & 4: Full Pipeline Complete with Measurement Analytics",
        "mode": mode,
        "gps_mode": georef_results.get("mode", "RTK_GPS_CONSTRAINED"),
        "total_wall_clock_sec": total_duration,
        "under_15_min_budget": total_duration < 900.0,
        "stage_durations_sec": timings,
        "data_contracts": {
            "keyframes_dir": keyframes_dir,
            "keyframes_count": extract_stats["keyframes_extracted"],
            "frame_gps_csv": frame_gps_csv,
            "masks_dir": masks_dir,
            "sparse_dir": sfm_results["sparse_dir"],
            "georef_json": georef_json,
            "dense_raw_ply": dense_raw_ply,
            "dense_filtered_ply": dense_filtered_ply,
            "mesh_raw_ply": mesh_raw_ply,
            "textured_mesh_obj": textured_obj,
            "model_georeferenced_obj": georef_obj,
            "pointcloud_georeferenced_ply": georef_ply,
            "classified_las": classified_las,
            "classified_semantic_ply": classified_ply,
            "deliverables_dir": deliverables_dir,
            "measurement_report": measurement_report_path,
            "coverage_report": coverage_report_path
        },
        "georef_metrics": {
            "scale_factor": georef_results["scale"],
            "rms_error_meters": georef_results["rms_error_meters"],
            "aligned_views": georef_results["num_aligned_views"]
        },
        "geometry_metrics": {
            "raw_points": fusion_stats["initial_points"],
            "filtered_points": fusion_stats["final_points"],
            "mesh_vertices": mesh_stats.get("num_vertices", 0),
            "mesh_faces": mesh_stats.get("num_faces", 0)
        },
        "structural_measurements": measurement_stats.get("structural_dimensions", {}),
        "occlusion_coverage": coverage_stats,
        "classification_metrics": cls_stats.get("distribution", {})
    }

    audit_path = os.path.join(output_dir, "audit_report.json")
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    dims = measurement_stats.get("structural_dimensions", {})
    print("\n" + "=" * 80)
    print(" PHASE 4 COMPLETE: 3D MODEL, MEASUREMENTS & GIS DELIVERABLES READY")
    print(f" Total Wall-Clock Time   : {total_duration:.2f} seconds (<15m target: {'PASS' if total_duration < 900 else 'FAIL'})")
    print(f" Building Footprint Area : {dims.get('footprint_area_sqm', 0.0)} m²")
    print(f" Building Peak Height    : {dims.get('peak_height_meters', 0.0)} m")
    print(f" Building Volume Est.    : {dims.get('estimated_volume_cubic_meters', 0.0)} m³")
    print(f" Honest Coverage Index   : {coverage_stats.get('coverage_percentage', 'N/A')}")
    print(f" Sim(3) Alignment RMSE   : {georef_results['rms_error_meters']} meters")
    print(f" Filtered Point Cloud    : {fusion_stats['final_points']:,} points")
    print(f" Reconstructed Faces     : {mesh_stats.get('num_faces', 0):,} triangles")
    print(f" Final Deliverables Dir  : {deliverables_dir}")
    print(f" Audit Report Saved To   : {audit_path}")
    print("=" * 80)

    return audit_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIH26158 Phase 1, 2, 3 & 4 Pipeline")
    parser.add_argument("--video", required=True, help="Path to input flight video")
    parser.add_argument("--telemetry", required=False, default=None, help="Path to DJI SRT or GPS file")
    parser.add_argument("--output_dir", default="workspace_run_002", help="Output directory")
    parser.add_argument("--fps", type=float, default=2.0, help="Keyframe sampling rate")
    parser.add_argument("--max_frames", type=int, default=500, help="Max frames to keep")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="Depth branch mode")
    parser.add_argument("--skip_masking", action="store_true", help="Skip dynamic masking")
    parser.add_argument("--no_gps", action="store_true", help="Run in GPS-denied fallback mode")
    args = parser.parse_args()

    run_full_pipeline(
        video_path=args.video,
        telemetry_path=args.telemetry,
        output_dir=args.output_dir,
        target_fps=args.fps,
        max_frames=args.max_frames,
        mode=args.mode,
        skip_masking=args.skip_masking,
        no_gps=args.no_gps
    )

