# PIPELINE AUDIT REPORT: Current Implementation Reality

> **Target Video**: `D:\drone3d\video\dronevid.mp4`  
> **Status**: Comprehensive code inspection of all active scripts in `d:\sih\src\`  
> **Audited On**: September 24, 2026

---

## 1. What is the VERY FIRST processing step performed on the video?

The very first processing step is executed by `extract_keyframes()` in [src/ingest/keyframe_extractor.py](file:///d:/sih/src/ingest/keyframe_extractor.py#L24-L97).
- It opens the video file using **OpenCV** (`cv2.VideoCapture(video_path)`).
- It queries the video properties: FPS (`cap.get(cv2.CAP_PROP_FPS)`), total frame count, and dimensions (width × height).
- It calculates a frame sampling step: `sample_step = max(1, int(round(video_fps / target_fps)))`.
- It loops through the video frames, reading each frame with `cap.read()`.

---

## 2. How are frames extracted?

- **Tool**: **OpenCV (`cv2.VideoCapture`)**. **FFmpeg is NOT used** (FFmpeg is mentioned in documentation, but the code relies entirely on OpenCV's Python bindings).
- **Sampling Strategy**: **Fixed-stride temporal sampling**.
  - Default `target_fps = 2.0 Hz` (1 frame every 0.5 seconds).
  - On a 30 FPS video, `sample_step = 15`. It inspects every 15th frame (`frame_idx % sample_step == 0`).

---

## 3. Does the current implementation perform intelligent keyframe selection?

- **Sharpness / Laplacian scoring**: **YES**.
  - For each sampled frame, it converts to grayscale and calculates `cv2.Laplacian(gray, cv2.CV_64F).var()`.
  - Any frame with `sharpness < min_sharpness` (default `20.0`) is rejected.
  - If the remaining sharp frames exceed `max_frames` (default `500`), it divides the candidate list into `max_frames` uniform temporal buckets and selects the frame with the highest Laplacian variance in each bucket.
  - Selected frames undergo CLAHE contrast enhancement (`cv2.createCLAHE(clipLimit=2.0)`) on the L-channel in LAB space.
- **Duplicate removal**: **NO**. There is no SSIM, perceptual hashing, or pixel-difference thresholding against previous frames.
- **Motion / parallax selection**: **NO**. It does NOT compute optical flow, feature matches, or essential matrix geometry to detect camera baseline/parallax. Frame selection is strictly time-based and blur-based.

---

## 4. After frame extraction, what happens next?

1. **Stage 2: Telemetry Synchronization** ([src/ingest/srt_parser.py](file:///d:/sih/src/ingest/srt_parser.py)):
   - If an SRT/CSV file is provided, it parses latitude, longitude, and altitude telemetry, matches them to keyframe timestamps, and writes `stage_01_ingest/frame_gps.csv`.
   - If no telemetry is passed, it switches to `GPS_DENIED_FALLBACK` (scale = 1.0).
2. **Stage 3: Dynamic Object Masking** ([src/masking/mask_generator.py](file:///d:/sih/src/masking/mask_generator.py)):
   - Loads `yolov8m-seg.pt` (or `yolov8n-seg.pt`) via Ultralytics YOLO.
   - Infers on each extracted keyframe to segment dynamic COCO classes (cars, trucks, buses, people).
   - Generates binary mask PNGs (`stage_02_masking/masks/frame_XXXXX.png` where dynamic objects = 255, background = 0).

---

## 5. Is COLMAP actually being used?

**NO, COLMAP IS NOT CURRENTLY BEING EXECUTED.**

In [src/sfm/glomap_runner.py](file:///d:/sih/src/sfm/glomap_runner.py#L51-L98):
- The script checks `check_binary_available("colmap")` (`shutil.which("colmap")`).
- On this system, `colmap` is **not installed in the system PATH**.
- Therefore, the COLMAP code block is completely bypassed and drops into lines 98–100:
  ```python
  print(f"[SfM] Notice: Neither COLMAP nor GLOMAP found in PATH.")
  print(f"[SfM] Generating development synthetic camera poses for downstream integration testing...")
  self._generate_synthetic_sparse_model(image_dir, sparse_dir)
  ```

*(If COLMAP were installed, the commands programmed in the wrapper are:*
- *Feature Extraction: `colmap feature_extractor --database_path database.db --image_path keyframes/ --ImageReader.camera_model OPENCV --ImageReader.single_camera 1 --SiftExtraction.use_gpu 1`*
- *Matching: `colmap sequential_matcher --database_path database.db --SequentialMatching.overlap 10 --SequentialMatching.quadratic_overlap 1 --SiftMatching.use_gpu 1`)*

---

## 6. Is a real COLMAP database being created?

**NO.** Because `colmap` is not in PATH, `colmap feature_extractor` never runs. No SQLite database (`database.db`) is created or written to.

---

## 7. Is COLMAP actually estimating camera poses / performing SfM?

**NO.** Because COLMAP is absent, `_generate_synthetic_sparse_model()` in `glomap_runner.py` writes hardcoded mock text files (`cameras.txt`, `images.txt`, `points3D.txt`):
- Pinhole camera: focal length 1500, principal point (960, 540).
- Poses: Identity rotation $q = (1, 0, 0, 0)$ with a synthetic linear translation loop:
  $$t_x = \text{idx} \times 2.5, \quad t_y = \text{idx} \times 2.0 + 0.2 \sin(\text{idx} \times 0.5), \quad t_z = 3.5 + 0.05 \cos(\text{idx} \times 0.2)$$
This creates a synthetic straight diagonal flight path.

---

## 8. Is GLOMAP being used?

**GLOMAP IS NOT BEING USED.**
- `shutil.which("glomap")` evaluates to `None`. No GLOMAP binary exists or runs.

---

## 9. Is Depth Anything V2 being used?

**NO. Depth Anything V2 is completely absent from the codebase.**
- There is no model weight, no import, and no inference code for Depth Anything anywhere in `src/`.
- The active class in [src/depth/depth_interface.py](file:///d:/sih/src/depth/depth_interface.py) is `FastDepthEngine`, which uses OpenCV's classical CPU stereo matcher: `cv2.StereoSGBM_create()`.

---

## 10. Is Open3D being used?

**YES.** Open3D (`v0.20.0`) is installed and imported in [src/pointcloud/pointcloud_fusion.py](file:///d:/sih/src/pointcloud/pointcloud_fusion.py#L82-L105).
- **What it actually does**:
  1. Loads points into `o3d.geometry.PointCloud()`.
  2. Runs voxel grid downsampling: `pcd.voxel_down_sample(voxel_size=0.08)`.
  3. Runs statistical outlier removal: `pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)`.
  4. Writes filtered PLY: `o3d.io.write_point_cloud(...)`.
- **What it does NOT do**: It is NOT used for TSDF volume integration, depth image reconstruction, or ICP camera tracking.

---

## 11. Is OpenMVS being used?

**NO. OpenMVS IS NOT BEING USED.**
- `shutil.which("DensifyPointCloud")`, `shutil.which("ReconstructMesh")`, and `shutil.which("TextureMesh")` are all `None`.
- The pipeline falls back to SciPy 2.5D Delaunay triangulation and an OpenCV image-tiling canvas.

---

## 12. What exact process creates the final .ply point cloud that is currently viewed?

1. **`FastDepthEngine` ([src/depth/depth_interface.py](file:///d:/sih/src/depth/depth_interface.py))**:
   - Takes consecutive video frames $(i, i+1)$ and converts them to grayscale.
   - Computes an un-rectified horizontal disparity map via `cv2.StereoSGBM`.
   - Inverts disparity to depth: $\text{depth} = \text{cam\_alt} / \text{ratio}$.
   - Back-projects pixels $(u, v)$ to 3D camera coordinates:
     $$x_{\text{cam}} = (u - c_x) \cdot \text{depth} / f_x, \quad y_{\text{cam}} = (v - c_y) \cdot \text{depth} / f_y, \quad z_{\text{cam}} = \text{depth}$$
   - Transforms into world coordinates using the **synthetically generated camera centers** ($c_1 = [-2.5 \times i, -2.0 \times i, -3.5]$).
   - Writes `stage_05_depth/dense_raw.ply`.
2. **`fuse_and_filter_pointcloud()` ([src/pointcloud/pointcloud_fusion.py](file:///d:/sih/src/pointcloud/pointcloud_fusion.py))**:
   - Reads `dense_raw.ply`, downsamples voxels to 8cm, removes statistical noise, and saves `stage_06_pointcloud/dense_filtered.ply`.
3. **`georeference_ply_pointcloud()` ([src/georef/mesh_georeferencer.py](file:///d:/sih/src/georef/mesh_georeferencer.py))**:
   - Applies the 7-DoF Sim(3) transformation matrix (identity scale = 1.0 in fallback mode) to produce `stage_06_pointcloud/pointcloud_georeferenced.ply`.
4. **`classify_pointcloud()` ([src/pointcloud/pointcloud_classifier.py](file:///d:/sih/src/pointcloud/pointcloud_classifier.py))**:
   - Calculates height-above-ground and k-NN covariance eigenvalues to assign ASPRS classes (Ground, Building, Vegetation, Road), producing `stage_08_classify/classified_semantic.ply` and `classified.las`.

---

## 13. Is the current point cloud A, B, C, or D?

**D. Something else.**
It is:
- **OpenCV CPU StereoSGBM** computed directly on consecutive un-rectified video frames,
- Back-projected into 3D using **hardcoded synthetic linear camera trajectory poses**,
- Voxel-downsampled and statistical-outlier-filtered with **Open3D**.

---

## 14. Current Pipeline Arrow Diagram

```text
Input Drone Video (.mp4)
      │
      ▼
[Stage 1] OpenCV VideoCapture + Laplacian Variance Scoring + CLAHE
      │ (outputs: keyframes/frame_XXXXX.png)
      ▼
[Stage 2] DJI SRT Parser / GPS Matcher (skipped if no SRT -> scale=1.0)
      │ (outputs: frame_gps.csv)
      ▼
[Stage 3] YOLOv8-seg Dynamic Object Segmentation
      │ (outputs: masks/frame_XXXXX.png)
      ▼
[Stage 4 & 5] SfM Fallback Generator (COLMAP & GLOMAP missing in PATH)
      │ (outputs: synthetic cameras.txt, images.txt, points3D.txt)
      ▼
[Stage 6] 7-DoF Umeyama Sim(3) Alignment (GPS-denied fallback -> scale=1.0)
      │ (outputs: georef.json)
      ▼
[Stage 7] FastDepthEngine: OpenCV StereoSGBM + Synthetic Camera Unprojection
      │ (outputs: dense_raw.ply)
      ▼
[Stage 8] Open3D Voxel Downsampling (0.08m) + Statistical Outlier Removal
      │ (outputs: dense_filtered.ply)
      ▼
[Stage 9 & 10] SciPy 2.5D Delaunay Plane Triangulation (OpenMVS missing)
      │ (outputs: mesh_raw.ply)
      ▼
[Stage 11] Native Orthographic UV Texture Projection & Canvas Tiling
      │ (outputs: textured_mesh.obj, .mtl, .png)
      ▼
[Stage 12] Sim(3) Georeferencing Application
      │ (outputs: model_georeferenced.obj, pointcloud_georeferenced.ply)
      ▼
[Stage 13] Point Cloud Semantic Classification (HAG + cKDTree Covariance)
      │ (outputs: classified.las, classified_semantic.ply)
      ▼
[Stage 14] Deliverables Export & Metrics Reporting
      │ (outputs: orthomosaic.tif, measurement_report.json, coverage_report.json)
      ▼
Interactive WebGL 3D Visualizer (Three.js on FastAPI :8000)
```

---

## 15. Complete Stage-by-Stage Breakdown

| Stage | Script / File | Tool / Library Used | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **1. Keyframe Extraction** | `src/ingest/keyframe_extractor.py` | `OpenCV`, `NumPy` | Video (`.mp4`) | Extracted frames (`keyframes/*.png`) |
| **2. Telemetry Sync** | `src/ingest/srt_parser.py` | `regex`, `csv`, `NumPy` | Telemetry (`.srt` / `.csv`) | Synchronized GPS (`frame_gps.csv`) |
| **3. Dynamic Masking** | `src/masking/mask_generator.py` | `Ultralytics YOLOv8-seg` | `keyframes/*.png` | Dynamic binary masks (`masks/*.png`) |
| **4–5. Camera Poses (SfM)** | `src/sfm/glomap_runner.py` | **Synthetic Python Generator** *(COLMAP/GLOMAP missing)* | `keyframes/`, `masks/` | Synthetic poses (`sparse/0/images.txt`) |
| **6. Sim(3) GPS Alignment** | `src/georef/umeyama_aligner.py` | `NumPy`, `SciPy` (Umeyama SVD) | `camera_centers`, `frame_gps.csv` | Transform matrix (`georef.json`) |
| **7. Dense Depth** | `src/depth/depth_interface.py` | `OpenCV (StereoSGBM)` *(Depth Anything missing)* | `keyframes/*.png`, `sparse/` | Raw point cloud (`dense_raw.ply`) |
| **8. Point Cloud Filtering** | `src/pointcloud/pointcloud_fusion.py` | `Open3D (v0.20.0)` | `dense_raw.ply` | Clean point cloud (`dense_filtered.ply`) |
| **9–10. Surface Meshing** | `src/mesh/mesh_generator.py` | `SciPy (Delaunay 2.5D)` *(OpenMVS missing)* | `dense_filtered.ply` | Triangle mesh (`mesh_raw.ply`) |
| **11. Texture Mapping** | `src/mesh/texture_mapper.py` | `OpenCV`, `NumPy` *(OpenMVS missing)* | `mesh_raw.ply`, `keyframes/` | Textured OBJ/MTL (`textured_mesh.obj`) |
| **12. Georeference Apply** | `src/georef/mesh_georeferencer.py` | `NumPy` | `textured_mesh.obj`, `georef.json` | `model_georeferenced.obj` |
| **13. ASPRS Classification**| `src/pointcloud/pointcloud_classifier.py`| `SciPy cKDTree`, `laspy` | `pointcloud_georeferenced.ply` | `classified.las`, `classified_semantic.ply` |
| **14. GIS Deliverables** | `src/export/gis_exporter.py` | `tifffile`, `cv2` | Deliverable files | `orthomosaic.tif`, `measurement_report.json` |

---

## 16. Does the current implementation actually follow the intended architecture?

**NO.**

The intended architecture:
```text
Video ➔ Keyframe Selection ➔ COLMAP Feature Extraction ➔ COLMAP Feature Matching ➔ COLMAP/GLOMAP SfM ➔ Camera Poses + Sparse Point Cloud ➔ Dense Depth / OpenMVS ➔ Open3D Fusion ➔ Filtering ➔ Mesh
```

**Why it deviates**:
1. **No Real SfM**: COLMAP and GLOMAP binaries are not installed in the Windows environment. As a result, no real features are extracted, no matching takes place, and no bundle adjustment is solved. Camera poses are purely hardcoded synthetic vectors.
2. **No Depth Anything V2**: Depth Anything V2 is not implemented in Python; it only exists as a design concept in documentation.
3. **No OpenMVS**: OpenMVS is not installed in the system PATH; meshing is done via 2.5D Delaunay in SciPy, and texturing is done via simple image canvas tiling.
4. **Current Status**: The codebase is currently operating in **Mock / Fallback mode**, designed to validate downstream data contracts (file formats, bounding boxes, web viewer, and LAS classification) rather than performing true photogrammetric triangulation.
