# Implementation Plan: Architecture v3 (SIH 2026, PS 158)

*Written 2026-09-26. Companion to ARCHITECTURE_V3.md. Idea-round submission is 2026-09-30 (from PHASE_PLAN.md). Team of four, workstreams A to D as before.*

---

## Status (updated 2026-09-26)

| Item | State | Where |
|---|---|---|
| 0.1 environment | done in WSL2: Miniforge env `drone3d`, COLMAP 4.0.4 CUDA 12.9 from conda-forge, Python 3.11 stack, CUDA torch | `scripts/setup_env.sh`, `scripts/check_env.py`, `Dockerfile`, `Makefile` |
| 0.2 real 3D test footage | synthetic 3D flight with exact ground truth rendered; ODM converter written; team flight still to do | `tools/synth3d.py`, `test_data/synth3d_pass/`, `tools/images_to_video_srt.py` |
| Keyframes: parallax selection, raw + enh | done, tested | `src/ingest/keyframes.py` |
| Telemetry: DJI new/old/Phantom SRT, CSV, yaw wrap, alt source | done, tested | `src/ingest/telemetry.py`, `src/ingest/intrinsics.py` |
| Dynamic masks with dilation, COLMAP convention | done | `src/semantics/masks.py` |
| SfM without fallback, version-agnostic COLMAP CLI or pycolmap | done | `src/sfm/colmap_runner.py`, `src/sfm/model_io.py` |
| Georef: RANSAC Sim(3), hold-out RMSE, collinearity check, gimbal / ground-plane orientation prior | done, tested | `src/georef/solver.py` |
| GPS-prior bundle adjustment | not started (Phase 1) | |
| LIVE lane: Depth Anything V2, per-frame affine fit to SfM tracks, multi-view consistency | done; oracle predictor for validation | `src/depth/lanes.py`, `src/depth/common.py` |
| SURVEY lane: COLMAP PatchMatch geometric | done, benchmarked on the synthetic flight | `src/depth/lanes.py` |
| Pi3X lane: multi-view depth conditioned on SfM poses, K, sparse depth | done, best synthetic DSM accuracy | `src/depth/pi3_predictor.py` |
| TSDF fusion with masks, view counts, coverage | done | `src/fusion/tsdf.py` |
| DSM, DTM (PMF), true ortho, LAS 1.4 with CRS, GeoTIFF in UTM | done, tested | `src/products/rasters.py`, `src/products/geo.py` |
| Tier-1 completion: footprint extrusion with provenance, buildings GeoJSON | done, tested | `src/completion/extrusion.py` |
| GLB export | done | `src/products/glb.py` |
| Stage runner: gates, caching, status, report | done | `src/core/stage.py`, `src/pipeline.py` |
| Synthetic evaluation harness | done | `eval/synth_eval.py` |
| Regularised, textured buildings with floors, doors, windows | done | `src/completion/` |
| Dashboard: runs, 3D modes, measure tools, building cards | done | `tools/build_dashboard.py`, `tools/dashboard_template.html` |
| Upload server: drag-and-drop video, quality presets and tuning options, restart recovery | done, tested for all three lanes | `webapp/server_v3.py` |
| Docker image for any host (GPU or CPU, amd64 or arm64), compose file, model prefetch | done | `Dockerfile`, `docker-compose.yml`, `docker/README.md` |

---

## 0. The two things that block everything else

Do these before any feature work. Nothing in the pipeline can be judged until both are true.

### 0.1 A working GPU Linux environment (owner: lead, 1 day)

The current machine runs Python 3.14 with CPU-only torch and has no photogrammetry binaries. WSL2 Ubuntu 22.04 and Docker Desktop are already installed.

```bash
# inside WSL2 Ubuntu
nvidia-smi                                   # must list the RTX 5050
conda create -n drone3d python=3.11 -y && conda activate drone3d
conda install -c conda-forge "colmap=*=cuda*" glomap -y     # verify the CUDA variant resolves; else plain colmap
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # pick the wheel matching the driver
pip install pycolmap open3d trimesh xatlas ultralytics transformers accelerate \
            pymap3d pyproj rasterio laspy shapely scipy opencv-python-headless \
            cloth-simulation-filter fastapi uvicorn python-multipart pytest pyyaml
python -c "import torch, pycolmap, open3d; print(torch.cuda.is_available(), pycolmap.__version__)"
colmap -h && glomap -h
```

Add `scripts/check_env.py` that asserts CUDA torch, COLMAP, GLOMAP, and pycolmap, and make `pipeline.py` call it at startup. Fail unless `--allow-cpu` is passed.

Also produce a `Dockerfile` (base `colmap/colmap:latest`, which ships CUDA COLMAP) with the same pip set, so the demo laptop and any teammate's machine run the identical image. Keep both paths; WSL2 for daily work, the container for the venue.

**Done when:** `python scripts/check_env.py` passes and `colmap feature_extractor` runs on the GPU on 20 images in under 10 s.

### 0.2 Real test footage with 3D content (owner: A, 1 to 2 days)

The current synthetic clips are flat drawings and must not be used for anything except unit tests of parsers.

Three sources, in priority order:
1. **Team-recorded flight.** Any DJI aircraft with "video captions" enabled writes the SRT. One 2 to 3 minute pass at 50 to 80 m over an area with 2 or more buildings, a road, and trees. Measure one building's footprint and height on the ground (tape or building plan) for accuracy checks. Fly one straight pass and one lawnmower pass.
2. **Public image datasets converted to a video + SRT.** OpenDroneMap sample datasets (images with EXIF GPS) can be encoded into an MP4 at 2 fps with a generated SRT from the EXIF. This exercises the full pipeline end to end today. Write `tools/images_to_video_srt.py`.
3. **Synthetic with real 3D.** Replace the 2D-canvas generators with a renderer that flies a camera over a textured mesh (Open3D `OffscreenRenderer` or Blender): a heightmap terrain plus boxes with roof textures. Export exact poses and intrinsics. This gives ground truth for georef and measurement accuracy tests and runs in CI.

**Done when:** `test_data/` holds at least one real clip with SRT, one ODM-derived clip, and the synthetic-3D generator, each with a `README` stating what it is.

---

## 1. Phase 0: idea-round proof (2026-09-26 to 2026-09-30)

Goal: one genuine reconstruction from real footage, with a screenshot and an honest georef RMSE, for the dossier. Do not touch texturing, classification, or the web app this week.

| Day | Task | Owner | Output |
|---|---|---|---|
| Sat 26 | Section 0.1 environment; Section 0.2 source 2 (ODM clip) | Lead, A | env passes; `test_data/odm_*/` |
| Sun 27 | `src/ingest/keyframe_extractor.py`: ffmpeg decode, sharpness normalisation, parallax selection, raw+enh output | A | 150 to 300 keyframes from the ODM clip |
| Sun 27 | `src/sfm/colmap_runner.py` (rename from glomap_runner): remove the synthetic fallback entirely, use `pycolmap` to read models, export `poses.json` + `tracks.npz`; mask inversion for COLMAP | B | registered ≥ 80% frames, reproj ≤ 1 px |
| Mon 28 | `src/georef/`: RANSAC Umeyama, hold-out RMSE, collinearity check, orientation term (ground-plane version first) | B | `georef.json` with hold-out RMSE and degeneracy flag |
| Mon 28 | SURVEY lane on COLMAP PatchMatch + `stereo_fusion`; TSDF fusion + marching cubes in `src/fusion/tsdf.py` | C | `tsdf_mesh_raw.ply` that looks like the site in MeshLab |
| Tue 29 | Run end to end on the ODM clip and, if available, the team clip; capture screenshots; record stage timings | C, D | figures + timing table |
| Tue 29 | Dossier: replace v2 diagram with v3, include the audit table from ARCHITECTURE_V3.md section 0 as "what we learned", the RMSE, the screenshots | D | dossier draft |
| Wed 30 | Review, rehearse, submit | All | submission |

Delete or quarantine now: `_generate_synthetic_sparse_model`, `FastDepthEngine` (SGBM), the atlas-tiling texture mapper, the `webapp/models/*.obj` renders of the flat scene. Keep them in git history only.

---

## 2. Phase 1: pose backbone and georeferencing hardened (week 1, Oct 1 to Oct 7)

**Workstream A (ingest, semantics)**
- Telemetry parser: add DJI `FrameCnt/DiffTime` format, `focal_len`, `abs_alt`/`rel_alt`, generic CSV column mapping, time-of-day rebasing. Unit tests with three real SRT samples.
- `intrinsics_prior.json` from `focal_len` + sensor table; pass to COLMAP as `ImageReader.camera_params`.
- Video-to-log offset estimation (cross-correlation of image speed and GPS speed) behind a flag; default off for DJI SRT.
- Dynamic masks on GPU with dilation; sky masks (SegFormer-B0 ADE20K, class `sky`, `water`); union mask written in COLMAP convention.

**Workstream B (SfM, georef)**
- Sequential matcher with vocabulary-tree loop detection; download the vocab tree once into `models/`.
- GLOMAP with incremental-mapper fallback and a recorded decision.
- Orientation prior from gimbal angles when present (second variant of the term from Phase 0).
- GPS-prior bundle adjustment via `pycolmap` if the installed COLMAP exposes pose priors; otherwise document and skip.
- `04_georef/gate.json` with all metrics from ARCHITECTURE_V3 section 3.4.

**Workstream C (geometry)**
- LIVE lane: Depth Anything V2 Large fp16 via `transformers`; per-frame scale/shift fit to SfM tracks with RANSAC; multi-view consistency check; write depth + confidence `.npy`.
- Shared TSDF fusion reads either lane. Voxel size from config. Masks applied per view.
- Benchmark both lanes on 300 frames and write the numbers into `docs/benchmarks.md`.

**Workstream D (orchestration, evaluation)**
- New `src/pipeline.py`: stage registry, `run.yaml`, hash-based skip, `status.json`, `report.json` with gates; `--until` and `--profile`.
- `scripts/check_env.py`, `Dockerfile`, `Makefile` targets (`make env`, `make test`, `make demo`).
- Evaluation harness `eval/`: hold-out RMSE, reprojection error, density, timing; runs on the synthetic-3D dataset in CI (GitHub Actions, CPU-only stages) and on real clips locally.

**Checkpoint (Oct 7):** `python -m src.pipeline --config configs/live.yaml --until fusion` produces a TSDF mesh in ENU from the team clip in under 6 minutes with all gates passing.

---

## 3. Phase 2: products that judges can measure on (weeks 2 to 3, Oct 8 to Oct 21)

**Workstream C**
- DSM/DTM: rasterise fused cloud, CSF ground filter, IDW hole fill, GeoTIFF with UTM CRS via rasterio. Verify in QGIS.
- True orthomosaic: per-cell best-view selection with depth-map visibility, feathered blending. Compare visually with the DSM-draped mesh.
- Textured mesh, built-in path: per-face view scoring, neighbour-majority label smoothing, `xatlas` UVs, texel projection, per-view colour balance against the ortho. GLB export with `enu_origin` extras. OpenMVS `TextureMesh` path behind a flag if the binary is available.

**Workstream A**
- Scene-label model selection: evaluate SegFormer ADE20K class-mapped vs a UAVid/LoveDA fine-tune on 20 hand-checked frames from the team clip; pick by IoU on building/road/tree. Store labels as PNG.

**Workstream D**
- 3D classification by projection vote + vectorised geometric features; LAS 1.4 PF7 with CRS VLR. Timing target: 5 M points in under 60 s.
- Measurements: DSM connected components per building, alpha-shape footprints, DTM-relative heights, volumes; road centreline length; canopy area. GeoJSON + CSV.
- Coverage: per-face view count and triangulation angle from depth maps; GLB vertex attribute; `coverage.json`.

**Workstream B**
- Accuracy study on the team clip: compare measured building height/footprint to the pipeline output for straight-pass vs lawnmower; with and without orientation prior; consumer GPS noise injection. Write `docs/accuracy.md`. This is the strongest slide in the finale.

**Checkpoint (Oct 21):** deliverables folder contains GLB, OBJ, LAS, DSM, DTM, ortho, GeoJSON, report. Ortho and footprints overlay correctly on a satellite basemap in QGIS. Building height within the target tolerance on the measured building.

---

## 4. Phase 3: web demo and robustness (week 4, Oct 22 to Oct 28)

**Workstream D**
- FastAPI: keep upload/status/results; serve `status.json` stage progress; serve GLB, ortho tiles (pre-tiled PNG pyramid or COG via `rio-tiler`), GeoJSON.
- Viewer: Three.js `GLTFLoader` with coverage overlay toggle and a measurement tool (point-to-point distance, polygon area, height by clicking two points); Leaflet map with ortho tiles and footprints; link camera between them.
- Remove hardcoded `models/mesh_35m.obj` style lists; the viewer takes a `run_id`.

**All**
- Stress set: straight pass, lawnmower, oblique gimbal (−45°), heavy traffic road, low light, a clip with a 20 s GPS dropout (simulated by deleting SRT blocks). Each must end in a truthful `report.json` (PARTIAL or FAILED with the failing gate named), never a crash and never a fake success.
- GPS-denied mode: pipeline completes up to fusion with scale from the intrinsics prior and altitude, report marks georef as `UNAVAILABLE`, viewer shows the model without a map.

**Checkpoint (Oct 28):** a teammate who did not write the code runs `make demo` on a fresh clone in the container and reaches the viewer in one command.

---

## 5. Phase 4: performance, polish, finale (weeks 5 to 6, Oct 29 to Nov 11)

- Time-budget controller: measured throughput after SfM decides LIVE vs SURVEY; log the decision.
- Profile and cut: ffmpeg hardware decode, batched depth inference, TSDF voxel tuning, mesh decimation for the GLB (target ≤ 30 MB).
- Optional experiment (one person, time-boxed to 3 days): VGGT or MapAnything as a pose-plus-depth initialiser for the LIVE lane, compared against COLMAP+GLOMAP on the same clips. Adopt only if it passes the SfM gate on all stress clips and fits in 8 GB.
- Cached results for every stress clip on USB; rehearsed 10-minute live run; slides built from `report.json` numbers, not from estimates.

---

## 6. Module map (what changes where)

| Path | Action | Notes |
|---|---|---|
| `src/ingest/keyframe_extractor.py` | rewrite | ffmpeg decode, parallax selection, raw+enh |
| `src/ingest/srt_parser.py` | extend | formats, focal_len, offset estimation, tests |
| `src/ingest/intrinsics.py` | new | prior from telemetry / model table |
| `src/masking/mask_generator.py` | extend | GPU enforcement, dilation, sky, COLMAP mask convention |
| `src/semantics/scene_labels.py` | new | 2D scene segmentation |
| `src/sfm/glomap_runner.py` | replace with `src/sfm/colmap_runner.py` | no fallback; pycolmap I/O; tracks export |
| `src/georef/umeyama_aligner.py` | keep function, wrap | RANSAC, hold-out, degeneracy check |
| `src/georef/orientation_prior.py` | new | gimbal / ground-plane term, NLLS refine |
| `src/georef/gps_ba.py` | new | pose-prior BA via pycolmap (if available) |
| `src/depth/depth_interface.py` | delete `FastDepthEngine`; keep `ColmapMVSEngine` | rename to `survey_lane.py` |
| `src/depth/live_lane.py` | new | DA-V2 + scale/shift + consistency |
| `src/fusion/tsdf.py` | new | Open3D TSDF, masks, marching cubes, view_count |
| `src/pointcloud/pointcloud_fusion.py` | shrink | downsample + SOR only; mask logic moves to TSDF |
| `src/mesh/mesh_generator.py` | delete Poisson path | mesh comes from TSDF or OpenMVS |
| `src/mesh/texture_mapper.py` | rewrite | view selection + xatlas + projection |
| `src/products/rasters.py` | new | DSM, DTM (CSF), ortho, GeoTIFF |
| `src/pointcloud/pointcloud_classifier.py` | rewrite | projection vote + vectorised features, LAS 1.4 + CRS |
| `src/export/measurement_reporter.py` | rewrite | instances, alpha shapes, DTM-relative heights, GeoJSON |
| `src/utils/fallback_modes.py` | replace | real visibility coverage in `src/products/coverage.py` |
| `src/export/gis_exporter.py` | rewrite | package products + recipient README |
| `src/pipeline_runner.py` | replace with `src/pipeline.py` | stage registry, gates, caching, status |
| `src/utils/generate_*_flight.py` | replace | 3D synthetic renderer with ground truth |
| `tools/images_to_video_srt.py` | new | ODM datasets to video + SRT |
| `scripts/check_env.py`, `Dockerfile`, `Makefile`, `configs/*.yaml` | new | environment and profiles |
| `eval/` | new | metrics harness, CI |
| `webapp/` | extend | GLB viewer, map, measurement tools, run_id routing |
| `tests/` | rewrite | parsers and math on synthetic data; integration on the 3D synthetic clip |
| `webapp/models/*.obj`, `runs/test_run/` | remove from repo | outputs of the flat scene; add to `.gitignore` |

---

## 7. Definition of done, per stage

A stage is done only when all four hold:
1. It writes its declared outputs and a `gate.json`, and the gate fails on a deliberately broken input (test exists).
2. It runs on the GPU where applicable and logs device and wall time.
3. It has a unit test on synthetic data and an integration run on a real clip recorded in `docs/benchmarks.md`.
4. It has no fallback that produces plausible-looking output from missing inputs.

---

## 8. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| conda-forge CUDA COLMAP/GLOMAP does not resolve for the driver | medium | `colmap/colmap` Docker image is the fallback and is the venue plan anyway |
| 8 GB VRAM too small for PatchMatch at full resolution | high | `max_image_size 1600`; SURVEY lane is offline; LIVE lane is the demo |
| Depth Anything scale fit unstable on textureless roofs | medium | fit on inverse depth with RANSAC; drop frames with < 30 anchors; TSDF averaging hides single-frame errors |
| Straight-pass roll ambiguity leaves "up" wrong | high without fix | orientation prior is mandatory; the degeneracy flag is asserted in tests |
| Team cannot fly before the idea round | medium | ODM-derived clip and 3D synthetic clip cover the dossier; real clip for the finale |
| Licence questions on Ultralytics/OpenMVS/VGGT | low for SIH | subprocess boundaries; note swaps in the report |
| Python 3.14 on Windows keeps biting | certain if kept | all work in the 3.11 WSL2 env; Windows is for editing only |

---

## 9. First commands to run (today)

```bash
# 1. environment (WSL2)
bash scripts/setup_env.sh && python scripts/check_env.py

# 2. make an ODM clip
python tools/images_to_video_srt.py --images ~/datasets/odm_brighton/images --out test_data/odm_brighton

# 3. run to fusion, LIVE profile
python -m src.pipeline --config configs/live.yaml --video test_data/odm_brighton/flight.mp4 \
       --telemetry test_data/odm_brighton/flight.srt --run-id odm_live --until fusion

# 4. look
python tools/view_mesh.py runs/odm_live/06_fusion/tsdf_mesh_raw.ply
cat runs/odm_live/report.json
```
