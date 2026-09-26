# SIH 2026 PS 158: Single-Pass Drone Video to Georeferenced 3D

AI-enabled reconstruction of a georeferenced, metrically accurate, textured 3D model from one drone flight video plus GPS/flight metadata. Organisation: NTRO.

## Status (2026-09-26)

The v3 pipeline (`src/pipeline.py`) replaces the v2 runner. It runs real SfM (COLMAP), georeferences with an orientation prior so a straight single pass has a defined "up", builds per-frame metric depth, fuses it with TSDF, and writes DSM, DTM, true orthomosaic, LAS, and GLB products. Buildings are completed by footprint extrusion with per-face provenance. Every stage has a gate; a failed gate stops the run and `report.json` says which one.

The v2 runner (`src/pipeline_runner.py`) is kept for reference only. Its outputs are not usable; ARCHITECTURE_V3.md section 0 explains why.

## Documents

| File | Purpose |
|---|---|
| `ARCHITECTURE_V3.md` | Root-cause audit of v2, v3 design, stage specs, completion ladder, evaluation criteria |
| `IMPLEMENTATION_PLAN.md` | Phased plan through the finale, module map, status of each item |
| `ARCHITECTURE.md`, `PHASE_PLAN.md`, `PIPELINE_AUDIT_README.md` | v2 material, superseded |

## Setup (WSL2 or Linux)

Windows Application Control on the team laptop blocks the native DLLs of pycolmap and rasterio, so the pipeline runs in WSL2.

```bash
bash scripts/setup_env.sh                 # Miniforge, COLMAP (CUDA) from conda-forge, Python deps, CUDA torch
source ~/miniforge3/etc/profile.d/conda.sh && conda activate drone3d
python scripts/check_env.py --profile live
```

## Run

```bash
# real footage, LIVE lane (Depth Anything V2 + TSDF)
python -m src.pipeline --config configs/live.yaml --video flight.mp4 --telemetry flight.srt --run-id site1

# validation on the synthetic 3D flight (ground-truth depth replaces the network)
python tools/synth3d.py --out test_data/synth3d_pass          # already rendered in the repo
python -m src.pipeline --config configs/synth_oracle.yaml --video test_data/synth3d_pass/flight.mp4 \
    --telemetry test_data/synth3d_pass/flight.srt --gt test_data/synth3d_pass --run-id synth_oracle
python eval/synth_eval.py --run runs/synth_oracle --gt test_data/synth3d_pass

# geotagged photo sets (e.g. OpenDroneMap samples) as video + SRT
python tools/images_to_video_srt.py --images path/to/images --out test_data/odm_clip
```

### Dashboard with drag-and-drop

```bash
python webapp/server_v3.py        # inside WSL, drone3d env active
```

Open `http://localhost:8765` on Windows. Drop an MP4/MOV, plus the DJI `.SRT` if you have it, and press Reconstruct. The run appears in the dashboard when it finishes, in about 4 minutes for a 25 s 2.7K clip on the RTX 5050. Without an `.SRT` the model is not georeferenced and its scale comes from the flying height you enter. `python tools/build_dashboard.py --runs ...` rebuilds the static page.

### Building regularisation

The completion stage replaces each detected building with a clean primitive. The footprint becomes a rectangle, triangle or right-angled outline, whichever fits the detection best. The roof becomes flat, single-slope or gable planes, fitted robustly to the height model, and walls are vertical. Roofs are textured from the orthomosaic, and walls take colour from the observed surface. Outputs: `08_completion/buildings_regularized.glb` and `model_hybrid.glb` (terrain and trees from the reconstruction, buildings as primitives). Turn it off with `--set completion.regularize=false`.

Useful flags: `--until georef` stops early, `--from depth` reuses earlier stages, `--force` reruns everything, `--set dense.lane=survey` overrides any config key.

## Outputs (`runs/<run-id>/`)

| Path | Content |
|---|---|
| `report.json`, `status.json` | run status, per-stage gates, timings, decisions |
| `03_sfm/` | COLMAP database, sparse model, `poses.json`, `tracks.npz` |
| `04_georef/georef.json` | SfM to ENU similarity, hold-out RMSE, degeneracy flag, UTM EPSG |
| `05_depth/` | undistorted images, per-frame metric depth and confidence, `cameras.json` |
| `06_fusion/` | `tsdf_mesh.ply`, `model.glb`, `dense_cloud.ply`, per-vertex view counts |
| `07_products/` | `dsm.tif`, `dtm.tif`, `ortho.tif` (UTM GeoTIFF), `dense_cloud.las` |
| `08_completion/` | `buildings_lod2.ply/.glb` (orange observed roof, blue inferred wall), `buildings.geojson` |

## Tests

```bash
python -m pytest tests/v3 -m "not slow"   # unit tests (most also run on Windows)
python -m pytest tests/v3 -m slow         # end-to-end on the synthetic flight, needs COLMAP
```
