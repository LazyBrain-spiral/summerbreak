# Running PS158 v3 in Docker

One image runs the whole pipeline (COLMAP 4.0.4, Depth Anything V2, Pi3X, OWLv2,
LaMa, open3d) plus the dashboard and upload server. It works on:

| Host | GPU | Command |
|---|---|---|
| Linux with NVIDIA driver >= 570 and NVIDIA Container Toolkit | yes | `docker compose up --build -d` |
| Windows 10/11, Docker Desktop (WSL2 backend), NVIDIA driver >= 570 | yes | `docker compose up --build -d` |
| Any x86-64 or arm64 machine without an NVIDIA GPU (incl. macOS) | no | `docker compose --profile cpu up --build -d app-cpu` |

Then open http://localhost:8765 and drop a video (plus its DJI `.SRT` if you
have one) on the upload panel.

Without a GPU the server detects it, runs every model on the CPU and disables
the stereo (SURVEY) lane, which needs CUDA PatchMatch. Expect CPU runs to take
20 minutes or more for a short clip; Pi3X on CPU is only practical for very
short clips.

## First run and offline use

Model weights (about 6.5 GB, Pi3X alone is 5 GB) download on first use into the
`cache` volume. To download them up front, for example before moving to an
air-gapped machine:

```bash
docker compose run --rm app python scripts/prefetch_models.py          # all models
docker compose run --rm app python scripts/prefetch_models.py --no-pi3x
```

Air-gapped transfer: `docker save ps158:latest | gzip > ps158.tar.gz`, and copy
the `ps158_cache` volume contents (`docker run --rm -v ps158_cache:/c -v "$PWD:/o" ubuntu tar czf /o/cache.tgz -C /c .`).

## Command-line runs

Put videos in `./data` (mounted at `/data`):

```bash
docker compose run --rm app python -m src.pipeline --config configs/live.yaml \
    --video /data/flight.mp4 --telemetry /data/flight.SRT --run-id flight \
    --set dense.predictor=pi3x
docker compose run --rm app pytest -q tests/v3 -m "not slow"
docker compose run --rm app python scripts/check_env.py --profile live
```

Results land in the `runs` volume (`/app/runs/<run-id>`). Copy them out with
`docker compose cp app:/app/runs/flight ./flight`.

## Upload tuning options

The dashboard's upload panel has a Quality preset and an Advanced section. The
same fields are accepted by `POST /api/runs` (multipart form):

| Field | Values | Effect |
|---|---|---|
| `lane` | `live`, `pi3x`, `survey` | depth method: Depth Anything (fast), Pi3X multi-view (best accuracy), COLMAP PatchMatch stereo (GPU only) |
| `detail` | `draft`, `standard`, `high` | TSDF voxel / DSM cell / texture texel: 30/40/10 cm, 15/20/6 cm, 10/10/4 cm |
| `keyframes` | `auto`, `dense`, `sparse` | dense helps slow or short clips, sparse long fast flights |
| `altitude_m` | 5 to 1000 | camera height used for scale when there is no telemetry |
| `min_building_m` | 1 to 20 | smallest height accepted as a building |
| `facades`, `regularize`, `texture`, `masks`, `vegetation` | `true` / `false` | doors and windows, straightened buildings, video textures, ignore cars and people, tree filter |

Presets: Fast = live/standard/auto, Balanced = pi3x/standard/auto,
Best = pi3x/high/dense.

```bash
curl -F video=@flight.mp4 -F telemetry=@flight.SRT -F lane=pi3x -F detail=high localhost:8765/api/runs
curl localhost:8765/api/runs/<id>        # stage-by-stage progress
```

## Build options

| Build arg | Default | Use |
|---|---|---|
| `TORCH_INDEX` | cu128 on amd64, cpu on arm64 | e.g. `https://download.pytorch.org/whl/cu126` for older drivers (>= 560) |
| `PI3_COMMIT` | pinned commit | Pi3 source revision |

Python packages are pinned in `docker/constraints.txt` to the versions the
benchmark numbers were produced with.
