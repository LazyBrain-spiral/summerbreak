"""Local v3 server: the dashboard plus drag-and-drop reconstruction of new videos.

Run inside WSL (the pipeline needs the Linux GPU environment):

    source ~/miniforge3/etc/profile.d/conda.sh && conda activate drone3d
    python webapp/server_v3.py            # then open http://localhost:8765 on Windows

Endpoints
    GET  /                      dashboard (webapp/dashboard.html)
    GET  /api/health            {"ok": true, "caps": {cuda, gpu, colmap_cuda, pi3}}
    POST /api/runs              multipart: video (mp4/mov/m4v/avi/mkv), telemetry (srt/csv, optional),
                                altitude_m (used only without telemetry), lane (live|pi3x|survey),
                                detail (draft|standard|high), keyframes (auto|dense|sparse),
                                min_building_m, and booleans facades, regularize, texture,
                                masks, vegetation (all default true)
    GET  /api/runs              uploaded runs, newest first, with state and tuning
    GET  /api/runs/{id}         live state: stage, per-stage gates, error
    GET  /api/runs/{id}/data    dashboard data for a finished run
Jobs run one at a time (one GPU).
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402

RUNS = ROOT / "runs"
UPLOADS = RUNS / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)
STAGES = ["01_ingest", "02_masks", "03_sfm", "04_georef", "05_depth", "06_fusion", "07_products", "08_completion"]

app = FastAPI(title="PS158 v3 local server")


def _capabilities() -> Dict[str, Any]:
    """What this machine can run: CUDA torch (fast deep models) and a CUDA COLMAP (stereo lane)."""
    caps = {"cuda": False, "gpu": None, "colmap_cuda": False, "pi3": False}
    try:
        import torch

        caps["cuda"] = bool(torch.cuda.is_available())
        if caps["cuda"]:
            caps["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    exe = shutil.which("colmap")
    if exe:
        try:
            out = subprocess.run([exe, "help"], capture_output=True, text=True, timeout=30)
            txt = (out.stdout or "") + (out.stderr or "")
            caps["colmap_cuda"] = "with CUDA" in txt and caps["cuda"]
        except Exception:
            pass
    for d in (os.environ.get("PI3_DIR", ""), os.path.expanduser("~/Pi3"), "/opt/Pi3"):
        if d and os.path.isdir(os.path.join(d, "pi3")):
            caps["pi3"] = True
    return caps


CAPS = _capabilities()

DETAIL = {  # detail level -> overrides (standard = config defaults)
    "draft": {"fusion.voxel_m": 0.3, "products.gsd_m": 0.4, "dense.max_image_size": 768, "completion.texel_m": 0.1},
    "standard": {},
    "high": {"fusion.voxel_m": 0.1, "products.gsd_m": 0.1, "dense.max_image_size": 1280,
             "completion.texel_m": 0.04, "fusion.block_count": 220000},
}
KEYFRAMES = {
    "auto": {},
    "dense": {"ingest.min_disp_frac": 0.025, "ingest.max_gap_s": 1.0},
    "sparse": {"ingest.min_disp_frac": 0.08, "ingest.max_gap_s": 3.0},
}


def tuning_overrides(t: Dict[str, Any]) -> Dict[str, Any]:
    """Validated upload settings -> dotted config overrides."""
    o: Dict[str, Any] = {"georef.assumed_altitude_m": float(t["altitude_m"])}
    o.update(DETAIL[t["detail"]])
    o.update(KEYFRAMES[t["keyframes"]])
    if t["lane"] == "pi3x":
        o["dense.predictor"] = "pi3x"
    o["completion.min_height_m"] = float(t["min_building_m"])
    o["completion.detect_facades"] = bool(t["facades"])
    o["completion.regularize"] = bool(t["regularize"])
    o["completion.texture"] = bool(t["texture"])
    o["masks.enabled"] = bool(t["masks"])
    o["completion.vegetation_filter"] = bool(t["vegetation"])
    if not CAPS["cuda"]:
        o["allow_cpu"] = True
        o["sfm.use_gpu"] = False
    return o
jobs: "queue.Queue[str]" = queue.Queue()


def _state_path(rid: str) -> Path:
    return UPLOADS / rid / "server_state.json"


def _write_state(rid: str, **kw) -> None:
    p = _state_path(rid)
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw, updated=time.time())
    p.write_text(json.dumps(cur))


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _worker() -> None:
    while True:
        rid = jobs.get()
        meta = _read_json(_state_path(rid)) or {}
        run_dir = RUNS / rid
        cfg = "configs/survey.yaml" if meta.get("lane") == "survey" else "configs/live.yaml"
        cmd = [sys.executable, "-u", "-m", "src.pipeline", "--config", cfg, "--video", meta["video"],
               "--run-id", rid, "--skip-env-check", "--force"]
        for k, v in (meta.get("overrides") or {}).items():
            cmd += ["--set", f"{k}={str(v).lower() if isinstance(v, bool) else v}"]
        if meta.get("telemetry"):
            cmd += ["--telemetry", meta["telemetry"]]
        _write_state(rid, state="running", started=time.time())
        log = open(UPLOADS / rid / "pipeline.log", "w", encoding="utf-8")
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        log.close()
        rep = _read_json(run_dir / "report.json") or {}
        if proc.returncode != 0 or rep.get("status") not in ("SUCCESS", "PARTIAL_SUCCESS"):
            failed = rep.get("failed_stage")
            gate = _read_json(run_dir / failed / "gate.json") if failed else None
            why = ", ".join((gate or {}).get("failures", [])) or (gate or {}).get("error") or "see pipeline.log"
            _write_state(rid, state="failed", error=f"{failed or 'pipeline'}: {why}", finished=time.time())
            continue
        _write_state(rid, state="building")
        try:
            from build_dashboard import collect

            data = collect(str(run_dir), with_mesh=True, faces=90000)
            data["label"] = meta.get("label") or rid
            t = meta.get("tuning", {})
            lane_name = {"live": "LIVE (Depth Anything)", "pi3x": "LIVE+ (Pi3X)", "survey": "SURVEY (stereo)"}.get(
                meta.get("lane"), meta.get("lane"))
            data["desc"] = (f"Uploaded {meta.get('filename')}, {lane_name}, {t.get('detail', 'standard')} detail, "
                            f"{t.get('keyframes', 'auto')} keyframes, "
                            + ("with telemetry" if meta.get("telemetry") else
                               f"no telemetry (scale from an assumed {meta.get('altitude_m')} m camera height)"))
            data["tuning"] = t
            (run_dir / "dashboard_run.json").write_text(json.dumps(data, separators=(",", ":")))
            _write_state(rid, state="done", finished=time.time())
        except Exception as exc:
            _write_state(rid, state="failed", error=f"dashboard export: {exc}", finished=time.time())


def _recover() -> None:
    """After a restart: re-queue waiting jobs, fail the ones that were cut off mid-run."""
    pending = []
    for d in UPLOADS.iterdir():
        st = _read_json(d / "server_state.json")
        if not st:
            continue
        if st.get("state") == "queued":
            pending.append((st.get("created", 0), d.name))
        elif st.get("state") in ("running", "building"):
            _write_state(d.name, state="failed", error="interrupted: the server restarted during this run; upload again",
                         finished=time.time())
    for _, rid in sorted(pending):
        jobs.put(rid)


_recover()
threading.Thread(target=_worker, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(ROOT / "webapp" / "dashboard.html", media_type="text/html")


@app.get("/api/health")
def health():
    return {"ok": True, "pipeline": "v3", "queued": jobs.qsize(), "caps": CAPS}


@app.post("/api/runs")
async def create_run(video: UploadFile = File(...), telemetry: Optional[UploadFile] = File(None),
                     altitude_m: float = Form(60.0), lane: str = Form("live"), detail: str = Form("standard"),
                     keyframes: str = Form("auto"), min_building_m: float = Form(2.5),
                     facades: bool = Form(True), regularize: bool = Form(True), texture: bool = Form(True),
                     masks: bool = Form(True), vegetation: bool = Form(True)):
    if lane not in ("live", "pi3x", "survey"):
        raise HTTPException(400, "lane must be live, pi3x or survey")
    if lane == "survey" and not CAPS["colmap_cuda"]:
        raise HTTPException(400, "The stereo (SURVEY) lane needs an NVIDIA GPU and a CUDA build of COLMAP")
    if lane == "pi3x" and not CAPS["pi3"]:
        raise HTTPException(400, "Pi3X is not installed on this machine (see scripts/setup_env.sh)")
    if detail not in DETAIL or keyframes not in KEYFRAMES:
        raise HTTPException(400, "detail must be draft/standard/high; keyframes auto/dense/sparse")
    if not (5.0 <= altitude_m <= 1000.0) or not (1.0 <= min_building_m <= 20.0):
        raise HTTPException(400, "flying height 5-1000 m; minimum building height 1-20 m")
    tuning = dict(altitude_m=altitude_m, lane=lane, detail=detail, keyframes=keyframes, min_building_m=min_building_m,
                  facades=facades, regularize=regularize, texture=texture, masks=masks, vegetation=vegetation)
    name = os.path.basename(video.filename or "video.mp4")
    if not re.search(r"\.(mp4|mov|m4v|avi|mkv)$", name, re.I):
        raise HTTPException(400, "Upload a video file (.mp4, .mov, .m4v, .avi or .mkv)")
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", os.path.splitext(name)[0])[:40] or "video"
    rid = f"upload_{stem}_{uuid.uuid4().hex[:6]}"
    d = UPLOADS / rid
    d.mkdir(parents=True)
    vpath = d / ("input" + os.path.splitext(name)[1].lower())
    with open(vpath, "wb") as f:
        shutil.copyfileobj(video.file, f, length=8 << 20)
    tpath = None
    if telemetry is not None and telemetry.filename:
        tname = os.path.basename(telemetry.filename)
        if not re.search(r"\.(srt|csv)$", tname, re.I):
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, "Telemetry must be a DJI .SRT or a .CSV flight log")
        tpath = d / ("telemetry" + os.path.splitext(tname)[1].lower())
        with open(tpath, "wb") as f:
            shutil.copyfileobj(telemetry.file, f)
    _write_state(rid, state="queued", filename=name, label=stem.replace("_", " "), video=str(vpath),
                 telemetry=str(tpath) if tpath else None, altitude_m=altitude_m, lane=lane, tuning=tuning,
                 overrides=tuning_overrides(tuning), created=time.time())
    jobs.put(rid)
    return {"id": rid, "position": jobs.qsize()}


@app.get("/api/runs")
def list_runs():
    out = []
    for d in UPLOADS.iterdir():
        st = _read_json(d / "server_state.json")
        if st:
            out.append({"id": d.name, "state": st.get("state"), "label": st.get("label"),
                        "created": st.get("created", 0), "tuning": st.get("tuning")})
    return sorted(out, key=lambda r: -r["created"])  # newest first


@app.get("/api/runs/{rid}")
def run_state(rid: str):
    st = _read_json(_state_path(rid))
    if st is None:
        raise HTTPException(404, "unknown run")
    run_dir = RUNS / rid
    status = _read_json(run_dir / "status.json") or {}
    stages = {}
    for s in STAGES:
        g = _read_json(run_dir / s / "gate.json")
        if g is not None:
            stages[s] = {"passed": bool(g.get("passed")), "seconds": g.get("seconds")}
    return {"id": rid, "state": st.get("state"), "error": st.get("error"), "stage": status.get("stage"),
            "stages": stages, "elapsed_s": round(time.time() - st.get("started", time.time()), 1)
            if st.get("state") == "running" else None}


@app.get("/api/runs/{rid}/data")
def run_data(rid: str):
    p = RUNS / rid / "dashboard_run.json"
    if not p.exists():
        raise HTTPException(404, "run not finished")
    return FileResponse(p, media_type="application/json")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")  # set HOST=0.0.0.0 inside a container
    port = int(os.environ.get("PORT", "8765"))
    print(f"PS158 v3 server on http://localhost:{port}  (drop a video on the dashboard)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
