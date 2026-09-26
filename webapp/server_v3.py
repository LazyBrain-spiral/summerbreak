"""Local v3 server: the dashboard plus drag-and-drop reconstruction of new videos.

Run inside WSL (the pipeline needs the Linux GPU environment):

    source ~/miniforge3/etc/profile.d/conda.sh && conda activate drone3d
    python webapp/server_v3.py            # then open http://localhost:8765 on Windows

Endpoints
    GET  /                      dashboard (webapp/dashboard.html)
    GET  /api/health            {"ok": true}
    POST /api/runs              multipart: video (mp4/mov), telemetry (srt/csv, optional),
                                altitude_m (used only without telemetry), lane (live|survey)
    GET  /api/runs              uploaded runs and their state
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
               "--run-id", rid, "--skip-env-check", "--force",
               "--set", f"georef.assumed_altitude_m={float(meta.get('altitude_m') or 60.0)}"]
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
            data["desc"] = (f"Uploaded {meta.get('filename')}, {meta.get('lane', 'live').upper()} lane, "
                            + ("with telemetry" if meta.get("telemetry") else
                               f"no telemetry (scale from an assumed {meta.get('altitude_m')} m camera height)"))
            (run_dir / "dashboard_run.json").write_text(json.dumps(data, separators=(",", ":")))
            _write_state(rid, state="done", finished=time.time())
        except Exception as exc:
            _write_state(rid, state="failed", error=f"dashboard export: {exc}", finished=time.time())


threading.Thread(target=_worker, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(ROOT / "webapp" / "dashboard.html", media_type="text/html")


@app.get("/api/health")
def health():
    return {"ok": True, "pipeline": "v3", "queued": jobs.qsize()}


@app.post("/api/runs")
async def create_run(video: UploadFile = File(...), telemetry: Optional[UploadFile] = File(None),
                     altitude_m: float = Form(60.0), lane: str = Form("live")):
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
            raise HTTPException(400, "Telemetry must be a DJI .SRT or a .CSV flight log")
        tpath = d / ("telemetry" + os.path.splitext(tname)[1].lower())
        with open(tpath, "wb") as f:
            shutil.copyfileobj(telemetry.file, f)
    _write_state(rid, state="queued", filename=name, label=stem.replace("_", " "), video=str(vpath),
                 telemetry=str(tpath) if tpath else None, altitude_m=altitude_m,
                 lane="survey" if lane == "survey" else "live", created=time.time())
    jobs.put(rid)
    return {"id": rid, "position": jobs.qsize()}


@app.get("/api/runs")
def list_runs():
    out = []
    for d in sorted(UPLOADS.iterdir()):
        st = _read_json(d / "server_state.json")
        if st:
            out.append({"id": d.name, "state": st.get("state"), "label": st.get("label")})
    return out


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

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    print(f"PS158 v3 server on http://localhost:{port}  (drop a video on the dashboard)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
