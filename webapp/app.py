"""
FastAPI Web Service Backend for SIH26158 3D Drone Reconstruction
Provides REST endpoints:
- POST /api/upload: Upload drone video & telemetry, triggers pipeline DAG
- GET /api/status/{job_id}: Real-time stage progress & duration telemetry
- GET /api/results/{job_id}: Fetch final deliverables (OBJ, classified LAS, orthomosaic)
- Serves static interactive 3D WebGL viewer
"""

import os
import uuid
import time
import shutil
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline_runner import run_full_pipeline

app = FastAPI(
    title="SIH26158 Drone 3D Reconstruction API",
    description="High-Speed Georeferenced Photogrammetry & Semantic Classification Engine",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jobs state dictionary in-memory
JOBS: Dict[str, Dict[str, Any]] = {}
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs", "api_runs")
os.makedirs(RUNS_DIR, exist_ok=True)


from typing import Dict, Any, Optional

def background_pipeline_job(job_id: str, video_path: str, telemetry_path: Optional[str], output_dir: str, mode: str):
    try:
        JOBS[job_id]["status"] = "PROCESSING"
        JOBS[job_id]["start_time"] = time.time()

        audit = run_full_pipeline(
            video_path=video_path,
            telemetry_path=telemetry_path,
            output_dir=output_dir,
            mode=mode,
            no_gps=(telemetry_path is None)
        )

        JOBS[job_id]["status"] = "COMPLETED"
        JOBS[job_id]["end_time"] = time.time()
        JOBS[job_id]["duration_sec"] = round(JOBS[job_id]["end_time"] - JOBS[job_id]["start_time"], 2)
        JOBS[job_id]["audit_report"] = audit
        JOBS[job_id]["deliverables_dir"] = os.path.join(output_dir, "deliverables")

    except Exception as e:
        JOBS[job_id]["status"] = "FAILED"
        JOBS[job_id]["error"] = str(e)


@app.post("/api/upload")
async def upload_flight_data(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    telemetry: Optional[UploadFile] = File(None),
    mode: str = "fast"
):
    job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    job_dir = os.path.join(RUNS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    video_ext = os.path.splitext(video.filename)[1] or ".mp4"
    video_save_path = os.path.join(job_dir, f"input_video{video_ext}")
    with open(video_save_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    tel_save_path = None
    if telemetry and telemetry.filename:
        tel_ext = os.path.splitext(telemetry.filename)[1] or ".srt"
        tel_save_path = os.path.join(job_dir, f"input_telemetry{tel_ext}")
        with open(tel_save_path, "wb") as f:
            shutil.copyfileobj(telemetry.file, f)

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "QUEUED",
        "created_at": time.time(),
        "video_filename": video.filename,
        "telemetry_filename": telemetry.filename if telemetry else None,
        "mode": mode,
        "job_dir": job_dir
    }

    background_tasks.add_task(
        background_pipeline_job,
        job_id,
        video_save_path,
        tel_save_path,
        job_dir,
        mode
    )

    return {
        "status": "ACCEPTED",
        "job_id": job_id,
        "message": "Flight video & telemetry queued for 3D reconstruction",
        "status_url": f"/api/status/{job_id}",
        "results_url": f"/api/results/{job_id}"
    }


@app.get("/api/status/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    job = JOBS[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "mode": job.get("mode"),
        "elapsed_sec": round(time.time() - job["created_at"], 1) if job["status"] == "PROCESSING" else job.get("duration_sec"),
        "error": job.get("error")
    }


@app.get("/api/results/{job_id}")
async def get_job_results(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    job = JOBS[job_id]
    if job["status"] != "COMPLETED":
        return {"job_id": job_id, "status": job["status"], "message": "Results not ready yet"}

    audit = job.get("audit_report", {})
    return {
        "job_id": job_id,
        "status": "COMPLETED",
        "total_wall_clock_sec": job.get("duration_sec"),
        "georef_metrics": audit.get("georef_metrics"),
        "geometry_metrics": audit.get("geometry_metrics"),
        "classification_metrics": audit.get("classification_metrics"),
        "downloads": {
            "model_obj": f"/api/download/{job_id}/model_georeferenced.obj",
            "classified_las": f"/api/download/{job_id}/classified_pointcloud.las",
            "orthomosaic_geotiff": f"/api/download/{job_id}/orthomosaic.tif",
            "audit_report": f"/api/download/{job_id}/audit_report.json"
        }
    }


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    job_dir = JOBS[job_id]["job_dir"]
    deliverables_path = os.path.join(job_dir, "deliverables", filename)
    audit_path = os.path.join(job_dir, filename)

    if os.path.exists(deliverables_path):
        return FileResponse(deliverables_path, filename=filename)
    elif os.path.exists(audit_path):
        return FileResponse(audit_path, filename=filename)
    else:
        raise HTTPException(status_code=404, detail=f"File {filename} not found")


# Mount static web viewer
STATIC_DIR = os.path.join(PROJECT_ROOT, "webapp")
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
