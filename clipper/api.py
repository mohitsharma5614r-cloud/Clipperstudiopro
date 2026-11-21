"""Phase 5: FastAPI REST API and Background Job Processing Engine.

Endpoints:
  POST /jobs                     Create clipping job (url + options)
  GET  /jobs/{id}                Get job status and progress
  GET  /jobs/{id}/clips          Get list of generated clips with scores and reasons
  GET  /jobs/{id}/clips/{cid}/stream    Stream video with range support for HTML5 player
  GET  /jobs/{id}/clips/{cid}/download  Download clip as attachment
  GET  /jobs/{id}/download-all   Download all clips as a single ZIP archive
  GET  /health                   Service health check
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import threading
import time
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clipper.analyze import detect_moments
from clipper.download import ingest_video
from clipper.render import render_all_clips
from clipper.signals import compute_video_signals
from clipper.utils import ensure_dir, get_config, get_workspace_dir, logger

app = FastAPI(
    title="AI Clipping Tool API",
    description="Free, local AI video clipping API using multi-signal moment detection",
    version="1.0.0",
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_no_cache_header(request: Request, call_next):
    """Prevent aggressive browser caching for static UI assets so styles always update immediately."""
    response = await call_next(request)
    path = request.url.path
    if path in ("/", "/index.html", "/style.css", "/app.js") or path.endswith((".css", ".js", ".html")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


class JobRequest(BaseModel):
    """Job creation request payload."""
    url: str = Field(..., description="YouTube URL or local video file path")
    num_clips: int = Field(5, ge=1, le=15, description="Number of clips to generate")
    min_length: float = Field(20.0, ge=5.0, le=120.0, description="Minimum clip duration in seconds")
    max_length: float = Field(60.0, ge=10.0, le=180.0, description="Maximum clip duration in seconds")
    aspect_ratio: str = Field("9:16", description="Target aspect ratio: 9:16, 16:9, or 1:1")


class JobState:
    """In-memory representation of a job state."""
    def __init__(self, job_id: str, request: JobRequest):
        self.job_id = job_id
        self.request = request
        self.status = "queued"  # queued, downloading, analyzing, rendering, completed, failed
        self.progress = 0       # 0 - 100
        self.stage = "Queued for processing"
        self.error: Optional[str] = None
        self.video_title: Optional[str] = None
        self.duration: Optional[float] = None
        self.created_at = time.time()
        self.updated_at = time.time()
        self.clips: List[Dict[str, Any]] = []

    def update(self, status: str, progress: int, stage: str, error: Optional[str] = None):
        self.status = status
        self.progress = progress
        self.stage = stage
        if error:
            self.error = error
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "url": self.request.url,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "error": self.error,
            "video_title": self.video_title,
            "duration": self.duration,
            "num_clips": self.request.num_clips,
            "aspect_ratio": self.request.aspect_ratio,
            "clips_count": len(self.clips),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# Global in-memory job registry
JOBS: Dict[str, JobState] = {}


def process_job_sync(job_id: str):
    """Synchronous worker function that executes the full clipping pipeline."""
    job = JOBS.get(job_id)
    if not job:
        return

    try:
        job.update("downloading", 5, "Initializing workspace & checking cache...")
        workspace_dir = get_workspace_dir()
        job_dir = ensure_dir(workspace_dir / job_id)
        clips_dir = ensure_dir(job_dir / "clips")
        logger.info("[%s] Starting ingestion for: %s", job_id, job.request.url)

        def on_download_progress(pct: int, msg: str):
            job.update("downloading", pct, msg)

        ingestion = ingest_video(
            job.request.url,
            job_id=job_id,
            workspace_dir=workspace_dir,
            progress_callback=on_download_progress,
        )
        job.video_title = ingestion.metadata.title
        job.duration = ingestion.metadata.duration
        job.update("downloading", 30, f"Ready: '{ingestion.metadata.title[:30]}...'")

        # Stage 2: Signal Extraction
        job.update("analyzing", 35, "Computing replay heatmap, audio energy & visual motion signals...")
        logger.info("[%s] Extracting signals...", job_id)
        signals = compute_video_signals(ingestion.metadata)
        job.update("analyzing", 60, "Signals computed. Detecting top engaging moments...")

        # Stage 3: Moment Detection
        logger.info("[%s] Detecting moments...", job_id)
        candidates = detect_moments(
            metadata=ingestion.metadata,
            signals=signals,
            num_clips=job.request.num_clips,
            min_length=job.request.min_length,
            max_length=job.request.max_length,
        )

        if not candidates:
            raise RuntimeError("No candidate moments detected for video.")

        # Stage 4: Rendering
        total_clips = len(candidates)
        job.update("rendering", 65, f"Rendering {total_clips} clips in {job.request.aspect_ratio}...")
        logger.info("[%s] Rendering %d clips...", job_id, total_clips)

        def on_render_progress(current_idx: int, total_count: int):
            percent = 65 + int((current_idx / total_count) * 30)
            job.update("rendering", percent, f"Rendering clip {current_idx} of {total_count}...")

        rendered_clips = render_all_clips(
            source_video_path=ingestion.video_path,
            candidates=candidates,
            output_dir=clips_dir,
            aspect_ratio=job.request.aspect_ratio,
            progress_callback=on_render_progress,
        )

        # Build clip metadata response
        clip_list = []
        for rc in rendered_clips:
            clip_list.append({
                "clip_id": rc.clip_id,
                "clip_index": rc.clip_index,
                "title": rc.title,
                "reason": rc.reason,
                "score": rc.score,
                "start_sec": rc.start_sec,
                "end_sec": rc.end_sec,
                "duration": rc.duration,
                "aspect_ratio": rc.aspect_ratio,
                "face_detected": rc.face_detected,
                "is_top_viral": rc.is_top_viral,
                "breakdown": rc.breakdown,
                "video_url": f"/jobs/{job_id}/clips/{rc.clip_id}/stream",
                "download_url": f"/jobs/{job_id}/clips/{rc.clip_id}/download",
            })

        job.clips = clip_list
        job.update("completed", 100, f"Successfully created {len(clip_list)} clips!")
        logger.info("[%s] Job completed successfully with %d clips", job_id, len(clip_list))

    except Exception as e:
        logger.exception("[%s] Job failed: %s", job_id, e)
        job.update("failed", job.progress, "Processing failed", error=str(e))


@app.post("/jobs")
async def create_job(req: JobRequest, background_tasks: BackgroundTasks):
    """Create and start a new video clipping job."""
    norm_url = req.url.strip().lower()
    for existing_id, existing_job in JOBS.items():
        if (
            existing_job.request.url.strip().lower() == norm_url
            and existing_job.request.num_clips == req.num_clips
            and existing_job.request.aspect_ratio == req.aspect_ratio
        ):
            # If already completed or actively in progress, reuse
            if existing_job.status in ("completed", "downloading", "analyzing", "rendering"):
                logger.info("Reusing active job %s (status: %s) for %s", existing_id, existing_job.status, req.url)
                return {"job_id": existing_id, "status": existing_job.status, "message": f"Matching job {existing_job.status}"}

    job_id = uuid.uuid4().hex[:12]
    job_state = JobState(job_id=job_id, request=req)
    JOBS[job_id] = job_state

    # Execute in dedicated daemon thread immediately (robust against event loop or serverless queue pauses)
    worker_thread = threading.Thread(target=process_job_sync, args=(job_id,), daemon=True)
    worker_thread.start()
    return {"job_id": job_id, "status": "queued", "message": "Job created and queued"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Retrieve the current status and progress of a job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/jobs/{job_id}/clips")
async def get_job_clips(job_id: str):
    """Retrieve the list of generated clips for a completed job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "completed":
        return {"status": job.status, "clips": [], "message": f"Job is currently {job.status}"}

    return {
        "job_id": job_id,
        "video_title": job.video_title,
        "clips_count": len(job.clips),
        "clips": job.clips,
    }


@app.get("/jobs/{job_id}/clips/{clip_id}/stream")
async def stream_clip(job_id: str, clip_id: str, request: Request):
    """Stream clip video with HTTP 206 partial content support for HTML5 video player."""
    workspace_dir = get_workspace_dir()
    video_path = workspace_dir / job_id / "clips" / f"{clip_id}.mp4"

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Clip video file not found")

    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range")

    if not range_header:
        return FileResponse(video_path, media_type="video/mp4")

    # Handle Range requests for video scrubbing
    try:
        h = range_header.replace("bytes=", "").split("-")
        start = int(h[0]) if h[0] else 0
        end = int(h[1]) if len(h) > 1 and h[1] else file_size - 1
    except ValueError:
        start = 0
        end = file_size - 1

    start = max(0, min(start, file_size - 1))
    end = max(start, min(end, file_size - 1))
    chunk_size = end - start + 1

    def iterfile():
        with open(video_path, "rb") as f:
            f.seek(start)
            bytes_left = chunk_size
            while bytes_left > 0:
                read_bytes = min(bytes_left, 1024 * 1024)
                data = f.read(read_bytes)
                if not data:
                    break
                bytes_left -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": "video/mp4",
    }
    return StreamingResponse(iterfile(), status_code=206, headers=headers)


@app.get("/jobs/{job_id}/clips/{clip_id}/download")
async def download_clip(job_id: str, clip_id: str):
    """Download a single clip as an MP4 attachment."""
    workspace_dir = get_workspace_dir()
    video_path = workspace_dir / job_id / "clips" / f"{clip_id}.mp4"

    if not video_path.is_file():
        raise HTTPException(status_code=404, detail="Clip file not found")

    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{job_id}_{clip_id}.mp4",
    )


@app.get("/jobs/{job_id}/download-all")
async def download_all_clips(job_id: str):
    """Download all rendered clips and metadata as a single ZIP file."""
    workspace_dir = get_workspace_dir()
    clips_dir = workspace_dir / job_id / "clips"

    if not clips_dir.is_dir():
        raise HTTPException(status_code=404, detail="No clips found for this job")

    files = list(clips_dir.glob("*.mp4")) + list(clips_dir.glob("*.json"))
    if not files:
        raise HTTPException(status_code=404, detail="No clip files found")

    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.name)

    mem_zip.seek(0)
    return StreamingResponse(
        mem_zip,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=clips_{job_id}.zip"},
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "1.0.0", "active_jobs": len(JOBS)}


# Mount static web UI
web_dir = Path(__file__).parent.parent / "web"
if web_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

