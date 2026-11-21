"""Phase 1: Video and Metadata Ingestion Module.

Handles downloading videos (YouTube or other yt-dlp supported sites) or ingesting local files,
extracting metadata (chapters, replay heatmap), and preparing 16kHz mono audio.
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import yt_dlp

from clipper.utils import ensure_dir, get_config, get_ffmpeg_path, logger, run_ffmpeg


@dataclass
class VideoMetadata:
    """Standardized video metadata."""
    job_id: str
    url_or_path: str
    title: str
    channel: str
    duration: float
    width: int
    height: int
    fps: float
    chapters: List[Dict[str, Any]]
    heatmap: Optional[List[Dict[str, float]]]
    has_heatmap: bool
    source_file: str
    audio_file: str


@dataclass
class IngestionResult:
    """Result of the ingestion phase."""
    job_id: str
    job_dir: Path
    video_path: Path
    audio_path: Path
    metadata_path: Path
    metadata: VideoMetadata


def get_video_properties(video_path: Path | str) -> tuple[int, int, float, float]:
    """Extract width, height, fps, and duration using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()
    return width, height, fps, duration


def extract_audio_16k(video_path: Path | str, output_wav_path: Path | str) -> Path:
    """Extract 16kHz mono 16-bit PCM WAV from video using FFmpeg."""
    output_wav_path = Path(output_wav_path)
    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(output_wav_path),
    ]
    logger.info("Extracting 16kHz mono audio to %s", output_wav_path)
    run_ffmpeg(args)

    if not output_wav_path.exists() or output_wav_path.stat().st_size == 0:
        raise RuntimeError(f"Audio extraction failed, empty or missing file: {output_wav_path}")

    return output_wav_path


def find_cached_video(
    url_or_path: str,
    workspace_dir: Path,
) -> Optional[tuple[Path, Path, Path, VideoMetadata]]:
    """Check if a video with the same URL was already ingested in workspace."""
    if not workspace_dir.exists():
        return None
    norm_url = url_or_path.strip().lower()
    base_match = norm_url.split("?")[0] if "youtu" in norm_url and "?" in norm_url else norm_url

    for job_folder in workspace_dir.iterdir():
        if not job_folder.is_dir():
            continue
        meta_file = job_folder / "metadata.json"
        audio_file = job_folder / "audio_16k.wav"
        src_files = list(job_folder.glob("source.*"))
        valid_src = [
            f for f in src_files
            if not f.name.endswith(".part") and f.suffix.lower() in [".mp4", ".mkv", ".webm", ".mov"]
        ]

        if meta_file.exists() and audio_file.exists() and valid_src:
            if valid_src[0].stat().st_size > 1024 * 1024 and audio_file.stat().st_size > 1024:
                try:
                    with open(meta_file, "r") as f:
                        data = json.load(f)
                    cached_url = data.get("url_or_path", "").strip().lower()
                    cached_base = cached_url.split("?")[0] if "youtu" in cached_url and "?" in cached_url else cached_url
                    if base_match == cached_base:
                        meta = VideoMetadata(**data)
                        return valid_src[0], audio_file, meta_file, meta
                except Exception:
                    pass
    return None


def ingest_video(
    url_or_path: str,
    job_id: Optional[str] = None,
    workspace_dir: Optional[str | Path] = None,
    progress_callback: Optional[Any] = None,
) -> IngestionResult:
    """Ingest a video from URL or local path.

    Downloads or copies video, extracts metadata including YouTube replay heatmap,
    and converts audio to 16kHz mono WAV. Reuses workspace cache if already downloaded.
    """
    cfg = get_config()
    if not job_id:
        job_id = uuid.uuid4().hex[:12]

    if workspace_dir is None:
        workspace_dir = Path(cfg.get("storage", {}).get("workspace_dir", "./workspace"))
    else:
        workspace_dir = Path(workspace_dir)

    job_dir = ensure_dir(workspace_dir / job_id)
    is_local_file = os.path.isfile(url_or_path)

    metadata_path = job_dir / "metadata.json"
    audio_path = job_dir / "audio_16k.wav"

    # Fast check: If video was already ingested in workspace, reuse immediately (0.00s)
    if not is_local_file:
        cached = find_cached_video(url_or_path, workspace_dir)
        if cached:
            c_video, c_audio, c_meta, c_metadata = cached
            logger.info("Reusing existing ingested video for URL %s from %s", url_or_path, c_video.parent.name)
            if progress_callback:
                progress_callback(30, f"Found cached video '{c_metadata.title[:30]}...' (instant load)")

            dest_video = job_dir / f"source{c_video.suffix}"
            if not dest_video.exists():
                try:
                    os.link(c_video, dest_video)
                except Exception:
                    shutil.copy2(c_video, dest_video)

            if not audio_path.exists():
                try:
                    os.link(c_audio, audio_path)
                except Exception:
                    shutil.copy2(c_audio, audio_path)

            meta_dict = asdict(c_metadata)
            meta_dict["job_id"] = job_id
            meta_dict["source_file"] = str(dest_video)
            meta_dict["audio_file"] = str(audio_path)
            with open(metadata_path, "w") as f:
                json.dump(meta_dict, f, indent=2)

            return IngestionResult(
                job_id=job_id,
                job_dir=job_dir,
                video_path=dest_video,
                audio_path=audio_path,
                metadata_path=metadata_path,
                metadata=VideoMetadata(**meta_dict),
            )

    if is_local_file:
        logger.info("Ingesting local file for job %s: %s", job_id, url_or_path)
        src_ext = Path(url_or_path).suffix or ".mp4"
        video_path = job_dir / f"source{src_ext}"
        if not video_path.exists():
            shutil.copy2(url_or_path, video_path)

        width, height, fps, duration = get_video_properties(video_path)
        title = Path(url_or_path).stem
        channel = "Local File"
        chapters: List[Dict[str, Any]] = []
        heatmap: Optional[List[Dict[str, float]]] = None
    else:
        logger.info("Downloading remote video for job %s from URL: %s", job_id, url_or_path)
        out_template = str(job_dir / "source.%(ext)s")

        # Real-time progress reporting hook for yt-dlp
        def make_progress_hook(cb):
            def hook(d):
                status = d.get("status")
                if status == "downloading" and cb:
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    downloaded = d.get("downloaded_bytes", 0)
                    speed = d.get("speed") or 0
                    eta = d.get("eta") or 0
                    pct = min(100, int((downloaded / total) * 100)) if total > 0 else 0
                    mb_down = downloaded / (1024 * 1024)
                    mb_tot = total / (1024 * 1024) if total else 0
                    mb_spd = (speed or 0) / (1024 * 1024)
                    eta_m = int((eta or 0) // 60)
                    eta_s = int((eta or 0) % 60)
                    overall = 5 + int(pct * 0.23)
                    tot_str = f"/{mb_tot:.1f}" if mb_tot > 0 else ""
                    cb(overall, f"Downloading: {pct}% ({mb_down:.1f}{tot_str} MB at {mb_spd:.1f} MB/s, ETA {eta_m:02d}:{eta_s:02d})")
                elif status == "finished" and cb:
                    cb(28, "Download complete! Extracting audio & preparing analysis...")
            return hook

        ffmpeg_bin = None
        try:
            ffmpeg_bin = get_ffmpeg_path()
        except Exception:
            pass

        base_ydl_opts = {
            "format": "best[height<=1080]/bestvideo[height<=1080]+bestaudio/best",
            "outtmpl": out_template,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "socket_timeout": 20,
            "retries": 3,
        }
        if ffmpeg_bin:
            base_ydl_opts["ffmpeg_location"] = ffmpeg_bin

        if progress_callback:
            base_ydl_opts["progress_hooks"] = [make_progress_hook(progress_callback)]

        # Priority strategies: 'android' client bypasses YouTube SABR 403 throttling instantly
        client_strategies = [
            ["android"],
            ["android", "web"],
            None,
        ]

        info = None
        last_error = None

        for clients in client_strategies:
            opts = dict(base_ydl_opts)
            if clients:
                opts["extractor_args"] = {"youtube": {"player_client": clients}}
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url_or_path, download=True)
                    if info is not None:
                        break
            except Exception as e:
                last_error = e
                logger.warning("yt-dlp attempt with clients %s failed: %s", clients, e)
                continue

        if info is None:
            raise RuntimeError(f"Could not download video: {last_error or 'Unknown error'}")

        # Locate downloaded source video
        matched_videos = list(job_dir.glob("source.*"))
        matched_videos = [v for v in matched_videos if v.suffix.lower() in [".mp4", ".mkv", ".webm", ".mov"]]
        if not matched_videos:
            raise FileNotFoundError(f"Downloaded video not found in {job_dir}")
        video_path = matched_videos[0]

        title = info.get("title", "Untitled Video")
        channel = info.get("uploader", info.get("channel", "Unknown Channel"))
        duration = float(info.get("duration") or 0.0)
        chapters = info.get("chapters") or []
        heatmap = info.get("heatmap")

        # Fallback to OpenCV if duration or dimensions missing from yt-dlp info
        width, height, fps, cv_dur = get_video_properties(video_path)
        if duration <= 0:
            duration = cv_dur

    has_heatmap = bool(heatmap and len(heatmap) > 0)
    logger.info(
        "Ingested video '%s' (%.1fs, %dx%d, %s heatmap)",
        title, duration, width, height, "with" if has_heatmap else "without"
    )

    # Extract 16kHz mono audio
    extract_audio_16k(video_path, audio_path)

    meta = VideoMetadata(
        job_id=job_id,
        url_or_path=url_or_path,
        title=title,
        channel=channel,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        chapters=chapters,
        heatmap=heatmap,
        has_heatmap=has_heatmap,
        source_file=str(video_path),
        audio_file=str(audio_path),
    )

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, indent=2)

    return IngestionResult(
        job_id=job_id,
        job_dir=job_dir,
        video_path=video_path,
        audio_path=audio_path,
        metadata_path=metadata_path,
        metadata=meta,
    )

