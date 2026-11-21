"""Utility helpers for AI Clipping Tool."""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any, Dict

# Suppress harmless urllib3 LibreSSL warning on macOS Python 3.9
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("clipper")

_CONFIG_CACHE: Dict[str, Any] | None = None


def get_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration from config.yaml with caching."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and config_path is None:
        return _CONFIG_CACHE

    if config_path is None:
        # Search relative to repo root or clipper module
        candidates = [
            Path("config.yaml"),
            Path(__file__).parent.parent / "config.yaml",
            Path(__file__).parent / "config.yaml",
        ]
        for c in candidates:
            if c.is_file():
                config_path = c
                break

    if config_path and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        logger.warning("config.yaml not found, using default configuration")
        cfg = {}

    if config_path is None:
        _CONFIG_CACHE = cfg
    return cfg


def get_ffmpeg_path() -> str:
    """Find FFmpeg binary path.

    Checks system PATH first, then falls back to imageio_ffmpeg bundled binary.
    """
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except ImportError:
        pass

    raise RuntimeError(
        "FFmpeg binary not found. Please install ffmpeg or install imageio-ffmpeg."
    )


def run_ffmpeg(args: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    """Run an FFmpeg command with standard error logging."""
    ffmpeg_exe = get_ffmpeg_path()
    cmd = [ffmpeg_exe, "-y", "-hide_banner", "-loglevel", "error"] + args
    logger.debug("Executing FFmpeg command: %s", " ".join(cmd))
    try:
        res = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=True,
        )
        return res
    except subprocess.CalledProcessError as e:
        logger.error("FFmpeg failed (exit %d): %s", e.returncode, e.stderr.strip())
        raise RuntimeError(f"FFmpeg error: {e.stderr.strip()}") from e


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    seconds = max(0.0, float(seconds))
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def cleanup_old_workspaces(workspace_dir: str | Path = "./workspace", max_age_hours: float = 24.0) -> int:
    """Remove workspace job directories older than max_age_hours."""
    if max_age_hours <= 0:
        return 0
    p = Path(workspace_dir)
    if not p.is_dir():
        return 0

    now = time.time()
    cutoff = now - (max_age_hours * 3600.0)
    removed_count = 0

    for item in p.iterdir():
        if item.is_dir():
            try:
                mtime = item.stat().st_mtime
                if mtime < cutoff:
                    shutil.rmtree(item, ignore_errors=True)
                    removed_count += 1
                    logger.info("Cleaned up expired workspace directory: %s", item.name)
            except Exception as e:
                logger.warning("Could not clean workspace %s: %s", item.name, e)

    return removed_count
