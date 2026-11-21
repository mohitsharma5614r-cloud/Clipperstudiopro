"""Phase 2: Signal Extraction Module.

Computes 1-second resolution time-series signals across the full video:
  a) Replay Heatmap (YouTube most-replayed or interpolated)
  b) Audio Energy (RMS loudness, spectral flux, sudden spikes)
  c) Visual Activity (motion intensity, scene cut density)
  d) Pacing & Silence (speech/sound activity, silence penalties)

Includes per-job caching so repeated analysis or parameter adjustments are instantaneous.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft

from clipper.download import VideoMetadata
from clipper.utils import get_config, logger


@dataclass
class VideoSignals:
    """Extracted time-series signals and boundary anchors."""
    duration: float
    total_seconds: int
    has_heatmap: bool
    heatmap: List[float]       # Length = total_seconds, values in [0, 1]
    audio_energy: List[float]  # Length = total_seconds, values in [0, 1]
    visual_motion: List[float] # Length = total_seconds, values in [0, 1]
    pacing: List[float]        # Length = total_seconds, values in [0, 1]
    scene_cuts: List[float]    # Timestamp list of detected scene cuts
    silences: List[List[float]]# List of [start_sec, end_sec] silence intervals


def normalize_signal(arr: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Min-max normalize an array to [0.0, 1.0]."""
    if len(arr) == 0:
        return arr
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    if max_val - min_val < eps:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - min_val) / (max_val - min_val)).astype(np.float32)


def extract_heatmap_signal(
    heatmap_data: Optional[List[Dict[str, Any]]],
    total_seconds: int,
) -> tuple[np.ndarray, bool]:
    """Interpolate YouTube heatmap points onto 1-second bins."""
    signal = np.zeros(total_seconds, dtype=np.float32)
    if not heatmap_data or len(heatmap_data) == 0:
        return signal, False

    try:
        # Each point typically has start_time, end_time, and value
        times = []
        vals = []
        for pt in heatmap_data:
            st = float(pt.get("start_time", 0.0))
            et = float(pt.get("end_time", st + 1.0))
            mid = (st + et) / 2.0
            val = float(pt.get("value", 0.0))
            times.append(mid)
            vals.append(val)

        if not times:
            return signal, False

        # Sort by time
        sorted_pairs = sorted(zip(times, vals), key=lambda x: x[0])
        times_arr = np.array([p[0] for p in sorted_pairs], dtype=np.float32)
        vals_arr = np.array([p[1] for p in sorted_pairs], dtype=np.float32)

        # Interpolate onto 0..total_seconds-1
        grid = np.arange(total_seconds, dtype=np.float32)
        interp_vals = np.interp(grid, times_arr, vals_arr, left=vals_arr[0], right=vals_arr[-1])
        return normalize_signal(interp_vals), True
    except Exception as e:
        logger.warning("Failed to parse heatmap data: %s", e)
        return signal, False


def extract_audio_signals(
    wav_path: Path | str,
    total_seconds: int,
    silence_thresh_db: float = -32.0,
    silence_min_dur: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, List[List[float]]]:
    """Compute audio energy signal, pacing signal, and detect silence intervals."""
    sr, data = wavfile.read(str(wav_path))

    # Convert to float mono in range [-1.0, 1.0]
    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_int = np.iinfo(data.dtype).max
        samples = data.astype(np.float32) / float(max_int)
    else:
        samples = data.astype(np.float32)

    total_samples = len(samples)
    samples_per_sec = sr

    rms_per_sec = np.zeros(total_seconds, dtype=np.float32)
    flux_per_sec = np.zeros(total_seconds, dtype=np.float32)
    pacing = np.ones(total_seconds, dtype=np.float32)

    # 1. High-speed Vectorized RMS loudness per second
    valid_sec_samples = min(total_samples, total_seconds * samples_per_sec)
    n_full_secs = valid_sec_samples // samples_per_sec
    if n_full_secs > 0:
        reshaped_sec = samples[:n_full_secs * samples_per_sec].reshape(n_full_secs, samples_per_sec)
        rms_per_sec[:n_full_secs] = np.sqrt(np.mean(reshaped_sec**2, axis=1))

    # 2. Vectorized spectral flux for applause/laughter/cheering/bursts
    # STFT frame length = 512. For long videos (>10 mins), use 2x downsampling for instant FFT
    if total_seconds > 600:
        stft_samples = samples[::2]
        stft_sr = sr // 2
        stft_noverlap = 0
    else:
        stft_samples = samples
        stft_sr = sr
        stft_noverlap = 256

    nperseg = min(512, len(stft_samples))
    if nperseg > 64:
        _, _, Zxx = stft(stft_samples, fs=stft_sr, nperseg=nperseg, noverlap=stft_noverlap)
        mag = np.abs(Zxx)
        diff = np.diff(mag, axis=1)
        pos_diff = np.maximum(0.0, diff)
        frame_flux = np.sum(pos_diff, axis=0)

        frames_per_sec = len(frame_flux) / max(1.0, len(stft_samples) / stft_sr)
        for t in range(total_seconds):
            f_start = int(t * frames_per_sec)
            f_end = int(min(len(frame_flux), (t + 1) * frames_per_sec))
            if f_end > f_start:
                flux_per_sec[t] = np.mean(frame_flux[f_start:f_end])

    # 3. Sudden loudness spikes relative to rolling average
    kernel_size = 5
    pad = kernel_size // 2
    padded_rms = np.pad(rms_per_sec, pad, mode="edge")
    rolling_rms = np.convolve(padded_rms, np.ones(kernel_size) / kernel_size, mode="valid")
    spikes = np.maximum(0.0, rms_per_sec - rolling_rms)

    # Combine audio components
    rms_norm = normalize_signal(rms_per_sec)
    flux_norm = normalize_signal(flux_per_sec)
    spike_norm = normalize_signal(spikes)
    audio_energy = 0.45 * rms_norm + 0.35 * flux_norm + 0.20 * spike_norm
    audio_energy = normalize_signal(audio_energy)

    # 4. Lightning Fast Vectorized Silence Detection and Pacing
    chunk_size = int(sr * 0.1)
    n_chunks = total_samples // chunk_size
    silence_blocks = np.zeros(n_chunks, dtype=bool)

    if n_chunks > 0:
        reshaped_chunks = samples[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
        rms_c = np.sqrt(np.mean(reshaped_chunks**2, axis=1)) + 1e-7
        db_c = 20.0 * np.log10(rms_c)
        silence_blocks = db_c < silence_thresh_db

    # Identify continuous silence intervals >= silence_min_dur
    min_silence_chunks = int(silence_min_dur / 0.1)
    silences: List[List[float]] = []
    in_silence = False
    start_c = 0

    for i, is_sil in enumerate(silence_blocks):
        if is_sil and not in_silence:
            in_silence = True
            start_c = i
        elif not is_sil and in_silence:
            in_silence = False
            dur_chunks = i - start_c
            if dur_chunks >= min_silence_chunks:
                silences.append([round(start_c * 0.1, 2), round(i * 0.1, 2)])

    if in_silence:
        dur_chunks = n_chunks - start_c
        if dur_chunks >= min_silence_chunks:
            silences.append([round(start_c * 0.1, 2), round(n_chunks * 0.1, 2)])

    # Penalize pacing during silences and long dead zones
    for sil_start, sil_end in silences:
        dur = sil_end - sil_start
        s_sec = int(math.floor(sil_start))
        e_sec = int(math.ceil(sil_end))
        for t in range(s_sec, min(total_seconds, e_sec)):
            penalty = 0.0 if dur > 2.0 else 0.2
            pacing[t] = min(pacing[t], penalty)

    # Penalize monotonous regions
    for t in range(total_seconds):
        t0 = max(0, t - 2)
        t1 = min(total_seconds, t + 3)
        window_std = np.std(rms_norm[t0:t1])
        if window_std < 0.02 and rms_norm[t] < 0.25:
            pacing[t] *= 0.5

    return audio_energy, pacing, silences


def extract_visual_signals(
    video_path: Path | str,
    total_seconds: int,
    sample_fps: float = 2.0,
    scene_cut_threshold: float = 28.0,
) -> tuple[np.ndarray, List[float]]:
    """Compute visual motion intensity, scene cuts, and cut density using high-speed sequential grabbing."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("Could not open video for visual analysis: %s", video_path)
        return np.zeros(total_seconds, dtype=np.float32), []

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Adaptive sample rate: high accuracy without lag on long videos
    if total_seconds > 7200:    # > 2 hours
        effective_sample_fps = 0.33  # 1 frame every 3s
    elif total_seconds > 1800:  # > 30 mins
        effective_sample_fps = 0.5   # 1 frame every 2s
    elif total_seconds > 300:   # > 5 mins
        effective_sample_fps = 1.0   # 1 frame every 1s
    else:
        effective_sample_fps = sample_fps

    frame_step = max(1, int(round(vid_fps / effective_sample_fps)))
    prev_gray: Optional[np.ndarray] = None
    motion_by_sec: List[List[float]] = [[] for _ in range(total_seconds)]
    scene_cuts: List[float] = []

    # High-speed sequential read using cap.grab() to avoid catastrophic keyframe seeking
    frame_idx = 0
    while True:
        if frame_idx % frame_step == 0:
            ret, frame = cap.read()
            if not ret:
                break
            t_sec = int(frame_idx / vid_fps)
            if t_sec >= total_seconds:
                break

            # Resize to 160x90 for blazing fast diff calculation
            small = cv2.resize(frame, (160, 90))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                mean_diff = float(np.mean(diff))
                motion_by_sec[t_sec].append(mean_diff)

                # Scene cut detection
                if mean_diff > scene_cut_threshold:
                    cut_time = round(frame_idx / vid_fps, 2)
                    if not scene_cuts or (cut_time - scene_cuts[-1]) > 0.8:
                        scene_cuts.append(cut_time)

            prev_gray = gray
        else:
            if not cap.grab():
                break
        frame_idx += 1

    cap.release()

    # Aggregate motion per second
    motion_arr = np.zeros(total_seconds, dtype=np.float32)
    for t in range(total_seconds):
        vals = motion_by_sec[t]
        motion_arr[t] = np.mean(vals) if vals else 0.0

    # Calculate scene cut density in rolling 4-second window
    cut_density = np.zeros(total_seconds, dtype=np.float32)
    for cut_t in scene_cuts:
        c_sec = int(cut_t)
        for offset in range(-2, 3):
            target_sec = c_sec + offset
            if 0 <= target_sec < total_seconds:
                cut_density[target_sec] += 1.0

    motion_norm = normalize_signal(motion_arr)
    cut_norm = normalize_signal(cut_density)

    visual_activity = 0.65 * motion_norm + 0.35 * cut_norm
    return normalize_signal(visual_activity), scene_cuts


def compute_video_signals(
    meta: VideoMetadata,
    cache_path: Optional[Path | str] = None,
    force_recompute: bool = False,
) -> VideoSignals:
    """Compute and cache all time-series signals for a video."""
    cfg = get_config()
    total_seconds = max(1, int(math.ceil(meta.duration)))

    if cache_path is None:
        cache_path = Path(meta.source_file).parent / "signals.json"
    else:
        cache_path = Path(cache_path)

    # Check cache
    if not force_recompute and cache_path.is_file() and cache_path.stat().st_size > 0:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.info("Loaded cached signals from %s", cache_path)
            return VideoSignals(
                duration=float(cached["duration"]),
                total_seconds=int(cached["total_seconds"]),
                has_heatmap=bool(cached["has_heatmap"]),
                heatmap=cached["heatmap"],
                audio_energy=cached["audio_energy"],
                visual_motion=cached["visual_motion"],
                pacing=cached["pacing"],
                scene_cuts=cached["scene_cuts"],
                silences=cached["silences"],
            )
        except Exception as e:
            logger.warning("Failed to load cached signals, recomputing: %s", e)

    logger.info("Extracting time-series signals for job %s (%d seconds)", meta.job_id, total_seconds)

    # Signal A: Heatmap
    heatmap_sig, has_heatmap = extract_heatmap_signal(meta.heatmap, total_seconds)

    # Signal B & D: Audio & Pacing
    silence_db = float(cfg.get("audio", {}).get("silence_thresh_db", -32.0))
    silence_dur = float(cfg.get("audio", {}).get("silence_min_dur_sec", 0.5))
    audio_sig, pacing_sig, silences = extract_audio_signals(
        meta.audio_file, total_seconds, silence_thresh_db=silence_db, silence_min_dur=silence_dur
    )

    # Signal C: Visual Activity & Scene Cuts
    sample_fps = float(cfg.get("visual", {}).get("sample_fps", 2.0))
    cut_thresh = float(cfg.get("visual", {}).get("scene_cut_threshold", 28.0))
    visual_sig, scene_cuts = extract_visual_signals(
        meta.source_file, total_seconds, sample_fps=sample_fps, scene_cut_threshold=cut_thresh
    )

    signals_obj = VideoSignals(
        duration=meta.duration,
        total_seconds=total_seconds,
        has_heatmap=has_heatmap,
        heatmap=heatmap_sig.tolist(),
        audio_energy=audio_sig.tolist(),
        visual_motion=visual_sig.tolist(),
        pacing=pacing_sig.tolist(),
        scene_cuts=scene_cuts,
        silences=silences,
    )

    # Cache to disk
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(asdict(signals_obj), f, indent=2)
        logger.info("Cached signals to %s", cache_path)
    except Exception as e:
        logger.warning("Could not write signals cache: %s", e)

    return signals_obj
