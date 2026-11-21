"""Phase 3: Moment Detection and Scoring Module.

Generates candidate video segments using a sliding window, scores them using
a weighted combination of heatmap, audio, visual, and pacing signals, snaps
boundaries to natural silence pauses or scene cuts, and applies Non-Maximum
Suppression (NMS) to select the top non-overlapping clips.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from clipper.download import VideoMetadata
from clipper.signals import VideoSignals
from clipper.utils import get_config, logger


@dataclass
class SignalBreakdown:
    """Individual signal scores (0-100) for a clip."""
    heatmap: float
    audio: float
    visual: float
    pacing: float


@dataclass
class MomentCandidate:
    """Detected moment candidate with boundaries and explanation."""
    clip_index: int
    start_sec: float
    end_sec: float
    duration: float
    score: float
    breakdown: SignalBreakdown
    title: str
    reason: str
    is_top_viral: bool = False


def compute_effective_weights(
    has_heatmap: bool,
    config_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Calculate normalized weights.

    If heatmap is not available, its weight is redistributed proportionally
    across audio, visual, and pacing.
    """
    if config_weights is None:
        cfg = get_config()
        config_weights = cfg.get("weights", {
            "heatmap": 0.35,
            "audio": 0.30,
            "visual": 0.20,
            "pacing": 0.15,
        })

    w_h = float(config_weights.get("heatmap", 0.35))
    w_a = float(config_weights.get("audio", 0.30))
    w_v = float(config_weights.get("visual", 0.20))
    w_p = float(config_weights.get("pacing", 0.15))

    if not has_heatmap or w_h <= 0.0:
        subtotal = w_a + w_v + w_p
        if subtotal <= 0:
            return {"heatmap": 0.0, "audio": 0.5, "visual": 0.3, "pacing": 0.2}
        return {
            "heatmap": 0.0,
            "audio": round(w_a / subtotal, 4),
            "visual": round(w_v / subtotal, 4),
            "pacing": round(w_p / subtotal, 4),
        }

    total = w_h + w_a + w_v + w_p
    return {
        "heatmap": round(w_h / total, 4),
        "audio": round(w_a / total, 4),
        "visual": round(w_v / total, 4),
        "pacing": round(w_p / total, 4),
    }


def find_nearest_anchor(
    target_time: float,
    anchors: List[float],
    max_distance: float,
) -> Optional[float]:
    """Find the closest anchor point within max_distance."""
    if not anchors:
        return None
    closest = None
    min_dist = float("inf")
    for a in anchors:
        d = abs(a - target_time)
        if d <= max_distance and d < min_dist:
            min_dist = d
            closest = a
    return closest


def snap_boundaries(
    raw_start: float,
    raw_end: float,
    total_duration: float,
    scene_cuts: List[float],
    silences: List[List[float]],
    snap_window: float = 2.5,
    padding: float = 0.3,
    min_length: float = 15.0,
    max_length: float = 60.0,
) -> Tuple[float, float]:
    """Snap start/end boundaries to nearest silence edges or scene cuts."""
    snapped_start = raw_start
    snapped_end = raw_end

    # --- Snap Start ---
    # Look for silence ending near raw_start (speech start) or silence starting near raw_start
    start_anchors: List[float] = []
    for s_start, s_end in silences:
        # If raw_start is inside silence, snap to silence end
        if s_start <= raw_start <= s_end:
            start_anchors.append(s_end)
        else:
            if abs(s_end - raw_start) <= snap_window:
                start_anchors.append(s_end)
            if abs(s_start - raw_start) <= snap_window:
                start_anchors.append(s_start)

    # Also consider scene cuts
    start_anchors.extend([c for c in scene_cuts if abs(c - raw_start) <= snap_window])

    nearest_start = find_nearest_anchor(raw_start, start_anchors, snap_window)
    if nearest_start is not None:
        snapped_start = nearest_start

    # --- Snap End ---
    # Look for silence starting near raw_end (speech pause) or scene cuts
    end_anchors: List[float] = []
    for s_start, s_end in silences:
        if s_start <= raw_end <= s_end:
            end_anchors.append(s_start)
        else:
            if abs(s_start - raw_end) <= snap_window:
                end_anchors.append(s_start)
            if abs(s_end - raw_end) <= snap_window:
                end_anchors.append(s_end)

    end_anchors.extend([c for c in scene_cuts if abs(c - raw_end) <= snap_window])

    nearest_end = find_nearest_anchor(raw_end, end_anchors, snap_window)
    if nearest_end is not None:
        snapped_end = nearest_end

    # Add small padding (e.g. 0.3s) at start/end
    padded_start = max(0.0, round(snapped_start - padding, 2))
    padded_end = min(total_duration, round(snapped_end + padding, 2))

    # Guardrails: verify duration constraint
    dur = padded_end - padded_start
    if dur < min_length:
        # Expand end or start to reach min_length
        diff = min_length - dur
        if padded_end + diff <= total_duration:
            padded_end = round(padded_end + diff, 2)
        elif padded_start - diff >= 0:
            padded_start = round(padded_start - diff, 2)
    elif dur > max_length:
        # Shrink to max_length
        padded_end = round(padded_start + max_length, 2)

    return padded_start, padded_end


def compute_iou(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    """Calculate temporal Intersection over Union (IoU) of two intervals."""
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    inter = max(0.0, inter_end - inter_start)
    if inter <= 0.0:
        return 0.0
    union = (end_a - start_a) + (end_b - start_b) - inter
    return inter / union if union > 0 else 0.0


def generate_moment_title(
    clip_index: int,
    start_sec: float,
    end_sec: float,
    metadata: VideoMetadata,
) -> str:
    """Assign chapter title if clip falls within a chapter, otherwise Part N."""
    midpoint = (start_sec + end_sec) / 2.0
    for chapter in metadata.chapters:
        ch_start = float(chapter.get("start_time", 0.0))
        ch_end = float(chapter.get("end_time", metadata.duration))
        if ch_start <= midpoint <= ch_end:
            ch_title = chapter.get("title", "").strip()
            if ch_title:
                return f"{ch_title} (Part {clip_index})"

    # Fallback to video title - Part N
    clean_title = metadata.title.strip()
    return f"{clean_title} - Clip {clip_index}"


def generate_moment_reason(
    breakdown: SignalBreakdown,
    has_heatmap: bool,
    is_global_peak: bool = False,
    has_strong_hook: bool = False,
    has_climax_burst: bool = False,
) -> str:
    """Generate concise human-readable explanation with virality indicators."""
    reasons = []

    if is_global_peak:
        reasons.append("🔥 #1 Global Replay Peak (Most Watched Moment)")
    elif has_heatmap and breakdown.heatmap >= 55.0:
        reasons.append(f"Replay peak ({int(breakdown.heatmap)}%)")

    if has_strong_hook:
        reasons.append("⚡ High Hook Energy in first 3s (Low Swipe-Away Risk)")

    if has_climax_burst:
        reasons.append("💥 Explosive Audio Climax / Reaction Burst")
    elif breakdown.audio >= 50.0:
        reasons.append("Loud audio energy / emphatic speech")
    elif breakdown.audio >= 35.0:
        reasons.append("Dynamic audio bursts")

    if breakdown.visual >= 50.0:
        reasons.append("Fast scene cuts & motion")
    elif breakdown.visual >= 35.0:
        reasons.append("Active visual motion")

    if breakdown.pacing >= 60.0:
        reasons.append("Continuous tight pacing")

    if not reasons:
        reasons.append("Balanced engagement signals")

    return " + ".join(reasons[:3])


def detect_moments(
    metadata: VideoMetadata,
    signals: VideoSignals,
    num_clips: int = 5,
    min_length: float = 20.0,
    max_length: float = 60.0,
    config_weights: Optional[Dict[str, float]] = None,
) -> List[MomentCandidate]:
    """Detect top engaging moments across the video.

    Uses sliding window across signals, boundary snapping to silences/scene cuts,
    and Non-Maximum Suppression (NMS) to eliminate overlaps.
    """
    cfg = get_config()
    clip_cfg = cfg.get("clips", {})
    total_dur = metadata.duration
    total_sec = signals.total_seconds

    # Adjust min/max length if video is shorter than default bounds
    actual_min = min(min_length, max(5.0, total_dur * 0.3))
    actual_max = min(max_length, total_dur)
    if actual_min > actual_max:
        actual_min = actual_max * 0.5

    step = float(clip_cfg.get("window_step", 2.0))
    snap_win = float(clip_cfg.get("boundary_snap_window", 2.5))
    padding = float(clip_cfg.get("boundary_padding", 0.3))
    iou_thresh = float(clip_cfg.get("nms_iou_threshold", 0.25))

    weights = compute_effective_weights(signals.has_heatmap, config_weights)
    logger.info("Scoring moments with weights: %s", weights)

    heatmap_arr = np.array(signals.heatmap, dtype=np.float32)
    audio_arr = np.array(signals.audio_energy, dtype=np.float32)
    visual_arr = np.array(signals.visual_motion, dtype=np.float32)
    pacing_arr = np.array(signals.pacing, dtype=np.float32)

    # Generate multi-scale window sizes
    window_sizes = []
    w = actual_min
    while w <= actual_max:
        window_sizes.append(w)
        w += max(10.0, (actual_max - actual_min) / 4.0)
    if actual_max not in window_sizes:
        window_sizes.append(actual_max)

    candidates: List[Dict[str, Any]] = []

    for win_size in window_sizes:
        curr_start = 0.0
        while curr_start + win_size <= total_dur:
            raw_start = curr_start
            raw_end = curr_start + win_size

            # Snap boundaries
            s_start, s_end = snap_boundaries(
                raw_start,
                raw_end,
                total_duration=total_dur,
                scene_cuts=signals.scene_cuts,
                silences=signals.silences,
                snap_window=snap_win,
                padding=padding,
                min_length=actual_min,
                max_length=actual_max,
            )

            # Extract window signal slice
            i0 = max(0, int(math.floor(s_start)))
            i1 = min(total_sec, int(math.ceil(s_end)))
            if i1 <= i0:
                curr_start += step
                continue

            h_val = float(np.mean(heatmap_arr[i0:i1])) if weights["heatmap"] > 0 else 0.0
            a_val = float(np.mean(audio_arr[i0:i1]))
            v_val = float(np.mean(visual_arr[i0:i1]))
            p_val = float(np.mean(pacing_arr[i0:i1]))

            # Base engagement score (0 - 100)
            raw_base_score = (
                weights["heatmap"] * h_val
                + weights["audio"] * a_val
                + weights["visual"] * v_val
                + weights["pacing"] * p_val
            ) * 100.0

            # ---------------- VIRALITY OPTIMIZATION ENGINE ----------------
            # 1. Hook Score (First 3 seconds energy)
            # Clips that start with high energy prevent users from swiping away on Shorts/TikTok
            hook_end = min(i1, i0 + 3)
            hook_audio = float(np.max(audio_arr[i0:hook_end])) if hook_end > i0 else a_val
            hook_visual = float(np.mean(visual_arr[i0:hook_end])) if hook_end > i0 else v_val
            has_strong_hook = (hook_audio >= 0.55 or hook_visual >= 0.45)
            hook_multiplier = 1.0 + (0.25 * hook_audio + 0.15 * hook_visual)

            # 2. Global Replay Peak Bonus
            # If the candidate contains the #1 global rewatch peak across the video, boost significantly
            is_global_peak = False
            peak_multiplier = 1.0
            if signals.has_heatmap and len(heatmap_arr) > 0:
                global_max_heat = float(np.max(heatmap_arr))
                clip_max_heat = float(np.max(heatmap_arr[i0:i1]))
                if global_max_heat > 0.1 and clip_max_heat >= (0.90 * global_max_heat):
                    is_global_peak = True
                    peak_multiplier = 1.35  # 35% virality bonus for most replayed moment

            # 3. Climax / Punchline Dynamic Burst
            # Detect dramatic peaks in audio (screaming, laughter, punchline, shocking revelation)
            clip_audio_peak = float(np.max(audio_arr[i0:i1]))
            clip_audio_mean = a_val
            dynamic_burst = clip_audio_peak - clip_audio_mean
            has_climax_burst = (dynamic_burst >= 0.30 and clip_audio_peak >= 0.60)
            climax_multiplier = 1.20 if has_climax_burst else 1.0

            # 4. Pacing Velocity Bonus
            velocity_multiplier = 1.10 if (v_val >= 0.40 and p_val >= 0.70) else 1.0

            # Calculate composite Virality Score (0 - 100)
            composite_score = raw_base_score * hook_multiplier * peak_multiplier * climax_multiplier * velocity_multiplier
            viral_score = round(float(np.clip(composite_score, 0.0, 99.8)), 1)

            breakdown = SignalBreakdown(
                heatmap=round(h_val * 100.0, 1),
                audio=round(a_val * 100.0, 1),
                visual=round(v_val * 100.0, 1),
                pacing=round(p_val * 100.0, 1),
            )

            candidates.append({
                "start": s_start,
                "end": s_end,
                "duration": round(s_end - s_start, 2),
                "score": viral_score,
                "breakdown": breakdown,
                "is_global_peak": is_global_peak,
                "has_strong_hook": has_strong_hook,
                "has_climax_burst": has_climax_burst,
            })

            curr_start += step

    if not candidates:
        logger.warning("No candidates generated; falling back to full video slice")
        candidates.append({
            "start": 0.0,
            "end": total_dur,
            "duration": total_dur,
            "score": 50.0,
            "breakdown": SignalBreakdown(50.0, 50.0, 50.0, 50.0),
        })

    # Sort descending by score
    candidates.sort(key=lambda c: c["score"], reverse=True)

    # Non-Maximum Suppression (NMS)
    selected: List[Dict[str, Any]] = []
    for cand in candidates:
        overlap = False
        for sel in selected:
            iou = compute_iou(cand["start"], cand["end"], sel["start"], sel["end"])
            if iou > iou_thresh:
                overlap = True
                break
            # Also prevent excessive absolute overlap (e.g. > 4 seconds)
            inter_dur = max(0.0, min(cand["end"], sel["end"]) - max(cand["start"], sel["start"]))
            if inter_dur > 4.0:
                overlap = True
                break

        if not overlap:
            selected.append(cand)
            if len(selected) >= num_clips:
                break

    # Build final MomentCandidate objects with index, title, reason, and viral flags
    results: List[MomentCandidate] = []
    for i, sel in enumerate(selected, start=1):
        title = generate_moment_title(i, sel["start"], sel["end"], metadata)
        reason = generate_moment_reason(
            sel["breakdown"],
            signals.has_heatmap,
            is_global_peak=sel.get("is_global_peak", False),
            has_strong_hook=sel.get("has_strong_hook", False),
            has_climax_burst=sel.get("has_climax_burst", False),
        )
        is_top = (i == 1)
        results.append(
            MomentCandidate(
                clip_index=i,
                start_sec=sel["start"],
                end_sec=sel["end"],
                duration=sel["duration"],
                score=sel["score"],
                breakdown=sel["breakdown"],
                title=title,
                reason=reason,
                is_top_viral=is_top,
            )
        )

    logger.info("Moment detection complete: %d clips selected", len(results))
    for r in results:
        logger.info(
            "Clip %d [%.1fs - %.1fs, dur %.1fs, score %.1f]: %s | %s",
            r.clip_index, r.start_sec, r.end_sec, r.duration, r.score, r.title, r.reason
        )

    return results

