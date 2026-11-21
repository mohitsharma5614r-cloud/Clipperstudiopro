"""CLI interface for AI Clipping Tool.

Usage:
  python -m clipper "https://www.youtube.com/watch?v=..." --clips 5 --aspect 9:16
  python -m clipper --serve --port 8000
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from clipper.analyze import detect_moments
from clipper.download import ingest_video
from clipper.render import render_all_clips
from clipper.signals import compute_video_signals
from clipper.utils import ensure_dir, format_timestamp, get_config, logger


def run_cli_pipeline(
    url_or_path: str,
    num_clips: int = 5,
    aspect_ratio: str = "9:16",
    min_length: float = 20.0,
    max_length: float = 60.0,
    output_dir: str = "./output",
):
    """Execute clipping pipeline via CLI with terminal progress updates."""
    print("=" * 65)
    print("🎬  AI CLIPPING TOOL (Local & Free Moment Detection)")
    print("=" * 65)
    print(f"• Input:        {url_or_path}")
    print(f"• Target Clips: {num_clips}")
    print(f"• Aspect Ratio: {aspect_ratio}")
    print(f"• Length Range: {min_length}s - {max_length}s")
    print(f"• Output Dir:   {output_dir}")
    print("-" * 65)

    # 1. Ingestion
    print("\n[1/4] 📥 Ingesting video & extracting metadata...")
    ingestion = ingest_video(url_or_path)
    meta = ingestion.metadata
    print(f"      ✓ Title:    {meta.title}")
    print(f"      ✓ Channel:  {meta.channel}")
    print(f"      ✓ Duration: {format_timestamp(meta.duration)} ({meta.duration:.1f}s)")
    print(f"      ✓ Heatmap:  {'Available' if meta.has_heatmap else 'Not available (reweighting signals)'}")

    # 2. Signal Extraction
    print("\n[2/4] 📊 Computing time-series signals (audio energy, visual activity, pacing)...")
    signals = compute_video_signals(meta)
    print(f"      ✓ Signals computed for {signals.total_seconds} seconds")
    print(f"      ✓ Detected {len(signals.scene_cuts)} scene cuts and {len(signals.silences)} audio pauses")

    # 3. Moment Detection
    print("\n[3/4] 🧠 Analyzing and ranking highest-engagement moments...")
    moments = detect_moments(
        metadata=meta,
        signals=signals,
        num_clips=num_clips,
        min_length=min_length,
        max_length=max_length,
    )

    if not moments:
        print("❌ Error: No candidate moments detected.")
        sys.exit(1)

    print(f"      ✓ Selected top {len(moments)} moments using Non-Max Suppression (NMS)")

    # 4. Rendering
    print(f"\n[4/4] 🎥 Rendering {len(moments)} clips in {aspect_ratio} with loudness normalization...")
    job_output = ensure_dir(output_dir)

    def on_progress(idx, total):
        print(f"      → Rendering clip {idx}/{total}...")

    rendered = render_all_clips(
        source_video_path=ingestion.video_path,
        candidates=moments,
        output_dir=job_output,
        aspect_ratio=aspect_ratio,
        progress_callback=on_progress,
    )

    # Copy metadata JSON alongside
    shutil.copy2(ingestion.metadata_path, job_output / "video_metadata.json")

    print("\n" + "=" * 65)
    print("🎉  CLIPS GENERATION COMPLETE!")
    print("=" * 65)
    print(f"{'#':<3} {'Score':<7} {'Time':<15} {'Dur':<6} {'Title':<25}")
    print("-" * 65)
    for rc in rendered:
        t_range = f"{format_timestamp(rc.start_sec)}-{format_timestamp(rc.end_sec)}"
        print(f"{rc.clip_index:<3} {rc.score:<7.1f} {t_range:<15} {rc.duration:<6.1f}s {rc.title[:24]:<25}")
        print(f"    ↳ Reason: {rc.reason}")
        print(f"    ↳ File:   {rc.video_path}")
    print("=" * 65)


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI web server."""
    import uvicorn
    print(f"🚀 Starting AI Clipping Tool Web UI on http://localhost:{port}")
    uvicorn.run("clipper.api:app", host=host, port=port, reload=reload)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="clipper",
        description="Free, local AI Clipping Tool (OpusClip alternative without transcription)",
    )
    parser.add_argument("url", nargs="?", default=None, help="Video URL (YouTube) or local video file path")
    parser.add_argument("--clips", type=int, default=5, help="Number of clips to generate (default: 5)")
    parser.add_argument("--aspect", choices=["9:16", "16:9", "1:1"], default="9:16", help="Target aspect ratio (default: 9:16)")
    parser.add_argument("--min-len", type=float, default=20.0, help="Minimum clip duration in seconds (default: 20)")
    parser.add_argument("--max-len", type=float, default=60.0, help="Maximum clip duration in seconds (default: 60)")
    parser.add_argument("--output", default="./output", help="Output directory for clips (default: ./output)")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI web server")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Web server port (default: 8000)")

    args = parser.parse_args()

    if args.serve:
        start_server(host=args.host, port=args.port)
    elif args.url:
        run_cli_pipeline(
            url_or_path=args.url,
            num_clips=args.clips,
            aspect_ratio=args.aspect,
            min_length=args.min_len,
            max_length=args.max_len,
            output_dir=args.output,
        )
    else:
        parser.print_help()
        print("\n💡 Tip: Run 'python -m clipper --serve' to open the Web UI, or provide a video URL.")


if __name__ == "__main__":
    main()

