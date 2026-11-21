# Local AI Video Clipping Tool

An open-source, local video clipping tool designed to identify high-engagement segments from long-form video and export vertical shorts (9:16), widescreen (16:9), or square (1:1) clips.

Operating without speech-to-text or transcription APIs, moment detection relies entirely on multimodal signals:
1. **Replay Heatmap**: Audience rewatch density curve.
2. **Audio Dynamics**: Loudness bursts, spectral flux, cheering, laughter, and dialogue energy.
3. **Visual Dynamics**: Scene cut frequency and frame-to-frame pixel motion.
4. **Pacing & Cadence**: Dialogue density and silence penalty scoring.

---

## Architecture Overview

```
+-------------------------------------------------------------------------+
|                              Input Source                               |
|        (YouTube URL, Supported Streaming Sites, or Local MP4/MKV)       |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                          Phase 1: Ingestion                             |
|  - yt-dlp stream & metadata retrieval (chapters, replay heatmap)        |
|  - 16 kHz mono 16-bit PCM audio extraction                              |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                      Phase 2: Signal Extraction                         |
|  [Replay Heatmap]       [Audio Spectral Flux]       [Visual Motion]     |
|   Audience peaks         RMS & sound spikes          Scene cuts & diff  |
|          \                       |                       /              |
|           +----------------------+----------------------+               |
|                                  |                                      |
|                       [Pacing & Silence Filter]                         |
|                    Penalize dead zones & static                         |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                      Phase 3: Moment Detection                          |
|  - Multi-scale sliding window candidate generation (15s - 60s)          |
|  - Dynamic signal weighting with automatic heatmap missing fallback     |
|  - Boundary snapping to natural silence pauses and scene cuts           |
|  - Temporal Non-Maximum Suppression (NMS) overlap reduction             |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                       Phase 4: Clip Rendering                           |
|  - Frame-accurate FFmpeg cuts                                           |
|  - Face detection & smooth subject centering (Haar Cascade / OpenCV)    |
|  - Blurred background fallback for landscape footage without faces      |
|  - -14 LUFS loudness normalization & optional edge fades                |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                       Phase 5: User Interfaces                          |
|       [FastAPI REST API]    [Modern Web Interface]    [Python CLI]      |
+-------------------------------------------------------------------------+
```

---

## Key Features

- **100% Local & Free**: Runs on standard consumer laptops with CPU. No external cloud endpoints or API tokens required.
- **No Speech-to-Text Bottleneck**: Instantaneous processing without Whisper or transcription models.
- **Smart Aspect Ratio Framing**:
  - `9:16` Vertical: Subject tracking centers speaker; falls back to blurred video background when no face is present.
  - `16:9` Landscape: Preserves original dimensions.
  - `1:1` Square: Centered square crop for social feeds.
- **Audio Mastering**: Broadcast-standard `-14 LUFS` loudness normalization with smooth fade-in and fade-out.
- **Caching**: Signal data is cached per job in `./workspace/<job_id>/signals.json` to make adjustments instantaneous.
- **REST API + Web UI + CLI**: Full coverage for automation, headless pipelines, and browser workflows.

---

## Installation

### Prerequisites

- Python 3.10+
- FFmpeg (bundled automatically via `imageio-ffmpeg` or installed via system package manager)

#### System FFmpeg Setup (Optional)
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt update && sudo apt install -y ffmpeg`
- **Windows**: `winget install Gyan.FFmpeg` or `choco install ffmpeg`

### Python Setup

```bash
# Clone the repository
git clone https://github.com/example/ai-clipper.git
cd ai-clipper

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Web Application

Launch the FastAPI application:

```bash
python -m clipper --serve --port 8000
```

Open `http://localhost:8000` in your browser. You can paste a video URL, select aspect ratios and clip counts, observe live stage progress, view the clips in an integrated player, and download them individually or as a `.zip` archive.

### 2. Command Line Interface (CLI)

Generate clips directly from the terminal:

```bash
# Vertical 9:16 clips from a video
python -m clipper "https://www.youtube.com/watch?v=EXAMPLE" --clips 5 --aspect 9:16

# Process a local video file
python -m clipper "./tests/sample_video.mp4" --clips 3 --aspect 9:16 --output ./my_clips
```

CLI Parameters:
- `url`: Video URL or local media path.
- `--clips`: Target number of moments (default: `5`).
- `--aspect`: Output framing: `9:16`, `16:9`, or `1:1` (default: `9:16`).
- `--min-len`: Minimum clip duration in seconds (default: `20.0`).
- `--max-len`: Maximum clip duration in seconds (default: `60.0`).
- `--output`: Destination folder for rendered clips (default: `./output`).
- `--serve`: Starts the Web UI server.

### 3. Docker Deployment

```bash
docker compose up --build
```

Access the interface at `http://localhost:8000`.

---

## Configuration (`config.yaml`)

Signal weights, thresholds, and audio presets can be adjusted in `config.yaml`:

```yaml
weights:
  heatmap: 0.35    # Replay curve weight (automatically reweighted if missing)
  audio: 0.30      # Sound loudness and spectral flux
  visual: 0.20     # Frame difference and scene cuts
  pacing: 0.15     # Speech-to-silence ratio

clips:
  default_count: 5
  min_length: 20.0
  max_length: 60.0
  boundary_snap_window: 2.5
  boundary_padding: 0.3
  nms_iou_threshold: 0.25

audio:
  silence_thresh_db: -32.0
  target_lufs: -14.0

render:
  default_aspect: "9:16"
  face_tracking:
    fallback_mode: "blur"  # "blur" or "center"
```

---

## Testing

Run the automated test suite covering signal scoring, boundary snapping, and NMS:

```bash
pytest tests/
```

---

## Troubleshooting

- **Audio Extraction Warning**: If using macOS default Python with LibreSSL, a standard `urllib3` notice may appear. This does not impact clipping and is suppressed by default.
- **FFmpeg Executable**: The application automatically locates system FFmpeg or uses the bundled `imageio-ffmpeg` binary. Ensure permissions allow binary execution.
- **Videos Without Heatmaps**: For new uploads or non-YouTube sources where heatmaps are unavailable, the system automatically redistributes the weight across audio, visual, and pacing signals.

---

## Roadmap

- [ ] Optional speech recognition captions module (local offline model).
- [ ] Multi-speaker active face switching for podcasts.
- [ ] GPU-accelerated video decoding/encoding pipelines.

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss intended modifications.

---

## Content Rights Notice & Disclaimer

Users are responsible for ensuring they possess all necessary rights, licenses, or permissions for any third-party video content processed through this tool. All usage must comply with the Terms of Service of content hosts and local copyright legislation.

