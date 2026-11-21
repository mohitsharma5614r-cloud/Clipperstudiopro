"""Generate a synthetic test MP4 video with varying motion, color bars, and audio tones."""
import math
import struct
import subprocess
import wave
from pathlib import Path
import cv2
import numpy as np
from clipper.utils import get_ffmpeg_path

def generate_test_media(output_path: str = "tests/sample_video.mp4", duration: float = 30.0, fps: int = 25):
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_video = str(p.with_suffix(".tmp.mp4"))
    temp_audio = str(p.with_suffix(".tmp.wav"))

    width, height = 640, 360
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, float(fps), (width, height))

    total_frames = int(duration * fps)
    for i in range(total_frames):
        t = i / fps
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Background gradient that shifts with time
        col = int((math.sin(t * 1.5) + 1.0) * 100) # 0 to 200
        val_b = max(0, min(255, 200 - col))
        frame[:, :] = (col, 60, val_b)

        # Draw moving circle/rectangle simulating motion and face-like shape
        center_x = int(width / 2 + math.sin(t * 2.0) * (width / 4))
        center_y = int(height / 2 + math.cos(t * 1.5) * 40)
        
        # Face-like oval
        cv2.ellipse(frame, (center_x, center_y), (40, 55), 0, 0, 360, (200, 220, 240), -1)
        # Eyes
        cv2.circle(frame, (center_x - 15, center_y - 10), 5, (50, 50, 50), -1)
        cv2.circle(frame, (center_x + 15, center_y - 10), 5, (50, 50, 50), -1)
        # Mouth
        cv2.ellipse(frame, (center_x, center_y + 20), (15, 8), 0, 0, 180, (50, 50, 200), -1)

        # Scene cut every 8 seconds (flash white for 2 frames)
        if (i % (8 * fps)) < 2:
            frame[:, :] = 255

        writer.write(frame)
    writer.release()

    # Generate audio with silence zones and loud bursts
    sample_rate = 16000
    total_samples = int(duration * sample_rate)
    with wave.open(temp_audio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        
        audio_data = []
        for s in range(total_samples):
            st = s / sample_rate
            # Silence between 5s and 9s, and 18s and 22s
            if (5.0 <= st < 9.0) or (18.0 <= st < 21.0):
                val = 0
            elif (10.0 <= st < 15.0) or (24.0 <= st < 28.0):
                # Loud bursts (applause/speech-like frequencies)
                val = int(math.sin(2 * math.pi * 440 * st) * 20000 + math.sin(2 * math.pi * 880 * st) * 10000)
            else:
                # Moderate tone
                val = int(math.sin(2 * math.pi * 300 * st) * 4000)
            val = max(-32767, min(32767, val))
            audio_data.append(struct.pack("<h", val))
        wf.writeframes(b"".join(audio_data))

    # Mux video and audio with ffmpeg
    ffmpeg = get_ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", temp_video,
        "-i", temp_audio,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p",
        str(p)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Cleanup temp
    Path(temp_video).unlink(missing_ok=True)
    Path(temp_audio).unlink(missing_ok=True)
    print(f"Generated test media: {p} ({duration}s)")

if __name__ == "__main__":
    generate_test_media()
