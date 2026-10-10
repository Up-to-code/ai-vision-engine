"""Animated presentation of independent detections, never cross-image tracking."""

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def make_film(analyses, output, seconds=50, fps=30):
    if seconds <= 0 or fps <= 0:
        raise ValueError("Duration and FPS must be positive")
    folders = sorted(p.parent for p in Path(analyses).glob("*/original/result.json"))
    if not folders:
        raise ValueError("No analyzed originals found")
    width, height = 1280, 720
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = shutil.which("ffmpeg")
    if not encoder:
        raise RuntimeError("Install ffmpeg to export an H.264 video")
    process = subprocess.Popen(
        [
            encoder,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    total = round(seconds * fps)
    cached_index = -1
    try:
        for frame in range(total):
            position = frame * len(folders) / total
            index = min(int(position), len(folders) - 1)
            phase = position - index
            if index != cached_index:
                folder = folders[index]
                original = cv2.imread(str(folder / "original.jpg"))
                annotation = cv2.imread(str(folder / "combined.jpg"))
                result = json.loads((folder / "result.json").read_text())
                if original is None or annotation is None:
                    raise ValueError(f"Missing images at {folder}")
                cached_index = index
            # Subtle image motion is presentation animation, not estimated object motion.
            blend = min(1.0, phase * 5)
            view = cv2.addWeighted(original, 1 - blend, annotation, blend, 0)
            scale = min(1216 / view.shape[1], 530 / view.shape[0]) * (1 + phase * 0.035)
            resized = cv2.resize(view, None, fx=scale, fy=scale)
            canvas = np.full((height, width, 3), (24, 19, 14), dtype=np.uint8)
            h, w = resized.shape[:2]
            x, y = (width - w) // 2, 112 + (530 - h) // 2
            # Current dimensions are capped below the canvas even with zoom.
            canvas[y : y + h, x : x + w] = resized
            cv2.putText(
                canvas,
                "AI VISION / STREET SCENE DETECTION",
                (32, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"{index + 1:03d} / {len(folders):03d}    {len(result['objects'])} objects    {result['processing_ms']:.0f} ms",
                (32, 82),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (80, 215, 245),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Independent still images | Animated overlays | No cross-image tracking",
                (32, 680),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.53,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.rectangle(
                canvas, (32, 704), (32 + int(1216 * (frame + 1) / total), 709), (90, 210, 80), -1
            )
            process.stdin.write(canvas.tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f"ffmpeg failed with exit {code}")
    return output
