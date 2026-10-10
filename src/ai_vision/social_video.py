"""Portrait social cut with full-frame backgrounds and original model predictions."""

from pathlib import Path
import json
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def cover(image, width, height):
    scale = max(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale)
    y, x = (resized.shape[0] - height) // 2, (resized.shape[1] - width) // 2
    return resized[y : y + height, x : x + width]


def render_social(analyses, output, seconds=30, fps=30):
    folders = sorted(p.parent for p in Path(analyses).glob("*/original/result.json"))
    if not folders:
        raise ValueError("No analyzed images")
    width, height = 720, 1280
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
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
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        stdin=subprocess.PIPE,
    )
    total = round(seconds * fps)
    last = -1
    font = ImageFont.load_default(size=48)
    small = ImageFont.load_default(size=24)
    medium = ImageFont.load_default(size=31)
    try:
        for frame in range(total):
            position = frame * len(folders) / total
            index = min(int(position), len(folders) - 1)
            phase = position - index
            if index != last:
                original = cv2.imread(str(folders[index] / "original.jpg"))
                annotation = cv2.imread(str(folders[index] / "combined.jpg"))
                if original is None or annotation is None:
                    raise ValueError(f"Missing image in {folders[index]}")
                result = json.loads((folders[index] / "result.json").read_text())
                background = cover(original, width, height)
                background = cv2.GaussianBlur(background, (0, 0), 24)
                background = (background.astype(np.float32) * 0.48).astype(np.uint8)
                last = index
            blend = min(1, phase * 4)
            image = cv2.addWeighted(original, 1 - blend, annotation, blend, 0)
            scale = min(width / image.shape[1], 780 / image.shape[0])
            image = cv2.resize(image, None, fx=scale, fy=scale)
            h, w = image.shape[:2]
            canvas = background.copy()
            x, y = (width - w) // 2, 300 + (780 - h) // 2
            canvas[y : y + h, x : x + w] = image
            pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil)
            title = "100 IMAGES.\nONE AI." if frame < 90 else "AI VISION"
            draw.multiline_text(
                (32, 145),
                title,
                font=font,
                fill="white",
                spacing=4,
                stroke_width=2,
                stroke_fill="black",
            )
            draw.text(
                (32, 265),
                f"{index + 1:03d}/100  |  {len(result['objects'])} detections",
                font=small,
                fill="#63f7df",
                stroke_width=1,
                stroke_fill="black",
            )
            calls = [
                "Real model predictions",
                "Blur, noise & low light tested",
                "People. Objects. Text. Depth.",
                "No camera used",
                "Open source on GitHub",
            ]
            draw.text(
                (32, 1090),
                calls[min(4, int(frame / fps) // 6)],
                font=medium,
                fill="white",
                stroke_width=2,
                stroke_fill="black",
            )
            draw.text((32, 1145), "Up-to-code / ai-vision-engine", font=small, fill="#63f7df")
            draw.rectangle((32, 1200, 32 + int(590 * (frame + 1) / total), 1206), fill="#63f7df")
            encoder.stdin.write(cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR).tobytes())
    finally:
        encoder.stdin.close()
        code = encoder.wait()
    if code:
        raise RuntimeError(f"FFmpeg exited {code}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/full-perception")
    parser.add_argument("--output", default="outputs/ai-vision-tiktok-30s.mp4")
    args = parser.parse_args()
    print(render_social(args.source, args.output))
