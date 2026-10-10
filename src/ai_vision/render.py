from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def overlay(image, result, *, objects=True, text=True, pose=True):
    output = image.copy()
    if objects:
        for obj in result.objects:
            x1, y1, x2, y2 = map(int, obj.bbox)
            color = (80, 210, 90) if obj.class_name == "person" else (245, 150, 65)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{obj.class_name} {obj.confidence:.2f}"
            if obj.track_id is not None:
                label += f" #{obj.track_id}"
            cv2.putText(
                output,
                label,
                (max(0, x1), max(16, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    if pose:
        for person in result.people:
            for joint in person.keypoints:
                if joint[2] > 0.5:
                    cv2.circle(output, (int(joint[0]), int(joint[1])), 3, (220, 80, 220), -1)
    if text:
        canvas = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", 14)
        except OSError:
            font = ImageFont.load_default()
        for region in result.text_regions:
            points = [tuple(map(int, p)) for p in region.polygon]
            draw.line(points + [points[0]], fill="yellow", width=2)
            draw.text(points[0], region.text, fill="yellow", font=font)
        output = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    return output


def save_analysis(image, result, artifacts, destination):
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    views = {
        "original": image,
        "objects": overlay(image, result, text=False, pose=False),
        "ocr": overlay(image, result, objects=False, pose=False),
        "pose": overlay(image, result, objects=False, text=False),
        "combined": overlay(artifacts.get("segmentation", image), result),
    }
    views.update(artifacts)
    for name, view in views.items():
        if not cv2.imwrite(str(folder / f"{name}.jpg"), view):
            raise RuntimeError(f"Could not save {name}")
    (folder / "result.json").write_text(result.model_dump_json(indent=2))
    # Disabled modules are recorded in JSON; no fabricated depth/segmentation images.
