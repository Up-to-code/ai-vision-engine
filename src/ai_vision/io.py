from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


def load_image(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        image = source.copy()
    else:
        image = cv2.imread(str(source))
    if image is None or image.size == 0 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a nonempty 3-channel BGR image: {source!s}")
    if image.dtype != np.uint8:
        raise ValueError("Image must be uint8")
    return image


@contextmanager
def capture(source: str | int):
    cap = cv2.VideoCapture(source)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source {source!r}")
        yield cap
    finally:
        cap.release()


def degradations(image: np.ndarray, seed: int = 42) -> dict[str, np.ndarray]:
    h, w = image.shape[:2]
    kernel = np.zeros((17, 17), dtype=np.float32)
    kernel[8] = 1 / 17
    ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 12])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    rng = np.random.default_rng(seed)
    return {
        "gaussian": cv2.GaussianBlur(image, (15, 15), 4),
        "motion": cv2.filter2D(image, -1, kernel),
        "low_resolution": cv2.resize(cv2.resize(image, (max(1, w // 4), max(1, h // 4))), (w, h)),
        "dark": (image.astype(np.float32) * 0.3).astype(np.uint8),
        "jpeg": cv2.imdecode(jpeg, cv2.IMREAD_COLOR),
        "noise": np.clip(image.astype(float) + rng.normal(0, 22, image.shape), 0, 255).astype(
            np.uint8
        ),
    }
