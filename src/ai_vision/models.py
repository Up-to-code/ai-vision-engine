import logging
from pathlib import Path
from time import perf_counter

from .config import select_device


class ModelManager:
    def __init__(self, device: str):
        self.device = select_device(device)
        self.loaded = {}
        self.modes = {}
        self.initialization_ms = {}

    def yolo(self, name: str):
        if name not in self.loaded:
            from ultralytics import YOLO

            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            start = perf_counter()
            # YOLO downloads recognized official filenames; unknown paths raise errors.
            self.loaded[name] = YOLO(str(path))
            self.initialization_ms[name] = (perf_counter() - start) * 1000
        return self.loaded[name]

    def predict(self, name: str, image, tracking: bool = False, **kwargs):
        model = self.yolo(name)
        if name in self.modes and self.modes[name] != tracking:
            model.reset_callbacks()
            model.predictor = None
        self.modes[name] = tracking
        fn = model.track if tracking else model.predict
        extra = {"persist": True, "tracker": "bytetrack.yaml"} if tracking else {}
        try:
            return fn(image, device=self.device, verbose=False, **extra, **kwargs)[0]
        except (RuntimeError, NotImplementedError) as exc:
            if self.device == "mps" and any(
                s in str(exc).lower() for s in ("mps", "not implemented")
            ):
                logging.warning("MPS operator failed; retrying on CPU: %s", exc)
                self.device = "cpu"
                model.to("cpu")
                return fn(image, device="cpu", verbose=False, **extra, **kwargs)[0]
            raise
