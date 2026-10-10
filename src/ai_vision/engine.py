import importlib.metadata
from pathlib import Path
from time import perf_counter

from .config import VisionConfig
from .io import load_image
from .models import ModelManager
from .optional import OptionalModules
from .schemas import Detection, PerceptionResult, Person
from .tracking import MotionHistory


class VisionEngine:
    """One engine per stream; image calls never reuse video tracking state."""

    def __init__(self, config: VisionConfig | None = None):
        self.config = config or VisionConfig()
        self.manager = ModelManager(self.config.device)
        self.optional = OptionalModules(self.config, self.manager)
        self.motion = MotionHistory()
        self.artifacts = {}

    def reset_tracking(self):
        self.motion = MotionHistory()
        model = self.manager.loaded.get(self.config.model)
        if model is not None and getattr(model, "predictor", None) is not None:
            for tracker in getattr(model.predictor, "trackers", []):
                tracker.reset()

    def analyze_image(self, image, source: str | None = None):
        self.reset_tracking()
        return self._analyze(
            image, source or (str(image) if isinstance(image, (str, Path)) else "image"), 0, None
        )

    def analyze_frame(
        self, frame, *, timestamp_seconds: float, frame_id: int = 0, source: str = "video"
    ):
        return self._analyze(frame, source, frame_id, timestamp_seconds)

    def _analyze(self, source_image, source, frame_id, timestamp_seconds):
        start = perf_counter()
        image = load_image(source_image)
        self.artifacts = {}
        prediction = self.manager.predict(
            self.config.model,
            image,
            tracking=timestamp_seconds is not None,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
        )
        result = PerceptionResult(
            source=source,
            frame_id=frame_id,
            models={
                "detector": self.config.model,
                "ultralytics": importlib.metadata.version("ultralytics"),
            },
            modules={"objects": "enabled"},
        )
        for box in prediction.boxes:
            bbox = tuple(box.xyxy[0].cpu().tolist())
            class_id = int(box.cls.item())
            result.objects.append(
                Detection(
                    class_name=prediction.names[class_id],
                    class_id=class_id,
                    confidence=float(box.conf.item()),
                    bbox=bbox,
                    center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                    track_id=int(box.id.item())
                    if timestamp_seconds is not None and box.id is not None
                    else None,
                )
            )
        if timestamp_seconds is not None:
            result.events = self.motion.update(result.objects, timestamp_seconds)
        result.people = [
            Person(bbox=obj.bbox) for obj in result.objects if obj.class_name == "person"
        ]
        for module in ("ocr", "pose", "face", "depth", "segmentation"):
            enabled = getattr(self.config, module)
            result.modules[module] = "enabled" if enabled else "disabled"
            if not enabled:
                continue
            # Requested initialization/inference failures propagate; never silently produce fake results.
            if module == "ocr":
                result.text_regions = self.optional.ocr(image)
                result.models[module] = "EasyOCR/" + ",".join(self.config.ocr_languages)
            elif module == "pose":
                result.people = self.optional.pose(image)
                result.models[module] = "yolo26n-pose.pt"
            elif module == "face":
                result.faces = self.optional.face(image)
                result.models[module] = self.config.face_model
            elif module == "depth":
                self.artifacts["depth"] = self.optional.depth(image)
                result.models[module] = "depth-anything/Depth-Anything-V2-Small-hf"
            elif module == "segmentation":
                self.artifacts[module], result.scene = self.optional.segmentation(
                    image, result.objects
                )
                result.models[module] = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
        result.processing_ms = (perf_counter() - start) * 1000
        return result
