from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class VisionConfig(BaseModel):
    model: str = "models/yolo26n.pt"
    device: str = "auto"
    confidence: float = Field(default=0.25, ge=0, le=1)
    image_size: int = Field(default=640, ge=32)
    ocr: bool = False
    ocr_languages: list[str] = Field(default_factory=lambda: ["en"])
    pose: bool = False
    face: bool = False
    depth: bool = False
    segmentation: bool = False
    face_model: str = "models/face_landmarker.task"

    @classmethod
    def load(cls, path: str | Path):
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


def select_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
