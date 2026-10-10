from datetime import UTC, datetime

from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_name: str
    class_id: int
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    track_id: int | None = None
    speed_pixels_per_second: float | None = None
    movement: str | None = None


class TextRegion(BaseModel):
    text: str
    confidence: float = Field(ge=0, le=1)
    polygon: list[tuple[float, float]]


class Person(BaseModel):
    bbox: tuple[float, float, float, float]
    keypoints: list[list[float]] = Field(default_factory=list)
    activity: str = "unknown"
    evidence: str | None = None
    expression: str = "unknown"


class PerceptionResult(BaseModel):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source: str
    frame_id: int = 0
    objects: list[Detection] = Field(default_factory=list)
    text_regions: list[TextRegion] = Field(default_factory=list)
    people: list[Person] = Field(default_factory=list)
    faces: list[dict] = Field(default_factory=list)
    scene: dict = Field(
        default_factory=lambda: {
            "road_visible": None,
            "sidewalk_visible": None,
            "candidate_walkable_fraction": None,
        }
    )
    modules: dict[str, str] = Field(default_factory=dict)
    models: dict[str, str] = Field(default_factory=dict)
    processing_ms: float = 0
    events: dict[str, list[int]] = Field(default_factory=dict)
