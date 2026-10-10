"""Typed local perception API. All optional models initialize lazily."""

from .config import VisionConfig
from .engine import VisionEngine

__all__ = ["VisionConfig", "VisionEngine"]
