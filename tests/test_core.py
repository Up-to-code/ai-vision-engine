import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from pydantic import ValidationError
from ai_vision import VisionConfig, VisionEngine
from ai_vision.io import load_image, capture, degradations
from ai_vision.schemas import Detection, PerceptionResult
from ai_vision.tracking import MotionHistory
from ai_vision.render import save_analysis


def detection(x=10, track_id=1):
    return Detection(
        class_name="person",
        class_id=0,
        confidence=0.9,
        bbox=(x, 0, x + 20, 40),
        center=(x + 10, 20),
        track_id=track_id,
    )


def test_schema_and_defaults():
    result = PerceptionResult(source="test", objects=[detection()])
    restored = PerceptionResult.model_validate_json(result.model_dump_json())
    assert restored.scene["road_visible"] is None
    assert restored.objects[0].speed_pixels_per_second is None
    with pytest.raises(ValidationError):
        VisionConfig(confidence=1.1)


def test_motion_temporal_evidence():
    history = MotionHistory()
    first = detection()
    assert history.update([first], 0)["newly_visible"] == [1]
    assert first.movement is None
    second = detection(30)
    history.update([second], 0.5)
    assert second.speed_pixels_per_second == 40
    assert second.movement == "left_to_right"
    assert history.update([], 1)["no_longer_visible"] == [1]
    assert history.previous == {}


def test_image_decode_and_degradations(tmp_path):
    with pytest.raises(ValueError):
        load_image(tmp_path / "missing.jpg")
    (tmp_path / "bad.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_image(tmp_path / "bad.jpg")
    image = np.full((32, 40, 3), 100, np.uint8)
    original = image.copy()
    versions = degradations(image)
    assert len(versions) == 6
    assert all(v.shape == image.shape for v in versions.values())
    np.testing.assert_array_equal(original, image)
    assert versions["dark"].mean() < image.mean()


def test_capture_releases_on_error():
    cap = MagicMock()
    cap.isOpened.return_value = False
    with patch("ai_vision.io.cv2.VideoCapture", return_value=cap):
        with pytest.raises(RuntimeError):
            with capture(0):
                pass
    cap.release.assert_called_once()


def test_pipeline_empty_predictions_and_artifacts(tmp_path):
    engine = VisionEngine(VisionConfig(device="cpu"))
    prediction = MagicMock()
    prediction.boxes = []
    with patch.object(engine.manager, "predict", return_value=prediction):
        result = engine.analyze_image(np.zeros((30, 40, 3), np.uint8))
    assert result.objects == []
    assert result.modules["depth"] == "disabled"
    save_analysis(np.zeros((30, 40, 3), np.uint8), result, {}, tmp_path)
    assert not (tmp_path / "depth.jpg").exists()
    assert json.loads((tmp_path / "result.json").read_text())["objects"] == []


def test_loading_failure_propagates():
    engine = VisionEngine(VisionConfig(device="cpu"))
    with patch.object(engine.manager, "predict", side_effect=FileNotFoundError("weights")):
        with pytest.raises(FileNotFoundError):
            engine.analyze_image(np.zeros((30, 40, 3), np.uint8))


def test_ocr_filters_uncertain_text():
    engine = VisionEngine(VisionConfig(device="cpu"))
    reader = MagicMock()
    reader.readtext.return_value = [
        ([(0, 0), (10, 0), (10, 10), (0, 10)], "30", 0.8),
        ([(0, 0)] * 4, "noise", 0.1),
    ]
    engine.optional.cache["ocr"] = reader
    assert [r.text for r in engine.optional.ocr(np.zeros((30, 40, 3), np.uint8))] == ["30"]


@pytest.mark.integration
def test_real_inference():
    from pathlib import Path

    images = sorted(Path("data/benchmarks/street_scenes/images").glob("*.jpg"))
    if not images:
        pytest.skip("Run ai-vision collect before integration test")
    engine = VisionEngine(VisionConfig(device="cpu"))
    result = engine.analyze_image(images[0])
    assert result.models["detector"].endswith("yolo26n.pt")
    assert result.processing_ms > 0
    assert all(o.track_id is None and o.movement is None for o in result.objects)
    PerceptionResult.model_validate_json(result.model_dump_json())


def test_capture_releases_when_consumer_fails():
    cap = MagicMock()
    cap.isOpened.return_value = True
    with patch("ai_vision.io.cv2.VideoCapture", return_value=cap):
        with pytest.raises(ValueError):
            with capture("fixture.mp4"):
                raise ValueError("consumer failure")
    cap.release.assert_called_once()


def test_requested_optional_failure_propagates():
    engine = VisionEngine(VisionConfig(device="cpu", depth=True))
    prediction = MagicMock()
    prediction.boxes = []
    with (
        patch.object(engine.manager, "predict", return_value=prediction),
        patch.object(engine.optional, "depth", side_effect=RuntimeError("model unavailable")),
    ):
        with pytest.raises(RuntimeError, match="model unavailable"):
            engine.analyze_image(np.zeros((30, 40, 3), np.uint8))


def test_mps_retry_is_limited_to_operator_errors():
    from ai_vision.models import ModelManager

    manager = ModelManager("mps")
    model = MagicMock()
    model.predict.side_effect = [RuntimeError("MPS operator not implemented"), ["result"]]
    with patch.object(manager, "yolo", return_value=model):
        assert manager.predict("test", np.zeros((30, 40, 3), np.uint8)) == "result"
    assert manager.device == "cpu"
    assert model.predict.call_count == 2


def test_film_requires_analyzed_images(tmp_path):
    from ai_vision.film import make_film

    with pytest.raises(ValueError, match="No analyzed"):
        make_film(tmp_path, tmp_path / "film.mp4")
