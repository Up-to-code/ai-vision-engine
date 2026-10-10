import math

from .schemas import Detection


class MotionHistory:
    """Motion is image-plane displacement; camera movement can affect it."""

    def __init__(self):
        self.previous: dict[int, tuple[tuple[float, float], float]] = {}
        self.visible: set[int] = set()

    def update(self, objects: list[Detection], timestamp: float) -> dict[str, list[int]]:
        current = {o.track_id for o in objects if o.track_id is not None}
        for obj in objects:
            if obj.track_id is None:
                continue
            old = self.previous.get(obj.track_id)
            if old and timestamp > old[1]:
                dx, dy = obj.center[0] - old[0][0], obj.center[1] - old[0][1]
                obj.speed_pixels_per_second = math.hypot(dx, dy) / (timestamp - old[1])
                if obj.speed_pixels_per_second < 5:
                    obj.movement = "stationary_in_image"
                elif abs(dx) >= abs(dy):
                    obj.movement = "left_to_right" if dx > 0 else "right_to_left"
                else:
                    obj.movement = "downward" if dy > 0 else "upward"
            self.previous[obj.track_id] = (obj.center, timestamp)
        events = {
            "newly_visible": sorted(current - self.visible),
            "no_longer_visible": sorted(self.visible - current),
        }
        self.visible = current
        # Bound memory and treat reappearing tracks as new motion observations.
        self.previous = {i: v for i, v in self.previous.items() if i in current}
        return events
