"""Small centroid tracker for public demo use."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot

from simple_detector import Detection


@dataclass
class Track:
    local_id: str
    bbox: tuple[int, int, int, int]
    age: int = 0
    missed: int = 0
    hits: int = 1
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class SimpleTracker:
    def __init__(self, max_distance: float = 80.0, max_missed: int = 10) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 1
        self.tracks: dict[str, Track] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        unmatched = set(range(len(detections)))
        used_tracks: set[str] = set()

        for track in list(self.tracks.values()):
            best_index = None
            best_distance = self.max_distance
            tx, ty = track.center
            for index in unmatched:
                dx1, dy1, dx2, dy2 = detections[index].bbox
                cx, cy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
                distance = hypot(cx - tx, cy - ty)
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            if best_index is None:
                track.missed += 1
                continue
            detection = detections[best_index]
            track.bbox = detection.bbox
            track.age += 1
            track.missed = 0
            track.hits += 1
            track.history.append(track.center)
            unmatched.remove(best_index)
            used_tracks.add(track.local_id)

        for index in unmatched:
            local_id = f"L{self._next_id:04d}"
            self._next_id += 1
            track = Track(local_id=local_id, bbox=detections[index].bbox)
            track.history.append(track.center)
            self.tracks[local_id] = track
            used_tracks.add(local_id)

        for local_id in list(self.tracks):
            if self.tracks[local_id].missed > self.max_missed:
                del self.tracks[local_id]

        return [track for track in self.tracks.values() if track.missed == 0]

