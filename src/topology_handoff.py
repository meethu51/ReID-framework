"""Manual handoff-zone helpers for the public demo."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HandoffZone:
    camera_id: str
    zone_id: str
    xyxy: tuple[int, int, int, int]


def load_zones(path: str | None) -> dict[str, list[HandoffZone]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    zone_items = data.get("zones", [])
    zones: dict[str, list[HandoffZone]] = {}
    for item in zone_items:
        if item.get("type", "rect") != "rect":
            continue
        camera_id = str(item["camera_id"])
        zone = HandoffZone(
            camera_id=camera_id,
            zone_id=str(item.get("zone_id", "zone")),
            xyxy=tuple(int(v) for v in item["xyxy"]),
        )
        zones.setdefault(camera_id, []).append(zone)
    return zones


def intersection_ratio(bbox: tuple[int, int, int, int], zone: HandoffZone) -> float:
    x1, y1, x2, y2 = bbox
    zx1, zy1, zx2, zy2 = zone.xyxy
    ix1, iy1 = max(x1, zx1), max(y1, zy1)
    ix2, iy2 = min(x2, zx2), min(y2, zy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, (x2 - x1) * (y2 - y1))
    return inter / area


def center_inside(bbox: tuple[int, int, int, int], zone: HandoffZone) -> bool:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    zx1, zy1, zx2, zy2 = zone.xyxy
    return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


def zone_confidence(camera_id: str, bbox: tuple[int, int, int, int], zones: dict[str, list[HandoffZone]]) -> float:
    scores = []
    for zone in zones.get(camera_id, []):
        overlap = intersection_ratio(bbox, zone)
        center_bonus = 0.35 if center_inside(bbox, zone) else 0.0
        scores.append(min(1.0, overlap + center_bonus))
    return max(scores, default=0.0)

