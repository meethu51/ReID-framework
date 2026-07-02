"""Public-safe multi-camera ReID demo.

This demo is intentionally small and generic. It supports webcams, video files,
and synthetic generated frames. It does not require CUDA or real ReID models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from global_id_manager import GlobalIDManager
from reid_embedding_stub import embedding_for_crop
from simple_detector import SimpleBlobDetector
from simple_tracker import SimpleTracker
from topology_handoff import load_zones, zone_confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Public-safe multi-camera ReID demo.")
    parser.add_argument("--sources", nargs="*", default=[])
    parser.add_argument("--camera-ids", nargs="*", default=[])
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--zone-file", default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def open_source(source: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(int(source) if source.isdigit() else source)


def synthetic_frame(camera_id: str, frame_index: int, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), 35, dtype=np.uint8)
    t = frame_index % 240
    if camera_id == "cam1":
        x = 40 + min(t, 180) * 2
        visible = t < 210
    else:
        x = 20 + max(0, t - 80) * 2
        visible = t > 70
    if visible:
        cv2.rectangle(frame, (int(x), 180), (int(x + 55), 300), (80, 220, 80), -1)
    cv2.putText(frame, camera_id, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
    return frame


def draw_zones(frame: np.ndarray, camera_id: str, zones: dict) -> None:
    for zone in zones.get(camera_id, []):
        x1, y1, x2, y2 = zone.xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 220), 2)
        cv2.putText(frame, zone.zone_id, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 220), 1)


def main() -> None:
    args = parse_args()
    synthetic = args.synthetic or not args.sources
    camera_ids = args.camera_ids or [f"cam{i + 1}" for i in range(max(2 if synthetic else len(args.sources), 1))]
    zones = load_zones(args.zone_file)
    detector = SimpleBlobDetector()
    trackers = {camera_id: SimpleTracker() for camera_id in camera_ids}
    global_ids = GlobalIDManager()
    captures = {} if synthetic else {camera_id: open_source(src) for camera_id, src in zip(camera_ids, args.sources)}

    print("ReID-Lite public demo started.")
    print("Use --synthetic for a fully public-safe generated demo.")

    for frame_index in range(args.max_frames):
        views = []
        for camera_id in camera_ids:
            if synthetic:
                frame = synthetic_frame(camera_id, frame_index, args.width, args.height)
            else:
                ok, frame = captures[camera_id].read()
                if not ok:
                    continue

            detections = detector.detect(frame)
            tracks = trackers[camera_id].update(detections)
            draw_zones(frame, camera_id, zones)

            for track in tracks:
                embedding = embedding_for_crop(frame, track.bbox)
                zconf = zone_confidence(camera_id, track.bbox, zones)
                gid = global_ids.assign(camera_id, track.local_id, embedding, frame_index, zconf)
                x1, y1, x2, y2 = track.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 180, 255), 2)
                label = f"{camera_id}:{track.local_id} -> {gid} zone={zconf:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            views.append(frame)

        if not views:
            break
        if args.show:
            canvas = np.hstack([cv2.resize(view, (args.width, args.height)) for view in views])
            cv2.imshow("ReID-Lite", canvas)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    for capture in captures.values():
        capture.release()
    cv2.destroyAllWindows()

    print("Summary:")
    print(f"  global_ids={len(global_ids.identities)}")
    print(f"  events={global_ids.events[-10:]}")


if __name__ == "__main__":
    main()

