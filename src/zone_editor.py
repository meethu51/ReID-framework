"""Simple rectangle handoff-zone editor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a public-safe handoff zone rectangle.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--display-width", type=int, default=960)
    parser.add_argument("--display-height", type=int, default=540)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    capture = cv2.VideoCapture(int(args.source) if args.source.isdigit() else args.source)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise SystemExit(f"Could not read source: {args.source}")

    original_h, original_w = frame.shape[:2]
    scale_x = original_w / args.display_width
    scale_y = original_h / args.display_height
    display = cv2.resize(frame, (args.display_width, args.display_height))
    drawing = {"start": None, "end": None}

    def redraw() -> None:
        canvas = display.copy()
        if drawing["start"] and drawing["end"]:
            cv2.rectangle(canvas, drawing["start"], drawing["end"], (0, 220, 220), 2)
        cv2.imshow("Draw handoff zone: drag, s=save, r=reset, q=quit", canvas)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing["start"] = (x, y)
            drawing["end"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing["start"]:
            drawing["end"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drawing["start"]:
            drawing["end"] = (x, y)
        redraw()

    cv2.namedWindow("Draw handoff zone: drag, s=save, r=reset, q=quit")
    cv2.setMouseCallback("Draw handoff zone: drag, s=save, r=reset, q=quit", on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            drawing["start"] = None
            drawing["end"] = None
            redraw()
        if key == ord("s") and drawing["start"] and drawing["end"]:
            x1, y1 = drawing["start"]
            x2, y2 = drawing["end"]
            ox1, ox2 = sorted((int(x1 * scale_x), int(x2 * scale_x)))
            oy1, oy2 = sorted((int(y1 * scale_y), int(y2 * scale_y)))
            data = {
                "zones": [
                    {
                        "camera_id": args.camera_id,
                        "zone_id": f"{args.camera_id}_handoff_zone",
                        "type": "rect",
                        "xyxy": [ox1, oy1, ox2, oy2],
                        "description": "Public-safe manually drawn handoff zone",
                    }
                ]
            }
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"Saved {output}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

