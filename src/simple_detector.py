"""Toy public-safe detector for colored person-like blobs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    confidence: float
    label: str = "person"


class SimpleBlobDetector:
    """Detects saturated colored blobs.

    This is intentionally not a production detector. It is only for public
    demos with synthetic rectangles or simple webcam scenes.
    """

    def __init__(self, min_area: float = 600.0) -> None:
        self.min_area = min_area

    def detect(self, frame: np.ndarray) -> list[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        mask = cv2.inRange(saturation, 60, 255) & cv2.inRange(value, 50, 255)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: list[Detection] = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            confidence = min(1.0, area / 5000.0)
            detections.append(Detection((x, y, x + w, y + h), confidence))
        return detections

