"""Deterministic embedding stub for public-safe ReID demos."""

from __future__ import annotations

import cv2
import numpy as np


def embedding_for_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(8, dtype=np.float32)

    resized = cv2.resize(crop, (16, 16))
    mean = resized.reshape(-1, 3).mean(axis=0) / 255.0
    std = resized.reshape(-1, 3).std(axis=0) / 255.0
    aspect = np.array([(x2 - x1) / max(1.0, y2 - y1), (x2 - x1) * (y2 - y1) / 10000.0])
    vector = np.concatenate([mean, std, aspect]).astype(np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-8 else vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(a, b) / denom)

