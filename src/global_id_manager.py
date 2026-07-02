"""Conservative global ID manager for public demo use."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from reid_embedding_stub import cosine_similarity


@dataclass
class GlobalIdentity:
    global_id: str
    embedding: np.ndarray
    camera_id: str
    local_id: str
    last_frame: int
    last_zone_confidence: float = 0.0


class GlobalIDManager:
    def __init__(
        self,
        similarity_threshold: float = 0.86,
        handoff_threshold: float = 0.72,
        max_frame_gap: int = 90,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.handoff_threshold = handoff_threshold
        self.max_frame_gap = max_frame_gap
        self._next_id = 1
        self.identities: dict[str, GlobalIdentity] = {}
        self.local_to_global: dict[tuple[str, str], str] = {}
        self.events: list[dict[str, object]] = []

    def assign(
        self,
        camera_id: str,
        local_id: str,
        embedding: np.ndarray,
        frame_index: int,
        zone_confidence: float,
    ) -> str:
        key = (camera_id, local_id)
        if key in self.local_to_global:
            global_id = self.local_to_global[key]
            self._update(global_id, camera_id, local_id, embedding, frame_index, zone_confidence)
            return global_id

        best_id = None
        best_score = 0.0
        best_similarity = 0.0
        for candidate in self.identities.values():
            if candidate.camera_id == camera_id and frame_index - candidate.last_frame < self.max_frame_gap:
                continue
            similarity = cosine_similarity(embedding, candidate.embedding)
            topology = max(zone_confidence, candidate.last_zone_confidence)
            recency = max(0.0, 1.0 - (frame_index - candidate.last_frame) / max(1, self.max_frame_gap))
            score = 0.65 * similarity + 0.25 * topology + 0.10 * recency
            if score > best_score:
                best_id = candidate.global_id
                best_score = score
                best_similarity = similarity

        if best_id and (best_similarity >= self.similarity_threshold or best_score >= self.handoff_threshold):
            self.local_to_global[key] = best_id
            self._update(best_id, camera_id, local_id, embedding, frame_index, zone_confidence)
            self.events.append({"event": "handoff_match", "global_id": best_id, "score": round(best_score, 3)})
            return best_id

        global_id = f"G{self._next_id:04d}"
        self._next_id += 1
        self.local_to_global[key] = global_id
        self.identities[global_id] = GlobalIdentity(global_id, embedding, camera_id, local_id, frame_index, zone_confidence)
        self.events.append({"event": "new_global_id", "global_id": global_id})
        return global_id

    def _update(
        self,
        global_id: str,
        camera_id: str,
        local_id: str,
        embedding: np.ndarray,
        frame_index: int,
        zone_confidence: float,
    ) -> None:
        identity = self.identities[global_id]
        identity.embedding = 0.85 * identity.embedding + 0.15 * embedding
        norm = np.linalg.norm(identity.embedding)
        if norm > 1e-8:
            identity.embedding = identity.embedding / norm
        identity.camera_id = camera_id
        identity.local_id = local_id
        identity.last_frame = frame_index
        identity.last_zone_confidence = zone_confidence

