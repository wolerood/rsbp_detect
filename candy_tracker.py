from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import math


BBox = Tuple[int, int, int, int]
Point = Tuple[int, int]


@dataclass
class Track:
    track_id: int
    bbox: BBox
    center: Point
    prev_center: Point
    hits: int = 1
    missed: int = 0
    counted: bool = False


class CandyTracker:
    """Simple centroid-based tracker for one-direction conveyor counting."""

    def __init__(
        self,
        line_x: int,
        max_distance: int = 50,
        min_hits: int = 3,
        max_missed: int = 5,
        direction: str = "left_to_right",
    ) -> None:
        self.line_x = line_x
        self.max_distance = max_distance
        self.min_hits = min_hits
        self.max_missed = max_missed
        self.direction = direction

        self._next_id = 1
        self._tracks: List[Track] = []
        self.total_count = 0

    @staticmethod
    def bbox_center(bbox: BBox) -> Point:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @staticmethod
    def _distance(p1: Point, p2: Point) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _create_track(self, bbox: BBox) -> None:
        center = self.bbox_center(bbox)
        self._tracks.append(
            Track(
                track_id=self._next_id,
                bbox=bbox,
                center=center,
                prev_center=center,
            )
        )
        self._next_id += 1

    def _crossed_line(self, track: Track) -> bool:
        prev_x = track.prev_center[0]
        curr_x = track.center[0]

        if self.direction == "left_to_right":
            return prev_x < self.line_x <= curr_x
        if self.direction == "right_to_left":
            return prev_x > self.line_x >= curr_x
        raise ValueError(f"Unsupported direction: {self.direction}")

    def update(self, detections: List[BBox]) -> List[Track]:
        """Update tracker with current-frame detections.

        Args:
            detections: list of bounding boxes (x1, y1, x2, y2)
        Returns:
            Current active tracks.
        """
        det_centers = [self.bbox_center(bbox) for bbox in detections]
        unmatched_dets = set(range(len(detections)))
        unmatched_tracks = set(range(len(self._tracks)))

        # Greedy matching by nearest centroid.
        pairs: List[Tuple[float, int, int]] = []
        for t_idx, track in enumerate(self._tracks):
            for d_idx, center in enumerate(det_centers):
                dist = self._distance(track.center, center)
                pairs.append((dist, t_idx, d_idx))

        pairs.sort(key=lambda item: item[0])

        for dist, t_idx, d_idx in pairs:
            if dist > self.max_distance:
                continue
            if t_idx not in unmatched_tracks or d_idx not in unmatched_dets:
                continue

            track = self._tracks[t_idx]
            track.prev_center = track.center
            track.center = det_centers[d_idx]
            track.bbox = detections[d_idx]
            track.hits += 1
            track.missed = 0

            if not track.counted and track.hits >= self.min_hits and self._crossed_line(track):
                track.counted = True
                self.total_count += 1

            unmatched_tracks.remove(t_idx)
            unmatched_dets.remove(d_idx)

        # Age unmatched tracks.
        for t_idx in list(unmatched_tracks):
            self._tracks[t_idx].missed += 1

        # Remove stale tracks.
        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]

        # Create tracks for unmatched detections.
        for d_idx in unmatched_dets:
            self._create_track(detections[d_idx])

        return list(self._tracks)
