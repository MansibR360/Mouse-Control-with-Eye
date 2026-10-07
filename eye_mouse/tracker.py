"""Face landmarks from MediaPipe Face Mesh, reduced to what the mouse needs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# MediaPipe Face Mesh landmark indices (with refine_landmarks=True).
IRIS_CENTERS = (468, 473)
NOSE_TIP = 1
EYES = (
    # top, bottom, inner corner, outer corner
    (159, 145, 133, 33),
    (386, 374, 362, 263),
)


@dataclass
class Face:
    pointer: tuple[float, float]  # normalized 0..1 point that drives the cursor
    left_ear: float  # eye aspect ratio of the eye on the left of the (mirrored) preview
    right_ear: float
    iris_px: list[tuple[int, int]]  # for drawing


class FaceTracker:
    def __init__(self, pointer: str = "iris") -> None:
        from mediapipe import solutions  # heavy import, keep it out of the tests

        self.pointer = pointer
        self._mesh = solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def close(self) -> None:
        self._mesh.close()

    def process(self, rgb_frame) -> Optional[Face]:
        from .gestures import eye_aspect_ratio

        result = self._mesh.process(rgb_frame)
        if not result.multi_face_landmarks:
            return None
        lm = result.multi_face_landmarks[0].landmark
        h, w = rgb_frame.shape[:2]

        def pt(i: int) -> tuple[float, float]:
            # Pixel units, so eye ratios aren't skewed by the frame's aspect ratio.
            return lm[i].x * w, lm[i].y * h

        if self.pointer == "nose":
            pointer = (lm[NOSE_TIP].x, lm[NOSE_TIP].y)
        else:
            a, b = IRIS_CENTERS
            pointer = ((lm[a].x + lm[b].x) / 2, (lm[a].y + lm[b].y) / 2)

        ears = []
        for top, bottom, inner, outer in EYES:
            ear = eye_aspect_ratio(pt(top), pt(bottom), pt(inner), pt(outer))
            ears.append((pt(inner)[0] + pt(outer)[0], ear))
        # The frame is mirrored, so the eye drawn on the left is the user's left eye.
        ears.sort()

        return Face(
            pointer=pointer,
            left_ear=ears[0][1],
            right_ear=ears[1][1],
            iris_px=[(int(lm[i].x * w), int(lm[i].y * h)) for i in IRIS_CENTERS],
        )
