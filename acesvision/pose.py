"""Local MediaPipe whole-body pose detection for the shared scene pipeline.

The dependency stays lazy: a missing optional ``pose_landmarker.task`` must
never prevent camera preview, object detection, or recording from starting.
The live GUI exposes that condition as a disabled/degraded pose stage instead.
"""
from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import cv2


# 33 MediaPipe points, in source-pixel coordinates.  Visibility stays with the
# point so the renderer can refuse an occluded limb rather than invent one.
BodyPose = namedtuple("BodyPose", "landmarks")

DEFAULT_MODEL_PATH = (Path(__file__).resolve().parent.parent / "models" /
                      "pose_landmarker.task")


class PoseDetector:
    """One-person MediaPipe Pose detector with an inspectable availability state."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH, *, num_poses=1):
        self.model_path = Path(model_path)
        self.error = ""
        self._landmarker = None
        if not self.model_path.is_file():
            self.error = f"Pose model unavailable: {self.model_path.name} is not installed"
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            self._mp = mp
            self._landmarker = vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(
                        model_asset_path=str(self.model_path)),
                    num_poses=max(1, int(num_poses)),
                ))
        except Exception as exc:  # Model/runtime issue must not kill the GUI.
            self.error = f"Pose model unavailable: {exc}"

    @property
    def available(self):
        return self._landmarker is not None

    def detect(self, frame):
        if self._landmarker is None:
            return []
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        rows = []
        for landmarks in result.pose_landmarks:
            rows.append(BodyPose(tuple(
                (float(point.x) * width, float(point.y) * height,
                 float(getattr(point, "visibility", 1.0)))
                for point in landmarks
            )))
        return rows

    def close(self):
        close = getattr(self._landmarker, "close", None)
        if close is not None:
            close()
        self._landmarker = None
