"""Latest-frame asynchronous object, face, and gesture perception."""
from __future__ import annotations

import threading
import time
from collections import deque

from .contracts import SceneFrame
from .perception import YoloSubprocessDetector


class FaceGestureProcessor:
    """Keep capture fluid while local models process only the newest frame."""

    def __init__(self, face_detector=None, gesture_detector=None,
                 object_detector=None, object_detector_factory=None,
                 detect_every: int = 1, face_hz: float = 2.0,
                 gesture_hz: float = 15.0):
        if face_detector is None:
            from engine import build_detector
            face_detector = build_detector()
        if gesture_detector is None:
            from gestures import GestureDetector
            gesture_detector = GestureDetector()
        self.face_detector = face_detector
        self.gesture_detector = gesture_detector
        self.object_detector = object_detector or YoloSubprocessDetector()
        self.object_detector_factory = (object_detector_factory or
                                        (lambda model: YoloSubprocessDetector(model=model)))
        self.detect_every = max(1, int(detect_every))
        self.face_interval_s = 1.0 / max(0.1, float(face_hz))
        self.gesture_interval_s = 1.0 / max(0.1, float(gesture_hz))
        self._condition = threading.Condition()
        self._pending = None
        self._requested_model = None
        self._stopping = False
        self._objects = []
        self._faces = []
        self._gestures = []
        self._metadata = {
            "object_model": "warming up",
            "face_engine": getattr(self.face_detector, "engine", "unknown"),
            "security_authorized": False,
            "inference_status": "warming_up",
            "inference_fps": 0.0,
            "inference_ms": 0.0,
            "inference_sequence": -1,
            "dropped_inference_frames": 0,
            "face_refresh_hz": float(face_hz),
            "gesture_refresh_hz": float(gesture_hz),
        }
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vision-inference"
        )
        self._thread.start()

    def __call__(self, frame, source, sequence, captured_at):
        if sequence % self.detect_every == 0:
            with self._condition:
                if self._pending is not None:
                    self._metadata["dropped_inference_frames"] += 1
                self._pending = (frame.copy(), source, sequence, captured_at)
                self._condition.notify()
        with self._condition:
            metadata = dict(self._metadata)
            objects = list(self._objects)
            faces = list(self._faces)
            gestures = list(self._gestures)
        return SceneFrame(
            source=source,
            sequence=sequence,
            captured_at=captured_at,
            raw=frame,
            objects=objects,
            faces=faces,
            gestures=gestures,
            metadata=metadata,
        )

    def metrics(self):
        with self._condition:
            return dict(self._metadata)

    def set_object_model(self, model):
        with self._condition:
            self._requested_model = str(model)
            self._metadata["object_model"] = "switching model"
            self._metadata["inference_status"] = "warming_up"
            self._condition.notify()

    def _run(self):
        previous_finished = None
        cycle_intervals = deque(maxlen=30)
        cycle_latencies = deque(maxlen=30)
        last_face_at = 0.0
        last_gesture_at = 0.0
        last_people_signature = ()
        faces = []
        gestures = []
        while True:
            with self._condition:
                while (self._pending is None and self._requested_model is None
                       and not self._stopping):
                    self._condition.wait()
                if self._stopping:
                    return
                requested_model = self._requested_model
                self._requested_model = None
                if requested_model is not None:
                    old_detector = self.object_detector
                    self.object_detector = self.object_detector_factory(requested_model)
                    self._metadata["object_model"] = "warming up"
                    self._metadata["inference_status"] = "warming_up"
                    close = getattr(old_detector, "close", None)
                    if close is not None:
                        close()
                if self._pending is None:
                    continue
                frame, source, sequence, captured_at = self._pending
                self._pending = None
            started = time.monotonic()
            errors = []
            timings = {}
            model_id = "unavailable"
            object_started = time.monotonic()
            try:
                objects, timings, model_id = self.object_detector.detect(frame)
            except Exception as exc:
                objects = []
                errors.append(f"objects: {exc}")
            object_ms = (time.monotonic() - object_started) * 1000.0
            now = time.monotonic()
            people_signature = tuple(sorted(
                item.track_id for item in objects
                if item.label == "person" and item.track_id is not None
            ))
            face_ms = 0.0
            face_refreshed = False
            if not any(item.label == "person" for item in objects):
                faces = []
                last_people_signature = ()
            elif (people_signature != last_people_signature or
                  now - last_face_at >= self.face_interval_s):
                face_started = time.monotonic()
                try:
                    faces = self._faces_for_people(frame, objects)
                except Exception as exc:
                    faces = []
                    errors.append(f"faces: {exc}")
                face_ms = (time.monotonic() - face_started) * 1000.0
                face_refreshed = True
                last_face_at = time.monotonic()
                last_people_signature = people_signature
            gesture_ms = 0.0
            gesture_refreshed = False
            if now - last_gesture_at >= self.gesture_interval_s:
                gesture_started = time.monotonic()
                try:
                    # Faces come first in this loop on purpose: the Shush pose
                    # is only separable from Pointing_Up by where the fingertip
                    # sits relative to the mouth. Faces refresh at face_hz and
                    # gestures at the faster gesture_hz, so these boxes can be
                    # up to one face interval old — good enough for proximity.
                    gestures = self.gesture_detector.detect(frame, faces=faces)
                except Exception as exc:
                    gestures = []
                    errors.append(f"gestures: {exc}")
                gesture_ms = (time.monotonic() - gesture_started) * 1000.0
                gesture_refreshed = True
                last_gesture_at = time.monotonic()
            finished = time.monotonic()
            elapsed_ms = (finished - started) * 1000.0
            cycle_latencies.append(elapsed_ms)
            if previous_finished is not None and finished > previous_finished:
                cycle_intervals.append(finished - previous_finished)
            inference_fps = (len(cycle_intervals) / sum(cycle_intervals)
                             if cycle_intervals else 0.0)
            average_inference_ms = sum(cycle_latencies) / len(cycle_latencies)
            previous_finished = finished
            with self._condition:
                self._objects = objects
                self._faces = faces
                self._gestures = gestures
                dropped = self._metadata["dropped_inference_frames"]
                self._metadata = {
                    "object_model": model_id,
                    "face_engine": getattr(self.face_detector, "engine", "unknown"),
                    "gesture_engine": "mediapipe:gesture_recognizer",
                    "security_authorized": False,
                    "inference_status": "degraded" if errors else "live",
                    "inference_error": " | ".join(errors),
                    "inference_fps": inference_fps,
                    "inference_ms": average_inference_ms,
                    "latest_inference_ms": elapsed_ms,
                    "model_inference_ms": float(timings.get("inference", 0.0)),
                    "object_stage_ms": object_ms,
                    "face_stage_ms": face_ms,
                    "face_refreshed": face_refreshed,
                    "face_refresh_hz": 1.0 / self.face_interval_s,
                    "gesture_stage_ms": gesture_ms,
                    "gesture_refreshed": gesture_refreshed,
                    "gesture_refresh_hz": 1.0 / self.gesture_interval_s,
                    "inference_sequence": sequence,
                    "inference_age_ms": max(0.0, (finished - captured_at) * 1000.0),
                    "dropped_inference_frames": dropped,
                }

    def _faces_for_people(self, frame, objects):
        people = [item for item in objects if item.label == "person"]
        if not people:
            return []
        height, width = frame.shape[:2]
        faces = []
        for person in people:
            x1, y1 = max(0, person.x), max(0, person.y)
            x2 = min(width, person.x + person.w)
            y2 = min(height, person.y + person.h)
            if x2 <= x1 or y2 <= y1:
                continue
            for face in self.face_detector(frame[y1:y2, x1:x2]):
                if hasattr(face, "_replace"):
                    face = face._replace(x=face.x + x1, y=face.y + y1)
                faces.append(face)
        return faces

    def close(self):
        with self._condition:
            self._stopping = True
            self._condition.notify()
        close = getattr(self.object_detector, "close", None)
        if close is not None:
            close()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)
        close_gesture = getattr(self.gesture_detector, "close", None)
        if close_gesture is not None:
            close_gesture()
