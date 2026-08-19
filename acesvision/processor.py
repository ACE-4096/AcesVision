"""Latest-frame asynchronous object, face, and gesture perception."""
from __future__ import annotations

import threading
import time
from collections import deque

from .contracts import SceneFrame
from .perception import YoloSubprocessDetector

#: The three perception stages, in the order the inference loop runs them.
#: Shared with the GUI so the control panel and the loop cannot disagree about
#: what exists.
STAGES = ("object", "face", "gesture")

DEFAULT_DETECT_EVERY = 1
DEFAULT_FACE_HZ = 2.0
DEFAULT_GESTURE_HZ = 15.0

#: A refresh interval is 1/hz, so a rate of zero is an infinite interval and a
#: negative rate is a time machine. Clamp rather than divide by zero on a knob
#: the operator can drag.
MIN_STAGE_HZ = 0.1

#: What ``object_model`` reads while the object stage is switched off. The
#: metadata never claims a model is running when nothing is.
OBJECT_STAGE_OFF = "stage disabled"


def _checked_stage(stage):
    """Reject an unknown stage id loudly. A typo must not silently do nothing."""
    if stage not in STAGES:
        raise ValueError(
            f"unknown perception stage {stage!r}; expected one of "
            + ", ".join(STAGES))
    return stage


class FaceGestureProcessor:
    """Keep capture fluid while local models process only the newest frame.

    Every knob here is live. ``detect_every``, the two refresh rates and the
    per-stage enables are read by the inference loop on *every* cycle, so all
    of them are held under ``self._condition`` and read into locals inside one
    lock acquisition per cycle. A stage must never run against a half-applied
    change, and a caller must never have to restart capture to retune.
    """

    def __init__(self, face_detector=None, gesture_detector=None,
                 object_detector=None, object_detector_factory=None,
                 detect_every: int = DEFAULT_DETECT_EVERY,
                 face_hz: float = DEFAULT_FACE_HZ,
                 gesture_hz: float = DEFAULT_GESTURE_HZ):
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
        self._condition = threading.Condition()
        # Guarded by _condition. Public reads go through the properties below.
        self._detect_every = max(1, int(detect_every))
        self._face_interval_s = 1.0 / max(MIN_STAGE_HZ, float(face_hz))
        self._gesture_interval_s = 1.0 / max(MIN_STAGE_HZ, float(gesture_hz))
        self._stage_enabled = {stage: True for stage in STAGES}
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
            "object_stage_ms": 0.0,
            "object_refreshed": False,
            "object_detect_every": self._detect_every,
            "object_enabled": True,
            "face_stage_ms": 0.0,
            "face_refreshed": False,
            "face_refresh_hz": 1.0 / self._face_interval_s,
            "face_enabled": True,
            "gesture_stage_ms": 0.0,
            "gesture_refreshed": False,
            "gesture_refresh_hz": 1.0 / self._gesture_interval_s,
            "gesture_enabled": True,
        }
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vision-inference"
        )
        self._thread.start()

    def __call__(self, frame, source, sequence, captured_at):
        # One acquisition: the frame divisor is a live knob, so reading it and
        # acting on it must not straddle a set_detect_every() from the GUI.
        with self._condition:
            if sequence % self._detect_every == 0:
                if self._pending is not None:
                    self._metadata["dropped_inference_frames"] += 1
                self._pending = (frame.copy(), source, sequence, captured_at)
                self._condition.notify()
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

    # ---- live knobs --------------------------------------------------------
    #
    # All four setters share set_object_model's idiom exactly: mutate under
    # _condition, mirror the new value into _metadata so metrics() reports it
    # immediately rather than only after the next completed cycle, then notify
    # the worker. The read side is the single lock acquisition at the top of
    # _run's loop body.

    @property
    def detect_every(self):
        """Submit one captured frame in every N to the inference loop."""
        with self._condition:
            return self._detect_every

    @property
    def face_interval_s(self):
        with self._condition:
            return self._face_interval_s

    @property
    def gesture_interval_s(self):
        with self._condition:
            return self._gesture_interval_s

    def stage_enabled(self, stage):
        with self._condition:
            return self._stage_enabled[_checked_stage(stage)]

    def set_object_model(self, model):
        with self._condition:
            self._requested_model = str(model)
            self._metadata["object_model"] = "switching model"
            self._metadata["inference_status"] = "warming_up"
            self._condition.notify()

    def set_detect_every(self, frames):
        """Change the capture-to-inference divisor on the running loop."""
        with self._condition:
            self._detect_every = max(1, int(frames))
            self._metadata["object_detect_every"] = self._detect_every
            self._condition.notify()

    def set_face_hz(self, hz):
        """Change how often the face stage re-runs, live."""
        with self._condition:
            self._face_interval_s = 1.0 / max(MIN_STAGE_HZ, float(hz))
            self._metadata["face_refresh_hz"] = 1.0 / self._face_interval_s
            self._condition.notify()

    def set_gesture_hz(self, hz):
        """Change how often the gesture stage re-runs, live."""
        with self._condition:
            self._gesture_interval_s = 1.0 / max(MIN_STAGE_HZ, float(hz))
            self._metadata["gesture_refresh_hz"] = 1.0 / self._gesture_interval_s
            self._condition.notify()

    def set_stage_enabled(self, stage, enabled):
        """Switch one perception stage off, or back on, without a restart.

        A disabled stage is skipped outright and its results are cleared, not
        frozen: a stale box that keeps rendering is indistinguishable from a
        live one, and the whole point of switching a stage off is that its
        output stops being claimed.
        """
        stage = _checked_stage(stage)
        with self._condition:
            self._stage_enabled[stage] = bool(enabled)
            self._metadata[f"{stage}_enabled"] = bool(enabled)
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
                # Read every live knob this loop uses once, under the lock that
                # publishes them, so one cycle runs against one coherent
                # configuration. (detect_every is applied in __call__, on the
                # capture thread, and is read there under the same lock.)
                face_interval_s = self._face_interval_s
                gesture_interval_s = self._gesture_interval_s
                enabled = dict(self._stage_enabled)
            started = time.monotonic()
            errors = []
            timings = {}
            model_id = "unavailable"
            object_ms = 0.0
            object_refreshed = False
            if not enabled["object"]:
                # Nothing downstream may keep pretending: no objects means no
                # person boxes, which means the face stage has nowhere to look.
                objects = []
                model_id = OBJECT_STAGE_OFF
            else:
                object_started = time.monotonic()
                try:
                    objects, timings, model_id = self.object_detector.detect(frame)
                except Exception as exc:
                    objects = []
                    errors.append(f"objects: {exc}")
                object_ms = (time.monotonic() - object_started) * 1000.0
                object_refreshed = True
            now = time.monotonic()
            people_signature = tuple(sorted(
                item.track_id for item in objects
                if item.label == "person" and item.track_id is not None
            ))
            face_ms = 0.0
            face_refreshed = False
            if not enabled["face"]:
                faces = []
                last_people_signature = ()
            elif not any(item.label == "person" for item in objects):
                faces = []
                last_people_signature = ()
            elif (people_signature != last_people_signature or
                  now - last_face_at >= face_interval_s):
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
            if not enabled["gesture"]:
                gestures = []
            elif now - last_gesture_at >= gesture_interval_s:
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
                    # Costs are what the cycle just measured; the rate and
                    # enable fields are re-read here, live, rather than
                    # replayed from the locals this cycle ran under. A knob
                    # moved mid-cycle would otherwise be published back at its
                    # old value for one cycle and snap the slider backwards.
                    "object_stage_ms": object_ms,
                    "object_refreshed": object_refreshed,
                    "object_detect_every": self._detect_every,
                    "object_enabled": self._stage_enabled["object"],
                    "face_stage_ms": face_ms,
                    "face_refreshed": face_refreshed,
                    "face_refresh_hz": 1.0 / self._face_interval_s,
                    "face_enabled": self._stage_enabled["face"],
                    "gesture_stage_ms": gesture_ms,
                    "gesture_refreshed": gesture_refreshed,
                    "gesture_refresh_hz": 1.0 / self._gesture_interval_s,
                    "gesture_enabled": self._stage_enabled["gesture"],
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
