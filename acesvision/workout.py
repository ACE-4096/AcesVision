"""Noise-filtered, landmark-driven workout analysis.

This module owns temporal judgement.  Pose detection says where joints were in
one image; it does not decide that one noisy frame was a completed rep.  A
rep here must cross both ends of an exercise's range, stay there long enough,
and return through hysteresis.  That makes the count deliberately conservative.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, hypot


VISIBILITY_MIN = 0.55
EMA_ALPHA = 0.45
MIN_PHASE_S = 0.25


@dataclass(frozen=True)
class Exercise:
    id: str
    label: str
    joints: tuple[tuple[int, int, int], ...]
    rest_angle: float
    active_angle: float


EXERCISES = (
    Exercise("curl", "Bicep curl", ((11, 13, 15), (12, 14, 16)), 150, 55),
    Exercise("squat", "Squat", ((23, 25, 27), (24, 26, 28)), 165, 105),
    Exercise("pushup", "Push-up", ((11, 13, 15), (12, 14, 16)), 155, 95),
    Exercise("press", "Shoulder press", ((11, 13, 15), (12, 14, 16)), 75, 155),
)
BY_ID = {exercise.id: exercise for exercise in EXERCISES}


def angle(a, b, c):
    """The 2D angle ABC in degrees, with no third-party numerical dependency."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    length = hypot(bax, bay) * hypot(bcx, bcy)
    if length < 1e-6:
        return None
    cosine = max(-1.0, min(1.0, (bax * bcx + bay * bcy) / length))
    return degrees(acos(cosine))


def pose_angle(pose, joints):
    """Average visible left/right working joint angles, or return no signal."""
    landmarks = getattr(pose, "landmarks", ())
    values = []
    for first, pivot, last in joints:
        if max(first, pivot, last) >= len(landmarks):
            continue
        points = (landmarks[first], landmarks[pivot], landmarks[last])
        if any(len(point) < 3 or float(point[2]) < VISIBILITY_MIN
               for point in points):
            continue
        value = angle(*points)
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


class WorkoutAnalyzer:
    """One exercise counter with filtering, dwell time, and full-cycle gating."""

    def __init__(self, exercise="squat", *, enabled=False):
        self._exercise = BY_ID[exercise]
        self.enabled = bool(enabled)
        self.reset()

    @property
    def exercise(self):
        return self._exercise

    def set_exercise(self, exercise):
        if exercise not in BY_ID:
            raise ValueError(f"unknown exercise {exercise!r}")
        if exercise != self._exercise.id:
            self._exercise = BY_ID[exercise]
            self.reset()

    def reset(self):
        self.count = 0
        self.phase = "find rest"
        self.filtered_angle = None
        self._candidate = None
        self._candidate_at = None

    def update(self, poses, captured_at):
        """Advance from the newest complete pose set and return UI-safe data."""
        raw = pose_angle(poses[0], self._exercise.joints) if poses else None
        if not self.enabled:
            return self._state(raw, "Workout paused")
        if raw is None:
            self._candidate = None
            self._candidate_at = None
            return self._state(None, "Keep the working joints visible")

        if self.filtered_angle is None:
            self.filtered_angle = raw
        else:
            self.filtered_angle += EMA_ALPHA * (raw - self.filtered_angle)
        current = self.filtered_angle
        direction_down = self._exercise.active_angle < self._exercise.rest_angle
        at_rest = (current >= self._exercise.rest_angle if direction_down
                   else current <= self._exercise.rest_angle)
        at_active = (current <= self._exercise.active_angle if direction_down
                     else current >= self._exercise.active_angle)
        target = "rest" if at_rest else "active" if at_active else None
        if target != self._candidate:
            self._candidate = target
            self._candidate_at = captured_at if target else None
        stable = (target is not None and self._candidate_at is not None and
                  captured_at - self._candidate_at >= MIN_PHASE_S)
        counted = False
        if stable and target == "rest" and self.phase == "find rest":
            self.phase = "rest"
        elif stable and target == "active" and self.phase == "rest":
            self.phase = "active"
        elif stable and target == "rest" and self.phase == "active":
            self.count += 1
            self.phase = "rest"
            counted = True
        feedback = ("Rep counted" if counted else
                    "Move to the start position" if self.phase == "find rest" else
                    "Controlled movement")
        return self._state(raw, feedback)

    def _state(self, raw, feedback):
        current = self.filtered_angle
        span = self._exercise.active_angle - self._exercise.rest_angle
        progress = (0.0 if current is None else
                    max(0.0, min(1.0, (current - self._exercise.rest_angle) /
                                      (span or 1.0))))
        return {
            "workout_enabled": self.enabled,
            "workout_exercise": self._exercise.id,
            "workout_label": self._exercise.label,
            "workout_reps": self.count,
            "workout_phase": self.phase,
            "workout_angle": current,
            "workout_raw_angle": raw,
            "workout_progress": progress,
            "workout_feedback": feedback,
            "workout_filter": ("EMA angle smoothing + 250 ms endpoint dwell + "
                               "full-range hysteresis"),
        }
