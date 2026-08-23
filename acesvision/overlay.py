"""Output-specific overlay profiles and OpenCV reference renderer."""
from __future__ import annotations

from dataclasses import dataclass

import cv2

from .contracts import SceneFrame


@dataclass(frozen=True)
class OverlayProfile:
    id: str = "minimal"
    show_objects: bool = True
    show_faces: bool = True
    show_gestures: bool = True
    show_landmarks: bool = True
    show_pose: bool = True
    show_confidence: bool = True
    known_colour: tuple[int, int, int] = (0, 200, 0)
    unknown_colour: tuple[int, int, int] = (40, 40, 220)
    gesture_colour: tuple[int, int, int] = (230, 160, 30)
    object_colour: tuple[int, int, int] = (255, 170, 60)
    line_width: int = 2
    font_scale: float = 0.6


CLEAN = OverlayProfile(id="clean", show_objects=False, show_faces=False,
                       show_gestures=False, show_landmarks=False, show_pose=False)
MINIMAL = OverlayProfile()
BROADCAST = OverlayProfile(id="broadcast", line_width=3, font_scale=0.8)
SECURITY = OverlayProfile(id="security", show_confidence=True, line_width=3)
PROFILES = {
    profile.id: profile
    for profile in (CLEAN, MINIMAL, BROADCAST, SECURITY)
}


def render(scene: SceneFrame, profile: OverlayProfile = MINIMAL):
    frame = scene.raw.copy()
    if profile.show_objects:
        for item in scene.objects:
            label = str(getattr(item, "label", "Object"))
            track_id = getattr(item, "track_id", None)
            if track_id is not None:
                label += f" #{track_id}"
            score = getattr(item, "score", None)
            if profile.show_confidence and score is not None:
                label += f" {score:.0%}"
            _box(frame, item, label, profile.object_colour, profile)
    if profile.show_faces:
        for face in scene.faces:
            known = bool(getattr(face, "known", False))
            colour = profile.known_colour if known else profile.unknown_colour
            name = getattr(face, "name", None) or "Unknown"
            confidence = getattr(face, "conf", None)
            label = name
            if profile.show_confidence and confidence is not None:
                label += f" {confidence:.2f}"
            _box(frame, face, label, colour, profile)
    if profile.show_gestures:
        for gesture in scene.gestures:
            label = str(getattr(gesture, "name", "Gesture"))
            if not label:
                # Landmark-only hands are deliberately not gesture boxes.
                continue
            score = getattr(gesture, "score", None)
            if profile.show_confidence and score is not None:
                label += f" {score:.0%}"
            _box(frame, gesture, label, profile.gesture_colour, profile)
    if profile.show_landmarks:
        for gesture in scene.gestures:
            _hand_skeleton(frame, getattr(gesture, "landmarks", ()),
                           profile.gesture_colour, profile,
                           float(getattr(gesture, "alpha", 1.0)))
    if profile.show_pose:
        for pose in scene.poses:
            _pose_skeleton(frame, getattr(pose, "landmarks", ()), profile)
    return frame


def _box(frame, item, label, colour, profile):
    """Draw one box, composited if the item asks to be drawn part-way in.

    ``alpha`` is the *only* concession this renderer makes to time, and it is
    not one it makes on its own: the number arrives on the item, from
    ``smoothing.SceneSmoother``, which is the single place that knows anything
    about previous frames. ``render`` stays a pure function of the scene it is
    handed — an item without an alpha (every ``Detection``, ``Face`` and
    ``Gesture`` as the detectors emit them) draws exactly as it always did.
    """
    alpha = float(getattr(item, "alpha", 1.0))
    if alpha <= 0.0:
        return
    x, y = int(item.x), int(item.y)
    w, h = int(item.w), int(item.h)
    if alpha >= 1.0:
        _draw(frame, x, y, w, h, label, colour, profile)
        return
    # Draw onto a copy and composite it back, so overlapping fading boxes
    # blend with the image rather than accumulating on each other.
    layer = frame.copy()
    _draw(layer, x, y, w, h, label, colour, profile)
    cv2.addWeighted(layer, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def _draw(frame, x, y, w, h, label, colour, profile):
    cv2.rectangle(frame, (x, y), (x + w, y + h), colour, profile.line_width)
    cv2.putText(frame, label, (x, max(18, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, profile.font_scale, colour,
                profile.line_width, cv2.LINE_AA)


# MediaPipe's fixed 21-landmark hand topology. Keeping the links here as data
# makes the renderer independent of MediaPipe at import time and lets fixtures
# exercise a small prefix of the topology.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


# MediaPipe's 33-point body topology.  Face-detail links are deliberately
# omitted: the pose stage is for posture and exercise form, not another face
# renderer.  Hands remain the dedicated 21-point GestureRecognizer overlay.
POSE_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19), (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (25, 27), (27, 29), (27, 31),
    (29, 31), (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
)
POSE_COLOUR = (210, 70, 255)
POSE_VISIBILITY = 0.45


def _hand_skeleton(frame, landmarks, colour, profile, alpha=1.0):
    """Draw a recognizer hand's actual joints and bones, respecting fades."""
    points = tuple((int(point[0]), int(point[1])) for point in landmarks or ())
    if not points or alpha <= 0.0:
        return
    target = frame if alpha >= 1.0 else frame.copy()
    for start, end in HAND_CONNECTIONS:
        if start < len(points) and end < len(points):
            cv2.line(target, points[start], points[end], colour,
                     max(1, profile.line_width), cv2.LINE_AA)
    radius = max(2, profile.line_width + 1)
    for point in points:
        cv2.circle(target, point, radius, colour, -1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(target, alpha, frame, 1.0 - alpha, 0.0, dst=frame)


def _pose_skeleton(frame, landmarks, profile):
    """Render visible MediaPipe body joints without drawing guessed limbs."""
    points = tuple(landmarks or ())
    if not points:
        return

    def visible(index):
        return (index < len(points) and len(points[index]) >= 3 and
                float(points[index][2]) >= POSE_VISIBILITY)

    for start, end in POSE_CONNECTIONS:
        if visible(start) and visible(end):
            cv2.line(frame, (int(points[start][0]), int(points[start][1])),
                     (int(points[end][0]), int(points[end][1])), POSE_COLOUR,
                     max(1, profile.line_width), cv2.LINE_AA)
    for index, point in enumerate(points):
        if visible(index):
            cv2.circle(frame, (int(point[0]), int(point[1])),
                       max(2, profile.line_width + 1), POSE_COLOUR,
                       -1, cv2.LINE_AA)
