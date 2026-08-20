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
    show_confidence: bool = True
    known_colour: tuple[int, int, int] = (0, 200, 0)
    unknown_colour: tuple[int, int, int] = (40, 40, 220)
    gesture_colour: tuple[int, int, int] = (230, 160, 30)
    object_colour: tuple[int, int, int] = (255, 170, 60)
    line_width: int = 2
    font_scale: float = 0.6


CLEAN = OverlayProfile(id="clean", show_objects=False, show_faces=False,
                       show_gestures=False)
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
            score = getattr(gesture, "score", None)
            if profile.show_confidence and score is not None:
                label += f" {score:.0%}"
            _box(frame, gesture, label, profile.gesture_colour, profile)
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
