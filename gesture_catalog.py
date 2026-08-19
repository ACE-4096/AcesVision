"""Shared gesture and action vocabulary for face-id, AcesVision and AceRGB.

Single source of truth for:

  * which gesture names exist — the seven MediaPipe built-ins plus the custom
    landmark-derived ``Middle_Finger`` and ``Shush``;
  * which typed actions a rule may bind to;
  * how to normalise a hand-typed name onto its canonical spelling;
  * the landmark geometry that recognises the two custom poses.

Ported from AceRGB's ``gesture/acergb_gesture.py`` (``GESTURES`` :132-133,
``ACTION_CATALOG`` :135-151 and the ``Middle_Finger`` landmark pose :43-57) so
the two projects share one vocabulary instead of two that silently disagree.
Free-text gesture names are how both live rules in ``rules.json`` ended up
permanently unfireable ("open_palm" never equals the emitted "Open_Palm").

The gesture half of that vocabulary is no longer written here. It is data in
``gestures.json``, loaded and validated by ``acesvision.catalog``, and
re-exported below so every existing caller keeps the same names. Doing it that
way gives the vocabulary a version and a content hash, which is what lets an
out-of-process subscriber check that it agrees with the emitter instead of
assuming it. The action catalog stays code: nothing outside this repo binds to
it yet, and it is the next thing to move if that changes.

Pure data and pure functions: no cv2, no mediapipe, no network or device I/O.
The one file read — ``gestures.json``, once, at import — happens inside
``acesvision.catalog``. Still safe to import from anywhere.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from acesvision.catalog import (
    CATALOG_SHA256,
    CATALOG_VERSION,
    GESTURE_IDS,
    GESTURES,
    GestureSpec,
    gesture_by_id,
    is_known_gesture,
    normalise_gesture,
    require_gesture,
)


@dataclass(frozen=True)
class ActionSpec:
    """One entry in the bindable-action catalog (AceRGB ACTION_CATALOG:135-151)."""

    id: str
    label: str
    target: str            # none | theme | device | device_color | device_palette | command

    def as_dict(self):
        return asdict(self)


# GESTURES and GESTURE_IDS are imported above from acesvision.catalog, which
# reads them from gestures.json. The two custom poses are named here because the
# geometry further down is what actually recognises them, and that code needs to
# refer to them by something other than a bare string literal.
MIDDLE_FINGER = "Middle_Finger"
SHUSH = "Shush"

ACTION_CATALOG = (
    ActionSpec("none", "No action", "none"),
    ActionSpec("lights_theme", "Load lighting theme", "theme"),
    ActionSpec("lights_next_theme", "Next lighting theme", "none"),
    ActionSpec("lights_off", "Turn all lights off", "none"),
    ActionSpec("lights_random", "Random global colour", "none"),
    ActionSpec("device_static", "Set device colour", "device_color"),
    ActionSpec("device_gradient", "Set device gradient", "device_palette"),
    ActionSpec("device_sync", "Sync device to global", "device"),
    ActionSpec("device_off", "Turn device off", "device"),
    ActionSpec("media_playpause", "Media play / pause", "none"),
    ActionSpec("media_next", "Next media track", "none"),
    ActionSpec("media_previous", "Previous media track", "none"),
    ActionSpec("volume_up", "Volume up", "none"),
    ActionSpec("volume_down", "Volume down", "none"),
    ActionSpec("volume_mute", "Toggle mute", "none"),
    ActionSpec("custom", "Advanced custom command", "command"),
)

ACTION_IDS = tuple(spec.id for spec in ACTION_CATALOG)

# Catalog action -> AcesVision typed (connector, action) pair. Only the actions
# that a typed connector can already express appear here; the rest stay shell
# rules until a connector exists. Kept honest by a test that checks every pair
# against acesvision.policy.CONNECTORS.
CONNECTOR_BINDINGS = {
    "lights_next_theme": ("acergb", "next_theme"),
    "lights_off": ("acergb", "off"),
    "media_playpause": ("mpris", "play_pause"),
    "media_next": ("mpris", "next"),
    "media_previous": ("mpris", "previous"),
    "volume_up": ("pipewire", "volume_up"),
    "volume_down": ("pipewire", "volume_down"),
    "volume_mute": ("pipewire", "mute"),
}

ANY_ACTOR = "*"

# MediaPipe hand-landmark indices: wrist = 0; (tip, pip) per finger.
_FINGERS = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}
_LANDMARK_COUNT = 21
_WRIST = 0
_INDEX_TIP, _INDEX_PIP = _FINGERS["index"]
_INDEX_MCP = 5           # the knuckle, below the pip on an upright finger

# Where the mouth sits inside a face box, as a fraction of the box height
# measured down from its top edge. Face detectors (YuNet, dlib) frame roughly
# hairline-to-chin, which puts the lips at ~0.70-0.75; 0.72 is that midpoint.
MOUTH_CENTRE_Y = 0.72

# How close the index fingertip must come to that mouth point to count as
# "at the lips". The distance is normalised by the face box — dx by its width,
# dy by its height — so it scales with how near the person stands to the
# camera. 0.5 is therefore an ellipse with semi-axes of half a face width and
# half a face height: horizontally the fingertip may sit anywhere across the
# face, vertically it may run from the eye line (0.22h) to just under the chin
# (1.22h). A hand raised to point at the ceiling puts the fingertip at or above
# the top of the head, which is dy <= -0.72 — comfortably outside.
MOUTH_RADIUS = 0.5


# gesture_by_id / normalise_gesture / is_known_gesture / require_gesture are
# imported above. They are methods of the loaded acesvision.catalog.GestureCatalog
# rather than functions written twice — a second copy of the folding rule is
# exactly how two spellings of one gesture start to disagree.
_ACTIONS_BY_ID = {spec.id: spec for spec in ACTION_CATALOG}


def action_by_id(action_id):
    return _ACTIONS_BY_ID.get(str(action_id))


def require_action(action_id):
    """Catalog action id, or ValueError naming the whole catalog."""
    spec = _ACTIONS_BY_ID.get(str(action_id))
    if spec is None:
        raise ValueError(
            f"unknown action: {action_id!r}. Known actions: "
            + ", ".join(ACTION_IDS)
        )
    return spec.id


def catalog_json():
    """Serialisable vocabulary for UI clients (QML combo boxes, applet, CLI).

    Carries the gesture catalog's version and content hash so a client that
    caches this payload can tell whether it is still current. ``GET /api/catalog``
    serves the gesture half of it to out-of-process subscribers.
    """
    return {
        "catalog_version": CATALOG_VERSION,
        "catalog_sha256": CATALOG_SHA256,
        "gestures": [spec.as_dict() for spec in GESTURES],
        "actions": [spec.as_dict() for spec in ACTION_CATALOG],
    }


def _extended(landmarks, tip, pip):
    """A finger is extended when its tip sits farther from the wrist than its
    pip joint — orientation-agnostic, so hand rotation does not matter."""
    def distance(index):
        return math.hypot(landmarks[index].x - landmarks[0].x,
                          landmarks[index].y - landmarks[0].y)

    return distance(tip) > distance(pip)


def _only_extended(landmarks, finger):
    """True when exactly ``finger`` is extended and the other three are curled."""
    extended = {name: _extended(landmarks, tip, pip)
                for name, (tip, pip) in _FINGERS.items()}
    return all(state == (name == finger) for name, state in extended.items())


def is_middle_finger(landmarks):
    """Middle finger extended and the other three curled — the 'salute'.

    ``landmarks`` is any sequence of 21 objects exposing ``.x`` and ``.y``
    (MediaPipe NormalizedLandmark, or a stand-in in tests).
    """
    if landmarks is None or len(landmarks) < _LANDMARK_COUNT:
        return False
    return _only_extended(landmarks, "middle")


def _index_points_up(landmarks):
    """The index finger stands vertically: tip above pip above knuckle.

    Image y grows downward, so "above" is a smaller y. This is deliberately
    stricter than ``_extended`` — extension alone is orientation-agnostic and
    would accept a finger pointing sideways or down at the chest.
    """
    return (landmarks[_INDEX_TIP].y < landmarks[_INDEX_PIP].y
            < landmarks[_INDEX_MCP].y
            and landmarks[_INDEX_TIP].y < landmarks[_WRIST].y)


def fingertip_near_mouth(landmarks, faces, frame_w, frame_h):
    """Is the index fingertip inside the mouth ellipse of any detected face?

    ``landmarks`` carry MediaPipe's normalised 0..1 coordinates; ``faces`` are
    ``engine.Face(x, y, w, h, name, conf, known)`` boxes in full-frame pixels
    (anything exposing ``.x/.y/.w/.h`` works), so the frame size is what maps
    one onto the other.
    """
    if not faces or not frame_w or not frame_h:
        return False
    tip_x = landmarks[_INDEX_TIP].x * float(frame_w)
    tip_y = landmarks[_INDEX_TIP].y * float(frame_h)
    for face in faces:
        width, height = float(face.w), float(face.h)
        if width <= 0 or height <= 0:
            continue
        mouth_x = float(face.x) + width / 2.0
        mouth_y = float(face.y) + MOUTH_CENTRE_Y * height
        if math.hypot((tip_x - mouth_x) / width,
                      (tip_y - mouth_y) / height) <= MOUTH_RADIUS:
            return True
    return False


def is_shush(landmarks, faces=None, frame_w=0, frame_h=0):
    """Index finger held vertically against the lips — the real-world shush.

    Three terms, all required:

      1. index extended, middle/ring/pinky curled;
      2. that index actually pointing *up*, not merely extended;
      3. its fingertip near the mouth of a detected face.

    Term 3 is the whole point. Terms 1 and 2 alone are exactly what MediaPipe
    already labels ``Pointing_Up`` — a shush and a point at the ceiling are the
    same hand. The face box is what tells them apart, so with no face in view
    this returns False and the built-in ``Pointing_Up`` label stands.
    """
    if landmarks is None or len(landmarks) < _LANDMARK_COUNT:
        return False
    if not _only_extended(landmarks, "index"):
        return False
    if not _index_points_up(landmarks):
        return False
    return fingertip_near_mouth(landmarks, faces, frame_w, frame_h)
