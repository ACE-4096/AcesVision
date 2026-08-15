"""Shared gesture and action vocabulary for face-id, AcesVision and AceRGB.

Single source of truth for:

  * which gesture names exist — the seven MediaPipe built-ins plus the custom
    landmark-derived ``Middle_Finger``;
  * which typed actions a rule may bind to;
  * how to normalise a hand-typed name onto its canonical spelling.

Ported from AceRGB's ``gesture/acergb_gesture.py`` (``GESTURES`` :132-133,
``ACTION_CATALOG`` :135-151 and the ``Middle_Finger`` landmark pose :43-57) so
the two projects share one vocabulary instead of two that silently disagree.
Free-text gesture names are how both live rules in ``rules.json`` ended up
permanently unfireable ("open_palm" never equals the emitted "Open_Palm").

Pure data and pure functions: no cv2, no mediapipe, no I/O. Safe to import from
anywhere, including modules AcesVision imports lazily.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GestureSpec:
    """One entry in the recognisable-gesture vocabulary."""

    id: str
    label: str
    builtin: bool = True   # True == the MediaPipe model emits this label itself

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ActionSpec:
    """One entry in the bindable-action catalog (AceRGB ACTION_CATALOG:135-151)."""

    id: str
    label: str
    target: str            # none | theme | device | device_color | device_palette | command

    def as_dict(self):
        return asdict(self)


GESTURES = (
    GestureSpec("Closed_Fist", "Closed fist"),
    GestureSpec("Open_Palm", "Open palm"),
    GestureSpec("Pointing_Up", "Pointing up"),
    GestureSpec("Thumb_Up", "Thumb up"),
    GestureSpec("Thumb_Down", "Thumb down"),
    GestureSpec("Victory", "Victory"),
    GestureSpec("ILoveYou", "I love you"),
    GestureSpec("Middle_Finger", "Middle finger", builtin=False),
)

GESTURE_IDS = tuple(spec.id for spec in GESTURES)
MIDDLE_FINGER = "Middle_Finger"

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


def _key(name):
    """Fold a hand-typed name onto a comparison key: case and separators ignored."""
    return "".join(character for character in str(name).lower()
                   if character.isalnum())


_GESTURES_BY_KEY = {_key(spec.id): spec for spec in GESTURES}
_ACTIONS_BY_ID = {spec.id: spec for spec in ACTION_CATALOG}


def gesture_by_id(gesture_id):
    return _GESTURES_BY_KEY.get(_key(gesture_id))


def normalise_gesture(name):
    """Return the canonical gesture id for a loosely typed name, or None."""
    spec = _GESTURES_BY_KEY.get(_key(name))
    return spec.id if spec else None


def is_known_gesture(name):
    return normalise_gesture(name) is not None


def require_gesture(name):
    """Canonical gesture id, or ValueError naming the whole vocabulary."""
    canonical = normalise_gesture(name)
    if canonical is None:
        raise ValueError(
            f"unknown gesture: {name!r}. Known gestures: "
            + ", ".join(GESTURE_IDS)
        )
    return canonical


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
    """Serialisable vocabulary for UI clients (QML combo boxes, applet, CLI)."""
    return {
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


def is_middle_finger(landmarks):
    """Middle finger extended and the other three curled — the 'salute'.

    ``landmarks`` is any sequence of 21 objects exposing ``.x`` and ``.y``
    (MediaPipe NormalizedLandmark, or a stand-in in tests).
    """
    if landmarks is None or len(landmarks) < _LANDMARK_COUNT:
        return False
    extended = {finger: _extended(landmarks, tip, pip)
                for finger, (tip, pip) in _FINGERS.items()}
    return (extended["middle"] and not extended["index"]
            and not extended["ring"] and not extended["pinky"])
