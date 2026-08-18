"""Gesture event output with hold and cooldown, without action execution."""
from __future__ import annotations

import logging
import time
from typing import Callable

from .contracts import SceneFrame

log = logging.getLogger(__name__)


def _centre(box):
    """Centre point of anything carrying x/y/w/h, or None."""
    try:
        return (float(box.x) + float(box.w) / 2.0,
                float(box.y) + float(box.h) / 2.0)
    except (AttributeError, TypeError, ValueError):
        return None


def select_gesture(gestures):
    """The gesture an event should be about: the most confident one.

    Taking ``gestures[0]`` and only when there was exactly one meant a
    two-handed person — the GestureDetector default is ``num_hands=2`` — fired
    nothing at all. A one-handed pose held by someone whose other hand happened
    to be in frame produced zero events, silently.

    Custom landmark poses (Middle_Finger, Shush) score a flat 1.0 in
    ``gesture_catalog``, so they keep the precedence they already have over the
    model's own labels. Ties fall to the earlier hand, which is stable across
    frames because MediaPipe keeps hand ordering.
    """
    best = None
    best_score = None
    for gesture in gestures or []:
        if not getattr(gesture, "name", ""):
            continue
        score = float(getattr(gesture, "score", 0.0) or 0.0)
        if best_score is None or score > best_score:
            best, best_score = gesture, score
    return best


def attribute_actor(faces, gesture):
    """Attribute a gesture to an enrolled person.

    Returns ``(actor, attribution, candidates)``:

    ``none``      no enrolled face in frame.
    ``unique``    exactly one enrolled face — it is the actor.
    ``nearest``   several enrolled faces — the one closest to the gesturing
                  hand. Reported so the caller can see the choice was made
                  under ambiguity rather than read it as certainty.
    ``ambiguous`` several enrolled faces and no usable geometry to separate
                  them. Actor is None, but the candidates are now named.

    Previously anything other than exactly one enrolled face gave ``None``, so
    two known people in frame meant every actor-scoped rule silently stopped
    matching with nothing logged.
    """
    known = [face for face in faces or [] if getattr(face, "known", False)]
    candidates = [getattr(face, "name", None) for face in known]
    if not known:
        return None, "none", []
    if len(known) == 1:
        return getattr(known[0], "name", None), "unique", candidates

    hand = _centre(gesture) if gesture is not None else None
    ranked = []
    for face in known:
        point = _centre(face)
        if point is None or hand is None:
            continue
        distance = (point[0] - hand[0]) ** 2 + (point[1] - hand[1]) ** 2
        ranked.append((distance, face))
    if not ranked:
        log.warning("gesture actor ambiguous: %d enrolled faces in frame (%s) "
                    "and no box geometry to separate them", len(known), candidates)
        return None, "ambiguous", candidates
    ranked.sort(key=lambda item: item[0])
    actor = getattr(ranked[0][1], "name", None)
    log.info("gesture actor attributed by proximity: %s of %s", actor, candidates)
    return actor, "nearest", candidates


class GestureEventOutput:
    def __init__(self, callback: Callable[[dict], None], hold_frames: int = 6,
                 cooldown_s: float = 1.5, clock=time.monotonic,
                 enabled: bool = False):
        self.callback = callback
        self.hold_frames = max(1, int(hold_frames))
        self.cooldown_s = float(cooldown_s)
        self.clock = clock
        self._name = ""
        self._held = 0
        self._last_fire = None
        # Off by default because the GUI owns a toggle for it. Any headless
        # caller must opt in explicitly — python -m acesvision emitted nothing
        # ever because nobody did.
        self.enabled = bool(enabled)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if not self.enabled:
            self._name = ""
            self._held = 0

    def publish(self, scene: SceneFrame) -> None:
        if not self.enabled:
            return
        gesture = select_gesture(scene.gestures)
        name = getattr(gesture, "name", "") if gesture is not None else ""
        if name and name == self._name:
            self._held += 1
        else:
            self._name = name
            self._held = 1 if name else 0
        now = self.clock()
        if not name or self._held != self.hold_frames:
            return
        if self._last_fire is not None and now - self._last_fire <= self.cooldown_s:
            return

        actor, attribution, candidates = attribute_actor(scene.faces, gesture)
        self.callback({
            "event": "gesture",
            "gesture": name,
            "confidence": float(getattr(gesture, "score", 0.0)),
            "held_frames": self._held,
            "actor": actor,
            "actor_attribution": attribution,
            "actor_candidates": candidates,
            "hands_in_frame": len(scene.gestures or []),
            "identity_state": "identified" if actor else "unknown",
            "liveness_state": "not_evaluated",
            "source": scene.source.id,
            "captured_at_monotonic": scene.captured_at,
            "security_authorized": False,
        })
        self._last_fire = now

    def close(self) -> None:
        pass
