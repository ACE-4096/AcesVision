"""Gesture event output with hold and cooldown, without action execution."""
from __future__ import annotations

import time
from typing import Callable

from .contracts import SceneFrame


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
        gesture = scene.gestures[0] if len(scene.gestures) == 1 else None
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

        known = [f for f in scene.faces if getattr(f, "known", False)]
        actor = known[0].name if len(known) == 1 else None
        self.callback({
            "event": "gesture",
            "gesture": name,
            "confidence": float(getattr(gesture, "score", 0.0)),
            "held_frames": self._held,
            "actor": actor,
            "identity_state": "identified" if actor else "unknown",
            "liveness_state": "not_evaluated",
            "source": scene.source.id,
            "captured_at_monotonic": scene.captured_at,
            "security_authorized": False,
        })
        self._last_fire = now

    def close(self) -> None:
        pass
