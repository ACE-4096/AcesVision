"""Temporal smoothing between the inference rate and the capture rate.

Why this exists
---------------
Wherever capture outruns inference, ``FaceGestureProcessor`` hands the same
detections to several consecutive frames. That is deliberate and it is what
keeps capture fluid: ``_objects``, ``_faces`` and ``_gestures`` hold the last
*completed* inference and are handed to every captured frame unchanged.

Take the operating point this filter was specified against — 60 fps capture,
16.7 ms a frame, against 40 fps inference, 25 ms a cycle. Roughly every third
captured frame carries a fresh detection and the two either side repeat it
verbatim. Boxes hold still, then jump, and at 60 fps that reads as flicker.

It is the overlay's problem, not the detector's: ``overlay.render`` draws
exactly what the last ``SceneFrame`` holds and has no notion of time at all.
Where inference already keeps up with capture there is nothing to fix, and this
filter says so by being the identity there — see "The time constant" below.

What this is
------------
A critically damped ease toward the newest result: a first-order lag, which is
the zero-overshoot case, with a time constant scaled by the measured
``inference_fps``.

It is **not interpolation** — there is no frame buffer, nothing is delayed to
put a known future result on the other side of the blend, and so no latency is
added. It is **not extrapolation** — a box never travels anywhere the detector
has not already been, so it cannot overshoot and snap back.

That distinction is the whole design. On a recording there is no on-screen
reference truth to compare against, so a box that lags a fraction of a frame
behind reads as "tracking", while one that overshoots and corrects reads as
broken. Given the choice, lag.

The time constant
-----------------
``tau = max(0, inference_interval - dt)`` — the part of the gap between
inference results that the capture loop has to fill with repeats. The eased box
therefore glides across the stale window and arrives about as the next real
detection lands.

The degenerate case falls out of the arithmetic rather than out of a special
case: when inference is at least as fast as capture, ``inference_interval <=
dt``, ``tau`` is zero, the ease factor is exactly 1.0 and the filter is the
identity. Same when ``inference_fps`` is not known yet (warming up): no
measured rate, no smoothing, nothing invented.

Per stage
---------
``objects``
    Matched by ``track_id`` — ``perception.Detection`` carries one — and eased
    in x/y/w/h. A track the newest result does not carry is held and faded out;
    a track that is new is faded in *at its true position*, never eased in from
    nowhere.

``faces``
    ``engine.Face`` has no track_id and the stage refreshes at 2 Hz. Position
    is deliberately **not** eased: a 500 ms ease drifts the box off the face it
    is naming, which is worse than a still box. Fade in and out only, matched
    on name plus overlap.

``gestures``
    15 Hz and transient. Fade only. There is deliberately no label debouncing
    here — ``events.GestureEventOutput`` already owns hold and cooldown, and a
    second definition of "the gesture is on" would eventually disagree with the
    one that fires the automations.

One smoother per output
-----------------------
Never share an instance between outputs. ``pipeline._OutputWorker`` is a
one-slot drop-old mailbox, so every output sees a *different subset* of frames.
A shared smoother would be raced across worker threads and would advance its
clock against frames a given output never received.

The lossless workers added for recording (``add_output(..., lossless=True)``)
make this stricter rather than looser: a recorder sees *every* frame while the
preview beside it sees a fraction of them, so the two are further apart than
two drop-old outputs ever were.

The stage switches
------------------
``processor.set_stage_enabled`` clears a disabled stage's results rather than
freezing them, and it does so on purpose: a stale box that keeps rendering is
indistinguishable from a live one. A fade-out applied to that clear would
resurrect boxes the operator has just switched off, which looks exactly like
broken stage controls. So the enable flags in ``scene.metadata`` are read
first, and a disabled stage is cleared instantly — no fade, no held ghosts, no
state kept to come back from.
"""
from __future__ import annotations

import math
import time
from typing import Any

from .contracts import SceneFrame

#: How long a newly seen track takes to reach full opacity.
FADE_IN_S = 0.15

#: How long a track that has left the newest result takes to disappear...
FADE_OUT_S = 0.25

#: ...hard-capped at this many inference intervals, so a ghost never outlives
#: the evidence for it by more than two chances to be re-detected.
FADE_OUT_INFERENCE_INTERVALS = 2.0

#: Overlap needed to call two same-named faces the same face across frames.
#: Faces are matched on name first; this only separates two people the gallery
#: gives the same name, and two "Unknown" faces from each other.
FACE_MATCH_IOU = 0.2

#: Hands move fast between gesture refreshes, so the bar to call it the same
#: hand is lower than for a face.
GESTURE_MATCH_IOU = 0.1

#: Returned by a key function for an item that cannot be tracked across frames
#: — an object detection with no track_id. It is rendered exactly as it
#: arrived, fully opaque, and no state is kept for it. Guessing which untracked
#: box is which would be the one thing this module refuses to do.
UNSMOOTHED = object()


def inference_interval_s(metadata) -> float:
    """Seconds between completed inference cycles, or 0.0 if not measured yet."""
    try:
        fps = float((metadata or {}).get("inference_fps", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(fps) or fps <= 0.0:
        return 0.0
    return 1.0 / fps


def ease_factor(dt: float, interval_s: float) -> float:
    """The fraction of the remaining distance to close this frame.

    ``1 - exp(-dt / tau)`` with ``tau = max(0, interval_s - dt)``. Always in
    [0, 1], so the eased value is always between where it was and where the
    detector says it is: convergence without overshoot, by construction.
    """
    if dt is None or dt <= 0.0:
        return 0.0
    tau = max(0.0, float(interval_s) - dt)
    if tau <= 0.0:
        return 1.0
    return min(1.0, 1.0 - math.exp(-dt / tau))


def fade_out_s(interval_s: float) -> float:
    """How long a fade-out lasts at this inference rate."""
    return min(FADE_OUT_S, FADE_OUT_INFERENCE_INTERVALS * max(0.0, float(interval_s)))


def iou(first, second) -> float:
    """Intersection over union of two things carrying x/y/w/h."""
    ax, ay = float(first.x), float(first.y)
    aw, ah = float(first.w), float(first.h)
    bx, by = float(second.x), float(second.y)
    bw, bh = float(second.w), float(second.h)
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0.0 else 0.0


class SmoothedItem:
    """One detection as the overlay should draw it: eased box plus an alpha.

    A proxy, not a copy. Everything the renderer reads that is not geometry —
    ``label``, ``track_id``, ``score``, ``name``, ``conf``, ``known`` — is
    forwarded to the detection this stands for, so a new field on ``Detection``
    or ``Face`` reaches the overlay without passing through here.
    """

    __slots__ = ("item", "x", "y", "w", "h", "alpha")

    def __init__(self, item: Any, x: float, y: float, w: float, h: float,
                 alpha: float = 1.0):
        self.item = item
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.alpha = alpha

    def __getattr__(self, name):
        # Only reached for names that are not slots. Private names are never
        # forwarded, so a half-built proxy raises instead of recursing.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.item, name)

    def __repr__(self):
        return (f"SmoothedItem({self.item!r}, x={self.x:.1f}, y={self.y:.1f}, "
                f"w={self.w:.1f}, h={self.h:.1f}, alpha={self.alpha:.2f})")


class _Track:
    """The smoother's memory of one detection between frames."""

    __slots__ = ("key", "item", "x", "y", "w", "h", "alpha")

    def __init__(self, key, item, alpha: float):
        self.key = key
        self.item = item
        self.x = float(item.x)
        self.y = float(item.y)
        self.w = float(item.w)
        self.h = float(item.h)
        self.alpha = alpha

    def snap_to(self, item) -> None:
        self.x, self.y = float(item.x), float(item.y)
        self.w, self.h = float(item.w), float(item.h)

    def ease_towards(self, item, factor: float) -> None:
        self.x += (float(item.x) - self.x) * factor
        self.y += (float(item.y) - self.y) * factor
        self.w += (float(item.w) - self.w) * factor
        self.h += (float(item.h) - self.h) * factor

    def rendered(self) -> SmoothedItem:
        return SmoothedItem(self.item, self.x, self.y, self.w, self.h, self.alpha)


def _object_key(item):
    track_id = getattr(item, "track_id", None)
    return UNSMOOTHED if track_id is None else ("track", track_id)


def _name_key(item):
    return ("name", str(getattr(item, "name", "") or ""))


class _Stage:
    """One stage's matching rule and whether its geometry may be eased."""

    __slots__ = ("metadata_flag", "key_of", "min_iou", "ease_position")

    def __init__(self, metadata_flag, key_of, min_iou, ease_position):
        self.metadata_flag = metadata_flag
        self.key_of = key_of
        self.min_iou = min_iou
        self.ease_position = ease_position


OBJECT_STAGE = _Stage("object_enabled", _object_key, None, True)
# min_iou None: a track_id is authoritative on its own, and a tracked object
# that teleports has still been asserted to be the same object.
FACE_STAGE = _Stage("face_enabled", _name_key, FACE_MATCH_IOU, False)
GESTURE_STAGE = _Stage("gesture_enabled", _name_key, GESTURE_MATCH_IOU, False)


class SceneSmoother:
    """Ease one output's view of the scene. One instance per output, never shared.

    Pure in the sense that matters: the frame it returns is a function of the
    scene, the state left by the scenes this instance was given before it, and
    the injected clock. It reads nothing global and writes nothing outside
    itself.
    """

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self._last_at: float | None = None
        self._source_id: str | None = None
        self._tracks: dict[str, list[_Track]] = {
            "objects": [], "faces": [], "gestures": [],
        }

    def reset(self) -> None:
        """Forget every track. Nothing survives a camera change or a restart."""
        self._last_at = None
        for tracks in self._tracks.values():
            del tracks[:]

    def apply(self, scene: SceneFrame) -> SceneFrame:
        """Return a new SceneFrame holding eased, alpha-carrying proxies.

        The scene handed in is not touched: its lists, its raw frame and its
        metadata come back out by reference, unchanged.
        """
        source_id = getattr(scene.source, "id", None)
        if self._source_id is not None and source_id != self._source_id:
            # A different camera. Nothing on screen was about this one.
            self.reset()
        self._source_id = source_id

        now = self.clock()
        dt = None if self._last_at is None else max(0.0, now - self._last_at)
        self._last_at = now

        metadata = scene.metadata or {}
        interval_s = inference_interval_s(metadata)
        factor = 1.0 if dt is None else ease_factor(dt, interval_s)
        fade_out = fade_out_s(interval_s)

        objects = self._advance("objects", OBJECT_STAGE, scene.objects,
                                metadata, dt, factor, fade_out)
        faces = self._advance("faces", FACE_STAGE, scene.faces,
                              metadata, dt, factor, fade_out)
        gestures = self._advance("gestures", GESTURE_STAGE, scene.gestures,
                                 metadata, dt, factor, fade_out)
        return SceneFrame(
            source=scene.source,
            sequence=scene.sequence,
            captured_at=scene.captured_at,
            raw=scene.raw,
            objects=objects,
            faces=faces,
            gestures=gestures,
            metadata=scene.metadata,
            contract_version=scene.contract_version,
        )

    # ---- internals ---------------------------------------------------------

    def _advance(self, slot, stage, items, metadata, dt, factor, fade_out):
        if not metadata.get(stage.metadata_flag, True):
            # Switched off. Cleared, not faded: see the module docstring.
            del self._tracks[slot][:]
            return []

        tracks = self._tracks[slot]
        claimed: set[int] = set()
        survivors: list[_Track] = []
        rendered: list[SmoothedItem] = []

        for item in items or []:
            key = stage.key_of(item)
            if key is UNSMOOTHED:
                rendered.append(SmoothedItem(item, float(item.x), float(item.y),
                                             float(item.w), float(item.h), 1.0))
                continue
            index = _match(tracks, claimed, key, item, stage.min_iou)
            if index is None:
                # Born at its true position. On the very first frame this
                # smoother sees there is no "mid-stream" to appear out of, so
                # it starts opaque rather than transparent.
                track = _Track(key, item, alpha=(1.0 if dt is None else 0.0))
            else:
                claimed.add(index)
                track = tracks[index]
                track.item = item
                if stage.ease_position and dt is not None:
                    track.ease_towards(item, factor)
                else:
                    track.snap_to(item)
            track.alpha = _faded_in(track.alpha, dt)
            survivors.append(track)
            rendered.append(track.rendered())

        for index, track in enumerate(tracks):
            if index in claimed:
                continue
            alpha = _faded_out(track.alpha, dt, fade_out)
            if alpha <= 0.0:
                continue
            track.alpha = alpha
            survivors.append(track)
            rendered.append(track.rendered())

        self._tracks[slot] = survivors
        return rendered


def _match(tracks, claimed, key, item, min_iou):
    """Index of the track ``item`` continues, or None if it is new."""
    best_index, best_iou = None, None
    for index, track in enumerate(tracks):
        if index in claimed or track.key != key:
            continue
        if min_iou is None:
            return index
        overlap = iou(track, item)
        if overlap < min_iou:
            continue
        if best_iou is None or overlap > best_iou:
            best_index, best_iou = index, overlap
    return best_index


def _faded_in(alpha, dt):
    if dt is None or FADE_IN_S <= 0.0:
        return 1.0
    return min(1.0, alpha + dt / FADE_IN_S)


def _faded_out(alpha, dt, duration_s):
    if dt is None or duration_s <= 0.0:
        return 0.0
    return alpha - dt / duration_s
