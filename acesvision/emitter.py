"""The event bus and the gesture-event projection that travels over it.

AcesVision publishes what it *saw*. It does not know what anybody does about
it. A subscriber — AceRGB, a home-automation bridge, a logger, a notebook — is
one HTTP connection away and needs no code in this repository.

Three objects, in dependency order:

``EventBus``
    The fan-out. A monotonic ``seq``, a bounded ring of recent events for
    replay, and one bounded queue per subscriber. It performs **no I/O of any
    kind** — no sockets, no files, no logging that could block. ``publish``
    takes a short lock, appends, and ``put_nowait``s. The transport
    (``server.py``) is what talks to a socket, on the subscriber's own thread.

``PublishFilter``
    What the operator will allow onto the wire, read from
    ``~/.config/acesvision/publish.json``. Fails closed and is never
    enumerable: no endpoint lists enrolled identities, and identity publishing
    can be switched off entirely.

``GestureEmitter``
    Projects one debounced gesture event and its scene onto the
    ``acesvision.gesture/1`` wire schema, applies the filter, and publishes.

**Why the bus must never block.** The capture loop is a single owner of one
camera (``pipeline.VisionPipeline``). It hands each scene to ``_OutputWorker``
threads through a one-slot mailbox that overwrites rather than waits, so a slow
consumer costs frames, never latency. This bus keeps the same contract one
level out: a subscriber that stops reading loses its **oldest** queued events
and is told how many in the ``dropped`` field. It cannot slow the emitter, and
it cannot make the emitter forget the newest thing that happened — which is the
one a gesture subscriber actually cares about.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import socket
import stat
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path

from . import __version__
from .catalog import CATALOG, GestureCatalog

log = logging.getLogger(__name__)

#: The versioned wire schema of a gesture event. A subscriber that does not
#: know this exact string must refuse the event rather than guess.
SCHEMA_GESTURE = "acesvision.gesture/1"

#: The bootstrap frame sent once per connection. Versioned separately: the
#: handshake and the payload can move independently.
SCHEMA_HELLO = "acesvision.hello/1"

SUPPORTED_SCHEMAS = (SCHEMA_GESTURE,)

EVENT_GESTURE = "gesture"
EVENT_HELLO = "hello"

#: How many events the ring keeps for replay. At the default one-gesture-per
#: -1.5s cooldown this is roughly six minutes of history — long enough to cover
#: a subscriber restart, short enough to stay bounded.
RING_SIZE = 256

#: Per-subscriber queue depth. Beyond this the oldest queued event is dropped.
QUEUE_SIZE = 64

#: Concurrent subscribers allowed. A loopback emitter has no business feeding a
#: crowd, and an unbounded count is a trivial local resource exhaustion.
MAX_SUBSCRIBERS = 8

# --- identity_state: what the emitter knows about *who* gestured -------------
#: Exactly one enrolled face in frame. The only value a subscriber may treat as
#: an identification.
IDENTITY_IDENTIFIED = "identified"
#: No enrolled face in frame. Somebody gestured; the emitter does not know who.
IDENTITY_UNKNOWN = "unknown"
#: Several enrolled faces in frame. The emitter cannot say which one gestured,
#: and deliberately does not name the candidates.
IDENTITY_AMBIGUOUS = "ambiguous"
#: The operator has turned identity publishing off. Says nothing about the scene.
IDENTITY_DISABLED = "disabled"

#: Liveness is not implemented. The field exists so the contract has a place for
#: it and so no subscriber can mistake its absence for a pass.
LIVENESS_NOT_EVALUATED = "not_evaluated"

DEFAULT_PUBLISH_PATH = Path.home() / ".config" / "acesvision" / "publish.json"


class TooManySubscribers(RuntimeError):
    """``MAX_SUBSCRIBERS`` are already connected. The transport answers 503."""


@dataclass(frozen=True)
class EmitterIdentity:
    """Who is emitting, stamped on every event.

    ``instance`` is generated once per process start. It is what tells a
    subscriber that the emitter restarted — ``seq`` returning to 1 under a *new*
    instance is a restart, under the *same* instance it would be a bug.
    """

    id: str
    instance: str
    version: str
    host: str

    @classmethod
    def for_process(cls, emitter_id: str = "acesvision",
                    host: str | None = None) -> "EmitterIdentity":
        return cls(
            id=emitter_id,
            instance=str(uuid.uuid4()),
            version=__version__,
            host=host if host is not None else socket.gethostname(),
        )

    def as_dict(self) -> dict:
        return {"id": self.id, "instance": self.instance,
                "version": self.version, "host": self.host}


class _Closed:
    """Sentinel pushed into a subscriber queue to wake its reader on shutdown."""


CLOSED = _Closed()


class Subscription:
    """One subscriber's bounded mailbox.

    Drop-**oldest**, matching ``pipeline._OutputWorker``: when the queue is full
    the stalest event is discarded so the newest still arrives. A gesture that
    happened four seconds ago is worth less than the one that just happened, and
    a subscriber that has fallen behind is better off current than complete.

    ``dropped`` is monotonic per subscription and is stamped onto every event
    this subscriber receives, so a gap is visible in-band rather than having to
    be inferred.
    """

    def __init__(self, bus: "EventBus", maxsize: int = QUEUE_SIZE):
        self.id = str(uuid.uuid4())
        self.bus = bus
        self.dropped = 0
        self.closed = False
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)

    def offer(self, payload: dict) -> None:
        """Called by the publisher, under no lock of the subscriber's. Never blocks."""
        while True:
            try:
                self._queue.put_nowait(payload)
                return
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:      # a reader drained it in between
                    continue
                self.dropped += 1

    def get(self, timeout: float | None = None) -> dict | None:
        """The next event, or None on timeout or close.

        Runs on the subscriber's own thread. The ``dropped`` stamp is applied
        here — the same ``seq`` handed to two subscribers may carry two
        different counts, because it describes *that* subscriber's gap.
        """
        try:
            payload = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if payload is CLOSED:
            return None
        return dict(payload, dropped=self.dropped)

    def close(self) -> None:
        """Wake the reader immediately instead of leaving it on its timeout."""
        self.closed = True
        try:
            self._queue.put_nowait(CLOSED)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(CLOSED)
            except (queue.Empty, queue.Full):
                pass


class EventBus:
    """Monotonic sequence, bounded replay ring, bounded per-subscriber queues.

    Thread-safe. ``publish`` is called from the ``GestureEventOutput`` worker
    thread, never from the capture thread, and holds the lock only long enough
    to stamp a sequence number and hand the payload out.
    """

    def __init__(self, ring_size: int = RING_SIZE, queue_size: int = QUEUE_SIZE,
                 max_subscribers: int = MAX_SUBSCRIBERS):
        self.queue_size = queue_size
        self.max_subscribers = max_subscribers
        self._lock = threading.Lock()
        self._seq = 0
        self._ring: deque = deque(maxlen=ring_size)
        self._subscriptions: list[Subscription] = []

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscriptions)

    def publish(self, payload: dict) -> dict:
        """Stamp a sequence number, ring it, and offer it to every subscriber.

        Returns the stamped payload. Sequence numbers start at 1, so ``seq``
        0 unambiguously means "nothing has been published by this instance".
        """
        with self._lock:
            self._seq += 1
            stamped = dict(payload, seq=self._seq, dropped=0)
            self._ring.append(stamped)
            subscriptions = tuple(self._subscriptions)
        for subscription in subscriptions:
            subscription.offer(stamped)
        return stamped

    def subscribe(self, since: int | None = None):
        """Register a subscriber and take its replay slice atomically.

        Returns ``(subscription, replay, seq)``. Both halves happen under one
        lock on purpose: registering first and reading the ring afterwards would
        duplicate anything published in between, and reading first would lose
        it. The seam between history and live traffic is the one place a replay
        protocol usually leaks, so it is closed here rather than in the
        transport.

        ``replay`` is the events with ``seq > since`` still in the ring.
        """
        with self._lock:
            if len(self._subscriptions) >= self.max_subscribers:
                raise TooManySubscribers(
                    f"{len(self._subscriptions)} subscribers already connected "
                    f"(limit {self.max_subscribers})")
            subscription = Subscription(self, maxsize=self.queue_size)
            self._subscriptions.append(subscription)
            replay = self._slice(since)
            return subscription, replay, self._seq

    def unsubscribe(self, subscription: Subscription) -> None:
        with self._lock:
            if subscription in self._subscriptions:
                self._subscriptions.remove(subscription)
        subscription.close()

    def close(self) -> None:
        """Wake and release every subscriber. Used on shutdown."""
        with self._lock:
            subscriptions = tuple(self._subscriptions)
            self._subscriptions.clear()
        for subscription in subscriptions:
            subscription.close()

    def replay_since(self, since: int | None):
        """The ring slice after ``since``, for the polling fallback."""
        with self._lock:
            return self._slice(since)

    def oldest_seq(self) -> int:
        """Lowest ``seq`` still replayable, or 0 when the ring is empty."""
        with self._lock:
            return self._ring[0]["seq"] if self._ring else 0

    def _slice(self, since):
        """Ring entries after ``since``. Caller holds the lock."""
        if since is None:
            return []
        return [event for event in self._ring if event["seq"] > since]


@dataclass(frozen=True)
class PublishFilter:
    """What the operator allows onto the wire.

    Read from ``~/.config/acesvision/publish.json``; every field has a
    conservative default so a missing file is a working, loopback-only,
    identity-publishing emitter — the behaviour that already existed — and a
    *present* file can only narrow it.

    ``gestures`` distinguishes absent from empty on purpose. ``None`` (the key
    is missing or null) means "no allowlist, publish the whole vocabulary". A
    list means "exactly these", and an empty list therefore means "none" rather
    than "all". An allowlist that silently meant its own opposite is not a
    mistake worth being tolerant about.
    """

    enabled: bool = True
    gestures: tuple[str, ...] | None = None
    min_confidence: float = 0.0
    publish_identity: bool = True
    publish_source_label: bool = True
    #: ``(raw_name, reason)`` for allowlist entries that are not in the catalog.
    #: Quarantined rather than raised: an unknown name in an allowlist can only
    #: ever fail closed, and one typo should not take the emitter down.
    rejected: tuple = ()

    @classmethod
    def from_mapping(cls, raw, catalog: GestureCatalog = CATALOG) -> "PublishFilter":
        if not isinstance(raw, dict):
            raise ValueError("publish filter must be a JSON object")
        allowed = raw.get("gestures")
        rejected = []
        if allowed is None:
            gestures = None
        elif isinstance(allowed, list):
            names = []
            for name in allowed:
                canonical = catalog.normalise(name)
                if canonical is None:
                    rejected.append((name, "not in the gesture catalog"))
                elif canonical not in names:
                    names.append(canonical)
            gestures = tuple(names)
        else:
            raise ValueError("publish filter 'gestures' must be an array or null")

        return cls(
            enabled=bool(raw.get("enabled", True)),
            gestures=gestures,
            min_confidence=float(raw.get("min_confidence", 0.0) or 0.0),
            publish_identity=bool(raw.get("publish_identity", True)),
            publish_source_label=bool(raw.get("publish_source_label", True)),
            rejected=tuple(rejected),
        )

    @classmethod
    def load(cls, path=None, catalog: GestureCatalog = CATALOG) -> "PublishFilter":
        """Read the filter file, or return the defaults when it does not exist."""
        path = Path(path or DEFAULT_PUBLISH_PATH)
        if not path.exists():
            return cls()
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")),
                                catalog=catalog)

    def without_identity(self) -> "PublishFilter":
        """The same filter with identity publishing forced off.

        Applied when the emitter is bound to a non-loopback address: naming an
        enrolled person to whatever is on the other end of a LAN socket is a
        different decision from naming them to a process on the same machine,
        and it is not one a config file default gets to make.
        """
        return replace(self, publish_identity=False)

    def allows(self, gesture: str, confidence: float):
        """``(allowed, reason)``. ``reason`` is empty when allowed."""
        if not self.enabled:
            return False, "publishing disabled"
        if self.gestures is not None and gesture not in self.gestures:
            return False, "gesture not in the publish allowlist"
        if float(confidence or 0.0) < self.min_confidence:
            return False, (f"confidence {float(confidence or 0.0):.3f} below "
                           f"min_confidence {self.min_confidence:.3f}")
        return True, ""

    def as_dict(self) -> dict:
        """The filter as advertised in the hello frame.

        Deliberately never lists identities — only whether identity publishing
        is on. ``gestures`` is a vocabulary, not personal data.
        """
        return {
            "enabled": self.enabled,
            "gestures": list(self.gestures) if self.gestures is not None else None,
            "min_confidence": self.min_confidence,
            "publish_identity": self.publish_identity,
            "publish_source_label": self.publish_source_label,
        }


def identity_state(attribution: str, publish_identity: bool) -> str:
    """Project ``events.attribute_actor``'s attribution onto the wire enum.

    The event that goes to local rules keeps all four attributions and the
    candidate names. The wire keeps less, on purpose:

    ``unique``    -> identified. One enrolled face; the only certain case.
    ``nearest``   -> ambiguous.  Several enrolled faces, separated by which one
                    was closest to the gesturing hand. ``attribute_actor``
                    already says that is "made under ambiguity rather than read
                    as certainty" — so it is not published as an identification,
                    and a subscriber gated on ``identified`` will not act on a
                    proximity guess.
    ``ambiguous`` -> ambiguous.  Several enrolled faces, no usable geometry.
    ``none``      -> unknown.    Nobody enrolled in frame.

    ``publish_identity=False`` overrides all four with ``disabled``, which says
    what the operator chose and nothing about the scene.
    """
    if not publish_identity:
        return IDENTITY_DISABLED
    if attribution == "unique":
        return IDENTITY_IDENTIFIED
    if attribution in ("nearest", "ambiguous"):
        return IDENTITY_AMBIGUOUS
    return IDENTITY_UNKNOWN


def source_payload(source, publish_label: bool) -> dict:
    """The ``source`` block. Never the raw URL.

    ``SourceSpec.safe_label`` strips userinfo and query from a network URL, so
    an RTSP camera's embedded credentials cannot reach a subscriber. With
    ``publish_source_label`` off the label is null as well — the id and kind are
    enough to route on and neither carries a hostname.
    """
    return {
        "id": source.id,
        "kind": source.kind,
        "label": source.safe_label() if publish_label else None,
        "trusted_device": bool(source.trusted_device),
    }


class GestureEmitter:
    """Projects debounced gesture events onto the wire and publishes them.

    Wired into ``events.GestureEventOutput``, which already runs on an
    ``_OutputWorker`` thread rather than the capture thread. The projection is
    pure and the publish is non-blocking, so nothing here can cost a frame.
    """

    def __init__(self, bus: EventBus, *, identity: EmitterIdentity | None = None,
                 catalog: GestureCatalog = CATALOG,
                 publish_filter: PublishFilter | None = None,
                 publishing: bool = True, clock=time.time):
        self.bus = bus
        self.identity = identity or EmitterIdentity.for_process()
        self.catalog = catalog
        self.filter = publish_filter or PublishFilter()
        #: False under ``--no-emit``: subscribers still connect and still get a
        #: hello and keepalives, so "switched off" reads differently from "dead".
        self.publishing = bool(publishing)
        self.clock = clock
        self.suppressed = 0

    def build(self, event: dict, scene) -> dict | None:
        """The ``acesvision.gesture/1`` payload, or None if the filter blocks it.

        ``seq`` and ``dropped`` are absent here — the bus stamps the first and
        each subscription stamps the second.
        """
        gesture = str(event.get("gesture") or "")
        confidence = float(event.get("confidence") or 0.0)
        allowed, reason = self.filter.allows(gesture, confidence)
        if not allowed:
            self.suppressed += 1
            log.debug("gesture %s not published: %s", gesture, reason)
            return None

        state = identity_state(str(event.get("actor_attribution") or "none"),
                               self.filter.publish_identity)
        # The actor's name rides along only when the emitter is certain and the
        # operator allows it. Under `ambiguous` the candidates are not listed:
        # naming two people to say it might be either of them publishes both.
        actor = event.get("actor") if state == IDENTITY_IDENTIFIED else None

        return {
            "schema": SCHEMA_GESTURE,
            "type": EVENT_GESTURE,
            "emitted_at": self.clock(),
            "emitter": self.identity.as_dict(),
            "catalog": self.catalog.stamp(),
            "gesture": gesture,
            "confidence": confidence,
            "held_frames": int(event.get("held_frames") or 0),
            "actor": actor,
            "identity_state": state,
            "liveness_state": str(event.get("liveness_state")
                                  or LIVENESS_NOT_EVALUATED),
            "security_authorized": bool(event.get("security_authorized", False)),
            "source": source_payload(scene.source,
                                     self.filter.publish_source_label),
            "frame_sequence": int(scene.sequence),
            "captured_at_monotonic": float(event.get("captured_at_monotonic")
                                           or scene.captured_at),
        }

    def publish_gesture(self, event: dict, scene) -> dict | None:
        """Build and publish. Returns the stamped payload, or None."""
        if not self.publishing:
            return None
        payload = self.build(event, scene)
        if payload is None:
            return None
        return self.bus.publish(payload)

    def hello(self, seq: int, requested_since: int | None = None,
              oldest_seq: int = 0) -> dict:
        """The bootstrap frame: one round trip and a subscriber knows everything.

        Identity, the whole catalog, the current sequence, the schemas this
        emitter speaks, and — when replay was requested — whether the ring could
        actually satisfy it. ``gap: true`` means events were lost before the
        subscriber connected, which is a fact it can act on; the alternative is
        silently handing it a short history that looks complete.
        """
        payload = {
            "schema": SCHEMA_HELLO,
            "type": EVENT_HELLO,
            "emitted_at": self.clock(),
            "emitter": self.identity.as_dict(),
            "catalog": self.catalog.as_payload(),
            "seq": seq,
            "supported_schemas": list(SUPPORTED_SCHEMAS),
            "publishing": self.publishing,
            "publish_filter": self.filter.as_dict(),
        }
        if requested_since is not None:
            payload["replay"] = {
                "requested_since": requested_since,
                "oldest_available": oldest_seq,
                # A ring that has never held the requested point cannot prove
                # it delivered everything after it.
                "gap": bool(oldest_seq and requested_since + 1 < oldest_seq),
            }
        return payload


def harden_directory(path: Path) -> None:
    """Make sure a directory holding a secret is 0700. Only ever narrows.

    ``mkdir(mode=0o700, exist_ok=True)`` applies the mode **only when it creates
    the directory** — an existing one keeps whatever it had, silently. Since
    ``~/.config/acesvision`` is usually created by the rule store long before
    the first token lands in it, the mode argument alone would have been a
    comment that looked like a control.
    """
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        path.chmod(mode & ~(stat.S_IRWXG | stat.S_IRWXO))
        log.info("tightened %s from %o to 0700; it holds the emitter token",
                 path, mode)


def write_secret_file(path: Path, content: str) -> None:
    """Create a 0600 file inside a 0700 directory, failing if it already exists.

    ``O_EXCL`` so a symlink planted at the path is refused rather than followed,
    and the mode is passed to ``open`` rather than chmod-ed afterwards so the
    file is never briefly world-readable.
    """
    harden_directory(path.parent)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def read_secret_file(path: Path) -> str:
    """Read a file that must not be readable by anyone else.

    A token the rest of the machine can read is not a token, so a permissive
    mode is refused loudly instead of being silently repaired — repairing it
    would hide that something already had the chance to read it.
    """
    mode = path.stat().st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(
            f"{path} is accessible to group or others (mode "
            f"{stat.filemode(mode)}). Refusing to use it: delete it and let "
            "AcesVision generate a new one.")
    return path.read_text(encoding="utf-8").strip()
