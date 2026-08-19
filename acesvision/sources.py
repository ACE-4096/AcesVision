"""Capture adapters for local webcams and DroidCam/network feeds."""
from __future__ import annotations

import sys
import threading
from typing import Any, Callable

import cv2

import camera as local_camera

from .contracts import SourceSpec


class LatestFrameReader:
    """Pull and decode a network capture on its own thread, keeping the newest
    frame only.

    Presents the small slice of the cv2.VideoCapture interface the pipeline
    uses — isOpened/read/release/set/get — so VisionPipeline needs no knowledge
    that it exists, and its source-switching logic is untouched: a switch still
    releases the capture, which stops this thread with it.

    ### Why

    A network capture read straight from the capture loop delivers frames *in
    order and without dropping any*. When the consumer is slower than the phone
    — which it always is, because the consumer is YOLO plus ArcFace — the
    frames it has not read yet do not disappear. They queue, in the socket
    buffer and in the FFMPEG demuxer, and every frame the pipeline then draws a
    box on is further into the past than the one before it. Measured against a
    18.9 FPS 720p MJPEG source with a 90 ms consumer: throughput 10.9 FPS, no
    frames skipped, and the delivered frame **8.45 seconds behind live after a
    20 second run**, still growing. Face recognition on an eight-second-old
    frame is not slow, it is wrong.

    Decode cost is a real but secondary part of that: a 1280x720 JPEG costs
    3.45 ms to decode on this host, a 290 FPS ceiling. Moving decode off the
    capture loop is worth doing and this does it, but the reason this class
    exists is the staleness, not the milliseconds.

    ### The contract

    Drop-old, exactly as ``_OutputWorker`` in pipeline.py and ``EventBus`` in
    emitter.py already do it: one slot, newest wins, and the count of what was
    discarded is kept rather than hidden. The producer never waits for the
    consumer, and the consumer never receives a frame older than the newest one
    that has arrived.

    MJPEG is what makes this legitimate. Every frame is independently coded, so
    discarding one costs nothing and corrupts nothing. That property is the
    reason not to "modernise" this to H.264: an H.264 stream carries decoder
    state across frames, a dropped frame is a reference some later frame needs,
    and drop-old — the contract this whole pipeline is built on — stops being
    free. Lower bandwidth would be paid for in the one behaviour that matters
    here.

    ### Where it is not used

    Webcams. ``VisionPipeline._apply_camera_controls`` calls ``cap.set()`` on
    the capture while the loop runs; OpenCV's VideoCapture is not safe to
    ``set()`` on one thread while another ``read()``s it. Network sources take
    no camera controls, so the hazard does not arise there. The boundary is
    deliberate, not an oversight.
    """

    def __init__(self, capture, *, read_timeout_s: float = 5.0):
        self._capture = capture
        self._read_timeout_s = read_timeout_s
        self._condition = threading.Condition()
        self._latest = None
        self._sequence = 0
        self._delivered = 0
        self._failed = False
        self._stopping = False
        self.dropped = 0
        self._thread = threading.Thread(target=self._pump, daemon=True,
                                        name="vision-source-reader")
        self._thread.start()

    def _pump(self) -> None:
        try:
            while True:
                with self._condition:
                    if self._stopping:
                        return
                ok, frame = self._capture.read()
                with self._condition:
                    if self._stopping:
                        return
                    if not ok or frame is None:
                        self._failed = True
                        self._condition.notify_all()
                        return
                    if self._latest is not None:
                        # The consumer never saw the frame in the slot. That is
                        # the design, and it is counted rather than silent.
                        self.dropped += 1
                    self._latest = frame
                    self._sequence += 1
                    self._condition.notify_all()
        except Exception:                   # noqa: BLE001 - a dead source is a
            with self._condition:           # reconnect, never a crashed thread
                self._failed = True
                self._condition.notify_all()

    def isOpened(self) -> bool:             # noqa: N802 - cv2 spelling
        with self._condition:
            if self._failed or self._stopping:
                return False
        return bool(self._capture.isOpened())

    def read(self):
        """The newest frame that has arrived since the last call.

        Waits only when nothing new has arrived yet, which is ordinary idling
        on a live source and not backpressure — a frame already in the slot is
        returned immediately. Returns ``(False, None)`` when the source has
        died or has gone quiet for longer than ``read_timeout_s``, which is the
        signal VisionPipeline already treats as "reconnect".
        """
        deadline = None
        with self._condition:
            while True:
                if self._sequence != self._delivered and self._latest is not None:
                    frame = self._latest
                    self._latest = None
                    self._delivered = self._sequence
                    return True, frame
                if self._failed or self._stopping:
                    return False, None
                if deadline is None:
                    deadline = self._read_timeout_s
                if not self._condition.wait(timeout=deadline):
                    return False, None

    def release(self) -> None:
        with self._condition:
            self._stopping = True
            self._latest = None
            self._condition.notify_all()
        # The pump may be parked inside a blocking capture.read(); releasing the
        # capture is what unblocks it, so never wait for the thread first.
        try:
            self._capture.release()
        finally:
            self._thread.join(timeout=2.0)

    def set(self, *args):
        return self._capture.set(*args)

    def get(self, *args):
        return self._capture.get(*args)


def open_source(
    source: SourceSpec,
    *,
    webcam_opener: Callable[[int | str | None], Any] | None = None,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
    on_error: Callable[[str], None] | None = None,
    reader_factory: Callable[[Any], Any] | None = LatestFrameReader,
):
    """Open a source and return a capture, or None when it is unavailable.

    None is still the "unavailable" signal the pipeline retries on, but the
    reason is never discarded: camera.py distinguishes busy from missing and
    that text reaches on_error (or stderr). A device held by another process
    used to surface as "No working camera found", which read as a broken app
    rather than contention.

    Network sources come back wrapped in a LatestFrameReader; pass
    ``reader_factory=None`` for the raw, in-order capture.
    """
    if source.kind == "webcam":
        try:
            device = source.device_path or source.index
            if webcam_opener is not None:
                return webcam_opener(device)
            return local_camera.open_camera(device, manual_exposure=False)
        except (RuntimeError, ValueError, OSError) as exc:
            _report(on_error, source, str(exc))
            return None

    cap = capture_factory(source.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        _report(on_error, source, "capture did not open")
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap if reader_factory is None else reader_factory(cap)


def _report(on_error, source, reason):
    if on_error is not None:
        on_error(reason)
    else:
        print(f"[source] {source.id}: {reason}", file=sys.stderr)
