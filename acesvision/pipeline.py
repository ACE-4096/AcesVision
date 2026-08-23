"""Single-owner capture loop and per-output frame fan-out.

Two fan-out modes, one worker class
-----------------------------------
``_OutputWorker`` is a one-slot **drop-old** mailbox by default, and that is
right for everything that draws a picture: a preview or a virtual camera wants
the *newest* frame, and a frame it could not keep up with is worth less than
the one behind it. Producers never wait for those consumers.

A recorder is the opposite. Drop-old on a recorder writes a file with silent
time gaps in it — the encoder falls behind for 200 ms, the frames from that
window are simply never written, and nothing anywhere says so. So
``add_output(output, lossless=True)`` swaps the one slot for a bounded
``queue.Queue`` and a blocking ``put``. A slow encoder then applies
backpressure to the capture loop: capture fps visibly dips, the recording stays
continuous, and every recorded frame is a real one.

The queue is bounded rather than infinite on purpose. Unbounded, an encoder
that has genuinely wedged would grow the queue until the machine died, and the
frames in it would be minutes stale by the time they were written. Bounded, a
wedged encoder eventually forces a decision, and the decision this module makes
is: drop the frame, **count it**, publish the count in pipeline metrics, and
tell the output itself so it can put the number in its own records. Silent loss
is the one outcome that is not allowed.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import cv2

from .contracts import SceneFrame, SourceSpec
from .sources import open_source

#: Frames a lossless worker will hold for a slow output before it has to
#: choose between blocking the capture loop and admitting a loss. ~2 s at
#: 60 fps, which is long enough to ride out an encoder hiccup and short enough
#: that a wedged one is noticed rather than buffered into next week.
LOSSLESS_QUEUE_FRAMES = 120

#: How long ``submit`` blocks the capture loop when that queue is full. This
#: *is* the backpressure: the capture loop waits here, so capture fps dips and
#: the operator sees it. Past this, the encoder is not busy, it is wedged.
LOSSLESS_PUT_TIMEOUT_S = 0.05

#: How long a stopping lossless worker gets to drain what is already queued
#: before the rest is counted as dropped. A recorder's queued frames are real
#: footage; throwing them away at stop would put the gap at the end of the file
#: instead of the middle, which is no better.
LOSSLESS_DRAIN_TIMEOUT_S = 5.0

#: How often a lossless worker wakes to notice ``stop()``. A ``queue.Queue``
#: has no "wait for an item or a flag" primitive, and a sentinel ``put`` would
#: itself block on the full queue this exists to survive.
LOSSLESS_POLL_S = 0.05

#: Seconds ``VisionPipeline`` waits for a worker to finish at shutdown. A
#: drop-old worker has at most one frame left and is done immediately; a
#: lossless one may have a queue to drain and an encoder to finalise.
DROP_OLD_JOIN_S = 2.0
LOSSLESS_JOIN_S = LOSSLESS_DRAIN_TIMEOUT_S + 2.0

# Rotation is deliberately applied before perception rather than just painted
# onto the preview. That keeps boxes, hand/body landmarks, recordings and
# sidecar geometry in one coordinate system — essential for portrait footage.
ROTATION_DEGREES = (0, 90, 180, 270)


def rotate_frame(frame, degrees: int):
    """Return ``frame`` in the requested clockwise presentation orientation."""
    degrees = int(degrees)
    if degrees == 0:
        return frame
    if degrees == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError("rotation must be one of 0, 90, 180, or 270 degrees")


class FrameOutput(Protocol):
    def publish(self, scene: SceneFrame) -> None: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class PipelineState:
    status: str
    source: SourceSpec
    sequence: int
    last_error: str
    metrics: dict[str, Any]


class _OutputWorker(threading.Thread):
    """One output's mailbox and delivery thread.

    Default (``lossless=False``): a one-slot drop-old mailbox. Producers never
    wait for slow frame consumers, and a frame that arrives while the previous
    one is still unread replaces it and is counted in ``dropped``.

    ``lossless=True``: a bounded queue and a blocking ``submit``, so the
    capture loop waits for the output instead of overwriting its work. See the
    module docstring for why a recorder needs the second mode and a preview
    does not.

    ``dropped`` means the same thing in both modes — frames this output was
    never given — but it means something very different about the system. On a
    drop-old worker it is the normal operating state. On a lossless worker it
    is a fault, and it is published as one.
    """

    def __init__(self, output: FrameOutput, *, lossless: bool = False,
                 maxsize: int = LOSSLESS_QUEUE_FRAMES,
                 put_timeout_s: float = LOSSLESS_PUT_TIMEOUT_S,
                 drain_timeout_s: float = LOSSLESS_DRAIN_TIMEOUT_S):
        super().__init__(daemon=True, name=f"vision-output-{type(output).__name__}")
        self.output = output
        self.lossless = bool(lossless)
        self.last_error = ""
        self.dropped = 0
        self._put_timeout_s = put_timeout_s
        self._drain_timeout_s = drain_timeout_s
        self._drain_deadline = 0.0
        # Exactly one of these two is live, decided at construction. A worker
        # never changes mode: an output that swapped mailboxes mid-run would
        # have frames in the old one with nothing left to read them.
        self._queue: queue.Queue | None = (
            queue.Queue(maxsize=max(1, int(maxsize))) if self.lossless else None
        )
        self._condition = threading.Condition()
        self._latest: SceneFrame | None = None
        self._stopping = False
        # The drop counter is the one number two threads both write: the
        # capture thread on a full queue, the worker thread when it runs out
        # of drain time. Losing an increment to a lost-update race would
        # under-report loss, which is the failure this whole mode exists to
        # prevent, so it is not left to the GIL.
        self._drop_lock = threading.Lock()

    @property
    def output_name(self) -> str:
        return type(self.output).__name__

    @property
    def join_timeout_s(self) -> float:
        return LOSSLESS_JOIN_S if self.lossless else DROP_OLD_JOIN_S

    def submit(self, scene: SceneFrame) -> None:
        if self._queue is None:
            with self._condition:
                if self._latest is not None:
                    self.dropped += 1
                self._latest = scene
                self._condition.notify()
            return
        try:
            self._queue.put(scene, timeout=self._put_timeout_s)
        except queue.Full:
            # Waited out the whole backpressure budget and the consumer has not
            # taken one frame. That is not "busy", that is wedged.
            self._count_drop()

    def stop(self) -> None:
        # Set the deadline before the flag, so the draining consumer can never
        # observe "stopping" with a deadline of zero already in the past.
        self._drain_deadline = time.monotonic() + self._drain_timeout_s
        with self._condition:
            self._stopping = True
            self._condition.notify()

    def _count_drop(self) -> None:
        """Count a frame this output never saw, and tell the output about it.

        The count lives here because only the worker knows a frame was refused;
        the output is offered it because only the output writes the file that
        has the gap in it. ``note_dropped`` is optional — it is not part of the
        ``FrameOutput`` protocol, and an output that does not care about loss
        (a preview) does not have to grow a method to say so.
        """
        with self._drop_lock:
            self.dropped += 1
        note = getattr(self.output, "note_dropped", None)
        if note is None:
            return
        try:
            note(1)
        except Exception as exc:
            self.last_error = str(exc)

    def _deliver(self, scene: SceneFrame) -> None:
        try:
            self.output.publish(scene)
            self.last_error = ""
        except Exception as exc:
            self.last_error = str(exc)

    def run(self) -> None:
        try:
            if self._queue is None:
                self._run_drop_old()
            else:
                self._run_lossless()
        finally:
            # Both modes land here, and a recorder depends on it: this is where
            # ffmpeg's stdin is closed and the moov atom gets written. A
            # lossless worker that skipped this would leave an unplayable file.
            try:
                self.output.close()
            except Exception as exc:
                self.last_error = str(exc)

    def _run_drop_old(self) -> None:
        while True:
            with self._condition:
                while self._latest is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    break
                scene = self._latest
                self._latest = None
            self._deliver(scene)

    def _drain_abandoned(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
            self._count_drop()

    def _run_lossless(self) -> None:
        while True:
            try:
                scene = self._queue.get(timeout=LOSSLESS_POLL_S)
            except queue.Empty:
                if self._stopping:
                    # A producer can still be inside a blocking submit() as
                    # this returns — remove_output stops a worker while the
                    # capture loop is running. Anything that lands after this
                    # point is abandoned, so account for it rather than let it
                    # disappear.
                    self._drain_abandoned()
                    return
                continue
            if self._stopping and time.monotonic() > self._drain_deadline:
                # Out of drain time, but the queue still has to be emptied
                # rather than abandoned: every frame in it is accounted for as
                # a drop, so the tail of the recording is short by a number
                # somebody can read rather than by an amount nobody can.
                self._count_drop()
                continue
            self._deliver(scene)


class VisionPipeline(threading.Thread):
    """Own one switchable camera and publish each scene to independent outputs."""

    def __init__(
        self,
        source: SourceSpec,
        processor: Callable[[Any, SourceSpec, int, float], SceneFrame],
        outputs: list[FrameOutput] | None = None,
        *,
        opener: Callable[[SourceSpec], Any] | None = None,
        retry_min_s: float = 0.5,
        retry_max_s: float = 10.0,
    ):
        super().__init__(daemon=True, name="vision-capture")
        self.processor = processor
        self._open_error = ""
        self.opener = opener or self._default_opener
        self.retry_min_s = retry_min_s
        self.retry_max_s = retry_max_s
        self._source = source
        self._source_generation = 0
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._status = "stopped"
        self._sequence = 0
        self._last_error = ""
        self._metrics = {"capture_fps": 0.0, "rotation_degrees": 0}
        self._rotation_degrees = 0
        self._camera_controls = {
            "auto_exposure": True,
            "exposure": 166,
            "brightness": 0,
            "contrast": 0,
            "gamma": 100,
        }
        self._camera_control_generation = 0
        self._workers = [_OutputWorker(out) for out in (outputs or [])]
        self._workers_lock = threading.Lock()
        self._started_workers = False

    def _default_opener(self, source):
        """open_source, with the open failure reason routed into pipeline state."""
        return open_source(source, on_error=self._note_open_error)

    def _note_open_error(self, reason):
        self._open_error = str(reason)

    def add_output(self, output: FrameOutput, lossless: bool = False) -> None:
        """Fan this pipeline's scenes out to ``output`` on its own thread.

        ``lossless=True`` gives the output a bounded queue and backpressure
        instead of the one-slot drop-old mailbox. Use it for anything that
        writes a file; leave it off for anything that draws a picture.
        """
        worker = _OutputWorker(output, lossless=lossless)
        with self._workers_lock:
            self._workers.append(worker)
            should_start = self._started_workers
        if should_start:
            worker.start()

    def remove_output(self, output: FrameOutput) -> bool:
        with self._workers_lock:
            worker = next((w for w in self._workers if w.output is output), None)
            if worker is None:
                return False
            self._workers.remove(worker)
        worker.stop()
        if worker.is_alive():
            worker.join(timeout=worker.join_timeout_s)
        return True

    def switch_source(self, source: SourceSpec) -> None:
        with self._state_lock:
            self._source = source
            self._source_generation += 1

    def set_camera_controls(self, *, auto_exposure=True, exposure=166,
                            brightness=0, contrast=0, gamma=100) -> None:
        controls = {
            "auto_exposure": bool(auto_exposure),
            "exposure": max(3, min(333, int(exposure))),
            "brightness": max(-64, min(64, int(brightness))),
            "contrast": max(0, min(95, int(contrast))),
            "gamma": max(100, min(300, int(gamma))),
        }
        with self._state_lock:
            self._camera_controls = controls
            self._camera_control_generation += 1

    def set_rotation(self, degrees: int) -> None:
        """Orient frames before perception and every downstream output."""
        degrees = int(degrees)
        if degrees not in ROTATION_DEGREES:
            raise ValueError("rotation must be one of 0, 90, 180, or 270 degrees")
        with self._state_lock:
            self._rotation_degrees = degrees

    def state(self) -> PipelineState:
        with self._state_lock:
            return PipelineState(self._status, self._source,
                                 self._sequence, self._last_error,
                                 dict(self._metrics))

    def stop(self) -> None:
        self._stop_event.set()

    def _set_state(self, *, status=None, error=None, sequence=None,
                   metrics=None) -> None:
        with self._state_lock:
            if status is not None:
                self._status = status
            if error is not None:
                self._last_error = error
            if sequence is not None:
                self._sequence = sequence
            if metrics is not None:
                self._metrics = dict(metrics)

    def run(self) -> None:
        with self._workers_lock:
            workers = list(self._workers)
            self._started_workers = True
        for worker in workers:
            worker.start()
        cap = None
        opened_generation = -1
        applied_control_generation = -1
        retry_s = self.retry_min_s
        fps_started = time.monotonic()
        fps_frames = 0
        capture_fps = 0.0
        self._set_state(status="connecting")
        try:
            while not self._stop_event.is_set():
                with self._state_lock:
                    source = self._source
                    generation = self._source_generation
                    rotation_degrees = self._rotation_degrees

                if generation != opened_generation and cap is not None:
                    cap.release()
                    cap = None
                    applied_control_generation = -1

                if cap is None or not cap.isOpened():
                    self._set_state(status="reconnecting")
                    self._open_error = ""
                    cap = self.opener(source)
                    opened_generation = generation
                    if cap is None or not cap.isOpened():
                        if cap is not None:
                            cap.release()
                        cap = None
                        # Prefer the specific reason (busy vs missing) over the
                        # generic one; the generic text hid device contention.
                        reason = self._open_error or "is unavailable or in use"
                        self._set_state(
                            error=f"{source.safe_label()} {reason}"
                        )
                        if self._stop_event.wait(retry_s):
                            break
                        retry_s = min(retry_s * 2, self.retry_max_s)
                        continue
                    retry_s = self.retry_min_s

                with self._state_lock:
                    control_generation = self._camera_control_generation
                    camera_controls = dict(self._camera_controls)
                if (source.kind == "webcam" and
                        control_generation != applied_control_generation):
                    self._apply_camera_controls(cap, camera_controls)
                    applied_control_generation = control_generation

                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.release()
                    cap = None
                    self._set_state(status="reconnecting")
                    continue

                frame = rotate_frame(frame, rotation_degrees)

                captured_at = time.monotonic()
                try:
                    scene = self.processor(frame, source, self._sequence, captured_at)
                except Exception as exc:
                    self._set_state(error=f"processor: {exc}")
                    self._sequence += 1
                    continue
                scene.metadata = {**scene.metadata,
                                  "rotation_degrees": rotation_degrees}

                with self._workers_lock:
                    workers = tuple(self._workers)
                for worker in workers:
                    worker.submit(scene)
                self._sequence += 1
                fps_frames += 1
                now = time.monotonic()
                if now - fps_started >= 0.5:
                    capture_fps = fps_frames / (now - fps_started)
                    fps_started, fps_frames = now, 0
                metrics = {"capture_fps": capture_fps}
                if self._sequence % 30 == 0:
                    quality = self._frame_quality(frame, camera_controls)
                    with self._state_lock:
                        self._metrics.update(quality)
                with self._state_lock:
                    quality_metrics = {
                        key: self._metrics[key] for key in
                        ("image_mean", "image_std", "image_warning")
                        if key in self._metrics
                    }
                metrics.update(quality_metrics)
                metrics.update(self._output_drop_metrics(workers))
                metrics["camera_controls"] = camera_controls
                metrics["rotation_degrees"] = rotation_degrees
                processor_metrics = getattr(self.processor, "metrics", None)
                if processor_metrics is not None:
                    metrics.update(processor_metrics())
                self._set_state(status="live", error="", sequence=self._sequence,
                                metrics=metrics)
        finally:
            if cap is not None:
                cap.release()
            with self._workers_lock:
                workers = tuple(self._workers)
                self._started_workers = False
            for worker in workers:
                worker.stop()
            for worker in workers:
                worker.join(timeout=worker.join_timeout_s)
            close_processor = getattr(self.processor, "close", None)
            if close_processor is not None:
                close_processor()
            self._set_state(status="stopped")

    @staticmethod
    def _output_drop_metrics(workers):
        """Frames a lossless output was refused, published for the GUI.

        Only lossless workers are counted. A drop-old worker sheds frames as
        its entire design — a preview behind a 60 fps camera drops most of them
        and is working perfectly — so folding those into one total would report
        a healthy system as lossy and bury the number that actually means
        something.
        """
        drops: dict[str, int] = {}
        seen: dict[str, int] = {}
        for worker in workers:
            if not worker.lossless:
                continue
            base = worker.output_name
            seen[base] = seen.get(base, 0) + 1
            # Two recorders would otherwise collide on one key and one of the
            # two counts would vanish, which is the exact failure this is here
            # to prevent.
            name = base if seen[base] == 1 else f"{base}-{seen[base]}"
            drops[name] = worker.dropped
        return {
            "dropped_output_frames": sum(drops.values()),
            "output_drops": drops,
        }

    @staticmethod
    def _apply_camera_controls(cap, controls):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                3 if controls["auto_exposure"] else 1)
        if not controls["auto_exposure"]:
            cap.set(cv2.CAP_PROP_EXPOSURE, controls["exposure"])
        cap.set(cv2.CAP_PROP_BRIGHTNESS, controls["brightness"])
        cap.set(cv2.CAP_PROP_CONTRAST, controls["contrast"])
        cap.set(cv2.CAP_PROP_GAMMA, controls["gamma"])

    @staticmethod
    def _frame_quality(frame, controls=None):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, std = cv2.meanStdDev(gray)
        mean_value, std_value = float(mean[0][0]), float(std[0][0])
        controls = controls or {}
        if mean_value < 25 and std_value < 3 and controls.get("brightness", 0) <= -48:
            warning = "Brightness is near minimum and crushing the image. Use Reset tuning."
        elif (mean_value < 25 and std_value < 3 and
              controls.get("auto_exposure") is False):
            warning = "Manual exposure is producing a black frame. Enable Automatic exposure or use Reset tuning."
        elif mean_value < 25 and std_value < 3:
            warning = "Camera image is nearly black and uniform. Check the privacy shutter or lens."
        elif mean_value < 40:
            warning = "Camera image is dark. Adjust brightness in Live image tuning."
        else:
            warning = ""
        return {
            "image_mean": mean_value,
            "image_std": std_value,
            "image_warning": warning,
        }
