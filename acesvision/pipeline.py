"""Single-owner capture loop and drop-old frame fan-out."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import cv2

from .contracts import SceneFrame, SourceSpec
from .sources import open_source


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
    """One-slot output mailbox. Producers never wait for slow frame consumers."""

    def __init__(self, output: FrameOutput):
        super().__init__(daemon=True, name=f"vision-output-{type(output).__name__}")
        self.output = output
        self.last_error = ""
        self.dropped = 0
        self._condition = threading.Condition()
        self._latest: SceneFrame | None = None
        self._stopping = False

    def submit(self, scene: SceneFrame) -> None:
        with self._condition:
            if self._latest is not None:
                self.dropped += 1
            self._latest = scene
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify()

    def run(self) -> None:
        try:
            while True:
                with self._condition:
                    while self._latest is None and not self._stopping:
                        self._condition.wait()
                    if self._stopping:
                        break
                    scene = self._latest
                    self._latest = None
                try:
                    self.output.publish(scene)
                    self.last_error = ""
                except Exception as exc:
                    self.last_error = str(exc)
        finally:
            try:
                self.output.close()
            except Exception as exc:
                self.last_error = str(exc)


class VisionPipeline(threading.Thread):
    """Own one switchable camera and publish each scene to independent outputs."""

    def __init__(
        self,
        source: SourceSpec,
        processor: Callable[[Any, SourceSpec, int, float], SceneFrame],
        outputs: list[FrameOutput] | None = None,
        *,
        opener: Callable[[SourceSpec], Any] = open_source,
        retry_min_s: float = 0.5,
        retry_max_s: float = 10.0,
    ):
        super().__init__(daemon=True, name="vision-capture")
        self.processor = processor
        self.opener = opener
        self.retry_min_s = retry_min_s
        self.retry_max_s = retry_max_s
        self._source = source
        self._source_generation = 0
        self._state_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._status = "stopped"
        self._sequence = 0
        self._last_error = ""
        self._metrics = {"capture_fps": 0.0}
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

    def add_output(self, output: FrameOutput) -> None:
        worker = _OutputWorker(output)
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
            worker.join(timeout=2.0)
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

                if generation != opened_generation and cap is not None:
                    cap.release()
                    cap = None
                    applied_control_generation = -1

                if cap is None or not cap.isOpened():
                    self._set_state(status="reconnecting")
                    cap = self.opener(source)
                    opened_generation = generation
                    if cap is None or not cap.isOpened():
                        if cap is not None:
                            cap.release()
                        cap = None
                        self._set_state(
                            error=f"{source.safe_label()} is unavailable or in use"
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

                captured_at = time.monotonic()
                try:
                    scene = self.processor(frame, source, self._sequence, captured_at)
                except Exception as exc:
                    self._set_state(error=f"processor: {exc}")
                    self._sequence += 1
                    continue

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
                metrics["camera_controls"] = camera_controls
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
                worker.join(timeout=2.0)
            close_processor = getattr(self.processor, "close", None)
            if close_processor is not None:
                close_processor()
            self._set_state(status="stopped")

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
