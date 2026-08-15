"""GUI preview, callback, and OBS output adapters."""
from __future__ import annotations

import threading
from typing import Callable

import cv2

from .contracts import SceneFrame
from .overlay import MINIMAL, OverlayProfile, render


class LatestFrameOutput:
    def __init__(self, profile: OverlayProfile = MINIMAL, jpeg_quality: int = 88):
        self.profile = profile
        self.jpeg_quality = jpeg_quality
        self._lock = threading.Lock()
        self._scene = None
        self._jpeg = b""

    def publish(self, scene: SceneFrame) -> None:
        frame = render(scene, self.profile)
        ok, buf = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        with self._lock:
            self._scene = scene
            if ok:
                self._jpeg = buf.tobytes()

    def set_profile(self, profile: OverlayProfile) -> None:
        self.profile = profile

    def snapshot(self):
        with self._lock:
            return self._scene, self._jpeg

    def close(self) -> None:
        pass


class CallbackOutput:
    def __init__(self, callback: Callable[[SceneFrame], None]):
        self.callback = callback

    def publish(self, scene: SceneFrame) -> None:
        self.callback(scene)

    def close(self) -> None:
        pass


class ObsVirtualCameraOutput:
    def __init__(self, profile: OverlayProfile = MINIMAL,
                 device: str | None = None, fps: int = 30):
        self.profile = profile
        self.device = device
        self.fps = fps
        self._camera = None
        self._size = None

    def _open(self, frame) -> None:
        import pyvirtualcam

        height, width = frame.shape[:2]
        kwargs = dict(width=width, height=height, fps=self.fps,
                      fmt=pyvirtualcam.PixelFormat.BGR)
        if self.device:
            kwargs["device"] = self.device
        self._camera = pyvirtualcam.Camera(**kwargs)
        self._size = (width, height)

    def publish(self, scene: SceneFrame) -> None:
        frame = render(scene, self.profile)
        if self._camera is None:
            self._open(frame)
        if frame.shape[1::-1] != self._size:
            frame = cv2.resize(frame, self._size)
        self._camera.send(frame)
        self._camera.sleep_until_next_frame()

    def set_profile(self, profile: OverlayProfile) -> None:
        self.profile = profile

    def close(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None
            self._size = None
