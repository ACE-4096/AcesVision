"""Capture adapters for local webcams and DroidCam/network feeds."""
from __future__ import annotations

from typing import Any, Callable

import cv2

import camera as local_camera

from .contracts import SourceSpec


def open_source(
    source: SourceSpec,
    *,
    webcam_opener: Callable[[int | str | None], Any] | None = None,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
):
    """Open a source and return a capture, or None when it is unavailable."""
    if source.kind == "webcam":
        try:
            device = source.device_path or source.index
            if webcam_opener is not None:
                return webcam_opener(device)
            return local_camera.open_camera(device, manual_exposure=False)
        except (RuntimeError, ValueError, OSError):
            return None

    cap = capture_factory(source.url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return None
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap
