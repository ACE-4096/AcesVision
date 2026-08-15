"""Capture adapters for local webcams and DroidCam/network feeds."""
from __future__ import annotations

import sys
from typing import Any, Callable

import cv2

import camera as local_camera

from .contracts import SourceSpec


def open_source(
    source: SourceSpec,
    *,
    webcam_opener: Callable[[int | str | None], Any] | None = None,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
    on_error: Callable[[str], None] | None = None,
):
    """Open a source and return a capture, or None when it is unavailable.

    None is still the "unavailable" signal the pipeline retries on, but the
    reason is never discarded: camera.py distinguishes busy from missing and
    that text reaches on_error (or stderr). A device held by another process
    used to surface as "No working camera found", which read as a broken app
    rather than contention.
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
    return cap


def _report(on_error, source, reason):
    if on_error is not None:
        on_error(reason)
    else:
        print(f"[source] {source.id}: {reason}", file=sys.stderr)
