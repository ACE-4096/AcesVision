"""Webcam helper.

This machine has two physical cameras behind one USB webcam:
  - video9 / video10 -> colour (RGB) sensor
  - video11 / video12 -> IR / greyscale sensor (face-unlock)
plus video20 (the OBS loopback).

Two gotchas this handles:
  1. The colour node only yields frames with the V4L2 backend + MJPG format
     (its default YUYV mode hands OpenCV nothing), so we force both.
  2. A naive probe grabs the IR node first and you get a greyscale picture.
     open_camera() checks each candidate's colour content and prefers a real
     colour camera, only falling back to greyscale if nothing else works.

Overrides (env vars):
    FACE_ID_CAM=9          force a specific index (skips colour preference)
    FACE_ID_W / FACE_ID_H  requested capture resolution (default 1280x720)
"""
import os

import cv2
import numpy as np

PREF_W = int(os.environ.get("FACE_ID_W", "1280"))
PREF_H = int(os.environ.get("FACE_ID_H", "720"))
PREF_FPS = int(os.environ.get("FACE_ID_FPS", "30"))
MANUAL_EXPOSURE = int(os.environ.get("ACESVISION_EXPOSURE", "166"))
# Probe colour nodes first, then common indices, then IR nodes as last resort.
CANDIDATES = [9, 10, 0, 1, 2, 11, 12, 13]
CHROMA_MIN = 6.0   # mean channel spread above this == a colour image


def _chroma(frame):
    b, g, r = cv2.split(frame.astype("int16"))
    return float((np.abs(b - g) + np.abs(g - r) + np.abs(b - r)).mean())


def _try_open(device, manual_exposure=False):
    """Open an index/path with V4L2+MJPG, returning (cap, frame) or none."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None, None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREF_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREF_H)
    cap.set(cv2.CAP_PROP_FPS, PREF_FPS)
    if manual_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    frame = None
    valid_frames = 0
    for _ in range(8):              # let format and exposure controls settle
        ok, frame = cap.read()
        if ok and frame is not None:
            valid_frames += 1
            if valid_frames >= 3:
                return cap, frame
    cap.release()
    return None, None


def open_camera(preferred=None, manual_exposure=False):
    """Return an opened cv2.VideoCapture (colour preferred), or raise."""
    env = os.environ.get("FACE_ID_CAM")
    forced = [x for x in (preferred, int(env) if env is not None else None)
              if x is not None]

    # Forced index: honour it exactly, no colour filtering.
    if forced:
        for device in forced:
            cap, frame = _try_open(device, manual_exposure=manual_exposure)
            if cap is not None:
                h, w = frame.shape[:2]
                tag = "colour" if _chroma(frame) > CHROMA_MIN else "greyscale"
                print(f"[camera] forced device {device} ({w}x{h}, {tag})")
                return cap
        raise RuntimeError(f"Forced camera device {forced} would not open.")

    # Probe, preferring a colour camera; remember first greyscale as fallback.
    fallback = None
    seen = set()
    for idx in CANDIDATES:
        if idx in seen:
            continue
        seen.add(idx)
        cap, frame = _try_open(idx, manual_exposure=manual_exposure)
        if cap is None:
            continue
        h, w = frame.shape[:2]
        if _chroma(frame) > CHROMA_MIN:
            print(f"[camera] using colour index {idx} ({w}x{h})")
            return cap
        if fallback is None:
            fallback = (cap, idx, w, h)   # keep one greyscale handle open
        else:
            cap.release()

    if fallback:
        cap, idx, w, h = fallback
        print(f"[camera] WARNING: no colour camera found — falling back to "
              f"greyscale index {idx} ({w}x{h}). Force colour with FACE_ID_CAM=9.")
        return cap

    raise RuntimeError(
        "No working camera found. Try FACE_ID_CAM=<n> (check `ls /dev/video*`)."
    )
