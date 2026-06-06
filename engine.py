"""Shared face-recognition engine.

build_detector() returns a callable  detect(frame_bgr) -> list[Face]  for the
chosen engine. One detector instance is self-contained and safe to use from a
single thread; give each camera thread its own instance.

Engines:
    'yunet' (default) — YuNet detection (~5-40ms, angle-robust) + dlib encoding.
                        Best of both: fast like HOG, robust like CNN. Keeps your
                        existing dlib enrolments valid (encoding is unchanged).
    'dlib'            — face_recognition HOG/CNN detection + dlib encoding.
    'lbph'            — the 2016 OpenCV LBPH recogniser (no dlib).

    Face(x, y, w, h, name, conf, known)
        box in full-frame pixels; name is the matched person or None;
        conf is the match distance (lower = better).
"""
import os
import threading
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

KNOWN_DIR = Path(__file__).parent / "known_faces"
YUNET_PATH = Path(__file__).parent / "models" / "face_detection_yunet.onnx"
IMG_EXT = {".jpg", ".jpeg", ".png"}

Face = namedtuple("Face", "x y w h name conf known")

# dlib's detector/encoder (and cv2.face) are process-global C++ objects that are
# NOT thread-safe — concurrent calls from multiple camera threads segfault.
# Every detect() call is serialised through this single process-wide lock.
_LOCK = threading.Lock()

# Enrolled encodings are static during a run; load once and share (read-only)
# across all camera detectors instead of re-loading per camera.
_KNOWN_CACHE = None


def _people_dirs():
    if not KNOWN_DIR.exists():
        return []
    return sorted(p for p in KNOWN_DIR.iterdir() if p.is_dir())


def build_detector(engine=None, scale=None, model=None):
    engine = (engine or os.environ.get("FACE_ID_ENGINE", "yunet")).lower()
    if engine == "lbph":
        return _build_lbph()
    if engine == "dlib":
        return _build_dlib(scale, model)
    return _build_yunet()


def _yunet_locate(yn, arr_rgb):
    """YuNet face boxes (dlib order) on an RGB image. ~40ms, angle-robust."""
    if yn is None:
        return []
    bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    yn.setInputSize((w, h))
    _, faces = yn.detect(bgr)
    out = []
    if faces is not None:
        for f in faces:
            x, y, bw, bh = (int(v) for v in f[:4])
            x, y = max(0, x), max(0, y)
            out.append((y, x + bw, y + bh, x))
    return out


def _load_known_encodings(face_recognition):
    """Return (encodings, names) for every enrolled photo. Cached.

    HOG first (fast, frontal); fall back to YuNet (~40ms, angle-robust) for
    awkward poses, e.g. enrolled off an ESP32. No slow CNN path.
    """
    global _KNOWN_CACHE
    if _KNOWN_CACHE is not None:
        return _KNOWN_CACHE
    yn = (cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.5, 0.3, 5000)
          if YUNET_PATH.exists() else None)
    encs, names = [], []
    for d in _people_dirs():
        for img in sorted(d.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            arr = face_recognition.load_image_file(str(img))
            locs = face_recognition.face_locations(arr, model="hog")
            if not locs:
                locs = _yunet_locate(yn, arr)
            if not locs:
                print(f"[warn] no face in {img}, skipping")
                continue
            found = face_recognition.face_encodings(arr, locs[:1])
            if found:
                encs.append(found[0])
                names.append(d.name)
    _KNOWN_CACHE = (encs, names)
    return _KNOWN_CACHE


def _match(encs, names, enc, tol):
    """Nearest enrolled encoding -> (name, dist, known). Euclidean, like dlib."""
    if not encs:
        return None, 0.0, False
    dists = np.linalg.norm(np.asarray(encs) - enc, axis=1)
    bi = int(np.argmin(dists))
    if dists[bi] <= tol:
        return names[bi], float(dists[bi]), True
    return None, float(dists[bi]), False


def _build_yunet():
    import face_recognition
    if not YUNET_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model missing at {YUNET_PATH}.\nDownload it:\n  curl -sL -o "
            f"{YUNET_PATH} https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx")
    tol = float(os.environ.get("FACE_ID_TOLERANCE", "0.6"))
    score = float(os.environ.get("FACE_ID_YUNET_SCORE", "0.6"))
    yn = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), score, 0.3, 5000)
    encs, names = _load_known_encodings(face_recognition)

    def detect(frame):
        with _LOCK:
            h, w = frame.shape[:2]
            yn.setInputSize((w, h))
            _, faces = yn.detect(frame)
            boxes = []
            if faces is not None:
                for f in faces:
                    x, y, bw, bh = (int(v) for v in f[:4])
                    x, y = max(0, x), max(0, y)
                    boxes.append((y, x + bw, y + bh, x))   # dlib order
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fencs = face_recognition.face_encodings(rgb, boxes) if boxes else []
        out = []
        for (top, right, bottom, left), e in zip(boxes, fencs):
            name, conf, known = _match(encs, names, e, tol)
            out.append(Face(left, top, right - left, bottom - top, name, conf, known))
        return out

    detect.people = sorted(set(names))
    detect.engine = "yunet"
    return detect


def _build_dlib(scale=None, model=None):
    import face_recognition
    tol = float(os.environ.get("FACE_ID_TOLERANCE", "0.6"))
    # 'hog' = fast, frontal-only. 'cnn' = slow (CPU) but handles odd angles,
    # caps, and uneven lighting — worth it for awkwardly-mounted room cameras.
    model = model or os.environ.get("FACE_ID_MODEL", "hog")
    # scale: 1.0 = detect at full frame (best for low-res network cams);
    # <1.0 = downscale first for speed (fine for a hi-res local webcam).
    if scale is None:
        scale = float(os.environ.get("FACE_ID_SCALE", "0.5"))
    inv = 1.0 / scale

    encs, names = _load_known_encodings(face_recognition)

    def detect(frame):
        with _LOCK:
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model=model)
            fencs = face_recognition.face_encodings(rgb, locs)
        out = []
        for (top, right, bottom, left), e in zip(locs, fencs):
            name, conf, known = None, 0.0, False
            if encs:
                dists = face_recognition.face_distance(encs, e)
                bi = int(np.argmin(dists))
                if dists[bi] <= tol:
                    name, conf, known = names[bi], float(dists[bi]), True
            x, y = int(left * inv), int(top * inv)
            out.append(Face(x, y, int((right - left) * inv),
                            int((bottom - top) * inv), name, conf, known))
        return out

    detect.people = sorted(set(names))
    detect.engine = "dlib"
    return detect


def _build_lbph():
    thr = float(os.environ.get("FACE_ID_LBPH_THRESH", "70"))
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    images, labels, names = [], [], []
    for d in _people_dirs():
        label = len(names)
        names.append(d.name)
        for img in sorted(d.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            g = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            faces = cascade.detectMultiScale(g, 1.3, 5)
            if len(faces) == 1:
                x, y, w, h = faces[0]
                images.append(g[y:y + h, x:x + w])
                labels.append(label)

    rec = None
    if images:
        rec = cv2.face.LBPHFaceRecognizer_create()
        rec.train(images, np.array(labels))

    def detect(frame):
        if rec is None:
            return []
        with _LOCK:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            out = []
            for (x, y, w, h) in cascade.detectMultiScale(gray, 1.3, 5):
                lbl, conf = rec.predict(gray[y:y + h, x:x + w])
                if conf <= thr:
                    out.append(Face(x, y, w, h, names[lbl], float(conf), True))
                else:
                    out.append(Face(x, y, w, h, None, float(conf), False))
        return out

    detect.people = sorted(set(names))
    detect.engine = "lbph"
    return detect
