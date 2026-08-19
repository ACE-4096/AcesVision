"""Shared face-recognition engine.

build_detector() returns a callable  detect(frame_bgr) -> list[Face]  for the
chosen engine. One detector instance is self-contained and safe to use from a
single thread; give each camera thread its own instance.

Engines (``FACE_ID_ENGINE``):
    'arcface' (default) — YuNet detection + ArcFace ONNX embedding. ~12 ms
                          (w600k_mbf) to ~34 ms (w600k_r50) for the whole
                          stage, against 77.6 ms for the dlib pipeline it
                          replaces. Scores cosine similarity.
    'yunet'            — YuNet detection + dlib ResNet encoding. The previous
                          default. 77.6 ms at 640x480, of which 65.9 ms is the
                          dlib encoder. Scores Euclidean distance.
    'dlib'             — face_recognition HOG/CNN detection + dlib encoding.
    'lbph'             — the 2016 OpenCV LBPH recogniser (no dlib).

    The dlib and LBPH engines are kept deliberately. They are the record of
    how this problem was solved before, they still work, and they are one
    environment variable away.

    Face(x, y, w, h, name, conf, known, metric)
        box in full-frame pixels; name is the matched person or None.

        ``conf`` is the match score and ``metric`` says what it means. This
        is not decoration: the metric direction FLIPPED when ArcFace arrived.
        For 'dlib' and 'yunet' conf is a Euclidean distance and lower is
        better; for 'arcface' it is a cosine similarity and HIGHER is better.
        Read the direction from ``matching.METRICS[face.metric]``, never from
        memory, and never compare two faces' conf unless their metrics match.

Threshold selection is per engine and lives in ``matching.py``. There is no
shared tolerance constant, because 0.50 means "strict" to dlib and "accept
essentially every stranger" to ArcFace.
"""
import os
import threading
from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np

import matching

KNOWN_DIR = Path(__file__).parent / "known_faces"
YUNET_PATH = Path(__file__).parent / "models" / "face_detection_yunet.onnx"
IMG_EXT = {".jpg", ".jpeg", ".png"}

DEFAULT_ENGINE = "arcface"

Face = namedtuple("Face", "x y w h name conf known metric")

# dlib's detector/encoder (and cv2.face) are process-global C++ objects that
# are NOT thread-safe — concurrent calls from multiple camera threads segfault.
# Every dlib and LBPH detect() call is serialised through this single
# process-wide lock, which also serialises every camera thread in the process.
#
# The ArcFace path does NOT take this lock and must not be made to. An
# onnxruntime InferenceSession is safe to call concurrently, and each detector
# owns its own cv2.FaceDetectorYN, so there is nothing shared to protect. The
# lock is a dlib workaround, not a house style.
_LOCK = threading.Lock()

# Enrolled embeddings are static during a run: load once, share read-only
# across every camera detector rather than re-encoding per camera.
#
# Keyed by EMBEDDING SPACE, not by nothing. A gallery of dlib 128-d encodings
# and a gallery of ArcFace 512-d embeddings are different spaces, and so are
# w600k_r50 and w600k_mbf. A single unkeyed cache would hand whichever gallery
# was built first to whichever engine asked second — silently, since a stale
# gallery still returns a plausible-looking number. Keying makes that
# impossible: a switch is a cache miss, not a wrong answer.
_KNOWN_CACHE = {}
_CACHE_LOCK = threading.Lock()

DLIB_EMBEDDING_SPACE = "dlib-resnet-v1-128/yunet-first-hog-fallback"


def _people_dirs():
    if not KNOWN_DIR.exists():
        return []
    return sorted(p for p in KNOWN_DIR.iterdir() if p.is_dir())


def enrolled_photos():
    """Every (person_name, image_path) pair on disk, in a stable order.

    There is no cached encoding artifact anywhere — no .npy, .pkl or .npz.
    The gallery is rebuilt from ``known_faces/<Name>/*.jpg`` at process start,
    every time. That is what makes re-enrolment free when the embedder
    changes: there is nothing on disk to invalidate, and nothing that can
    outlive the model that produced it.
    """
    for person in _people_dirs():
        for image in sorted(person.iterdir()):
            if image.suffix.lower() in IMG_EXT:
                yield person.name, image


def known_embeddings(space, encode_path):
    """The enrolled gallery for one embedding space. Cached per space.

    ``encode_path(Path) -> vector | None``. Called once per photo, only on a
    cache miss.
    """
    with _CACHE_LOCK:
        cached = _KNOWN_CACHE.get(space)
    if cached is not None:
        return cached

    vectors, names = [], []
    for name, image in enrolled_photos():
        vector = encode_path(image)
        if vector is None:
            print(f"[warn] no face in {image}, skipping")
            continue
        vectors.append(vector)
        names.append(name)

    with _CACHE_LOCK:
        # Another thread may have won the race; its gallery is equally valid,
        # and sharing one is the point of the cache.
        return _KNOWN_CACHE.setdefault(space, (vectors, names))


def clear_known_cache():
    """Drop every cached gallery. For tests and for re-enrolment in-process."""
    with _CACHE_LOCK:
        _KNOWN_CACHE.clear()


def build_detector(engine=None, scale=None, model=None):
    engine = (engine or os.environ.get("FACE_ID_ENGINE", DEFAULT_ENGINE)).lower()
    if engine == "lbph":
        return _build_lbph()
    if engine == "dlib":
        return _build_dlib(scale, model)
    if engine == "yunet":
        return _build_yunet()
    if engine == "arcface":
        return _build_arcface()
    raise ValueError(
        f"unknown FACE_ID_ENGINE {engine!r}; "
        "choose arcface (default), yunet, dlib or lbph"
    )


# ---------------------------------------------------------------------------
# ArcFace: YuNet detection + ONNX embedding. The default.
# ---------------------------------------------------------------------------

def arcface_gallery(variant=None, pipeline=None):
    """``(embeddings, names, pipeline)`` for the enrolled ArcFace gallery.

    The gallery pipeline and the query pipeline are the same class, so
    enrolment and query cannot drift apart: same detector, same landmark
    ordering, same alignment, same weights. That invariant used to be three
    hand-copied blocks in three files plus a comment asking editors to keep
    them in step; when they drifted it cost 0.13 of separation and voided a
    calibration (ticket a3c3c709). ``scan_photos`` and ``calibrate_threshold``
    enrol through this same object for exactly that reason.
    """
    import arcface

    variant = variant or os.environ.get("FACE_ID_ARCFACE_MODEL",
                                        arcface.DEFAULT_VARIANT)
    pipeline = pipeline or arcface.ArcFacePipeline.load(variant)
    encs, names = known_embeddings(pipeline.embedding_space(),
                                   pipeline.encode_file)
    return encs, names, pipeline


def _build_arcface(variant=None, pipeline=None):
    import arcface

    threshold = matching.threshold_for("arcface").require_evidence()
    encs, names, gallery_pipeline = arcface_gallery(variant, pipeline)
    variant = gallery_pipeline.variant.name
    space = gallery_pipeline.embedding_space()

    # A second pipeline for the live path so the camera thread does not share
    # a cv2.FaceDetectorYN with enrolment or with another camera. The ONNX
    # session is shared deliberately — it is thread-safe and it is the
    # expensive part.
    query_pipeline = (
        gallery_pipeline if pipeline is not None
        else arcface.ArcFacePipeline(gallery_pipeline.embedder)
    )

    def detect(frame):
        # No lock. See the _LOCK comment: this path has nothing to serialise,
        # and taking it would re-impose the bottleneck ArcFace removes.
        out = []
        for found in query_pipeline.detect(frame):
            name, conf, known = matching.match(encs, names, found.embedding,
                                               threshold)
            out.append(Face(found.x, found.y, found.w, found.h,
                            name, conf, known, threshold.metric))
        return out

    detect.people = sorted(set(names))
    detect.engine = "arcface"
    detect.variant = variant
    detect.metric = threshold.metric
    detect.threshold = threshold
    detect.embedding_space = space
    return detect


# ---------------------------------------------------------------------------
# dlib-encoder engines
# ---------------------------------------------------------------------------

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

    DETECTOR ORDER: YuNet first (angle-robust, ~40ms), HOG fallback.
    This MUST match the detection order used in scan_photos._encode_image
    and calibrate_threshold's dlib path so that enrolled embeddings and query
    embeddings are produced from comparable bounding boxes. Mixing
    HOG-enrolled vs YuNet-queried embeddings adds ~0.13 to genuine distances
    and collapses the separation gap — ticket a3c3c709.

    The ArcFace path does not repeat this arrangement; it shares one
    ``arcface.ArcFacePipeline`` between enrolment and query instead, so there
    is no second copy to keep in step.
    """
    yn = (cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.5, 0.3, 5000)
          if YUNET_PATH.exists() else None)

    def encode_path(image):
        arr = face_recognition.load_image_file(str(image))
        # YuNet first — same order as scan_photos._encode_image query path
        locs = _yunet_locate(yn, arr)
        if not locs:
            locs = face_recognition.face_locations(arr, model="hog")
        if not locs:
            return None
        found = face_recognition.face_encodings(arr, locs[:1])
        return found[0] if found else None

    return known_embeddings(DLIB_EMBEDDING_SPACE, encode_path)


def _match(encs, names, enc, threshold):
    """Nearest enrolled encoding -> (name, score, known).

    ``threshold`` is a ``matching.Threshold``; a bare float is refused,
    because a float carries neither its metric nor its direction.
    """
    return matching.match(encs, names, enc, threshold)


def _build_yunet():
    import face_recognition
    if not YUNET_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model missing at {YUNET_PATH}.\nDownload it:\n  curl -sL -o "
            f"{YUNET_PATH} https://github.com/opencv/opencv_zoo/raw/main/models/"
            "face_detection_yunet/face_detection_yunet_2023mar.onnx")
    # 0.50 Euclidean: calibrated (ticket a3c3c709, 2026-06-06).
    # YuNet-first pipeline: genuine max=0.452, impostor min=0.500 (n=2500 LFW).
    # Clean gap [0.452, 0.500]. FAR=0%/Recall=100% at 0.50.
    # Prior default 0.60 accepted ~8.2% of strangers — do not restore.
    threshold = matching.threshold_for("yunet")
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
            name, conf, known = matching.match(encs, names, e, threshold)
            out.append(Face(left, top, right - left, bottom - top,
                            name, conf, known, threshold.metric))
        return out

    detect.people = sorted(set(names))
    detect.engine = "yunet"
    detect.metric = threshold.metric
    detect.threshold = threshold
    detect.embedding_space = DLIB_EMBEDDING_SPACE
    return detect


def _build_dlib(scale=None, model=None):
    import face_recognition
    # 0.50 Euclidean: calibrated — see _build_yunet comment + ticket a3c3c709.
    threshold = matching.threshold_for("dlib")
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
            name, conf, known = matching.match(encs, names, e, threshold)
            x, y = int(left * inv), int(top * inv)
            out.append(Face(x, y, int((right - left) * inv),
                            int((bottom - top) * inv),
                            name, conf, known, threshold.metric))
        return out

    detect.people = sorted(set(names))
    detect.engine = "dlib"
    detect.metric = threshold.metric
    detect.threshold = threshold
    detect.embedding_space = DLIB_EMBEDDING_SPACE
    return detect


def _build_lbph():
    threshold = matching.threshold_for("lbph")
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
                known = threshold.accepts(conf, threshold.metric)
                out.append(Face(x, y, w, h, names[lbl] if known else None,
                                float(conf), known, threshold.metric))
        return out

    detect.people = sorted(set(names))
    detect.engine = "lbph"
    detect.metric = threshold.metric
    detect.threshold = threshold
    detect.embedding_space = "lbph-histogram/haar-frontalface"
    return detect
