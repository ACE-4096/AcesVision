"""Hand-gesture detection via MediaPipe GestureRecognizer.

Recognises 7 built-in gestures out of the box (no training):
    Closed_Fist  Open_Palm  Pointing_Up  Thumb_Up  Thumb_Down  Victory  ILoveYou

GestureDetector.detect(frame_bgr) -> list of Gesture(name, score, box).
Each detector instance is single-thread; give each camera worker its own.

A literal finger-*snap* is not reliably detectable from video (sub-100ms, subtle
signature) — use one of the static gestures above as your trigger instead.
"""
import threading
from collections import namedtuple
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL = Path(__file__).parent / "models" / "gesture_recognizer.task"
Gesture = namedtuple("Gesture", "name score x y w h")
_GLOCK = threading.Lock()   # serialise detect across camera threads, just in case


class GestureDetector:
    def __init__(self, num_hands=2, min_score=0.6):
        if not MODEL.exists():
            raise FileNotFoundError(
                f"Gesture model missing at {MODEL}. Download it:\n  curl -sL -o "
                f"{MODEL} https://storage.googleapis.com/mediapipe-models/"
                "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task")
        opts = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(model_asset_path=str(MODEL)),
            num_hands=num_hands)
        self.rec = vision.GestureRecognizer.create_from_options(opts)
        self.min_score = min_score

    def detect(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _GLOCK:
            res = self.rec.recognize(img)
        out = []
        for cats, lms in zip(res.gestures, res.hand_landmarks):
            if not cats:
                continue
            g = cats[0]
            if g.category_name in ("None", "") or g.score < self.min_score:
                continue
            xs = [lm.x for lm in lms]
            ys = [lm.y for lm in lms]
            x0, y0 = int(min(xs) * w), int(min(ys) * h)
            x1, y1 = int(max(xs) * w), int(max(ys) * h)
            out.append(Gesture(g.category_name, float(g.score),
                               x0, y0, x1 - x0, y1 - y0))
        return out

    def close(self):
        self.rec.close()
