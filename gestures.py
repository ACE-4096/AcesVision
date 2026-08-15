"""Hand-gesture detection via MediaPipe GestureRecognizer.

Recognises 7 built-in gestures out of the box (no training):
    Closed_Fist  Open_Palm  Pointing_Up  Thumb_Up  Thumb_Down  Victory  ILoveYou

plus two custom poses derived from the hand landmarks rather than the model:
    Middle_Finger    (ported from AceRGB acergb_gesture.py:47-57)
    Shush            (index finger up AND at the lips of a detected face)

The full vocabulary lives in gesture_catalog.GESTURES — that module is the
single source of truth both this repo and AceRGB validate names against.

GestureDetector.detect(frame_bgr, faces) -> list of Gesture(name, score, box).
Pass the face boxes from engine.build_detector(); without them Shush cannot be
told apart from the model's Pointing_Up and will never be emitted.
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

from gesture_catalog import (MIDDLE_FINGER, SHUSH, is_middle_finger,
                             is_shush)

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

    def recognize(self, frame_bgr):
        """One MediaPipe pass -> (raw result, frame width, frame height).

        Public because the model's own labels are evidence in their own right:
        a tool that wants to see what MediaPipe said *before* the custom poses
        overrode it (verify_gestures_live.py does) would otherwise have to run
        inference twice, or reach into .rec. detect() is this plus classify.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _GLOCK:
            res = self.rec.recognize(img)
        return res, w, h

    def detect(self, frame_bgr, faces=None):
        res, w, h = self.recognize(frame_bgr)
        return classify_hands(res.hand_landmarks, res.gestures, w, h,
                              self.min_score, faces=faces)

    def close(self):
        self.rec.close()


def classify_hands(hand_landmarks, gesture_categories, w, h, min_score,
                   faces=None):
    """Map one recognizer result onto Gesture rows. Pure — no MediaPipe calls.

    The custom landmark poses win over the model's built-in label for that
    hand, matching AceRGB's _classify (acergb_gesture.py:60-80). Landmarks lead
    the iteration so a custom pose is still detected on a hand the model
    returned no category for.

    Shush takes strict precedence over Pointing_Up, and that is not cosmetic:
    the model labels a finger-to-the-lips hand Pointing_Up, which is bound to
    `ledctl next-theme` in automations.json. Without this precedence every
    shush would also cycle the lighting themes. `faces` is what makes the
    distinction possible — pass the boxes from engine.build_detector().
    """
    out = []
    for index, lms in enumerate(hand_landmarks or []):
        xs = [lm.x for lm in lms]
        ys = [lm.y for lm in lms]
        x0, y0 = int(min(xs) * w), int(min(ys) * h)
        x1, y1 = int(max(xs) * w), int(max(ys) * h)
        box = (x0, y0, x1 - x0, y1 - y0)

        if is_middle_finger(lms):
            out.append(Gesture(MIDDLE_FINGER, 1.0, *box))
            continue

        if is_shush(lms, faces, w, h):
            out.append(Gesture(SHUSH, 1.0, *box))
            continue

        cats = (gesture_categories[index]
                if gesture_categories and index < len(gesture_categories) else [])
        if not cats:
            continue
        g = cats[0]
        if g.category_name in ("None", "") or g.score < min_score:
            continue
        out.append(Gesture(g.category_name, float(g.score), *box))
    return out
