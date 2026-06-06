"""Find objects (YOLO) and measure them in millimetres (ArUco scale).

YOLO detects any object on any background — a person/head, a bottle, a part —
as a bounding box. An ArUco marker of known size in frame gives pixels-per-mm,
so each box is reported in real-world dimensions.

ObjectMeasurer.detect(frame_bgr) -> (list[Obj], px_per_mm or None)
    Obj(name, conf, x, y, w, h, w_mm, h_mm)   box in px; w_mm/h_mm None if no marker.

Accuracy caveat: a single 2D camera only measures truthfully on the marker's
plane — objects roughly co-planar with the marker and a square-on camera. Off
that plane (an object much closer/further than the marker) the mm scale drifts.
"""
import os
import threading
from collections import namedtuple

import cv2
import numpy as np

from engine import YUNET_PATH

MARKER_MM = float(os.environ.get("FACE_ID_MARKER_MM", "50"))
CONF = float(os.environ.get("FACE_ID_YOLO_CONF", "0.35"))
MODEL_NAME = os.environ.get("FACE_ID_YOLO_MODEL", "yolov8n.pt")
# Head is taller/wider than the tight face box; scale it up to approximate the
# whole head. 1.0 = the bare face box. ~1.5 height / ~1.1 width ≈ a head.
HEAD_W = float(os.environ.get("FACE_ID_HEAD_W", "1.15"))
HEAD_H = float(os.environ.get("FACE_ID_HEAD_H", "1.45"))
DICT = cv2.aruco.DICT_4X4_50

Obj = namedtuple("Obj", "name conf x y w h w_mm h_mm")
_LOCK = threading.Lock()   # detector inference serialised across camera threads


def _aruco_scale(aruco, gray, marker_mm):
    corners, ids, _ = aruco.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None
    c = corners[0].reshape(4, 2)
    sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides)) / marker_mm


def _annotate(frame, objs, px_per_mm):
    for o in objs:
        cv2.rectangle(frame, (o.x, o.y), (o.x + o.w, o.y + o.h), (0, 200, 0), 2)
        if o.w_mm is not None:
            label = f"{o.name}  {max(o.w_mm, o.h_mm):.0f} x {min(o.w_mm, o.h_mm):.0f} mm"
        else:
            label = f"{o.name}  (no scale)"
        y = max(o.y - 8, 16)
        cv2.rectangle(frame, (o.x, y - 16), (o.x + 9 * len(label), y + 2),
                      (0, 120, 0), cv2.FILLED)
        cv2.putText(frame, label, (o.x + 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1)
    msg = (f"scale {px_per_mm:.2f} px/mm" if px_per_mm
           else "no marker in view — hold the ArUco marker by your face for mm")
    cv2.putText(frame, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255) if px_per_mm else (0, 0, 255), 2)
    return frame


class FaceMeasurer:
    """Detect the face/head (YuNet) and measure it in mm against an ArUco marker.

    For accuracy, hold the marker in the SAME plane as your face (next to your
    cheek, same distance from the camera) — a marker on the desk is closer/further
    than your head and the scale would drift.
    """

    def __init__(self, marker_mm=MARKER_MM):
        self.yn = cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320),
                                            0.6, 0.3, 5000)
        self.aruco = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(DICT), cv2.aruco.DetectorParameters())
        self.marker_mm = marker_mm

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = frame.shape[:2]
        with _LOCK:
            px_per_mm = _aruco_scale(self.aruco, gray, self.marker_mm)
            self.yn.setInputSize((W, H))
            _, faces = self.yn.detect(frame)
        out = []
        if faces is not None:
            for f in faces:
                fx, fy, fw, fh = (float(v) for v in f[:4])
                # expand the tight face box toward a full-head box
                cx, cy = fx + fw / 2, fy + fh / 2
                hw, hh = fw * HEAD_W, fh * HEAD_H
                x, y = int(max(0, cx - hw / 2)), int(max(0, cy - hh / 2))
                w, h = int(min(W - x, hw)), int(min(H - y, hh))
                w_mm = w / px_per_mm if px_per_mm else None
                h_mm = h / px_per_mm if px_per_mm else None
                out.append(Obj("head", float(f[14]), x, y, w, h, w_mm, h_mm))
        return out, px_per_mm

    @staticmethod
    def annotate(frame, objs, px_per_mm):
        return _annotate(frame, objs, px_per_mm)


class ObjectMeasurer:
    def __init__(self, marker_mm=MARKER_MM, conf=CONF):
        from ultralytics import YOLO
        self.model = YOLO(MODEL_NAME)
        self.names = self.model.names
        self.aruco = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(DICT), cv2.aruco.DetectorParameters())
        self.marker_mm = marker_mm
        self.conf = conf

    def _scale(self, gray):
        corners, ids, _ = self.aruco.detectMarkers(gray)
        if ids is None or len(corners) == 0:
            return None
        c = corners[0].reshape(4, 2)
        sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
        return float(np.mean(sides)) / self.marker_mm

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        with _LOCK:
            px_per_mm = self._scale(gray)
            res = self.model(frame, verbose=False, conf=self.conf)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            w_mm = w / px_per_mm if px_per_mm else None
            h_mm = h / px_per_mm if px_per_mm else None
            out.append(Obj(self.names[int(b.cls)], float(b.conf), x1, y1, w, h,
                           w_mm, h_mm))
        return out, px_per_mm

    @staticmethod
    def annotate(frame, objs, px_per_mm):
        for o in objs:
            cv2.rectangle(frame, (o.x, o.y), (o.x + o.w, o.y + o.h), (0, 200, 0), 2)
            if o.w_mm is not None:
                label = f"{o.name}  {max(o.w_mm, o.h_mm):.0f} x {min(o.w_mm, o.h_mm):.0f} mm"
            else:
                label = f"{o.name} {o.conf:.2f}  (no scale)"
            y = max(o.y - 8, 16)
            cv2.rectangle(frame, (o.x, y - 16), (o.x + 9 * len(label), y + 2),
                          (0, 120, 0), cv2.FILLED)
            cv2.putText(frame, label, (o.x + 2, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1)
        msg = (f"scale {px_per_mm:.2f} px/mm" if px_per_mm
               else "no marker in view — place the ArUco marker to enable mm")
        cv2.putText(frame, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if px_per_mm else (0, 0, 255), 2)
        return frame
