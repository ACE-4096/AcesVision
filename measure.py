"""Identify + measure objects from a camera, in real-world millimetres.

How it works:
  1. An ArUco marker of known size (make_marker.py) gives pixels-per-mm.
  2. Objects on the surface are found by edge detection + contours.
  3. Each object's minimum-area rectangle is scaled by the marker -> mm.

    python measure.py                 # live window, needs marker + objects in view

A single 2D camera can only measure a flat plane truthfully — lay objects and
the marker flat, camera roughly square-on. For 3D box height you'd need depth.

Env: FACE_ID_MARKER_MM (default 50), FACE_ID_MIN_MM (ignore objects under, default 8).
"""
import os

import cv2
import numpy as np

from camera import open_camera

MARKER_MM = float(os.environ.get("FACE_ID_MARKER_MM", "50"))
MIN_MM = float(os.environ.get("FACE_ID_MIN_MM", "8"))
DICT = cv2.aruco.DICT_4X4_50


def make_detector():
    d = cv2.aruco.getPredefinedDictionary(DICT)
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


def find_scale(gray, detector):
    """Return (px_per_mm, marker_corners) or (None, None)."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(corners) == 0:
        return None, None
    c = corners[0].reshape(4, 2)
    sides = [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
    return float(np.mean(sides)) / MARKER_MM, c


def measure(frame, detector):
    """Return (list of (rotated_box_pts, w_mm, h_mm), px_per_mm or None)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    px_per_mm, marker = find_scale(gray, detector)
    if px_per_mm is None:
        return [], None
    mx, my, mw, mh = cv2.boundingRect(marker.astype(np.int32))

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, None, iterations=2)
    edges = cv2.erode(edges, None, iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_px_area = (MIN_MM * px_per_mm) ** 2
    out = []
    for cnt in cnts:
        if cv2.contourArea(cnt) < min_px_area:
            continue
        (cx, cy), (w, h), ang = cv2.minAreaRect(cnt)
        if mx - 8 <= cx <= mx + mw + 8 and my - 8 <= cy <= my + mh + 8:
            continue   # skip the marker itself
        box = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32)
        out.append((box, w / px_per_mm, h / px_per_mm))
    return out, px_per_mm


def annotate(frame, objs, px_per_mm):
    for box, w_mm, h_mm in objs:
        cv2.drawContours(frame, [box], -1, (0, 200, 0), 2)
        cx, cy = box.mean(axis=0).astype(int)
        big, small = max(w_mm, h_mm), min(w_mm, h_mm)
        cv2.putText(frame, f"{big:.0f} x {small:.0f} mm", (cx - 50, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 220, 30), 2)
    msg = (f"scale: {px_per_mm:.2f} px/mm  |  {len(objs)} object(s)"
           if px_per_mm else "no marker in view — place the ArUco marker flat")
    cv2.putText(frame, msg, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255) if px_per_mm else (0, 0, 255), 2)
    return frame


def main():
    detector = make_detector()
    cap = open_camera()
    print("measure.py — show the ArUco marker + objects, Q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        objs, ppm = measure(frame, detector)
        annotate(frame, objs, ppm)
        cv2.imshow("Measure - Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
