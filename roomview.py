"""Room View — many camera feeds in one window + a live "who's in the room" list.

Pulls from local webcams and ESP32-CAM (or any MJPEG/RTSP) network streams,
runs face recognition on each feed in its own thread, tiles them into a single
grid, and shows a sidebar listing everyone currently recognised across all
feeds (with which camera each person is on).

    python roomview.py                 # uses cameras.json next to this file
    python roomview.py my_cams.json    # explicit config path

Config (JSON list); each camera is one of:
    {"name": "Webcam",  "index": 9}                      # local V4L2 camera
    {"name": "Camera A", "url": "http://camera-a.local:81/stream"}  # ESP32-CAM MJPEG
    {"name": "Camera B", "url": "rtsp://user:password@camera-b.local/h264"}  # RTSP also OK

Keys:  Q or ESC = quit.

Env vars:
    FACE_ID_ENGINE=dlib        'dlib' (accurate) or 'lbph' (your 2016 system)
    FACE_ID_DETECT_EVERY=5     run recognition every N frames per feed
    FACE_ID_RECENCY=3.0        seconds a name stays "in the room" after last seen
    FACE_ID_CELL=480x270       per-feed tile size in the grid
    (plus the per-engine tuning vars from engine.py)
"""
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np

import camera
import matching
from engine import build_detector

DETECT_EVERY = int(os.environ.get("FACE_ID_DETECT_EVERY", "5"))
RECENCY = float(os.environ.get("FACE_ID_RECENCY", "3.0"))
_cw, _ch = (os.environ.get("FACE_ID_CELL", "480x270").split("x") + ["270"])[:2]
CELL_W, CELL_H = int(_cw), int(_ch)
SIDEBAR_W = 280
GREEN, RED, GREY = (0, 200, 0), (40, 40, 220), (150, 150, 150)


def open_source(spec):
    """Open a camera spec -> cv2.VideoCapture (or None)."""
    if "url" in spec:
        cap = cv2.VideoCapture(spec["url"], cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # keep latency low on net cams
        except Exception:
            pass
        return cap if cap.isOpened() else None
    if "index" in spec:
        try:
            return camera.open_camera(preferred=spec["index"])
        except RuntimeError:
            return None
    try:
        return camera.open_camera()
    except RuntimeError:
        return None


class Stream(threading.Thread):
    """One camera: reads frames, recognises faces, holds the latest annotated frame."""

    def __init__(self, spec):
        super().__init__(daemon=True)
        self.spec = spec
        self.name = spec.get("name", "cam")
        # Per-camera model + scale. 'cnn' handles odd angles but is slow, so it
        # defaults to a small scale; 'hog' network cams detect at full frame.
        model = spec.get("model")
        scale = spec.get("scale")
        if scale is None:
            if model == "cnn":
                scale = 0.3          # cnn is ~1.8s/frame at this size
            elif "url" in spec:
                scale = 1.0          # low-res network cam, hog at full frame
        self.detector = build_detector(scale=scale, model=model)  # own instance => thread-safe
        self.lock = threading.Lock()
        self.frame = None                        # latest annotated BGR frame
        self.last_seen = {}                      # name -> monotonic timestamp
        self.status = "connecting"
        self.running = True
        self._faces = []

    def _draw(self, frame, faces):
        for f in faces:
            color = GREEN if f.known else RED
            # f.conf is only meaningful alongside f.metric: for ArcFace it is
            # a cosine similarity where higher is better, for the dlib engines
            # a Euclidean distance where lower is better. The metric suffix is
            # drawn so a 0.62 on screen is never read as the wrong 0.62.
            label = (f"{f.name} ({matching.format_score(f.conf, f.metric)})"
                     if f.known else "Unknown")
            cv2.rectangle(frame, (f.x, f.y), (f.x + f.w, f.y + f.h), color, 2)
            cv2.putText(frame, label, (f.x, max(f.y - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def run(self):
        cap = open_source(self.spec)
        i = 0
        while self.running:
            if cap is None or not cap.isOpened():
                self.status = "reconnecting"
                time.sleep(1.0)
                cap = open_source(self.spec)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                self.status = "reconnecting"
                cap.release()
                time.sleep(0.5)
                cap = open_source(self.spec)
                continue

            self.status = "live"
            if i % DETECT_EVERY == 0:
                try:
                    self._faces = self.detector(frame)
                except Exception:
                    self._faces = []
                now = time.monotonic()
                with self.lock:
                    for f in self._faces:
                        if f.known:
                            self.last_seen[f.name] = now
            annotated = self._draw(frame, self._faces)
            with self.lock:
                self.frame = annotated
            i += 1

        if cap is not None:
            cap.release()

    def snapshot(self):
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
            seen = dict(self.last_seen)
        return frame, seen, self.status


def letterbox(frame, w, h):
    """Resize frame to fit w x h, preserving aspect, padding with black."""
    fh, fw = frame.shape[:2]
    s = min(w / fw, h / fh)
    nw, nh = int(fw * s), int(fh * s)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    x, y = (w - nw) // 2, (h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def make_tile(stream):
    frame, _, status = stream.snapshot()
    if frame is None:
        tile = np.zeros((CELL_H, CELL_W, 3), dtype=np.uint8)
        cv2.putText(tile, f"{status}...", (12, CELL_H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, GREY, 2)
    else:
        tile = letterbox(frame, CELL_W, CELL_H)
    # name banner + live/reconnecting dot
    cv2.rectangle(tile, (0, 0), (CELL_W, 24), (0, 0, 0), cv2.FILLED)
    dot = GREEN if status == "live" else RED
    cv2.circle(tile, (12, 12), 5, dot, cv2.FILLED)
    cv2.putText(tile, stream.name, (26, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return tile


def compose_grid(tiles):
    n = len(tiles)
    cols = max(1, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)
    canvas = np.zeros((rows * CELL_H, cols * CELL_W, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * CELL_H:(r + 1) * CELL_H, c * CELL_W:(c + 1) * CELL_W] = tile
    return canvas


def aggregate_people(streams):
    """name -> sorted list of camera names where seen within RECENCY."""
    now = time.monotonic()
    people = {}
    for s in streams:
        _, seen, _ = s.snapshot()
        for name, t in seen.items():
            if now - t <= RECENCY:
                people.setdefault(name, set()).add(s.name)
    return {k: sorted(v) for k, v in sorted(people.items())}


def render_sidebar(height, people):
    panel = np.full((height, SIDEBAR_W, 3), 24, dtype=np.uint8)
    cv2.putText(panel, f"In the room ({len(people)})", (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.line(panel, (12, 42), (SIDEBAR_W - 12, 42), (80, 80, 80), 1)
    y = 70
    if not people:
        cv2.putText(panel, "(nobody recognised)", (14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREY, 1)
    for name, cams in people.items():
        cv2.circle(panel, (18, y - 5), 5, GREEN, cv2.FILLED)
        cv2.putText(panel, name, (32, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1)
        cv2.putText(panel, ", ".join(cams), (32, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, GREY, 1)
        y += 48
    return panel


def load_config():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get(
        "FACE_ID_CAMERAS", os.path.join(os.path.dirname(__file__), "cameras.json"))
    if os.path.exists(path):
        with open(path) as fh:
            specs = json.load(fh)
        print(f"[config] {len(specs)} camera(s) from {path}")
        return specs
    print(f"[config] {path} not found — defaulting to the local colour webcam.")
    print("         Copy cameras.example.json -> cameras.json to add ESP32 feeds.")
    return [{"name": "Webcam"}]


def main():
    specs = load_config()
    streams = [Stream(s) for s in specs]
    print(f"[engine] {streams[0].detector.engine} | "
          f"known: {', '.join(streams[0].detector.people) or '(none — enrol first)'}")
    for s in streams:
        s.start()

    try:
        while True:
            grid = compose_grid([make_tile(s) for s in streams])
            sidebar = render_sidebar(grid.shape[0], aggregate_people(streams))
            cv2.imshow("Room View - Q to quit", np.hstack([grid, sidebar]))
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        for s in streams:
            s.running = False
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
