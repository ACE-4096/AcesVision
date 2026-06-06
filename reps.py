"""Count weightlifting reps from a camera, using body-pose joint angles.

    python reps.py                 # bicep curl (default)
    python reps.py squat
    python reps.py pushup
    python reps.py press

It finds your body (MediaPipe Pose), measures the working joint's angle, and
counts a rep each time that angle completes its full range (e.g. arm extended ->
curled -> extended). Big live counter, stage indicator, range bar. R resets, Q quits.

Add your own in EXERCISES: a joint triple (3 landmark indices) + rest/active angles.
"""
import os
import sys
from collections import namedtuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from camera import open_camera

MODEL = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker.task")

# MediaPipe Pose landmark indices used below:
#  shoulders 11/12, elbows 13/14, wrists 15/16, hips 23/24, knees 25/26, ankles 27/28
Ex = namedtuple("Ex", "name joints rest active")
EXERCISES = {
    "curl":   Ex("Bicep Curl", [(11, 13, 15), (12, 14, 16)], 150, 50),
    "squat":  Ex("Squat",      [(23, 25, 27), (24, 26, 28)], 168, 100),
    "pushup": Ex("Push-up",    [(11, 13, 15), (12, 14, 16)], 160, 95),
    "press":  Ex("Shoulder Press", [(11, 13, 15), (12, 14, 16)], 70, 160),
}


def angle3(a, b, c):
    a, b, c = (np.asarray(p, float) for p in (a, b, c))
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


class RepCounter:
    """State machine: count a rep on each rest -> active -> rest cycle."""

    def __init__(self, rest, active):
        self.rest, self.active = rest, active
        self.flexed = active < rest          # curls/squats flex to a smaller angle
        self.stage = "rest"
        self.count = 0

    def update(self, angle):
        reached = angle <= self.active if self.flexed else angle >= self.active
        back = angle >= self.rest if self.flexed else angle <= self.rest
        if reached:
            self.stage = "active"
        elif back and self.stage == "active":
            self.stage = "rest"
            self.count += 1
            return True
        return False

    def progress(self, angle):
        p = (angle - self.rest) / (self.active - self.rest + 1e-9)
        return max(0.0, min(1.0, p))


def landmarks_px(res, w, h):
    if not res.pose_landmarks:
        return None
    return [(lm.x * w, lm.y * h, lm.visibility) for lm in res.pose_landmarks[0]]


def joint_angle(pts, joints):
    vals = []
    for a, b, c in joints:
        if pts[a][2] > 0.4 and pts[b][2] > 0.4 and pts[c][2] > 0.4:
            vals.append(angle3(pts[a][:2], pts[b][:2], pts[c][:2]))
    return float(np.mean(vals)) if vals else None


def draw(frame, pts, ex, counter, angle, flash):
    h, w = frame.shape[:2]
    # faint full skeleton dots
    if pts:
        for x, y, v in pts:
            if v > 0.4:
                cv2.circle(frame, (int(x), int(y)), 3, (120, 120, 120), -1)
        # bold working joints
        for a, b, c in ex.joints:
            if pts[a][2] > 0.4 and pts[b][2] > 0.4 and pts[c][2] > 0.4:
                pa, pb, pc = (tuple(map(int, pts[i][:2])) for i in (a, b, c))
                cv2.line(frame, pa, pb, (0, 220, 0), 3)
                cv2.line(frame, pb, pc, (0, 220, 0), 3)
                cv2.circle(frame, pb, 8, (0, 220, 0), -1)

    # counter panel
    cv2.rectangle(frame, (0, 0), (240, 150), (25, 25, 25), cv2.FILLED)
    cv2.putText(frame, ex.name, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    col = (60, 240, 60) if flash else (255, 255, 255)
    cv2.putText(frame, str(counter.count), (12, 120), cv2.FONT_HERSHEY_SIMPLEX, 3.0, col, 6)
    cv2.putText(frame, f"REPS", (140, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
    stage_col = (60, 200, 255) if counter.stage == "active" else (200, 200, 200)
    cv2.putText(frame, counter.stage.upper(), (140, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, stage_col, 2)

    # range-of-motion bar (right edge)
    if angle is not None:
        p = counter.progress(angle)
        x0, y0, y1 = w - 40, 60, h - 60
        cv2.rectangle(frame, (x0, y0), (x0 + 22, y1), (60, 60, 60), 2)
        fill = int(y1 - p * (y1 - y0))
        cv2.rectangle(frame, (x0, fill), (x0 + 22, y1),
                      (0, 220, 0) if p > 0.85 else (0, 160, 220), cv2.FILLED)
        cv2.putText(frame, f"{angle:.0f}", (x0 - 6, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(frame, "step into frame", (260, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame


def main():
    key = sys.argv[1].lower() if len(sys.argv) > 1 else "curl"
    if key not in EXERCISES:
        print(f"unknown exercise '{key}'. choices: {', '.join(EXERCISES)}")
        return
    ex = EXERCISES[key]
    opts = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL), num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(opts)
    counter = RepCounter(ex.rest, ex.active)
    print(f"{ex.name}: stand in view, R resets, Q quits.")

    cap = open_camera()
    flash = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        pts = landmarks_px(res, frame.shape[1], frame.shape[0])
        angle = joint_angle(pts, ex.joints) if pts else None
        if angle is not None and counter.update(angle):
            flash = 6
        draw(frame, pts, ex, counter, angle, flash > 0)
        flash = max(0, flash - 1)
        cv2.imshow(f"Rep Counter — {ex.name} (R reset, Q quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            counter.count = 0
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
