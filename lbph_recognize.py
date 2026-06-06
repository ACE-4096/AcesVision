"""LBPH + Haar cascade face recognition — the 2015/2016 OpenCV approach.

This is a faithful, fixed-up version of the original 2016 script: Haar cascade
for face *detection* (cv2.CascadeClassifier.detectMultiScale) and OpenCV's LBPH
recognizer (cv2.face.LBPHFaceRecognizer) for *identification*. Needs
opencv-contrib-python (the `cv2.face` module). No dlib, no face_recognition.

It trains on startup from  known_faces/<Name>/*.jpg  (same folders enroll.py
fills), assigning each person an integer label, then runs the webcam loop.

    python lbph_recognize.py

Keys:  Q or ESC = quit.

Tuning (env vars):
    FACE_ID_CAM=10          force a camera index
    FACE_ID_LBPH_THRESH=70  max LBPH distance to accept a match.
                            LOWER = stricter. 0 = identical. Typical 40-80.
                            (Your 2016 code used 10, which almost never matched.)
"""
import os
from pathlib import Path

import cv2
import numpy as np

from camera import open_camera

KNOWN_DIR = Path(__file__).parent / "known_faces"
THRESHOLD = float(os.environ.get("FACE_ID_LBPH_THRESH", "70"))
IMG_EXT = {".jpg", ".jpeg", ".png"}
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def train(face_cascade):
    """Detect a face in each known photo, train LBPH. Returns (recognizer, names).

    names[label] -> person's name.
    """
    images, labels, names = [], [], []
    for person_dir in sorted(p for p in KNOWN_DIR.iterdir() if p.is_dir()):
        label = len(names)            # integer label per person
        names.append(person_dir.name)
        used = 0
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) != 1:
                print(f"[warn] {img_path.name}: {len(faces)} faces, skipping")
                continue
            x, y, w, h = faces[0]
            images.append(gray[y:y + h, x:x + w])
            labels.append(label)
            used += 1
        print(f"[train] {person_dir.name} (label {label}): {used} sample(s)")

    if not images:
        return None, names
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    return recognizer, names


def main():
    face_cascade = cv2.CascadeClassifier(CASCADE)
    recognizer, names = train(face_cascade)
    if recognizer is None:
        print('No training faces. Run:  python enroll.py "Your Name"')
        return
    print(f"[ready] {len(names)} person(s): {', '.join(names)} | "
          f"threshold={THRESHOLD}")

    cap = open_camera()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            label, conf = recognizer.predict(roi)   # conf: lower = better
            if conf <= THRESHOLD:
                name = f"{names[label]} ({conf:.0f})"
                color = (0, 200, 0)
            else:
                name = f"Unknown ({conf:.0f})"
                color = (40, 40, 220)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("LBPH Face ID - Q to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):     # Q or ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
