"""Live webcam face recognition.

Loads every photo under known_faces/<Name>/*.jpg, builds face encodings, then
runs the webcam: detects faces, draws bounding boxes, and labels each one with
the best-matching known name (or "Unknown").

    python recognize.py

Keys:  Q = quit.

Tuning (env vars):
    FACE_ID_CAM=10        force a camera index
    FACE_ID_TOLERANCE=0.6 match strictness; LOWER = stricter (0.5 is tight)
    FACE_ID_MODEL=hog     'hog' (fast, CPU) or 'cnn' (accurate, wants a GPU)
    FACE_ID_SCALE=0.25    downscale factor for detection speed
"""
import os
from pathlib import Path

import cv2
import numpy as np
import face_recognition

from camera import open_camera

KNOWN_DIR = Path(__file__).parent / "known_faces"
TOLERANCE = float(os.environ.get("FACE_ID_TOLERANCE", "0.6"))
MODEL = os.environ.get("FACE_ID_MODEL", "hog")
SCALE = float(os.environ.get("FACE_ID_SCALE", "0.5"))
IMG_EXT = {".jpg", ".jpeg", ".png"}


def load_known_faces():
    """Return (encodings, names) from known_faces/<Name>/*.jpg."""
    encodings, names = [], []
    if not KNOWN_DIR.exists():
        return encodings, names
    for person_dir in sorted(p for p in KNOWN_DIR.iterdir() if p.is_dir()):
        per_person = 0
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXT:
                continue
            image = face_recognition.load_image_file(str(img_path))
            found = face_recognition.face_encodings(image)
            if not found:
                print(f"[warn] no face in {img_path}, skipping")
                continue
            encodings.append(found[0])
            names.append(person_dir.name)
            per_person += 1
        if per_person:
            print(f"[load] {person_dir.name}: {per_person} encoding(s)")
    return encodings, names


def main():
    known_encodings, known_names = load_known_faces()
    if not known_encodings:
        print("No known faces yet. Run:  python enroll.py \"Your Name\"")
        return
    print(f"[load] {len(known_encodings)} encoding(s) across "
          f"{len(set(known_names))} person(s). Model={MODEL} tol={TOLERANCE}")

    cap = open_camera()
    inv = 1.0 / SCALE          # scale boxes back up to full-res frame
    locations, face_names = [], []
    process = True             # process every other frame to keep it smooth

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if process:
            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model=MODEL)
            encs = face_recognition.face_encodings(rgb, locations)

            face_names = []
            for enc in encs:
                name = "Unknown"
                dists = face_recognition.face_distance(known_encodings, enc)
                if len(dists) > 0:
                    best = int(np.argmin(dists))
                    if dists[best] <= TOLERANCE:
                        name = f"{known_names[best]} ({dists[best]:.2f})"
                face_names.append(name)
        process = not process

        for (top, right, bottom, left), name in zip(locations, face_names):
            top, right = int(top * inv), int(right * inv)
            bottom, left = int(bottom * inv), int(left * inv)
            known = not name.startswith("Unknown")
            color = (0, 200, 0) if known else (40, 40, 220)  # BGR
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 28), (right, bottom),
                          color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 8),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Face ID - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
