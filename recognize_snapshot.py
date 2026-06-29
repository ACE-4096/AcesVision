"""recognize_snapshot.py — name the faces in a single image, for watch_person.

Runs the face engine (engine.build_detector) on one image and prints the
recognised enrolled names, one per line. Prints nothing if no enrolled face is
present (callers treat that as "Unknown").

    python recognize_snapshot.py /path/to/frame.jpg

Kept separate so it can run under the face-id venv (which has a working dlib /
face_recognition) while watch_person.py runs YOLO under a torch-capable venv.
"""
import sys
from pathlib import Path

import cv2

_REPO = Path(__file__).parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from engine import build_detector


def main():
    if len(sys.argv) < 2:
        print("usage: recognize_snapshot.py <image>", file=sys.stderr)
        return 2
    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"could not read {sys.argv[1]}", file=sys.stderr)
        return 1
    det = build_detector()
    names = sorted({f.name for f in det(img) if f.known and f.name})
    for n in names:
        print(n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
