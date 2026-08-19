"""Enrol a person's face from the local webcam OR a network camera (ESP32-CAM).

    python enroll.py "Toby"                              # local colour webcam
    python enroll.py "Toby" --source ESP32-CAM           # a name from cameras.json
    python enroll.py "Toby" --source http://192.168.68.111:81/stream
    python enroll.py "Toby" --source 9 --model hog       # explicit index + model

SPACE captures (saves a face crop only when exactly one face is found), Q quits.
Enrol at the distances and angles you'll actually be recognised at — and from
the SAME camera, since a face encodes differently across cameras/lighting.

Network cams are usually mounted at an awkward angle, so detection defaults to
the slower-but-robust CNN model: after SPACE there's a ~2s pause while it checks.

This script is engine-agnostic and deliberately so: it saves face-crop JPEGs
and nothing else. No embedding is computed here and none is written to disk,
so the crops stay valid across an engine or model change — whichever engine is
selected re-encodes them from scratch at process start. dlib is used here only
to *find* the face to crop; the recognition pipeline re-detects with its own
detector when it enrols, which is what keeps enrolment and query in the same
embedding space (engine.py, arcface.ArcFacePipeline).
"""
import argparse
import json
from pathlib import Path

import cv2
import face_recognition

import camera

KNOWN_DIR = Path(__file__).parent / "known_faces"
MARGIN = 0.4   # fraction of the face box added as crop margin


def resolve_source(arg):
    if arg is None:
        return {"kind": "local"}
    cfg = Path(__file__).parent / "cameras.json"
    if cfg.exists():
        for s in json.load(open(cfg)):
            if s.get("name", "").lower() == str(arg).lower():
                if "url" in s:
                    return {"kind": "url", "url": s["url"]}
                if "index" in s:
                    return {"kind": "index", "index": s["index"]}
                return {"kind": "local"}   # named entry, auto-detect colour cam
    a = str(arg)
    if a.startswith(("http", "rtsp")):
        return {"kind": "url", "url": a}
    if a.isdigit():
        return {"kind": "index", "index": int(a)}
    raise SystemExit(f"unknown --source '{arg}' (not a cameras.json name, URL, or index)")


def open_source(src):
    if src["kind"] == "url":
        cap = cv2.VideoCapture(src["url"], cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap
    if src["kind"] == "index":
        return camera.open_camera(preferred=src["index"])
    return camera.open_camera()


def detect(frame, model, scale):
    small = frame if scale == 1.0 else cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    locs = face_recognition.face_locations(rgb, model=model)
    inv = 1.0 / scale
    return [(int(t * inv), int(r * inv), int(b * inv), int(l * inv))
            for (t, r, b, l) in locs]


def crop_face(frame, box):
    t, r, b, l = box
    h, w = frame.shape[:2]
    mh, mw = int((b - t) * MARGIN), int((r - l) * MARGIN)
    crop = frame[max(0, t - mh):min(h, b + mh), max(0, l - mw):min(w, r + mw)]
    # Cap size so the CNN load-time fallback stays fast; 400px keeps ample
    # detail (the encoder works at 150px internally).
    big = max(crop.shape[:2])
    if big > 400:
        s = 400 / big
        crop = cv2.resize(crop, (0, 0), fx=s, fy=s)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--source", default=None,
                    help="cameras.json name, URL, or index (default: local webcam)")
    ap.add_argument("--model", default=None, choices=["hog", "cnn"])
    args = ap.parse_args()

    src = resolve_source(args.source)
    model = args.model or ("cnn" if src["kind"] == "url" else "hog")
    scale = 1.0 if model == "hog" else 0.3   # cnn at full res is ~20s/frame
    live_preview = model == "hog"            # cnn is too slow to box every frame

    person_dir = KNOWN_DIR / args.name.strip()
    person_dir.mkdir(parents=True, exist_ok=True)
    count = len(list(person_dir.glob("*.jpg")))
    where = args.source or "local webcam"
    print(f"Enrolling '{args.name}' from {where} (model={model}). "
          f"{count} photo(s) already saved.")
    print("SPACE = capture, Q = quit." + ("" if live_preview else
          "  (cnn: box is checked on capture, ~2s pause)"))

    cap = open_source(src)
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            if cv2.waitKey(100) & 0xFF == ord("q"):
                break
            continue
        view = frame.copy()
        if live_preview:
            for (t, r, b, l) in detect(frame, model, scale):
                cv2.rectangle(view, (l, t), (r, b), (0, 200, 0), 2)
        cv2.putText(view, f"{args.name}: {count} saved | {model}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Enroll - SPACE capture, Q quit", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord(" "):
            if not live_preview:
                print("  checking (cnn)...")
            boxes = detect(frame, model, scale)
            if len(boxes) != 1:
                print(f"  skipped: need exactly 1 face, saw {len(boxes)}")
                continue
            crop = crop_face(frame, boxes[0])
            count += 1
            out = person_dir / f"{count:02d}.jpg"
            cv2.imwrite(str(out), crop)
            print(f"  saved {out}  ({crop.shape[1]}x{crop.shape[0]})")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. '{args.name}' now has {count} photo(s) in {person_dir}")


if __name__ == "__main__":
    main()
