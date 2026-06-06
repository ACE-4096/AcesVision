"""Face recognition -> OBS virtual camera bridge.

Reads the real webcam, draws recognition boxes, and republishes the annotated
frames to a v4l2loopback virtual camera. In OBS you then add that device as a
"Video Capture Device" source and the boxes are baked into the feed.

Pipeline:  real webcam  ->  this script (detect + draw)  ->  /dev/videoN (loopback)  ->  OBS

PREREQUISITE (one-time, needs sudo — see README "OBS integration"):
    sudo apt install v4l2loopback-dkms v4l2loopback-utils
    sudo modprobe v4l2loopback video_nr=20 card_label="FaceID Cam" exclusive_caps=1

Then:
    python obs_virtualcam.py

Detection is expensive, so it runs every FACE_ID_DETECT_EVERY frames; the last
known boxes are redrawn on every frame to keep the output smooth.

Env vars:
    FACE_ID_ENGINE=lbph        'lbph' (your 2016 system) or 'dlib' (more accurate)
    FACE_ID_VCAM=/dev/video20  loopback device to publish to (else auto-detect)
    FACE_ID_CAM=10             real camera index (auto-probed otherwise)
    FACE_ID_DETECT_EVERY=5     run detection every N frames (higher = faster/laggier boxes)
    FACE_ID_LBPH_THRESH=70     LBPH match distance (lower = stricter)   [lbph]
    FACE_ID_TOLERANCE=0.6      dlib match distance (lower = stricter)   [dlib]
    FACE_ID_MODEL=hog          'hog' or 'cnn'                           [dlib]
    FACE_ID_SCALE=0.25         detection downscale                     [dlib]
"""
import os

import cv2
import numpy as np
import pyvirtualcam

from camera import open_camera

ENGINE = os.environ.get("FACE_ID_ENGINE", "lbph").lower()
VCAM_DEVICE = os.environ.get("FACE_ID_VCAM")            # None -> auto-detect
DETECT_EVERY = int(os.environ.get("FACE_ID_DETECT_EVERY", "5"))


# --- LBPH engine (your 2016 system) ----------------------------------------
def make_lbph():
    import lbph_recognize as L
    threshold = float(os.environ.get("FACE_ID_LBPH_THRESH", "70"))
    cascade = cv2.CascadeClassifier(L.CASCADE)
    recognizer, names = L.train(cascade)
    if recognizer is None:
        return None

    def detect(frame):
        """Return list of (x, y, w, h, label, color)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        out = []
        for (x, y, w, h) in cascade.detectMultiScale(gray, 1.3, 5):
            lbl, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf <= threshold:
                out.append((x, y, w, h, f"{names[lbl]} ({conf:.0f})", (0, 200, 0)))
            else:
                out.append((x, y, w, h, f"Unknown ({conf:.0f})", (40, 40, 220)))
        return out

    return detect


# --- dlib engine (face_recognition) ----------------------------------------
def make_dlib():
    import face_recognition
    from recognize import load_known_faces
    tol = float(os.environ.get("FACE_ID_TOLERANCE", "0.6"))
    model = os.environ.get("FACE_ID_MODEL", "hog")
    scale = float(os.environ.get("FACE_ID_SCALE", "0.5"))
    inv = 1.0 / scale
    known_enc, known_names = load_known_faces()
    if not known_enc:
        return None

    def detect(frame):
        small = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb, model=model)
        encs = face_recognition.face_encodings(rgb, locs)
        out = []
        for (top, right, bottom, left), enc in zip(locs, encs):
            dists = face_recognition.face_distance(known_enc, enc)
            name, color = "Unknown", (40, 40, 220)
            if len(dists):
                best = int(np.argmin(dists))
                if dists[best] <= tol:
                    name = f"{known_names[best]} ({dists[best]:.2f})"
                    color = (0, 200, 0)
            x, y = int(left * inv), int(top * inv)
            w = int((right - left) * inv)
            h = int((bottom - top) * inv)
            out.append((x, y, w, h, name, color))
        return out

    return detect


def draw(frame, boxes):
    for (x, y, w, h, label, color) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(y - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def main():
    detect = make_dlib() if ENGINE == "dlib" else make_lbph()
    if detect is None:
        print('No enrolled faces. Run:  python enroll.py "Your Name"')
        return
    print(f"[engine] {ENGINE} | detecting every {DETECT_EVERY} frame(s)")

    cap = open_camera()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 30

    cam_kwargs = dict(width=width, height=height, fps=fps,
                      fmt=pyvirtualcam.PixelFormat.BGR)
    if VCAM_DEVICE:
        cam_kwargs["device"] = VCAM_DEVICE

    try:
        vcam = pyvirtualcam.Camera(**cam_kwargs)
    except RuntimeError as e:
        print(f"[error] could not open virtual camera: {e}")
        print("Is v4l2loopback loaded? See README 'OBS integration'.")
        cap.release()
        return

    with vcam:
        print(f"[virtualcam] publishing {width}x{height}@{fps} -> {vcam.device}")
        print("Add this device in OBS as a 'Video Capture Device'. Ctrl+C to stop.")
        boxes = []
        i = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                if i % DETECT_EVERY == 0:
                    boxes = detect(frame)
                i += 1
                draw(frame, boxes)
                vcam.send(frame)
                vcam.sleep_until_next_frame()
        except KeyboardInterrupt:
            print("\n[stop] shutting down")
        finally:
            cap.release()


if __name__ == "__main__":
    main()
