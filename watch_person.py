"""watch_person.py — alert when a person comes into view on a camera.

Taps the existing face-id stack: opens one or many cameras (DroidCam phone,
Reolink security cameras, ESP32-CAM, local webcams), runs YOLO person detection
(a person triggers even with no visible face), then labels *who* it is with the
face engine (engine.py). On a new arrival it fires a desktop pop-up and/or a
Telegram push with a snapshot, tagged with which camera saw them.

    python watch_person.py                       # all cams in watch_cameras.json,
                                                 # else local webcam index 0
    python watch_person.py --source 0            # one local webcam, index 0
    python watch_person.py --source http://phone.local:4747/video
    python watch_person.py --cameras my_cams.json   # explicit multi-cam config
    WATCH_ALERT=telegram python watch_person.py  # phone push only

Camera config (JSON list) — each entry is one of:
    {"name": "DroidCam",   "url": "http://phone.local:4747/video"}
    {"name": "Front Door", "type": "reolink", "ip": "camera.local",
                           "user": "admin", "password": "secret", "stream": "sub"}
    {"name": "Lounge",     "url": "rtsp://user:pass@camera.local:554/h264Preview_01_sub"}
    {"name": "Webcam",     "index": 0}
See reolink.py for the full Reolink spec, and watch_cameras.example.json.

Why YOLO and not the presence path: presence.py only reacts to *recognised*
enrolled faces. "A person is in view" must fire for strangers and for backs of
heads too, so detection is YOLO's `person` class; the face engine only adds the
name when a face happens to be visible.

Telegram: reads TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID from the environment, a
local (gitignored) `.env`, or the explicit `ACESVISION_SECRETS_FILE`. Use a
dedicated @BotFather bot or any token you control; get your chat id from
@userinfobot. Leave creds unset to silently skip Telegram (desktop alerts still
fire).

Env knobs:
    WATCH_SOURCE        camera source (overridden by --source). Default local webcam (0).
    WATCH_ALERT         both | telegram | desktop                     (default both)
    WATCH_CONF          YOLO person confidence 0-1                    (default 0.40)
    WATCH_DETECT_EVERY  run YOLO every N frames                       (default 5)
    WATCH_MIN_SIGHTINGS detections before "arrived" (debounce)        (default 2)
    WATCH_CLEAR_AFTER   seconds with no person before "left"          (default 15)
    WATCH_COOLDOWN      min seconds between alerts                     (default 60)
    WATCH_SNAPSHOT      1 = attach annotated frame to Telegram        (default 1)
    ACESVISION_SECRETS_FILE  optional path to a dotenv-style credential file
    plus the engine.py face vars (FACE_ID_TOLERANCE, FACE_ID_ENGINE, ...).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

_REPO = Path(__file__).parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import reolink  # noqa: E402  Reolink RTSP/snapshot URL builder
import roomview  # noqa: E402  reuse open_source (network/local camera opener)

DEFAULT_SOURCE = os.environ.get("WATCH_SOURCE", "0")
DEFAULT_CAMERAS = _REPO / "watch_cameras.json"
_YOLO_LOCK = threading.Lock()  # ultralytics inference serialised across cam threads
# Recognition runs in the face-id venv (working dlib); this watcher runs YOLO in
# a torch-capable venv. We shell out to recognize_snapshot.py for the "who".
FACE_ID_PYTHON = os.environ.get(
    "FACE_ID_PYTHON", str(_REPO / ".venv" / "bin" / "python"))
RECOGNIZER = str(_REPO / "recognize_snapshot.py")
ALERT_MODE = os.environ.get("WATCH_ALERT", "both").lower()
CONF = float(os.environ.get("WATCH_CONF", "0.40"))
DETECT_EVERY = int(os.environ.get("WATCH_DETECT_EVERY", "5"))
MIN_SIGHTINGS = int(os.environ.get("WATCH_MIN_SIGHTINGS", "2"))
CLEAR_AFTER = float(os.environ.get("WATCH_CLEAR_AFTER", "15"))
COOLDOWN = float(os.environ.get("WATCH_COOLDOWN", "60"))
SNAPSHOT = os.environ.get("WATCH_SNAPSHOT", "1") == "1"
PERSON_CLASS = 0  # COCO class id for "person"


# --- Telegram --------------------------------------------------------------
def _load_telegram_creds():
    """(token, chat_id). Env vars win, then the local .env or an explicit file.
    Either may be None — in which case the Telegram channel is skipped."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID") or os.environ.get("TELEGRAM_OWNER_CHAT_ID")
    candidates = [_REPO / ".env"]
    if os.environ.get("ACESVISION_SECRETS_FILE"):
        candidates.append(Path(os.environ["ACESVISION_SECRETS_FILE"]))
    for env_path in candidates:
        if token and chat:
            break
        if not env_path.exists():
            continue
        for line in env_path.read_text(errors="ignore").splitlines():
            s = line.strip()
            if s.startswith("export "):
                s = s[7:]
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k == "TELEGRAM_BOT_TOKEN" and not token:
                token = v
            elif k in ("TELEGRAM_CHAT_ID", "TELEGRAM_OWNER_CHAT_ID") and not chat:
                chat = v
    return (token or None), (chat or None)


def _send_telegram(text, image_path=None):
    import requests
    token, chat = _load_telegram_creds()
    if not (token and chat):
        print("    [telegram] no creds found — skipping")
        return
    try:
        if image_path and Path(image_path).exists():
            with open(image_path, "rb") as f:
                r = requests.post(
                    f"https://api.telegram.org/bot{token}/sendPhoto",
                    data={"chat_id": chat, "caption": text},
                    files={"photo": f}, timeout=15)
        else:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat, "text": text}, timeout=15)
        print(f"    [telegram] {'ok' if r.ok else 'HTTP ' + str(r.status_code)}")
    except Exception as e:
        print(f"    [telegram] FAILED: {e}")


def _notify_desktop(title, body):
    try:
        subprocess.Popen(["notify-send", "-u", "critical", title, body])
    except Exception as e:
        print(f"    [desktop] FAILED: {e}")


def alert(label, frame, camera="camera"):
    stamp = datetime.now().strftime("%H:%M:%S")
    title = f"Person in view — {camera}"
    body = f"{label} at {stamp}"
    print(f"[ALERT] [{camera}] {body}")
    if ALERT_MODE in ("both", "desktop"):
        _notify_desktop(title, body)
    if ALERT_MODE in ("both", "telegram"):
        img = None
        if SNAPSHOT and frame is not None:
            img = str(Path("/tmp") / f"watch_person_{int(time.time())}.jpg")
            cv2.imwrite(img, frame)
        _send_telegram(f"\U0001f6a8 {title}: {body}", img)


# --- recognition: who is this person? --------------------------------------
def label_people(frame):
    """Recognised names in frame, else 'Unknown'. Taps the face engine by
    shelling out to recognize_snapshot.py under the face-id venv."""
    if not (Path(FACE_ID_PYTHON).exists() and Path(RECOGNIZER).exists()):
        return "Unknown"
    snap = str(Path("/tmp") / f"watch_recog_{int(time.time())}.jpg")
    try:
        cv2.imwrite(snap, frame)
        out = subprocess.run([FACE_ID_PYTHON, RECOGNIZER, snap],
                             capture_output=True, text=True, timeout=30)
        names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
        return ", ".join(names) if names else "Unknown"
    except Exception as e:
        print(f"    [face] recognise error: {e}")
        return "Unknown"
    finally:
        try:
            os.remove(snap)
        except OSError:
            pass


def annotate(frame, boxes, label):
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)
    return frame


# --- camera config ---------------------------------------------------------
def resolve_spec(entry):
    """Map a config entry to (display_name, openable_spec for roomview).

    entry may be a raw string ('0' or a URL) or a dict (url / index / reolink)."""
    if isinstance(entry, str):
        entry = ({"index": int(entry)} if entry.isdigit() else {"url": entry})
    name = entry.get("name", "camera")
    if str(entry.get("type", "")).lower() == "reolink":
        return name, {"url": reolink.rtsp_url(entry)}
    if "url" in entry:
        return name, {"url": entry["url"]}
    if "index" in entry:
        return name, {"index": int(entry["index"])}
    return name, {}  # bare {"name": ...} -> roomview auto-probes a local webcam


def load_cameras(path):
    """Read a cameras JSON list -> [(name, spec), ...]."""
    entries = json.loads(Path(path).read_text())
    return [resolve_spec(e) for e in entries]


def _redact(spec):
    """Hide credentials in an rtsp:// URL for logging."""
    url = spec.get("url", "")
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url or f"index {spec.get('index', '?')}" or "auto"


# --- one camera, one thread ------------------------------------------------
class CameraWatcher(threading.Thread):
    def __init__(self, name, spec, yolo):
        super().__init__(daemon=True)
        self.name = name
        self.spec = spec
        self.yolo = yolo
        self.running = True

    def run(self):
        cap = roomview.open_source(self.spec)
        present = False
        last_seen = last_alert = 0.0
        sightings = i = 0
        while self.running:
            if cap is None or not cap.isOpened():
                print(f"[{self.name}] stream down — retrying in 2s...")
                time.sleep(2.0)
                cap = roomview.open_source(self.spec)
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                time.sleep(0.5)
                cap = roomview.open_source(self.spec)
                continue

            if i % DETECT_EVERY == 0:
                now = time.monotonic()
                with _YOLO_LOCK:
                    res = self.yolo(frame, verbose=False, conf=CONF,
                                    classes=[PERSON_CLASS])[0]
                boxes = [tuple(int(v) for v in b.xyxy[0].tolist()) for b in res.boxes]
                if boxes:
                    last_seen = now
                    sightings += 1
                    if not present and sightings >= MIN_SIGHTINGS:
                        present = True
                        if now - last_alert >= COOLDOWN:
                            last_alert = now
                            label = label_people(frame)
                            alert(label, annotate(frame.copy(), boxes, label),
                                  camera=self.name)
                else:
                    sightings = 0
                    if present and now - last_seen > CLEAR_AFTER:
                        present = False
                        print(f"[{self.name}] clear ({datetime.now():%H:%M:%S})")
            i += 1
        if cap is not None:
            cap.release()


def main():
    ap = argparse.ArgumentParser(description="Alert when a person is in view.")
    ap.add_argument("--source", help="single camera URL or local index.")
    ap.add_argument("--cameras", help="path to a cameras JSON config (multi-cam).")
    args = ap.parse_args()

    if args.source:
        cams = [resolve_spec(args.source)]
    elif args.cameras:
        cams = load_cameras(args.cameras)
    elif DEFAULT_CAMERAS.exists():
        cams = load_cameras(DEFAULT_CAMERAS)
    else:
        source = DEFAULT_SOURCE.strip()
        default_spec = ({"name": "Webcam", "index": int(source)}
                        if source.isdigit()
                        else {"name": "Network camera", "url": source})
        cams = [resolve_spec(default_spec)]

    from ultralytics import YOLO
    model_path = str(_REPO / "yolov8n.pt")
    yolo = YOLO(model_path if Path(model_path).exists() else "yolov8n.pt")
    face_ok = Path(FACE_ID_PYTHON).exists() and Path(RECOGNIZER).exists()
    print(f"[watch] YOLO={Path(model_path).name} | "
          f"face_recog={'on' if face_ok else 'off'} | alert={ALERT_MODE}")
    for name, spec in cams:
        print(f"    watching {name}: {_redact(spec)}")

    workers = [CameraWatcher(name, spec, yolo) for name, spec in cams]
    for w in workers:
        w.start()
    try:
        while any(w.is_alive() for w in workers):
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[watch] stopping...")
        for w in workers:
            w.running = False


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[watch] stopped")
