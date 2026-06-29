# face-id

Webcam face recognition — detects faces, draws bounding boxes, and labels you
and your child (or "Unknown"). Two interchangeable engines:

| Engine | Script | What it is |
|--------|--------|-----------|
| **LBPH + Haar** | `lbph_recognize.py` | The **2015/2016 OpenCV** approach — your original system, cleaned up. `cv2.face.LBPHFaceRecognizer` + Haar cascade. |
| **dlib / face_recognition** | `recognize.py` | The modern stack (dlib HOG detector + ResNet encoder). More accurate, especially for a child's face. |

Both read the same enrolled photos and both already work in this venv.

## Setup (done already, for reference)

```bash
cd face-id
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1. Enrol the people to recognise

```bash
python enroll.py "Toby"
python enroll.py "Emma"      # your child's name
```

SPACE grabs a frame (only when exactly one face is visible), Q quits. Aim for
**3-6 shots each**, varied angle / expression / lighting. Photos land in
`known_faces/<Name>/`. (You can also drop existing photos in there by hand —
one clear face per image.)

## 2. Run recognition

Your 2016 LBPH system:
```bash
python lbph_recognize.py
```

Or the more accurate dlib version:
```bash
python recognize.py
```

Green box = recognised (with confidence), red = Unknown. **Q** to quit.

## Tuning (env vars)

| Var | Applies to | Default | Meaning |
|-----|-----------|---------|---------|
| `FACE_ID_CAM` | both | auto-probe | Force camera index (this box uses ~9-12, not 0) |
| `FACE_ID_LBPH_THRESH` | LBPH | `70` | Max match distance. **Lower = stricter**, 0 = identical. (Your 2016 `10` almost never matched.) |
| `FACE_ID_TOLERANCE` | dlib | `0.6` | Match strictness, **lower = stricter**. Try `0.5`. |
| `FACE_ID_MODEL` | dlib | `hog` | `hog` (CPU) or `cnn` (needs GPU, more robust) |
| `FACE_ID_SCALE` | dlib | `0.25` | Detection downscale; smaller = faster |

Example:
```bash
FACE_ID_CAM=10 FACE_ID_LBPH_THRESH=60 python lbph_recognize.py
```

## LBPH vs dlib — which to use?

- **LBPH** is what you built in 2016: light, no GPU, trains instantly. But it's
  sensitive to lighting/pose and the confidence numbers need tuning per setup.
- **dlib** (`recognize.py`) is markedly more reliable, particularly for telling
  a parent and child apart. If accuracy matters more than nostalgia, use it.

## OBS integration (overlay boxes on your OBS webcam feed)

`obs_virtualcam.py` reads the real webcam, draws the recognition boxes, and
republishes the result to a **virtual camera**. OBS then adds that virtual
camera as a normal source — boxes baked in. This is far more robust than an
in-OBS Python script (which would have to load dlib/opencv into OBS's embedded
Python).

```
real webcam  ->  obs_virtualcam.py (detect + draw)  ->  /dev/video20 (loopback)  ->  OBS
```

### One-time setup (needs sudo — kernel module)

```bash
sudo apt install v4l2loopback-dkms v4l2loopback-utils
# create a loopback device at /dev/video20 (avoids your real cams at 9-12 and OBS's own virtualcam)
sudo modprobe v4l2loopback video_nr=20 card_label="FaceID Cam" exclusive_caps=1
```

To make the device survive reboots, persist the modprobe options:

```bash
echo 'v4l2loopback' | sudo tee /etc/modules-load.d/v4l2loopback.conf
printf 'options v4l2loopback video_nr=20 card_label="FaceID Cam" exclusive_caps=1\n' \
  | sudo tee /etc/modprobe.d/v4l2loopback.conf
```

### Run the bridge

```bash
source .venv/bin/activate
python obs_virtualcam.py                       # LBPH (default)
# or the more accurate engine:
FACE_ID_ENGINE=dlib python obs_virtualcam.py
# pin the loopback target explicitly if auto-detect picks the wrong one:
FACE_ID_VCAM=/dev/video20 python obs_virtualcam.py
```

### Wire it into OBS

1. Start the bridge (above) — leave it running.
2. In OBS: **Sources → + → Video Capture Device (V4L2)**.
3. Pick **"FaceID Cam"** (`/dev/video20`) as the device.
4. You'll see your webcam with live recognition boxes.

Notes:
- Detection runs every `FACE_ID_DETECT_EVERY` frames (default 5) so the feed
  stays smooth; the boxes refresh a few times a second, the video at full fps.
- Don't point OBS at your *real* camera at the same time you point the bridge at
  it — only one process can hold a V4L2 device. Bridge owns the real cam; OBS
  owns the loopback.

## Room View — many feeds in one window + who's in the room

`roomview.py` pulls multiple cameras (your local colour webcam **and**
ESP32-CAM / any MJPEG / RTSP network streams), recognises faces on each, tiles
them into one grid, and shows a live **"In the room"** sidebar listing everyone
recognised across all feeds and which camera they're on.

```bash
cp cameras.example.json cameras.json   # then edit it
python roomview.py                      # or: python roomview.py my_cams.json
```

`cameras.json` is a list; each camera is one of:

```json
[
  { "name": "Webcam", "index": 9 },
  { "name": "Lounge", "url": "http://192.168.68.50:81/stream" },
  { "name": "Door",   "url": "rtsp://user:pass@192.168.68.51/h264" }
]
```

- **ESP32-CAM**: the stock `CameraWebServer` Arduino sketch streams MJPEG at
  `http://<device-ip>:81/stream` (control UI is on port 80). Put that URL in
  `url`. Quick test outside this app: `ffplay http://<ip>:81/stream`.
- Each feed runs in its own thread with **auto-reconnect** — a feed that drops
  shows "reconnecting…" and rejoins on its own; the others keep running.
- A name stays in the sidebar for `FACE_ID_RECENCY` seconds (default 3) after it
  was last seen, so the list doesn't flicker frame-to-frame.

Tuning (env vars): `FACE_ID_ENGINE` (dlib/lbph), `FACE_ID_DETECT_EVERY` (frames
between recognitions per feed), `FACE_ID_RECENCY` (seconds), `FACE_ID_CELL`
(tile size, e.g. `640x360`), plus the per-engine vars above.

> ESP32-CAMs are low-res and grainy — use `FACE_ID_ENGINE=dlib` for the network
> feeds; LBPH struggles with that image quality.

---

## Person-in-view alerter (`watch_person.py`)

**The ask:** alert me whenever a person comes into view on a camera — using the
DroidCam phone feed (`192.168.1.187`) and the Reolink security cameras — and tap
the existing face recognition to say *who* it is.

**How it works.** Detection is YOLOv8's `person` class, **not** the face engine —
so it fires for strangers and backs of heads too (the `presence.py` path only
reacts to *recognised enrolled* faces). On a debounced new arrival it runs the
calibrated YuNet/dlib engine to label who (or "Unknown") and sends an alert via
**desktop `notify-send` and/or Telegram** (with an annotated snapshot), tagged
with which camera saw them. One thread per camera, each auto-reconnecting.

```bash
./watch.sh                                   # all cams in watch_cameras.json, else DroidCam
./watch.sh --source 0                        # one local webcam
./watch.sh --source http://192.168.1.187:4747/video
./watch.sh --cameras watch_cameras.example.json
WATCH_ALERT=desktop ./watch.sh               # pop-ups only (no Telegram)
```

### ⚠️ Runtime: it needs a torch-capable venv (two-venv split)

`face-id/.venv` has working dlib/face_recognition but its **torch is broken**, so
YOLO can't run there. `watch.sh` therefore runs the watcher under the
**`cv-worker/.venv`** (working torch + ultralytics, AMD ROCm GPU) and shells out
to `face-id/.venv` for the recognition step (`recognize_snapshot.py`). The
launcher wires both automatically; override with `CV_WORKER_PYTHON` /
`FACE_ID_PYTHON` if either venv moves. (The old ROCm venv on the external drive
is gone — `cv-worker/.venv` is the live torch env.)

### Cameras (`watch_cameras.json`)

`cp watch_cameras.example.json watch_cameras.json` then edit. **This file holds
camera passwords and is gitignored.** Each entry is one of:

```json
[
  { "name": "DroidCam",   "url": "http://192.168.1.187:4747/video" },
  { "name": "Front Door", "type": "reolink", "ip": "192.168.1.50",
                          "user": "admin", "password": "secret",
                          "stream": "sub", "channel": 1 },
  { "name": "Lounge",     "url": "rtsp://user:pass@192.168.1.52:554/h264Preview_01_sub" },
  { "name": "Webcam",     "index": 0 }
]
```

- **Reolink** (`reolink.py`): builds `rtsp://…/h264Preview_<ch:02d>_<main|sub>`.
  `stream:"sub"` = lower-res substream (lighter, ideal for detection),
  `"main"` = full-res. `channel:1` for a standalone camera, the channel number
  behind an NVR. Credentials are URL-encoded and **redacted in logs**. Older
  firmware: add `"path":"Preview_01_sub"` to drop the `h264` prefix. `reolink.py`
  also exposes `snapshot_url()` (HTTP JPEG CGI).
- **DroidCam**: start the app on the phone; default MJPEG is `…:4747/video`
  (fallback `…:4747/mjpegfeed`). This box and the phone share `192.168.1.0/24`.

### Telegram (optional — desktop works with zero config)

Reads `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` from the env, else a local
(gitignored) `face-id/.env`, else `ops-hq/.env`. Use a **dedicated @BotFather
bot** (token from BotFather, chat id from @userinfobot) — do **not** reuse the
ops-hq poll-only gate bot. Leave creds unset and Telegram is silently skipped.

### Tuning (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `WATCH_ALERT` | `both` | `both` \| `telegram` \| `desktop` |
| `WATCH_CONF` | `0.40` | YOLO person confidence (0–1) |
| `WATCH_DETECT_EVERY` | `5` | run YOLO every N frames per camera |
| `WATCH_MIN_SIGHTINGS` | `2` | detections before "arrived" (debounce) |
| `WATCH_CLEAR_AFTER` | `15` | seconds with no person before "left" |
| `WATCH_COOLDOWN` | `60` | min seconds between alerts per camera |
| `WATCH_SNAPSHOT` | `1` | attach annotated frame to Telegram |

---

## Use Case A: scan your phone photo library for your face

`scan_photos.py` scans any folder (recursively) and finds every image
containing your enrolled face.  No cloud, no upload, runs entirely on CPU.

### Install the extra dep (HEIC/HEIF for iPhone photos)

```bash
source .venv/bin/activate
pip install pillow-heif          # already in requirements.txt
```

### Run it

```bash
source .venv/bin/activate

# Scan a folder and symlink matches to ./photo_matches/matches/
python scan_photos.py --source /path/to/DCIM

# Stricter match + hard-copy + 8 parallel workers
python scan_photos.py --source /path/to/DCIM --tolerance 0.50 --jobs 8 --copy

# Custom output directory
python scan_photos.py --source /path/to/DCIM --output ~/Desktop/my_photos
```

Flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--source <dir>` | auto-detect GVFS | Directory to scan (recurse). Omit to try auto-detecting a phone mount. |
| `--output <dir>` | `./photo_matches` | Where to write manifest + matches/ |
| `--tolerance <float>` | `0.6` | dlib distance — **lower = stricter**. 0.5 tight/precise, 0.6 relaxed/high recall |
| `--jobs <N>` | half CPU count | Parallel encoding workers |
| `--copy` | (symlink) | Hard-copy matched images instead of symlinking |

Output:
- `<output>/manifest.json` — list of all scanned images with `{path, matched, best_distance, num_faces}`
- `<output>/matches/` — symlinks (or copies) of every matched image

---

## Mounting your phone over USB

The scanner only needs a mounted folder passed to `--source`.  The steps below
produce that mount point.

### iPhone (HEIC photos)

**Option A — ifuse / libimobiledevice (recommended for scripting)**

```bash
# Install once
sudo apt install libimobiledevice-utils ifuse

# Pair the phone (trust prompt appears on screen)
idevicepair pair

# Mount
mkdir -p ~/iphone-dcim
ifuse ~/iphone-dcim

# Scan
python scan_photos.py --source ~/iphone-dcim/DCIM

# Unmount when done
fusermount -u ~/iphone-dcim
```

**Option B — GVFS / Nautilus auto-mount (plug-and-go)**

1. Plug in the iPhone via USB.
2. Unlock the phone and tap "Trust" when prompted.
3. Nautilus mounts it automatically under:

   ```
   /run/user/<uid>/gvfs/afc:host=<uuid>/
   ```

   Find the exact path:
   ```bash
   ls /run/user/$(id -u)/gvfs/
   # look for afc:host=... entry, then:
   ls "/run/user/$(id -u)/gvfs/afc:host=.../DCIM"
   ```

4. Run the scanner:
   ```bash
   python scan_photos.py --source "/run/user/$(id -u)/gvfs/afc:host=.../DCIM"
   ```

   Or just omit `--source` and the scanner will try to auto-detect it:
   ```bash
   python scan_photos.py
   ```

Note: iPhone photos are in **HEIC format** — `pillow-heif` handles them
transparently.  JPEG/PNG (AirDropped or Screenshots) also work.

### Android (JPEG / mixed)

Android phones mount via **MTP** and are exposed by GVFS:

```
/run/user/<uid>/gvfs/mtp:host=<model>/
```

Find your device:
```bash
ls /run/user/$(id -u)/gvfs/
```

Then scan:
```bash
python scan_photos.py --source "/run/user/$(id -u)/gvfs/mtp:host=.../Internal storage/DCIM"
```

Android photos are usually JPEG, so pillow-heif is not strictly needed, but
having it installed causes no harm.

### KDE Connect (wireless, no cable)

1. Install KDE Connect on both Linux and the phone.
2. Open the phone's DCIM in **Files → Remote Devices** on the phone side, or
   use the KDE Connect CLI to request the browse session.
3. The phone will appear under `~/.local/share/kdeconnect/` (SFTP mount) or
   under `/run/user/<uid>/gvfs/` after the SFTP plugin activates.
4. Pass that path to `--source`.

---

## Notes

- A child's face changes fast — re-enrol every few months.
- Everything runs locally; no images leave the machine.
- `cv2.face` requires **opencv-contrib-python**, not plain `opencv-python`.
- Biometric data (`known_faces/`) is gitignored — it never leaves this machine
  and is not committed to the repository.  If you lose it, re-run `enroll.py`.
- Model binaries (`models/`) are also gitignored.  Re-download with:
  ```bash
  curl -sL -o models/face_detection_yunet.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  ```
