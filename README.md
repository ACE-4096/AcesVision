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

## Notes

- A child's face changes fast — re-enrol every few months.
- Everything runs locally; no images leave the machine.
- `cv2.face` requires **opencv-contrib-python**, not plain `opencv-python`.
