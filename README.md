# face-id

Webcam face recognition — detects faces, draws bounding boxes, and labels you
and your child (or "Unknown").

**AcesVision is the supported entry point.** The recognition engines live in
`engine.py` and are selected with `FACE_ID_ENGINE`:

| Engine | `FACE_ID_ENGINE` | What it is | Face stage | Score |
|--------|------------------|-----------|-----------|-------|
| **ArcFace** | `arcface` (default) | YuNet detector + ArcFace ONNX embedder, CPU. | **44 ms** | cosine similarity |
| **YuNet** | `yunet` | YuNet detector + dlib ResNet encoder. The previous default. | 78 ms | Euclidean distance |
| **dlib / face_recognition** | `dlib` | HOG/CNN detector + dlib ResNet encoder. | — | Euclidean distance |
| **LBPH + Haar** | `lbph` | The **2015/2016 OpenCV** approach. Light, no GPU, trains instantly, but sensitive to lighting and pose. | — | chi-square |

All four read the same enrolled photos from `known_faces/`. The dlib and LBPH
engines are kept deliberately — they are the record of how this problem was
solved before, they still work, and they are one environment variable away.

### Why the recogniser changed, and not the detector

The face stage cost **77.6 ms** at 640x480 and **65.9 ms of that — 85% — was
the dlib encoder**. Detection was never the problem: YuNet finds a face in
about 6 ms. So YuNet stays and only the embedder is replaced. Measured on this
12-core host under normal working load (`load average ~8`), same frame, same
process shape:

| Engine | median | min | vs before |
|--------|--------|-----|-----------|
| `yunet` (dlib encoder) | 77.0 ms | 74.3 ms | baseline |
| `arcface` w600k_r50 | 44.0 ms | 36.6 ms | **1.75x faster** |
| `arcface` w600k_mbf | 21.6 ms | 15.7 ms | 3.6x faster |

`w600k_r50` ships. `w600k_mbf` is faster still and also reaches FAR 0%, but
with a 19% narrower separation gap (0.391 vs 0.480), and margin is what
absorbs faces the calibration never saw.

### The scores are not interchangeable

This is the part worth reading twice. The old calibrated tolerance `0.50` is a
dlib **Euclidean distance**, where *lower* is better and a threshold is a
ceiling. ArcFace scores **cosine similarity**, where *higher* is better and a
threshold is a floor. Reusing `0.50` across them would not be a slightly-wrong
setting, it would accept essentially every stranger.

So a threshold in `matching.py` is not a float — it is a value bound to a
metric, carrying its measured false-accept rate. A score computed in one
metric and compared against a threshold from another raises `MetricMismatch`
rather than returning an answer. Each engine has its own entry and its own
environment variable; there is no shared constant to reuse by accident.
`engine.Face` carries a `metric` field for the same reason: `face.conf` cannot
be read without it.

| Engine | Threshold | Metric | Evidence |
|--------|-----------|--------|----------|
| `arcface` | `>= 0.503` | cosine similarity | LFW n=2500 impostors, FAR **0.00%**, clean gap [0.2634, 0.7435] |
| `yunet` / `dlib` | `<= 0.50` | Euclidean distance | LFW n=2500, FAR 0%, clean gap [0.452, 0.500] (ticket a3c3c709) |
| `lbph` | `<= 70` | chi-square | never calibrated, and says so |

Re-derive any of these with `calibrate_threshold.py`, which downloads LFW
itself:

```bash
python calibrate_threshold.py --engine arcface --arcface-model w600k_r50
python calibrate_threshold.py --engine dlib
```

**What the ArcFace number does not establish.** The impostor side is well
sampled (2500 LFW identities). The genuine side is 66 photos of one person
under enrolment conditions, so "100% recall" describes that sample and is not
a general accuracy claim. Live frames score lower than enrolment photos; if
the enrolled person starts being missed, re-run the calibration against
live-condition genuines rather than nudging the number down.

### Everything runs on the CPU

`onnxruntime-rocm` on this host reports `ROCMExecutionProvider` from
`get_available_providers()` and then silently executes on the CPU, because
`libhipblas.so.3` and `libamdhip64.so.7` are absent (host is ROCm 6.3.1, the
wheel wants 7.x). That is the same "reports healthy, runs somewhere else"
hazard `acesvision/yolo_worker.py` documents for GPU devices.

`arcface.py` therefore never consults `get_available_providers()` — only
`InferenceSession.get_providers()`, what the live session actually bound — and
defaults to requesting the CPU and nothing else. Asking for a GPU provider is
opt-in via `FACE_ID_ARCFACE_PROVIDERS` and raises if the session does not bind
it.

> **Retired 2026-08-15.** The standalone CLI demos `recognize.py`,
> `lbph_recognize.py`, `obs_virtualcam.py` and `server.py` were removed. They
> wrapped what `engine.py` and AcesVision now provide directly. Recoverable at
> any time: `git show db88c10:recognize.py` (or `git revert` the retirement
> commit).

## AcesVision (in development)

AcesVision is the approved unified local vision application. One switchable
webcam or DroidCam input feeds the GUI preview, OBS virtual camera, and typed
gesture events without competing camera handles. See
[`docs/VISION_CONTROL_SPEC.md`](docs/VISION_CONTROL_SPEC.md).

The Phase 1 runner is dry-run for actions and does not enable a service:

```bash
python -m acesvision
python -m acesvision --source droidcam --url http://PHONE_IP:4747/video
python -m acesvision --obs
python -m acesvision --no-events            # suppress gesture events
python -m acesvision --hold-frames 3 --cooldown-s 0.5
python -m acesvision --print-token          # the token subscribers need
python -m acesvision --port 8765 --bind 127.0.0.1
python -m acesvision --no-emit              # serve the event API, publish nothing
```

Gesture events are emitted (dry-run) by default; the runner prints
`[events] enabled` at startup. The browser preview and the event API are both
on `http://127.0.0.1:8765`, and both require the token — including
`/latest.jpg`, which is live camera video.

### The event API

AcesVision publishes what it saw. It does not know what anybody does about it.
Gesture events go out over HTTP as Server-Sent Events, and a subscriber — a
lighting daemon, a home-automation bridge, a logger — needs no code in this
repository.

```bash
python -m acesvision --print-token
curl -N -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/api/events
```

[`docs/EVENTS.md`](docs/EVENTS.md) is the normative wire contract: the
`acesvision.gesture/1` schema, the four-value `identity_state`, replay and
reconnection, the publish filter, and the obligations a subscriber takes on.
Read it before writing one — in particular, `liveness_state` is always
`not_evaluated` and `security_authorized` is always `false`, so no correct
subscriber can bind a sensitive action to a gesture today.

### Gesture and action vocabulary

[`gestures.json`](gestures.json) holds the recognisable gestures — the seven
MediaPipe built-ins plus the two landmark-derived poses `Middle_Finger` and
`Shush` — with a `catalog_version` and a sha256 over its canonical
serialisation, so an out-of-process subscriber can pin the vocabulary it was
built against instead of assuming it. `acesvision/catalog.py` is the only
reader; `GET /api/catalog` serves it whole.

[`gesture_catalog.py`](gesture_catalog.py) re-exports that vocabulary, holds
the typed action catalog, and holds the landmark geometry that recognises the
two custom poses. Rules validate gesture and actor names against it at entry,
so a mistyped rule is rejected instead of being silently unfireable.

`Shush` is the real-world shush — index finger up, held to the lips. MediaPipe
labels that same hand `Pointing_Up`, which is bound to `ledctl next-theme`, so
the fingertip's distance to the mouth of a detected face is what separates the
two. It is measured in face-box units (`MOUTH_CENTRE_Y`, `MOUTH_RADIUS`) so it
scales with how near you stand. That means `Shush` needs face boxes:
`GestureDetector.detect(frame, faces=...)`. With no face in view the built-in
`Pointing_Up` stands, and the themes still cycle.

### Live gesture verification

Every gesture claim in this repo is proven against *synthetic* landmarks. No
gesture has ever been observed coming off a real camera. One command closes
that gap:

```bash
.venv/bin/python verify_gestures_live.py
```

It prompts you through all nine gestures on a timer (~75s), records what the
real detectors saw for every frame, and writes
`verification/live-gestures-<date>.md` plus two or three annotated frames per
gesture under `verification/frames/`. `verification/` is gitignored — those
frames contain your face and must never be committed.

It is observation only: no rule is loaded, no connector is constructed, nothing
is dispatched, and `~/.config/acesvision/rules.json` is not read. Enforced by
`LiveVerificationSafetyTests`, which parses the file and fails if it ever
imports the rule or connector machinery.

The headline number is the Shush occlusion test. A finger across the lips can
hide the mouth; if YuNet drops the face box at that instant `is_shush` returns
False and the frame degrades to `Pointing_Up`, which is bound to `next-theme` —
a failed shush would cycle your themes. The report counts every such frame and
fails `Shush` on a single one.

Exit codes: `0` all gestures verified · `1` at least one failed · `2` camera
held by another process (it names the holder and stops) · `3` no camera ·
`4` a model is missing.

### Camera selection

`acesvision/discovery.py` is the only device inventory. `camera.py` orders what
it finds — colour first, IR last, virtual/metadata nodes never — and opens one.
`FACE_ID_CAM=<index|/dev/path>` still forces a device. A contended device raises
`CameraBusyError` naming the holding process; an absent one raises
`CameraNotFoundError`. Diagnose with `fuser -v /dev/videoN`.

Launch the native desktop shell:

```bash
python -m acesvision.gui
```

The native shell currently includes live runtime status, webcam and DroidCam
switching, named physical-camera selectors, an on-demand local DroidCam scan,
live preview-side exposure/brightness/contrast/gamma controls, black/privacy-frame warnings,
GUI and OBS overlay profiles, custom box styles, gesture-event controls, and a
persistent typed dry-run rule editor. Security authorization remains locked
until the face verification and liveness phases pass their tests.

The perception spine now runs YOLO object detection and ByteTrack tracking in
its own subprocess (`acesvision.yolo_worker`) without blocking capture. YOLO26n is the
default; YOLO11n and YOLOv8n are selectable verified fallbacks. Face recognition
runs on tracked person crops and MediaPipe supplies Release 1 gestures. The GUI
shows capture FPS, inference FPS, latency, and the active model. See
[`docs/ACESVISION_VALIDATION.md`](docs/ACESVISION_VALIDATION.md) for measured
results and remaining security gates.

### Perception stages

The three stages — objects, faces, gestures — run in sequence on one inference
cycle, so the cycle cost is their sum. **Models and Security** carries a panel
that reports each stage's own measured cost and refresh rate, and lets you
switch a stage off or retune its rate against the running loop. Nothing
restarts: the setters mutate under the same lock the inference loop reads, and
the loop picks the new configuration up on its next cycle. The same three knobs
are available at start-up, on both entry points:

```bash
python -m acesvision.gui --detect-every 2 --face-hz 3 --gesture-hz 20
python -m acesvision --detect-every 2
```

The panel warns, visibly, whenever the face stage is switched off, starved of
refreshes, or starved indirectly by the object stage being switched off. That
is not a politeness. Shush is only separable from `Pointing_Up` by a fingertip
held near a *detected mouth*, so without a face box MediaPipe's own
`Pointing_Up` label stands — and `automations.example.json` binds `Pointing_Up`
to `ledctl next-theme`. Turning the face stage off does not drop a shush, it
turns every shush into a lighting change. A control panel that put that one
click away without saying so would be worse than no panel.

The first YOLO/ROCm cold start takes roughly 9-10 seconds on this host. After
warm-up the live RGB pipeline runs at **14.9 FPS**, and that ceiling is the
camera, not the pipeline: the USB2.0 FHD UVC webcam advertises 30 FPS through
`v4l2-ctl` but delivers 14.93 FPS on a raw `cv2` MJPG read with no inference
attached at all, at every resolution. Raw capture, plain capture, and the full
pipeline all measure 14.93-14.96 FPS. Warm GPU inference itself takes about
11 ms per frame, so there is roughly 55 ms of unused headroom per frame. (The
earlier "approximately 30 FPS" claim here was the driver's advertised rate,
not a measured one.) This is Ultralytics YOLO on AMD ROCm, not an NVIDIA or
TensorRT runtime.

The current Sunplus monitor webcam is locked to automatic exposure in the GUI.
Its driver advertises manual exposure and accepts values, but the sensor returns
black frames in that mode. Manual exposure remains an internal capability for a
future camera that passes calibration; it is not exposed for this device.

The camera selector shows the Linux camera number, hardware name, capture type,
and `/dev/videoN` path. AcesVision starts with the first real colour camera;
IR and virtual devices remain available for explicit selection.

### The network capture path: newest frame wins

A network source is read on its own thread and only the newest frame is kept
(`acesvision/sources.py`, `LatestFrameReader`). This is the same drop-old
contract `_OutputWorker` and `EventBus` already use, applied at the other end
of the pipeline.

Read straight from the capture loop, `cv2.VideoCapture` delivers every frame,
in order, and drops none. That sounds like a feature and is not. The consumer
here is YOLO plus ArcFace, it is slower than the phone, and the frames it has
not read do not evaporate — they queue in the socket buffer and the FFMPEG
demuxer, so every frame it eventually draws a box on is further into the past
than the one before it. Measured against a 1280x720 MJPEG source at 18.9 FPS
with a 90 ms consumer, over a real socket:

| | throughput | frame age at the end of a 25 s run | frames skipped |
|---|---|---|---|
| raw JPEG off the wire, no decode | 11.08 FPS | — | — |
| `cv2.VideoCapture`, in order | 10.86 FPS | **10.56 s** | 0 |
| threaded, newest frame only | 11.07 FPS | **0.06 s** | 193 |

Throughput is not the story — all three are within 2% of each other, and the
figure that matters is the middle column. Recognising a face on a ten-second-old
frame is not slow, it is wrong. The reader trades 193 frames nobody needed for
a live one.

Decode cost is real but second order: a 1280x720 JPEG costs **3.45 ms** to
decode on this host, a 290 FPS ceiling. Moving it off the capture loop is worth
doing and the reader does it, but decode was never where an 18.9 FPS source
became a 10 FPS pipeline. The consumer's own cost was.

**Why MJPEG stays.** H.264 would cut bandwidth and is hardware-decodable, and
it is still the wrong trade here. Every MJPEG frame is independently coded, so
throwing a stale one away costs nothing and corrupts nothing — which is the
single property the table above depends on. An H.264 stream carries decoder
state between frames; a dropped frame is a reference a later frame needs, and
drop-old stops being free. Lower bandwidth would be paid for in the one
behaviour that matters.

**Hardware decode: available, and slower.** The RX 6600's VCN block does decode
4:2:0 MJPEG — `ffmpeg -hwaccel vaapi -vaapi_device /dev/dri/renderD128` runs a
1280x720 MJPEG file at 16.9x realtime. Software decode of the same file runs at
**49.9x**, three times faster, because at this resolution libjpeg-turbo beats
the round trip through GPU memory. It is also unreachable from here regardless:
this OpenCV build reports `FFMPEG: YES`, `GStreamer: NO`, `VA: NO`, so
`cv2.VideoCapture` has no VAAPI path without rebuilding OpenCV. Not pursued, on
both counts. (`vainfo` is not installed and was not installed to find this out;
`ffmpeg -hwaccels` does list `vaapi` on this host.)

**Source resolution: unresolved, and marked as such.** Asking the phone for a
smaller frame would be the cheapest win available, and it is the one thing here
that is not settled. `/mjpegfeed?640x480` is reported to 404 on this DroidCam
build; that has not been re-checked, because the DroidCam endpoint was not
reachable when this was written and a stale result is worse than an open
question. What this build does honour is still to be determined, and if the
answer is "resolution is set in the phone app only", that is the answer and it
belongs here rather than in a wish.

Worth knowing before that work is done: nothing in this pipeline downscales for
inference. The detector JPEG-encodes the **full** frame for the YOLO worker
(`acesvision/perception.py`), and `FACE_ID_W`/`FACE_ID_H` default to 1280x720.
So a smaller source frame is a real saving end to end, not a pixel budget that
gets thrown away later — but it is a saving in the detector's encode and the
worker's inference, not in a decode step that was ever the bottleneck.

### DroidCam discovery: what it scans, and what it will not

The DroidCam scan is manual — it runs when you click Scan, never on startup —
and it is bounded three ways: private IPv4 only, one `/24` at most (254 hosts,
never a `/16` sweep), and a total deadline so a scan cannot hold the GUI open.

Which network it scans is decided from the machine's real interface table
(`/sys/class/net` plus `SIOCGIFADDR`), not from hostname resolution. A host
whose `/etc/hosts` maps its name to `127.0.1.1` — the Debian and Ubuntu default
— is exactly the case that used to make discovery find nothing at all.

Only a physical ethernet or wireless adapter that is up, with a private IPv4
address, is scanned. Loopback, `tun`/`tap`, `wg*`, `tailscale*`, `virbr*`,
`vnet*`, `docker*`, `br-*` and `veth*` are excluded by name **and** by
interface flags and hardware type. That exclusion is the point: a VPN peer, a
libvirt guest or a container network is somebody else's machine, and this
program has no business probing ports on it because you wanted to find your own
phone.

Each host gets **one second** to answer. That number is measured, not chosen.
The probe budget used to be 0.12 s, and 0.12 s does not reliably find a phone:
timing how long the development phone took to give a definitive TCP answer gave
a median of 211 ms and a maximum of 335 ms over ten single probes, and 99-252 ms
across six full `/24` sweeps. One of those sixteen measurements landed inside
0.12 s. The phone was awake, on the same subnet, the whole time — the delay is
Wi-Fi radio power saving, where the access point buffers a frame until the next
beacon, so it is a property of every phone this is meant to find. At 0.12 s,
whether the scan sees the phone is close to a coin toss, which reads from the
outside as "it found it, then it lost it".

One second is about three times the worst measurement and sits just under
Linux's 1 s initial SYN retransmit, so a probe still costs exactly one SYN. The
worker pool went from 32 to 128 to pay for it: a host that is not there burns
the whole timeout, and a `/24` is almost entirely hosts that are not there. On
this LAN a full `/24` sweep measures 0.97 s at the old 0.12 s / 32 workers,
6.33 s at 1 s / 32 workers, and **2.02 s at the 1 s / 128 workers now shipped**
— roughly one extra second of wall clock for an eight-fold wider answer window.

The plan is inspectable before a single packet is sent, and overridable:

```bash
python -m acesvision --list-networks    # what would be scanned, and why not the rest
python -m acesvision --scan-droidcam    # print the plan, then scan it

python -m acesvision --scan-droidcam --scan-timeout 3   # slow or congested link
python -m acesvision --scan-droidcam --scan-port 4848   # DroidCam moved in the app

ACESVISION_SCAN_INTERFACES=wlp3s0 python -m acesvision --scan-droidcam
ACESVISION_SCAN_NETWORKS=192.168.68.0/24 python -m acesvision --scan-droidcam
```

A scan that finds nothing now names the budget it was given, because "nothing
answered" and "nothing answered *in time*" are different answers and the second
one is fixable from the command line.

The Sources page shows the same target above the Scan button. Neither override
can widen the scan past a `/24` or reach a public network; both are refused with
a message rather than quietly narrowed. If no interface qualifies, discovery
says so — it does not return an empty list that reads like "no phone found".

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

```bash
python -m acesvision.gui                        # ArcFace (default)
FACE_ID_ENGINE=yunet python -m acesvision.gui   # the previous dlib-encoder pipeline
FACE_ID_ENGINE=lbph python -m acesvision.gui    # the 2016 approach
```

Green box = recognised (with confidence), red = Unknown.

## Tuning (env vars)

| Var | Applies to | Default | Meaning |
|-----|-----------|---------|---------|
| `FACE_ID_CAM` | both | auto-probe | Force camera index (this box uses changing V4L indexes) |
| `FACE_ID_W` / `FACE_ID_H` | camera | `1280x720` | Capture resolution; AcesVision defaults to the webcam's 30 FPS 720p MJPEG mode. |
| `FACE_ID_FPS` | camera | `30` | Requested physical-camera frame rate. |
| `ACESVISION_EXPOSURE` | AcesVision webcam | `166` | Starting value when manual exposure is selected; automatic exposure is the visible-image default. |
| `ACESVISION_SCAN_INTERFACES` | DroidCam discovery | auto-detect | Comma-separated interface names to scan instead of the auto-detected LAN adapter, e.g. `wlp3s0`. A name this host does not have is refused, not ignored. |
| `ACESVISION_SCAN_NETWORKS` | DroidCam discovery | from the interface | Comma-separated CIDRs to scan, e.g. `192.168.68.0/24`. Wins over everything. A public network or anything wider than a `/24` is refused. |
| `FACE_ID_ENGINE` | both | `arcface` | `arcface`, `yunet`, `dlib` or `lbph`. An unrecognised name is refused, not silently ignored. |
| `FACE_ID_ARCFACE_MODEL` | ArcFace | `w600k_r50` | `w600k_r50` (accurate) or `w600k_mbf` (2x faster, narrower margin) |
| `FACE_ID_ARCFACE_THRESHOLD` | ArcFace | `0.503` | Minimum cosine similarity. **Higher = stricter** — the opposite direction to the dlib knob below. An override has no measured FAR. |
| `FACE_ID_ARCFACE_THREADS` | ArcFace | half the cores, max 6 | ONNX intra-op threads. `0` hands the choice back to onnxruntime, which sizes it to every core and is measurably slower (48 ms vs 28 ms for r50 on 12 cores). |
| `FACE_ID_ARCFACE_PROVIDERS` | ArcFace | `CPUExecutionProvider` | Comma-separated ONNX providers. Anything the session fails to bind raises. |
| `FACE_ID_LBPH_THRESH` | LBPH | `70` | Max match distance. **Lower = stricter**, 0 = identical. (Your 2016 `10` almost never matched.) |
| `FACE_ID_TOLERANCE` | dlib, yunet | `0.50` | Match strictness, **lower = stricter**. Does not reach ArcFace: `0.50` means opposite things to the two engines. |
| `FACE_ID_MODEL` | dlib | `hog` | `hog` (CPU) or `cnn` (needs GPU, more robust) |
| `FACE_ID_SCALE` | dlib | `0.25` | Detection downscale; smaller = faster |

Example:
```bash
FACE_ID_CAM=1 FACE_ID_ENGINE=lbph FACE_ID_LBPH_THRESH=60 python -m acesvision.gui
```

## OBS integration (overlay boxes on your OBS webcam feed)

AcesVision's `ObsVirtualCameraOutput` (`acesvision/outputs.py`) republishes the
annotated frames to a **virtual camera**. OBS then adds that virtual camera as
a normal source — boxes baked in. Because AcesVision owns the one camera handle
and fans out to its outputs, the GUI preview and the OBS feed run from a single
capture.

```
real webcam  ->  AcesVision (capture + detect + draw)  ->  /dev/video20 (loopback)  ->  OBS
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
python -m acesvision --obs                      # ArcFace (default)
# or the previous dlib-encoder pipeline:
FACE_ID_ENGINE=yunet python -m acesvision --obs
```

Enable the OBS output from the GUI's Outputs screen, or pass `--obs` to the
headless runner.

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

### Runtime: one venv

`face-id/.venv` runs everything — dlib/face_recognition, Ultralytics, and torch
on the AMD ROCm GPU. There is no second environment and no dependency on any
other repository.

    torch==2.9.1+rocm6.3         # AMD RX 6600, HSA_OVERRIDE_GFX_VERSION=10.3.0
    torchvision==0.24.1+rocm6.3
    ultralytics==8.4.60
    opencv-contrib-python==4.13.0.92

Install torch and torchvision from the ROCm index, never from PyPI — the PyPI
build of `torchvision` drags a CUDA `torch` in behind it and silently replaces
the ROCm one:

```bash
.venv/bin/pip install torch==2.9.1+rocm6.3 torchvision==0.24.1+rocm6.3 \
  --index-url https://download.pytorch.org/whl/rocm6.3
.venv/bin/pip install --no-deps ultralytics==8.4.60
```

`ultralytics` is installed with `--no-deps` deliberately: its metadata requires
plain `opencv-python`, which would shadow `opencv-contrib-python` and take
`cv2.face` (the LBPH engine) out with it. `pip check` therefore reports
"ultralytics requires opencv-python, which is not installed" — that one line is
expected and is the correct state.

Both `watch.sh` and the YOLO worker default to this venv's interpreter. Set
`ACESVISION_YOLO_PYTHON` (worker) or `FACE_ID_PYTHON` (recogniser) only if you
deliberately want to split the environments again.

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
- Model binaries (`models/`) are also gitignored, and nothing here ever
  downloads a model for you — the same rule `acesvision/perception.py` applies
  to the YOLO weights. Fetch them once:
  ```bash
  # YuNet detector — used by every engine except lbph
  curl -sL -o models/face_detection_yunet.onnx \
    https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

  # ArcFace recognisers, from the InsightFace release archives.
  # buffalo_l carries w600k_r50 (the default); buffalo_s carries w600k_mbf.
  curl -sL -o /tmp/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
  curl -sL -o /tmp/buffalo_s.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip
  unzip -jo /tmp/buffalo_l.zip w600k_r50.onnx -d models/
  unzip -jo /tmp/buffalo_s.zip w600k_mbf.onnx -d models/
  ```
  Verify what you got — these are the files this branch was calibrated against:
  ```
  8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  models/face_detection_yunet.onnx
  4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43  models/w600k_r50.onnx
  9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f  models/w600k_mbf.onnx
  ```
- There is no cached encoding artifact — no `.npy`, `.pkl` or `.npz` anywhere.
  The gallery is rebuilt from `known_faces/<Name>/*.jpg` at process start,
  every time, and cached in memory under a key naming the embedding space it
  came from. That is what makes switching engines free and what stops
  embeddings from one model being compared against queries from another.
