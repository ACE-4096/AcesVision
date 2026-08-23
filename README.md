# AcesVision

Local-first camera vision for Linux desktops: live video, object and face
overlays, hand and pose landmarks, gesture events, and optional desktop
automations. AcesVision runs on your computer and binds its preview and event
API to loopback by default.

> AcesVision processes sensitive camera and biometric data. It deliberately
> keeps enrolled faces, recordings, camera configuration, event databases,
> tokens, and downloaded model files out of Git. Read the privacy section
> before enrolling anyone.

## What it does

- A native Qt/Plasma desktop shell with a collapsible navigation rail and
  resizable live-video workspace.
- Local webcam and network-camera inputs, with physical cameras preferred over
  virtual loopback devices to avoid capture feedback.
- Object boxes, face labels, MediaPipe hand joints, and optional pose/body
  landmarks rendered into the preview or OBS virtual-camera output.
- A clean-overlay gesture mode that can hide visual annotations without
  stopping detection.
- Typed gesture events over authenticated Server-Sent Events (SSE).
- Optional local rules for MPRIS media playback, AcesRGB lighting, and overlay
  controls. Rules are dry-run by default.
- Face enrolment and recognition using local files under `known_faces/`.

The project is built for a single-user Linux desktop. It is not a hosted
biometric service and must not be exposed directly to the public internet.

## Quick start

### 1. Create an environment

```bash
git clone <your-fork-or-clone-url> acesvision
cd acesvision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Most features work on CPU. GPU acceleration is optional and hardware-specific;
the default configuration remains conservative when no usable accelerator is
present.

### 2. Download local models

Model binaries are intentionally not committed. Create the `models/` directory
and fetch the face-recognition models you want to use:

```bash
mkdir -p models

# YuNet detector — used by every engine except LBPH.
curl -sL -o models/face_detection_yunet.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx

# ArcFace recognisers. buffalo_l contains w600k_r50 (the default);
# buffalo_s contains w600k_mbf (faster, with a narrower margin).
curl -sL -o /tmp/buffalo_l.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
curl -sL -o /tmp/buffalo_s.zip \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip
unzip -jo /tmp/buffalo_l.zip w600k_r50.onnx -d models/
unzip -jo /tmp/buffalo_s.zip w600k_mbf.onnx -d models/
```

Obtain optional object-detection and pose models from their publishers. Do not
commit them, and review each model's licence and terms before distributing a
build that includes it.

### 3. Start the desktop app

```bash
python -m acesvision.gui
```

The app chooses a usable physical camera automatically. To select a camera
explicitly, use `FACE_ID_CAM`:

```bash
FACE_ID_CAM=/dev/video0 python -m acesvision.gui
FACE_ID_CAM=0 python -m acesvision.gui
```

If a camera is already held by another process, AcesVision reports the holder
instead of silently switching to an unexpected device.

## Desktop launcher

Install a normal application-menu entry and a user-level systemd service:

```bash
./scripts/install-desktop.sh
```

The installer generates a service unit for the current checkout and its
`.venv`. If Python lives elsewhere, pass it explicitly:

```bash
ACESVISION_PYTHON=/path/to/python ./scripts/install-desktop.sh
```

Then open **AcesVision** from your application launcher. Reopening it starts
the same managed service rather than creating a second process that competes
for the camera.

## Native recording

Use the **Record video** switch in the Live page's **Destinations** panel. It
records the exact overlaid feed shown by the app without opening another camera
process. Turning it off finalises a constant-frame-rate MP4 and writes a JSON
sidecar with the raw detections and timing for each frame.

Audio is deliberately off by default. In **Recording audio**, choose a
microphone for narration or an explicit PipeWire/Pulse **System audio** monitor
for desktop sound; the selector applies to the next recording. A system-audio
monitor can include calls and other applications, so AcesVision never enables
one automatically.

By default, recordings go to `~/Videos/AcesVision`. Set
`ACESVISION_RECORDINGS` before starting the desktop service to choose another
directory outside the checkout. Recordings can contain faces, rooms, and
recognition labels; keep the MP4 and its JSON sidecar together and delete both
when you no longer need them.

## Using the live view

The Live page is designed around the picture:

- The Session dashboard shows input, capture and inference rates, scene counts,
  recording state, and direct Source/Output controls in one place.
- Drag the splitters around the video to resize the workspace.
- Collapse the left navigation rail when you want a larger preview.
- Use the overlay controls to show or hide object boxes, face labels, hand
  joints, and the 33-point full-body pose skeleton independently.
- The clean-overlay control hides annotations only; it does not disable the
  perception stages or event output.
- After a model or source change, allow a short warm-up before judging whether
  detections are active.

The preview is intentionally presented at a smooth 15 FPS while capture and
inference continue at their configured rate. This avoids a desktop-renderer
flicker caused by replacing a decoded JPEG snapshot for every incoming frame.

Hand joints are drawn whenever MediaPipe supplies hand landmarks, including
hands that have not yet been classified as a named gesture. The **Body pose**
stage is independent of hand tracking: it uses the optional
`models/pose_landmarker.task` model to draw visible shoulders, hips, knees,
ankles, and the rest of MediaPipe Pose's 33 landmarks. For workout form or
posture, frame your entire body, avoid severe backlighting, and use the stage's
refresh-rate control to balance responsiveness against inference cost.

## Enrolment and recognition

Enrol a person locally:

```bash
python enroll.py "Your Name"
```

Press Space to save a frame when exactly one face is visible, and press Q to
quit. Use several well-lit images with varied angles. Files go to
`known_faces/<Name>/`; that directory is gitignored.

The default ArcFace engine uses cosine similarity. The older YuNet/dlib path
uses Euclidean distance, so their thresholds are deliberately not
interchangeable:

```bash
python -m acesvision.gui                         # ArcFace (default)
FACE_ID_ENGINE=yunet python -m acesvision.gui
FACE_ID_ENGINE=lbph python -m acesvision.gui
```

For a defensible threshold calibration, use your own enrolled directory:

```bash
python calibrate_threshold.py --person "Your Name" --engine arcface
```

Calibration downloads the LFW dataset to `/tmp` for the run. It does not
upload your face images.

## Gestures and automations

AcesVision recognises MediaPipe hand gestures plus the locally derived
`Middle_Finger` and `Shush` poses. Gesture output is typed and includes its
catalogue version and digest so subscribers can reject mismatched vocabularies.

The GUI lets you edit local rules. The built-in connectors include:

- MPRIS media actions such as play/pause and next track.
- AcesRGB actions, including next lighting theme.
- Overlay controls such as clean-overlay toggle.

Rules are dry-run by default. Treat real actions as opt-in: verify the gesture
in the live view, then deliberately arm only the rule you want.

For a no-action camera verification of all gestures:

```bash
.venv/bin/python verify_gestures_live.py
```

It stores annotated verification frames under `verification/`, which is
gitignored because it can contain your face.

## Event API and integrations

The headless runner and GUI publish gesture events locally over SSE. The API
requires a bearer token, including the live JPEG endpoint:

```bash
python -m acesvision --print-token
curl -N -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8765/api/events
```

Useful commands:

```bash
python -m acesvision --no-events              # preview/API, no emitted gestures
python -m acesvision --no-emit                # event API, no publishing
python -m acesvision --obs                    # enable OBS virtual-camera output
python -m acesvision --scan-droidcam          # explicitly scan a bounded LAN range
```

The event contract, authentication rules, source redaction, and subscriber
guidance are documented in [docs/EVENTS.md](docs/EVENTS.md). AcesVision refuses
browser `Origin` headers and non-loopback binding without an explicit flag; a
remote deployment needs its own threat model and access controls.

## Cameras

Use a physical webcam, a named camera from `cameras.json`, or a direct network
URL where supported:

```bash
python -m acesvision --source 0
python -m acesvision --source http://phone.local:4747/video
```

Start from the safe examples rather than committing real endpoints:

```bash
cp cameras.example.json cameras.json
cp watch_cameras.example.json watch_cameras.json
```

Both configuration files are ignored by Git. Camera URLs can contain
credentials, so treat them like passwords. DroidCam discovery is manual,
bounded to a private local `/24`, and never runs merely because the app starts.

## Privacy and data removal

These paths are intentionally local and ignored:

| Data | Location |
| --- | --- |
| Enrolled face photos | `known_faces/` |
| Camera configuration | `cameras.json`, `watch_cameras.json` |
| Local automations | `automations.json`, `~/.config/acesvision/rules.json` |
| API token | `~/.config/acesvision/emitter.token` |
| Verification frames | `verification/` |
| Event databases | `events.db`, `presence.db` |
| Downloaded models | `models/`, `*.pt` |

To remove a person, delete their directory under `known_faces/` and remove any
event history you no longer need. Restart AcesVision afterwards so its in-memory
gallery is rebuilt.

Never enrol or record another person without an appropriate lawful basis and
their informed consent.

## Development

Run the test suite before opening a pull request:

```bash
.venv/bin/python -m unittest
QT_QPA_PLATFORM=offscreen .venv/bin/python -m acesvision.gui --smoke-test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution expectations,
[SECURITY.md](SECURITY.md) for vulnerability reporting, and
[docs/VISION_CONTROL_SPEC.md](docs/VISION_CONTROL_SPEC.md) for the runtime and
UI design.

## Licence

AcesVision is licensed under the [GNU Affero General Public License v3.0](LICENSE).
Third-party dependencies and optional model weights retain their own licences.
