# AcesVision validation record

Date: 2026-08-10  
Host GPU: AMD Radeon RX 6600  
Runtime: PyTorch 2.9.1 with ROCm 6.3, Ultralytics 8.4.60

## Validated architecture

- AcesVision capture and Qt outputs run in the `face-id` environment.
- Ultralytics runs in a subprocess of this repo (`acesvision.yolo_worker`)
  through a framed, frame-tagged binary protocol, in the same `face-id`
  environment. Keeping it out of process keeps torch off the Qt import path;
  it is not a second environment.
- Capture publishes every available frame using latest-frame output slots.
- Inference consumes only the newest pending frame and cannot block capture.
- YOLO performs COCO object detection and ByteTrack tracking in one pass.
- Face recognition runs only on current YOLO person crops.
- MediaPipe gesture recognition shares the asynchronous perception worker.
- Model changes replace the YOLO worker without restarting capture or outputs.
- Security authorization remains disabled.

## Model benchmark

The benchmark used an installed Ultralytics sample image, a 640-pixel model
input, ROCm device 0, ByteTrack, one warm-up pass, and 25 measured passes.

| Model | Median wall time | P95 wall time | Median model inference | Detections |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n | 16.74 ms | 17.98 ms | 4.49 ms | 6 |
| YOLO11n | 18.31 ms | 19.39 ms | 6.48 ms | 5 |
| YOLO26n | 19.25 ms | 22.19 ms | 7.74 ms | 5 |

YOLO26n is the default because it is the current spec candidate and remains
comfortably inside the 30 FPS latency budget. YOLO11n and YOLOv8n remain local
fallback choices. This benchmark compares runtime, not detection accuracy.

## End-to-end synthetic capture test

A 640 by 480 synthetic capture was paced at 30 FPS through the real YOLO,
face, gesture, frame-spine, and shutdown path:

- Capture: 29.8 FPS.
- Combined asynchronous perception: 52.7 FPS.
- Combined perception latency: 19.0 ms.
- YOLO model inference: 4.5 ms for the then-selected YOLOv8n baseline.
- Pipeline and YOLO subprocess stopped cleanly.

## Device and model integrity

- Colour camera stable identity:
  `/dev/v4l/by-path/pci-0000:0d:00.3-usb-0:3.4.2:1.0-video-index0`.
- IR camera stable identity:
  `/dev/v4l/by-id/usb-SunplusIT_Inc_USB2.0_FHD_UVC_WebCam_01.00.00-video-index0`.
- YOLO26n, YOLO11n, and YOLOv8n files matched the SHA-256 values in
  `acesvision/model_manifest.json`.
- AcesVision excludes an installed model from the GUI if its checksum differs.

## Automated verification

- Python unit and integration tests.
- Python bytecode compilation.
- Qt QML lint.
- Qt offscreen GUI smoke launch.
- Git whitespace validation.
- Real ROCm subprocess transport, two sequential frames, and clean shutdown.

## Open validation gates

- The colour camera could not be used for the final live hardware pass because
  OBS had an active PipeWire camera source holding it open. IR was not treated
  as a substitute for RGB validation.
- YOLO model accuracy has not been compared on a labelled AcesVision dataset.
- SCRFD-class face detection, AdaFace or CVLFace embeddings, PAD, RGB and IR
  liveness, calibrated verification, and the attack suite remain future gates.
- No model in this build is allowed to authorize sensitive actions.

## Live bottleneck diagnosis and scheduling update

The first live RGB run showed roughly 10 capture FPS and 77 to 92 ms combined
perception latency even though YOLO26n itself used only 8.0 to 8.7 ms. A local
1080p stage benchmark measured approximately 221.5 ms for the migration dlib
face path, 12.2 ms for MediaPipe gestures, and 10.5 ms for MediaPipe on a
640-pixel input. FP16 reduced YOLO26n model time only from 7.10 to 6.79 ms, so
YOLO quantization was not the useful optimization on this AMD host.

AcesVision now schedules YOLO tracking continuously, gesture refresh at 15 Hz,
and migration face identity at 2 Hz or immediately when the set of person track
IDs changes. Identity and gesture results are cached between refreshes. Webcam
capture now requests 1280 by 720 MJPEG at 30 FPS with a one-frame buffer. These
rates affect labels and convenience gestures only; authorization remains locked.

Raw camera testing then isolated a firmware behavior: aperture-priority exposure
reported 30 FPS but delivered only 5.2 to 10.0 FPS at every tested MJPEG
resolution. Manual V4L exposure delivered 29.7 FPS at exposure values 50, 100,
and 166. Forcing a one-frame V4L queue then reduced this driver to 14.9 FPS, so
physical webcam capture retains the driver's normal queue while AcesVision
drops stale frames at its own processing and output boundaries. With manual
exposure 166 and the normal V4L queue, raw capture measured 30.0 FPS.
AcesVision retains that configuration as an optional smooth-manual profile.
Automatic exposure is the application default because the monitor camera
produced a black image at the initially tested manual values.

The final 20-second live RGB pipeline run measured 29.8 FPS median capture,
29.8 FPS median perception updates, 15.8 ms rolling combined perception
latency, and 7.6 ms median YOLO26n model inference. It stopped cleanly without
leaving the camera or YOLO worker open.

The subsequent brightness calibration found that the colour sensor was
returning an almost uniform black image: median luma about 15 with standard
deviation below 1. Exposure values 120 through 333 all retained 29.6 FPS but
could not create missing tonal detail. Conservative brightness, contrast, and
gamma profiles also did not recover detail. This is treated as a privacy/lens
state rather than ordinary underexposure. The Live page now provides exposure,
brightness, contrast, gamma, and automatic-exposure controls beside the preview.
Changes apply immediately, and the panel reports measured luma plus a specific
black/privacy-frame warning.

During the live GUI check, the applied controls were manual exposure 251 and
brightness -64, the minimum supported value. The resulting image had mean luma
2.0 and virtually no tonal variation. Restoring automatic exposure, brightness
0, contrast 0, and gamma 100 immediately returned real detail with mean luma
about 39.5 and standard deviation about 36.2. Capture returned to the camera's
automatic-exposure rate of roughly 10 FPS. The GUI now defaults to automatic
exposure, uses editable numeric controls instead of drag sliders, includes a
one-click Reset, and explains when manual exposure or minimum brightness is
causing the black frame.

Further live testing confirmed that this monitor camera produces a usable image
only in aperture-priority automatic exposure. AcesVision now marks manual
exposure unsupported for this device, locks the GUI toggle on, disables the
manual exposure value, and guards the backend against stale clients requesting
manual mode. Brightness, contrast, and gamma remain adjustable beside preview.
