# AcesVision Specification

Status: Draft for founder review  
Date: 2026-08-10  
Repository: `face-id`

## Implementation status

- Phase 0 specification and threat model: approved.
- Phase 1 shared runtime: implemented and unit-verified. Stable physical-device
  identities and the camera-free end-to-end path are verified; the final live
  RGB hardware pass remains blocked by an existing OBS PipeWire camera source.
- Phase 2 desktop GUI and safe actions: in progress. The Qt/QML shell, source
  controls, named webcam inventory, bounded private-network DroidCam discovery,
  live camera image tuning, black/privacy-frame warnings, overlay presets and
  custom box styling, gesture-event gate, typed dry-run
  rules, atomic rule persistence, model selection, and live performance
  telemetry are implemented.
- Accelerated perception: implemented and locally benchmarked. YOLO26n is the
  default ROCm object detector with ByteTrack; YOLO11n and YOLOv8n are verified
  fallbacks. Capture and inference are decoupled through latest-frame slots.
- Face verification, PAD, authorization, real connector execution, packaging,
  service installation, and hardware tests are not yet enabled.

## 1. Product definition

AcesVision is a local-first desktop vision application. It owns one active
camera feed, runs local inference once, and sends the resulting frames and
events to independent outputs:

1. A full desktop GUI.
2. An OBS virtual camera.
3. Gesture and presence automation.

The input can be a physical webcam or a DroidCam network feed. Changing the
input must not change recognition, gesture, rule, or output configuration.

The AceRGB Plasma applet remains a small status and shortcut surface. It does
not own a camera or run inference.

## 2. Goals

- One process owns the active camera. OBS, the GUI, and automation never
  compete for the same V4L2 device.
- Webcam and DroidCam behave consistently and recover from disconnects.
- Object detection, tracking, face analysis, and gestures run locally.
- A gesture can be associated with the person who performed it.
- Gestures, conditions, permissions, and actions are configured in the GUI.
- Face identity contributes to security decisions, not only name labels.
- Sensitive actions require stronger evidence than convenience actions.
- Every rejected or executed action has an understandable reason.
- Output failures are isolated from capture and other outputs.

## 3. First-release non-goals

- Cloud inference or automatic recording.
- Treating a face match alone as authentication.
- Multiple simultaneous input cameras. The design must permit this later.
- Door access, finance, production, credential, or destructive actions from a
  gesture without a separate strong authentication factor.
- Training foundation models from scratch.

## 4. User experience

### 4.1 Normal operation

1. Open AcesVision.
2. Select `Webcam` or `DroidCam`.
3. Enable any combination of `GUI preview`, `OBS output`, and `Actions`.
4. The same processed frame and scene state feed every enabled output.
5. The header shows source health, latency, model state, OBS state, action state,
   and the current verified person.

### 4.2 Source changes and reconnects

- Sources change without an application restart.
- The old capture is released before the new one opens.
- A missing source enters `reconnecting` with bounded backoff.
- Reconnecting the monitor webcam restores video, gestures, face analysis, GUI,
  and OBS without restarting AcesVision.
- Security authorization stays disabled until the approved physical camera and
  all required security signals are healthy.

### 4.3 Rule builder

The rule builder uses this form:

> When [gesture] is performed by [person or role] at [camera or room], while
> [conditions], run [action].

Each rule displays the gesture confidence, hold duration, cooldown, actor,
location, face and liveness requirements, action, risk class, confirmation
requirements, and last decision.

## 5. Architecture

```text
Physical webcam or DroidCam
             |
             v
       Capture adapter
             |
             v
         Frame spine
             |
       +-----+-------------------+------------------+
       |                         |                  |
       v                         v                  v
 Object/person              Face security      Gesture/pose
 detection + tracking       pipeline            pipeline
       |                         |                  |
       +------------+------------+------------------+
                    |
                    v
              Fused scene state
                    |
       +------------+-------------+----------------+
       |                          |                |
       v                          v                v
 Desktop GUI                OBS virtual cam    Policy engine
                                                  |
                                                  v
                                           Typed connectors
```

### 5.1 Runtime boundaries

- `visiond`: owns capture, inference, tracking, identity state, liveness,
  policies, outputs, and the event log.
- `acesvision`: Qt 6 and QML desktop client. It controls `visiond` through a
  versioned local API and never opens a camera.
- AceRGB applet: compact health, last event, output toggles, and an
  `Open AcesVision` button.
- Existing scripts remain compatibility entry points during migration but may
  not open the same camera while `visiond` owns it.

### 5.2 Shared frame contract

Each frame receives a monotonic sequence and timestamp. Its scene state holds:

- Source identity and health.
- Raw and annotated frame references.
- Object and person tracks.
- Face detection, quality, match, verification, and liveness results.
- Hand and body gesture results.
- Actor association for each gesture.
- Model versions and timings.

Outputs consume this contract. They do not capture frames or call models.
Bounding boxes and other visuals are not permanently drawn into the shared raw
frame. Each output renders scene geometry through its own overlay profile.

### 5.3 Backpressure

- Capture and GUI never wait for OBS or action delivery.
- Frame outputs use latest-frame slots and can drop stale frames.
- Events enter a bounded persistent queue and are not silently dropped.
- Tracking fills gaps between detector frames.
- A degraded output is isolated and reported.

## 6. Inputs

### 6.1 Physical webcam

- Auto-probe for convenience use and explicit index for compatibility.
- Stable udev device identity for security use.
- Requested resolution, frame rate, and pixel format.
- Retain the existing colour preference and MJPEG configuration.
- Pair the known colour and IR nodes of the monitor webcam as one trusted device.

### 6.2 DroidCam

- Direct HTTP, MJPEG, or RTSP URL.
- Convenience host, port, and path fields.
- Low-latency buffering and reconnect supervision.
- Credentials stored in the desktop secret store, not plain configuration.
- Credentials and tokens redacted from logs and the GUI.
- Valid for GUI, OBS, identification, and gesture control.
- Not trusted for privileged authorization until transport and injection threats
  have a separately approved design.
- Discovery scans only the operator's own physical LAN: one private `/24`,
  derived from the real interface table, restricted to an ethernet or wireless
  adapter that is up. VPN, container and hypervisor guest networks are excluded
  by name and by interface flags, never scanned.
- The scan target is visible before the scan runs and overridable by the
  operator. No override may widen the scan past a `/24` or reach a public
  network.

Release 1 uses one active source with switching. The source contract must allow
multiple cameras in a later release without redesign.

## 7. Model system

Each model adapter has an ID, version, checksum, license record, input and output
contract, and local benchmark. Models are never silently downloaded.

### 7.1 Objects and tracking

- Benchmark YOLO26n and YOLO11n on the RX 6600.
- Select the smallest model that meets agreed accuracy and latency targets.
- Keep ByteTrack initially, with BoT-SORT selectable.
- Record Ultralytics AGPL or enterprise obligations before adoption.

### 7.2 Face detection and alignment

- Replace the mixed YuNet and dlib path with one explicit detector, landmark,
  alignment, quality, and embedding pipeline.
- Benchmark a current SCRFD-class detector against YuNet on real webcam and
  DroidCam footage.
- Reject inadequate face size, blur, lighting, pose, occlusion, and ambiguous
  multiple-face captures.

### 7.3 Face embeddings

- The current dlib embedding remains a migration baseline only.
- Evaluate current CVLFace or AdaFace-class models for local verification,
  especially poor-quality and angled faces.
- Review code, training-data, and pretrained-weight licenses before selection.
- Use aligned multi-frame embeddings and calibrated cosine thresholds.
- Never reuse the current dlib threshold with a new embedding model.

### 7.4 Gestures

Release 1 supports MediaPipe static gestures, the existing middle-finger
landmark rule, and configurable confidence, hold, repeat, and cooldown.

Later releases add GUI-recorded custom static poses, temporal hand gestures,
and whole-body gestures. A custom gesture must pass a preview and conflict test
before activation.

## 8. Face security

### 8.1 Separate decisions

- Identification is a 1:N search for labels, presence, and event search.
- Verification is a 1:1 match against a claimed or tracked identity.
- Authorization combines verification, liveness, device trust, policy, and any
  required additional factor.

A display label is never an authorization result.

### 8.2 Presentation attack detection

The threat set includes printed photos, phone or monitor displays, replayed
video, physical and partial masks, face swaps, generated video, virtual cameras,
and direct frame injection.

Security mode combines:

1. Passive RGB presentation attack detection.
2. Colour and IR consistency on the approved paired webcam.
3. Temporal consistency on one continuous person track.
4. Random active challenges for privileged actions.
5. Physical capture-device provenance.

No open PAD model is called secure until it passes the local attack suite.
Serious access control requires independently evaluated or certified components.

### 8.3 Active challenges

- Challenges derive from a session nonce and expire quickly.
- Examples include blink twice, turn left, look up, or show a requested gesture.
- Reusing a response from an earlier session must fail.
- Convenience actions should not interrupt users with challenges.

### 8.4 Device trust

- Privileged authorization requires an allowlisted physical capture device.
- Virtual V4L2 devices, OBS outputs, and untrusted network feeds are rejected.
- Loss of a required IR node disables privileged authorization unless an
  explicit fallback policy says otherwise.
- A root-level local attacker is outside a user-space application's guarantee.

### 8.5 Metrics

- Face: FMR, FNMR, failure to acquire, and quality rejection rate.
- PAD: APCER, BPCER, and non-response by attack type.
- End to end: false authorization and false rejection by risk class.
- Break results down by lighting, camera, distance, pose, and relevant
  demographic factors.

Testing follows ISO/IEC 30107-3:2023 concepts, current NIST FRTE/FATE reporting,
and FIDO biometric guidance.

## 9. Actor and gesture association

1. YOLO creates a person track.
2. A face detection attaches to its containing person track.
3. Hand landmarks attach to the most plausible person track.
4. Identity and liveness belong to the track, not the entire frame.
5. An ambiguous hand cannot trigger an identity-restricted rule.

Example event:

```json
{
  "event": "gesture",
  "gesture": "Open_Palm",
  "confidence": 0.94,
  "held_ms": 520,
  "track_id": "office-1842",
  "actor": "SamplePerson",
  "identity_state": "verified",
  "liveness_state": "live",
  "source": "monitor-webcam",
  "room": "Office",
  "captured_at": "2026-08-10T10:30:00+12:00"
}
```

## 10. Policy and actions

| Class | Examples | Minimum evidence |
| --- | --- | --- |
| Convenience | lights, media, volume | gesture and cooldown |
| Personal | desktop switching, private notification | associated allowed actor |
| Sensitive | unlock, send a message | verified live actor plus confirmation or strong second factor |
| Prohibited by default | destructive, finance, production, credentials | no gesture-only execution |

Initial typed connectors:

- AceRGB through D-Bus, with `ledctl` compatibility.
- Media through MPRIS.
- Audio through PipeWire or `wpctl`.
- KDE actions through D-Bus.
- Home Assistant through scoped webhooks.
- Desktop notifications.

Each connector declares actions, parameters, risk, timeout, and permissions.
Secrets are references to the desktop secret store.

Arbitrary shell execution is not a safe default. A future advanced connector
may use an explicit executable and argument allowlist. It must not accept shell
syntax or interpolate untrusted recognition text.

Execution controls:

- Global and per-rule enable switches.
- Dry-run mode that records without executing.
- Per-rule cooldown and global rate limits.
- Confirmation for sensitive actions.
- Idempotency keys for queued or webhook actions.
- Recorded outcome and explanation for every attempt.

## 11. Outputs

### 11.1 GUI

- Annotated view by default, with raw, security, and tracking debug views.
- Never opens the camera directly.
- Continues if OBS or actions are unavailable.

### 11.2 OBS virtual camera

- Uses `pyvirtualcam` and `v4l2loopback`.
- Selectable raw or annotated feed.
- Configurable stable size and frame rate.
- Configurable holding frame during source reconnect.
- Failure is visible but does not stop inference, GUI, or actions.

### 11.3 Gesture and presence

- Emits typed events into the policy engine.
- Works while GUI preview and OBS are disabled.
- Does not execute while disabled or in dry-run mode.

### 11.4 Overlay compositor and styling

All drawn vision elements are configurable. The scene state stores normalized
geometry and semantic metadata; an output-specific compositor turns that data
into pixels or native GUI items.

Configurable layers include:

- Object, person, face, and hand bounding boxes.
- Face landmarks, hand landmarks, pose skeletons, and tracking trails.
- Names, object classes, confidence values, track IDs, and model timings.
- Identity, verification, liveness, source-trust, and action-state badges.
- Gesture name, confidence, hold-progress indicator, and cooldown state.
- Detection zones, tripwires, masks, guides, safe areas, and region labels.
- Optional timestamps, source names, watermarks, and diagnostic panels.
- Privacy layers such as face blur, pixelation, silhouette, or complete label
  suppression for unknown people.

Each layer supports applicable style properties:

- Visible or hidden.
- Colour by fixed value, detected class, actor, identity state, liveness state,
  risk, or confidence range.
- Line width, line style, corner treatment, fill, opacity, and shadow.
- Font family, size, weight, label padding, label position, and text format.
- Landmark size and skeleton line width.
- Tracking-trail length, fade, and smoothing.
- Confidence precision and threshold for displaying an item.
- Animation and transition settings where the output renderer supports them.

Styles are saved as named profiles. Initial profiles:

- `Clean`: no overlays.
- `Minimal`: subtle boxes and short labels.
- `Broadcast`: OBS-readable labels and high-contrast lines.
- `Security`: identity, liveness, device trust, and rejection reasons.
- `Debug`: all geometry, IDs, scores, timing, and model state.
- `Privacy`: hides or obscures identity-sensitive information.

Profiles are independently assigned to the GUI preview, OBS output, screenshots,
and recordings. Changing an OBS style must not change the GUI style. A user can
duplicate a built-in profile, edit it, preview changes live, reset individual
properties, and export or import a profile.

Conditional styling is evaluated from typed scene fields, not arbitrary code.
Examples:

- Known and verified person: green rounded box.
- Known but not live: amber box with `Liveness pending`.
- Unknown person: red box in the security view, blurred in the privacy view.
- Gesture approaching its hold threshold: progress ring from 0 to 100 percent.
- Rejected action: temporary reason badge beside the actor track.

The raw source frame remains available to outputs that require no annotation.
Rendered frames carry the overlay-profile ID and version for reproducibility.

## 12. GUI screens

- **Live:** preview, overlays, source, output toggles, tracks, identity,
  liveness, FPS, latency, GPU load, and dropped frames.
- **Overlay Studio:** layer visibility, conditional colours, boxes, labels,
  landmarks, trails, zones, privacy effects, live preview, named profiles, and
  separate GUI and OBS profile assignment.
- **Sources:** add/test webcam or DroidCam, pair colour and IR, approve a trusted
  physical device.
- **People:** guided multi-pose enrolment, template/model state, calibration,
  disable, re-enrol, export, and delete.
- **Gesture Studio:** confidence, hold progress, actor association, configuration,
  and later custom-gesture recording.
- **Rules:** visual builder, risk, dry-run test, and explanation trace.
- **Models and Security:** inventory, checksums, licenses, benchmarks, thresholds,
  PAD results, and guided attack tests.
- **Events:** searchable decisions, executions, rejections, and retention.

## 13. Data and privacy

- Localhost-only control API by default and no cloud inference.
- Recording off by default.
- Biometric templates encrypted at rest using the desktop secret store or
  hardware-backed storage where available.
- Explicit retention for enrolment images and optional event snapshots.
- Logs redact credentials, tokens, and biometric vectors.
- Person deletion removes templates, indexes, and retained enrolment material
  after explicit confirmation.

## 14. Compatibility

- Versioned configuration with atomic writes.
- Existing `cameras.json` remains importable.
- Existing `automations.json` imports as draft rules. Shell rules stay disabled
  until converted or explicitly approved.
- Existing AceRGB mappings convert to typed AceRGB and MPRIS connectors.
- The current gesture preview endpoint remains during applet migration.

## 15. Failure behavior

| Failure | Required behavior |
| --- | --- |
| Webcam disconnected | reconnect without restart; security actions disabled |
| DroidCam unavailable | bounded retry; credentials not logged |
| No scannable network | reported as its own state, with the interfaces skipped and why; never reported in the words used for a scanned network that held no device |
| Face model failed | no identity-restricted action |
| PAD or IR unavailable | no privileged authorization unless policy allows fallback |
| Gesture model failed | face, GUI, and OBS continue |
| OBS unavailable | GUI and actions continue |
| Connector failed | retain error event; pipeline continues |
| GUI closed | configured headless outputs continue |
| `visiond` stopped | release camera and virtual camera cleanly |

## 16. Initial performance targets

- GUI preview at least 25 FPS at 720p.
- Local webcam preview latency under 150 ms.
- Gesture action latency under 250 ms after hold completion.
- Multi-frame face verification within 750 ms.
- Capture recovery within 5 seconds after a device becomes available.
- One inference pipeline regardless of the number of enabled outputs.

Targets may change after a reproducible RX 6600 baseline.

## 17. Acceptance criteria

### Sources and outputs

- Enable all outputs on a webcam and observe one input handle.
- Disconnect and reconnect the monitor webcam; all outputs recover.
- Switch to DroidCam without restarting; all outputs use it.
- Stop OBS; GUI and actions continue.
- Assign different overlay profiles to the GUI and OBS and verify they render
  independently from one scene state.
- Select `Clean` and verify the output matches the unannotated input frame.
- Edit box, label, landmark, trail, zone, and privacy styles in the GUI and
  verify the live preview updates without restarting inference.
- Export and import an overlay profile with the same rendered configuration.

### Gestures and rules

- Configure an AceRGB theme gesture without editing files.
- Prove an unknown actor cannot fire a person-restricted gesture.
- Show dry-run and rejection explanations in the GUI.
- Prove cooldown prevents repeated execution while a pose is held.

### Face security

- Complete guided enrolment and calibrated verification.
- Test print, phone replay, prerecorded video, and virtual-camera injection.
- Reject identity-restricted actions when actor association is ambiguous.
- Disable privileged authorization if trusted capture or required IR disappears.
- Produce FMR/FNMR and APCER/BPCER evidence for the selected model stack.

### Safety

- Verify no cloud inference calls.
- Verify logs contain no credentials or biometric vectors.
- Imported and new actions default to dry-run.
- No service is enabled automatically.

## 18. Delivery phases

1. **Specification and threat model:** approve this document, attack scenarios,
   risk policy, and model licenses.
2. **Shared runtime:** frame/events contracts, webcam and DroidCam, reconnect,
   fan-out, GUI preview, OBS, and gesture outputs.
3. **Desktop GUI and safe actions:** Qt/QML screens, typed connectors, dry-run,
   and explanations.
4. **Current face pipeline:** benchmark harness, new detector and embeddings,
   enrolment, calibrated verification, and actor association.
5. **PAD and authorization:** RGB/IR pairing, passive PAD, active challenges,
   device trust, attack suite, and second-factor hooks.
6. **Polish:** custom gestures, multiple cameras, signed model manifests,
   packaging, and opt-in service installation.

## 19. Approved implementation decisions

1. Product name: `AcesVision` (decided 2026-08-10).
2. Full desktop GUI: Qt 6 and QML.
3. Release 1 source model: one active source with live switching.
4. Initial connectors: AceRGB, MPRIS, PipeWire, KDE, notifications, and Home
   Assistant webhook.
5. Sensitive actions require a passkey or explicit confirmation in addition to
   face verification. There is no gesture-only sensitive-action path.

Approved by the founder on 2026-08-10.

## 20. Primary references

- ISO/IEC 30107-3:2023: <https://www.iso.org/standard/79520.html>
- NIST FRTE 1:1: <https://pages.nist.gov/frvt/html/frvt11.html>
- NIST FATE PAD: <https://pages.nist.gov/frvt/html/frvt_pad.html>
- FIDO Face Verification:
  <https://fidoalliance.org/certification/identity-verification/face-verification/>
- Ultralytics YOLO26: <https://docs.ultralytics.com/models/yolo26>
- AdaFace: <https://github.com/mk-minchul/AdaFace>
- InsightFace and pretrained-model licensing:
  <https://github.com/deepinsight/insightface>
