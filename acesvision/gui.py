"""Qt 6 and QML desktop shell for AcesVision."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path
from urllib.parse import quote

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from gesture_catalog import ANY_ACTOR, GESTURE_IDS

from .connectors import OverlayConnector, default_registry
from .audio import (AudioLevelMeter, discover_audio_sources,
                    set_source_volume_percent, source_volume_percent)
from .contracts import SceneFrame, SourceSpec
from .emitter import EventBus, GestureEmitter, PublishFilter
from .events import GestureEventOutput
from .discovery import (
    discover_webcams,
    preferred_webcam,
    scan_droidcam,
    scan_plan,
)
from .outputs import LatestFrameOutput, ObsVirtualCameraOutput
from .overlay import MINIMAL, PROFILES, OverlayProfile
from .pipeline import ROTATION_DEGREES, VisionPipeline
from .policy import CONNECTORS, Rule, RuleEngine, RuleStore, known_actors
from .recording import RecordingError, RecordingOutput, resolve_path
from .server import VisionServer, load_or_create_token
from .processor import (
    DEFAULT_DETECT_EVERY,
    DEFAULT_FACE_HZ,
    DEFAULT_GESTURE_HZ,
    DEFAULT_POSE_HZ,
    FaceGestureProcessor,
)
from .workout import EXERCISES
from .perception import file_sha256

QML_PATH = Path(__file__).parent / "qml" / "Main.qml"
MODEL_MANIFEST_PATH = Path(__file__).parent / "model_manifest.json"

# A frame this dark and this flat is not a picture of anything. pipeline
# _frame_quality raises image_warning off the same two numbers; the GUI reuses
# that signal to decide, per device and at runtime, whether this camera can
# actually do manual exposure — instead of hardcoding one vendor's quirk into
# every user's build.
BLACK_FRAME_MEAN = 25.0
BLACK_FRAME_STD = 3.0

# No new frame for this long is a still image, not a video feed. Without this
# the last good frame sits on screen forever and reads as live.
PREVIEW_STALE_AFTER_S = 2.0

# QML ``Image`` is a JPEG snapshot renderer, not a video sink. Replacing its
# URL at 30 FPS made it cancel and restart decodes fast enough to visibly flash
# even while capture itself was perfectly steady. The capture and inference
# pipelines stay at full rate; only the desktop snapshot presentation is paced.
PREVIEW_MAX_FPS = 15.0
PREVIEW_REFRESH_S = 1.0 / PREVIEW_MAX_FPS

# The refresh timer runs at 100 ms. Filesystem and registry re-scans are far
# too coarse for that, so they ride a slower tick.
SLOW_REFRESH_TICKS = 20

# The three perception stages the operator can drive, in the order the
# inference loop runs them. Held as data because the panel, the two slots and
# the guard all have to agree about what exists; three copy-pasted branches
# would let them drift.
#
# `rate` means a different thing per stage and says so: the object stage is
# gated by a frame divisor (run on one frame in N), the other two by a refresh
# frequency. Both are the knob the runtime actually has — inventing a uniform
# "Hz" for the object stage would have made the number a lie.
PERCEPTION_STAGES = (
    {
        "id": "object",
        "label": "Objects — YOLO",
        "detail": ("Finds people. The face stage only searches inside the "
                   "person boxes this produces, so switching this off stops "
                   "faces too."),
        "rateLabel": "Run on one frame in",
        "rateKind": "every",
        "rateSuffix": " frames",
        "rateMin": 1.0,
        "rateMax": 10.0,
        "rateStep": 1.0,
    },
    {
        "id": "face",
        "label": "Faces — YuNet and dlib",
        "detail": ("Recognises enrolled people, and gives the gesture stage "
                   "the mouth position that separates Shush from Pointing_Up."),
        "rateLabel": "Refresh rate",
        "rateKind": "hz",
        "rateSuffix": " Hz",
        "rateMin": 0.2,
        "rateMax": 10.0,
        "rateStep": 0.1,
    },
    {
        "id": "gesture",
        "label": "Gestures — MediaPipe",
        "detail": "Classifies hands on every cycle the refresh rate is due.",
        "rateLabel": "Refresh rate",
        "rateKind": "hz",
        "rateSuffix": " Hz",
        "rateMin": 1.0,
        "rateMax": 30.0,
        "rateStep": 1.0,
    },
    {
        "id": "pose",
        "label": "Body pose — MediaPipe (33 joints)",
        "detail": ("Tracks shoulders, hips, knees, ankles and other whole-body "
                   "joints for posture and workout form. Keep your full body in frame."),
        "rateLabel": "Refresh rate",
        "rateKind": "hz",
        "rateSuffix": " Hz",
        "rateMin": 1.0,
        "rateMax": 20.0,
        "rateStep": 1.0,
    },
)

STAGE_IDS = tuple(stage["id"] for stage in PERCEPTION_STAGES)
STAGE_BY_ID = {stage["id"]: stage for stage in PERCEPTION_STAGES}

# The processor setter each stage's rate knob drives. The object stage's knob
# is a frame divisor and the other two are frequencies, so this cannot be one
# generic call.
STAGE_RATE_SETTERS = {
    "object": "set_detect_every",
    "face": "set_face_hz",
    "gesture": "set_gesture_hz",
    "pose": "set_pose_hz",
}

# Below this rate the newest face box the gesture stage can consult is over a
# second old, which is longer than a head stays still in conversation. This is
# a stated judgement, not a measurement — but see shush_degradation for why
# erring toward warning is the right side to be wrong on.
SAFE_FACE_HZ = 1.0

# What a lost Shush turns into, and what that then does. Named here so the
# warning copy and the tests quote the same two facts.
SHUSH_FALLBACK_GESTURE = "Pointing_Up"
SHUSH_FALLBACK_ACTION = "ledctl next-theme"


def shush_degradation(object_enabled, face_enabled, face_hz):
    """Name it when the perception knobs have silently rebound the lighting.

    Shush is only separable from Pointing_Up by where the fingertip sits
    relative to a detected mouth — that is term 3 of
    ``gesture_catalog.is_shush``, and with no face box it returns False and
    MediaPipe's own ``Pointing_Up`` label stands. ``automations.example.json``
    binds Pointing_Up to ``ledctl next-theme``.

    So turning the face stage off does not merely *drop* the shush. It
    converts every shush into a lighting-theme change. Starving the stage of
    refreshes is the same failure arriving slowly: the boxes the gesture stage
    consults go stale, stop covering the mouth, and the same fallback fires.

    Disabling the object stage does it too, at one remove — faces are only
    searched for inside person boxes, so no objects means no faces means no
    Shush.

    Returns the operator-facing warning, or "" when nothing is degraded.
    """
    reason = ""
    if not face_enabled:
        reason = "The face stage is switched off"
    elif not object_enabled:
        reason = ("The object stage is switched off, so the face stage has no "
                  "person boxes to search")
    elif float(face_hz) < SAFE_FACE_HZ:
        reason = (f"The face stage is refreshing at {float(face_hz):.1f} Hz, "
                  f"below the {SAFE_FACE_HZ:g} Hz needed for a face box to "
                  "still describe where the mouth is")
    if not reason:
        return ""
    return (f"{reason}. Shush is only told apart from {SHUSH_FALLBACK_GESTURE} "
            f"by a fingertip near a detected mouth, so a shush will now be "
            f"read as {SHUSH_FALLBACK_GESTURE} — which the example automations "
            f"bind to `{SHUSH_FALLBACK_ACTION}`. Shushing will cycle your "
            f"lights.")


def describe_compute_device(torch_module=None):
    """Name the inference device *this* machine actually exposes.

    Never names a specific card. Shipping "ROCm on the RX 6600" as UI copy was
    true only on the author's desktop and false everywhere else.
    """
    torch = torch_module if torch_module is not None else sys.modules.get("torch")
    if torch is None:
        return "this machine"
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            api = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
            return f"{name} via {api}"
    except Exception:                       # noqa: BLE001 - never break the UI
        return "this machine"
    return "the CPU on this machine"


class VisionBackend(QObject):
    statusChanged = Signal()
    sourceChanged = Signal()
    sequenceChanged = Signal()
    errorChanged = Signal()
    previewChanged = Signal()
    obsChanged = Signal()
    eventsChanged = Signal()
    recordingChanged = Signal()
    recordingFpsChanged = Signal()
    rotationChanged = Signal()
    audioSourcesChanged = Signal()
    audioControlChanged = Signal()
    workoutChanged = Signal()
    sceneCountsChanged = Signal()
    overlayChanged = Signal()
    gestureChanged = Signal()
    rulesChanged = Signal()
    decisionChanged = Signal()
    performanceChanged = Signal()
    imageChanged = Signal()
    webcamsChanged = Signal()
    droidCamsChanged = Signal()
    droidScanChanged = Signal()
    previewStaleChanged = Signal()
    exposureCapabilityChanged = Signal()
    cameraTuningChanged = Signal()
    connectorsChanged = Signal()
    actorsChanged = Signal()
    modelsChanged = Signal()
    gesturesChanged = Signal()
    overlayProfilesChanged = Signal()
    computeChanged = Signal()
    # Per-cycle stage measurements plus the knob positions they were measured
    # under. Fires on the 100 ms poll whenever any of it moves.
    stagesChanged = Signal()
    # The *set* of stages, which is fixed. Kept apart from stagesChanged on
    # purpose: a QML Repeater rebuilds every delegate when its model changes,
    # so driving the repeater off stageStats would tear down and rebuild the
    # switches and sliders ten times a second and make them undraggable.
    stageSetChanged = Signal()

    gestureFromWorker = Signal(str)
    droidScanFinished = Signal(str)
    audioMeterSample = Signal(str, float, str)
    # A rule is evaluated on the pipeline worker. Presentation state belongs
    # to Qt's GUI thread, so the local overlay connector asks that thread to
    # perform the toggle instead of touching QObjects from the worker.
    overlayToggleRequested = Signal()

    def __init__(self, preview_port=8765, initialize_models=True,
                 load_saved_rules=True, executor=None, parent=None,
                 clock=time.monotonic, processor=None,
                 detect_every=DEFAULT_DETECT_EVERY, face_hz=DEFAULT_FACE_HZ,
                 gesture_hz=DEFAULT_GESTURE_HZ, pose_hz=DEFAULT_POSE_HZ,
                 recording_factory=RecordingOutput,
                 recording_path_factory=resolve_path, audio_run=None,
                 audio_meter_factory=AudioLevelMeter):
        super().__init__(parent)
        self._clock = clock
        self._status = "starting"
        discovered_webcams = discover_webcams()
        default_webcam = preferred_webcam(discovered_webcams)
        self._preferred_webcam = default_webcam
        self._virtual_webcam_indexes = {
            device.index for device in discovered_webcams if device.kind == "virtual"
        }
        self._source = default_webcam.label if default_webcam else "Webcam (auto)"
        self._sequence = 0
        # Two independent error channels. The pipeline one is a live mirror of
        # capture state and is rewritten every tick; the action one is what a
        # button press or a failed connector said. Folding them into one string
        # meant a 100 ms timer erased every message a slot had just set, so a
        # dead daemon looked exactly like success.
        self._pipeline_error = ""
        self._action_error = ""
        self._preview_tick = 0
        self._last_preview_emit_at = None
        self._last_frame_at = None
        self._preview_stale = False
        self._slow_tick = 0
        self._obs_enabled = False
        self._recorder = None
        self._recording_factory = recording_factory
        self._recording_path_factory = recording_path_factory
        self._recording_status = "Recording off"
        # Auto is source-aware: a DroidCam normally delivers 60 FPS, while a
        # physical UVC webcam normally delivers 30. Output controls may force
        # either rate deliberately for the next recording.
        self._recording_rate = 0
        self._rotation_degrees = 0
        self._events_enabled = False
        self._overlay = "minimal"
        self._overlay_before_clean = "minimal"
        # Whether *this* session has an applied custom profile. Overlay Studio
        # needs a card for it, otherwise "Apply custom" leaves every card
        # reading "Select" and nothing reading "Active".
        self._custom_overlay = None
        self._last_gesture = "No gesture events yet"
        self._last_decision = "No rule decisions yet"
        self._last_outcome = "no decision"
        self._capture_fps = 0.0
        self._inference_fps = 0.0
        self._inference_ms = 0.0
        self._latest_inference_ms = 0.0
        self._model_inference_ms = 0.0
        self._model_summary = "Models warming up"
        self._object_count = 0
        self._face_count = 0
        self._gesture_count = 0
        self._pose_count = 0
        self._audio_sources = discover_audio_sources()
        self._audio_source = ""
        self._audio_run = audio_run
        self._audio_gain = 100
        self._audio_level_db = AudioLevelMeter.FLOOR_DB
        self._audio_meter_status = "Choose a microphone to show its live level"
        self._audio_meter = audio_meter_factory(self._emit_audio_meter_sample)
        self._workout_enabled = False
        self._workout_exercise = "squat"
        self._workout_reps = 0
        self._workout_phase = "find rest"
        self._workout_angle = 0.0
        self._workout_progress = 0.0
        self._workout_feedback = "Workout paused"
        self._workout_filter = ""
        # Seeded from the same knobs the processor is built with, so the panel
        # reads true before the first inference cycle publishes anything.
        self._stage_rate = {
            "object": float(max(1, int(detect_every))),
            "face": float(face_hz),
            "gesture": float(gesture_hz),
            "pose": float(pose_hz),
        }
        self._stage_enabled = {stage: True for stage in STAGE_IDS}
        self._stage_ms = {stage: 0.0 for stage in STAGE_IDS}
        self._stage_refreshed = {stage: False for stage in STAGE_IDS}
        self._image_warning = ""
        self._image_mean = 0.0
        self._auto_exposure = True
        # Manual exposure starts ENABLED and is withdrawn per device only when
        # this camera is observed returning a black, flat frame in manual mode.
        # It used to be a hardcoded False describing one Sunplus monitor webcam,
        # which greyed the control out for every user of every other camera.
        self._manual_exposure_supported = True
        self._manual_exposure_blocked = {}
        self._manual_exposure_notice = ""
        self._compute_device = describe_compute_device()
        self._exposure = 166
        self._brightness = 0
        self._contrast = 0
        self._gamma = 100
        self._model_options, self._model_paths = self._scan_models()
        self.rule_store = RuleStore()
        try:
            # Non-strict: repair rules that only differ by case/separator and
            # quarantine the rest, so one unfireable rule cannot blank the set.
            self._rules = (self.rule_store.load(strict=False)
                           if load_saved_rules else [])
            if self.rule_store.rejected:
                dropped = ", ".join(
                    f"{raw.get('gesture', '?')} ({reason})"
                    for raw, reason in self.rule_store.rejected)
                self._last_decision = f"Rules skipped as unfireable: {dropped}"
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self._rules = []
            self._last_decision = f"Saved rules not loaded: {exc}"
        # No engine-wide dry_run any more. Each rule carries its own, and every
        # rule loaded from a file written before that change defaults to True,
        # so nothing gets armed by this wiring.
        # An armed saved rule is an explicit opt-in to gesture processing. The
        # old GUI restarted with its event output off regardless, leaving an
        # armed rule looking live yet permanently unfireable.
        self._events_enabled = any(rule.enabled and not rule.dry_run
                                   for rule in self._rules)
        self.executor = default_registry() if executor is None else executor
        if executor is None:
            self.executor.register(OverlayConnector(
                self.overlayToggleRequested.emit))
        self.rule_engine = RuleEngine(self._rules, executor=self.executor)
        self._connector_names = list(self.executor.names())
        self._actor_names = [ANY_ACTOR] + known_actors()
        self._obs = None
        # AcesVision's own v4l2loopback output is a valid camera device at the
        # kernel level but never a valid *input* here: selecting it feeds our
        # rendered preview back into capture and creates visible flicker.
        self._webcams = [device.as_dict() for device in discovered_webcams
                         if device.kind != "virtual"]
        self._droid_cams = []
        self._droid_scan_active = False
        self._droid_scan_status = "Not scanned"

        source = SourceSpec.from_mapping({
            "id": "webcam",
            "name": default_webcam.name if default_webcam else "Webcam",
            "type": "webcam",
            "index": default_webcam.index if default_webcam else None,
            "device_path": default_webcam.stable_path if default_webcam else None,
        })
        self.latest = LatestFrameOutput(MINIMAL)
        # The GUI is the primary desktop runtime, not merely a preview. It
        # therefore owns the same authenticated event surface as headless
        # AcesVision: local rules execute in-process and subscribers (including
        # acergb-visiond) receive the projected SSE event from this exact frame.
        self.emitter = GestureEmitter(EventBus(), publish_filter=PublishFilter.load())
        self.gestures = GestureEventOutput(self._receive_gesture,
                                           enabled=self._events_enabled,
                                           emitter=self.emitter)
        # The GUI used to build FaceGestureProcessor() with no arguments while
        # __main__ passed all three knobs, so the desktop app was pinned to the
        # defaults and had no way to reach its own runtime controls.
        if processor is None:
            processor = FaceGestureProcessor(
                detect_every=detect_every, face_hz=face_hz,
                gesture_hz=gesture_hz, pose_hz=pose_hz,
            ) if initialize_models else (
                lambda frame, src, sequence, captured_at:
                SceneFrame(src, sequence, captured_at, frame)
            )
        self.processor = processor
        self.pipeline = VisionPipeline(source, processor, [self.latest, self.gestures])
        # The preview server carries a token now — /latest.jpg is live camera
        # video and used to be readable by any process on the machine that could
        # guess the port. The GUI is a local subscriber like any other and
        # authenticates the same way.
        token, _ = load_or_create_token()
        self.preview = VisionServer(self.latest, self.pipeline,
                                    port=preview_port, token=token,
                                    bus=self.emitter.bus, emitter=self.emitter)
        self.preview_url = (f"http://127.0.0.1:{preview_port}/latest.jpg"
                            f"?token={quote(token)}")

        self.gestureFromWorker.connect(self._set_gesture)
        self.droidScanFinished.connect(self._apply_droid_scan)
        self.overlayToggleRequested.connect(self.toggleCleanOverlay)
        self.audioMeterSample.connect(self._apply_audio_meter_sample)
        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._refresh)

    def start(self):
        self.pipeline.start()
        self.preview.start()
        self._poll.start()

    def stop(self):
        self._poll.stop()
        self._audio_meter.stop()
        if self.pipeline.is_alive():
            self.pipeline.stop()
            self.pipeline.join(timeout=5.0)
        self.preview.stop()

    def _set_action_error(self, message):
        """Raise an action/validation error that the refresh tick cannot erase.

        It stays on screen until another action fails or the operator dismisses
        it — the only two events that make the old message wrong.
        """
        message = str(message)
        if message == self._action_error:
            return
        visible = self.lastError
        self._action_error = message
        if self.lastError != visible:
            self.errorChanged.emit()

    @Slot()
    def dismissError(self):
        """Operator acknowledgement. The only thing that clears an action error."""
        self._set_action_error("")

    def _receive_gesture(self, event):
        decisions = self.rule_engine.evaluate(event)
        if decisions:
            decision = decisions[-1]
            self._last_decision = decision.describe()
            self._last_outcome = decision.outcome
            self.decisionChanged.emit()
            # A failed action is a first-class visible failure, not a log line.
            # The whole point of the typed connector is that a dead daemon
            # cannot look like success the way Popen("ledctl ...") did.
            if decision.outcome == "failed":
                self._set_action_error(
                    f"Action failed [{decision.error_kind}]: "
                    f"{decision.connector}.{decision.action} — {decision.reason}"
                )
        self.gestureFromWorker.emit(json.dumps(event, sort_keys=True))

    @Slot(str)
    def _set_gesture(self, event_json):
        event = json.loads(event_json)
        actor = event.get("actor") or "unknown actor"
        self._last_gesture = f"{event['gesture']} by {actor} ({self._last_outcome})"
        self.gestureChanged.emit()

    @Slot()
    def _refresh(self):
        now = self._clock()
        state = self.pipeline.state()
        if (state.source.kind == "webcam"
                and state.source.index in self._virtual_webcam_indexes
                and self._preferred_webcam is not None):
            # Never ingest our own v4l2loopback output. Besides wasting the
            # perception budget it recursively re-encodes the preview and
            # produces the visible feedback flicker the guard exists to stop.
            self.useWebcamIndex(self._preferred_webcam.index)
            return
        if state.status != self._status:
            self._status = state.status
            self.statusChanged.emit()
        label = state.source.safe_label()
        if label != self._source:
            self._source = label
            self._apply_exposure_capability(label)
            self.sourceChanged.emit()
            self.recordingFpsChanged.emit()
        if state.sequence != self._sequence:
            self._sequence = state.sequence
            self._last_frame_at = now
            self.sequenceChanged.emit()
            if (self._last_preview_emit_at is None
                    or now - self._last_preview_emit_at >= PREVIEW_REFRESH_S):
                self._preview_tick += 1
                self._last_preview_emit_at = now
                self.previewChanged.emit()
        self._update_preview_staleness(now)
        if state.last_error != self._pipeline_error:
            visible = self.lastError
            self._pipeline_error = state.last_error
            if self.lastError != visible:
                self.errorChanged.emit()
        if self._recorder is not None and self._recorder.error:
            message = f"Recording failed: {self._recorder.error}"
            if message != self._recording_status:
                self._recording_status = message
                self.recordingChanged.emit()
            self._set_action_error(message)
        self._slow_tick += 1
        if self._slow_tick >= SLOW_REFRESH_TICKS:
            self._slow_tick = 0
            self._refresh_slow_capabilities()
        metrics = state.metrics
        counts = (int(metrics.get("object_count", 0)),
                  int(metrics.get("face_count", 0)),
                  int(metrics.get("gesture_count", 0)),
                  int(metrics.get("pose_count", 0)))
        if counts != (self._object_count, self._face_count, self._gesture_count,
                      self._pose_count):
            (self._object_count, self._face_count, self._gesture_count,
             self._pose_count) = counts
            self.sceneCountsChanged.emit()
        capture_fps = float(metrics.get("capture_fps", 0.0))
        inference_fps = float(metrics.get("inference_fps", 0.0))
        inference_ms = float(metrics.get("inference_ms", 0.0))
        latest_ms = float(metrics.get("latest_inference_ms", 0.0))
        model_ms = float(metrics.get("model_inference_ms", 0.0))
        model_summary = str(metrics.get("object_model", "Models warming up"))
        performance = (round(capture_fps, 1), round(inference_fps, 1),
                       round(inference_ms, 1), round(latest_ms, 1),
                       round(model_ms, 1), model_summary)
        previous = (round(self._capture_fps, 1), round(self._inference_fps, 1),
                    round(self._inference_ms, 1),
                    round(self._latest_inference_ms, 1),
                    round(self._model_inference_ms, 1), self._model_summary)
        if performance != previous:
            self._capture_fps, self._inference_fps = capture_fps, inference_fps
            self._inference_ms, self._model_summary = inference_ms, model_summary
            self._latest_inference_ms = latest_ms
            self._model_inference_ms = model_ms
            self.performanceChanged.emit()
            if not self._recording_rate:
                self.recordingFpsChanged.emit()
        self._absorb_stage_metrics(metrics)
        workout = (
            bool(metrics.get("workout_enabled", self._workout_enabled)),
            str(metrics.get("workout_exercise", self._workout_exercise)),
            int(metrics.get("workout_reps", self._workout_reps)),
            str(metrics.get("workout_phase", self._workout_phase)),
            float(metrics.get("workout_angle", self._workout_angle) or 0.0),
            float(metrics.get("workout_progress", self._workout_progress)),
            str(metrics.get("workout_feedback", self._workout_feedback)),
            str(metrics.get("workout_filter", self._workout_filter)),
        )
        if workout != (self._workout_enabled, self._workout_exercise,
                       self._workout_reps, self._workout_phase,
                       self._workout_angle, self._workout_progress,
                       self._workout_feedback, self._workout_filter):
            (self._workout_enabled, self._workout_exercise,
             self._workout_reps, self._workout_phase, self._workout_angle,
             self._workout_progress, self._workout_feedback,
             self._workout_filter) = workout
            self.workoutChanged.emit()
        image_warning = str(metrics.get("image_warning", ""))
        image_mean = float(metrics.get("image_mean", 0.0))
        image_std = float(metrics.get("image_std", 0.0))
        if (image_warning, round(image_mean, 1)) != (
                self._image_warning, round(self._image_mean, 1)):
            self._image_warning = image_warning
            self._image_mean = image_mean
            self.imageChanged.emit()
        self._check_manual_exposure(label, image_mean, image_std,
                                    metrics.get("camera_controls"))

    # ---- preview freshness -------------------------------------------------

    def _update_preview_staleness(self, now):
        """A still frame that keeps rendering is indistinguishable from video.

        The preview Image only refetches when previewSource changes, and that
        only changes when the sequence moves. If capture stalls, the last good
        frame stays up forever and reads as live. Say so instead.
        """
        stale = (self._last_frame_at is not None
                 and self._status != "stopped"
                 and (now - self._last_frame_at) >= PREVIEW_STALE_AFTER_S)
        if stale != self._preview_stale:
            self._preview_stale = stale
            self.previewStaleChanged.emit()

    # ---- runtime capability detection --------------------------------------

    def _refresh_slow_capabilities(self):
        """Re-scan the things that can appear after start-up.

        Enrolled people, wired connectors and the compute device were all
        `constant=True`, so enrolling somebody or starting a connector daemon
        could never show up without restarting the app.
        """
        self.refreshActors()
        self.refreshConnectors()
        self.refreshAudioSources()
        device = describe_compute_device()
        if device != self._compute_device:
            self._compute_device = device
            self.computeChanged.emit()

    def _check_manual_exposure(self, device, image_mean, image_std, controls):
        """Withdraw manual exposure for a camera observed producing black frames.

        Reuses the same black/uniform signal pipeline._frame_quality raises, so
        the guard is a *detected* per-device condition rather than a constant
        baked in for every user.
        """
        if controls is not None and controls.get("auto_exposure", True):
            return
        if self._auto_exposure:
            return
        if image_mean >= BLACK_FRAME_MEAN or image_std >= BLACK_FRAME_STD:
            return
        if device in self._manual_exposure_blocked:
            return
        self._manual_exposure_blocked[device] = (
            "returned a black frame in manual exposure mode")
        self._apply_exposure_capability(device)
        # Put the camera back somewhere it can actually see.
        self.setCameraTuning(True, self._exposure, self._brightness,
                             self._contrast, self._gamma)
        self._set_action_error(
            f"{device} returned a black frame in manual exposure mode. "
            "Automatic exposure has been restored for this camera.")

    def _apply_exposure_capability(self, device):
        supported = device not in self._manual_exposure_blocked
        notice = "" if supported else (
            f"{device} {self._manual_exposure_blocked[device]}, so manual "
            "exposure is disabled for it. Other cameras are unaffected.")
        if (supported, notice) == (self._manual_exposure_supported,
                                   self._manual_exposure_notice):
            return
        self._manual_exposure_supported = supported
        self._manual_exposure_notice = notice
        self.exposureCapabilityChanged.emit()

    @Slot()
    def refreshActors(self):
        """Re-read known_faces/. Enrolling somebody must not need a restart."""
        actors = [ANY_ACTOR] + known_actors()
        if actors != self._actor_names:
            self._actor_names = actors
            self.actorsChanged.emit()

    @Slot()
    def refreshConnectors(self):
        names = list(self.executor.names())
        if names != self._connector_names:
            self._connector_names = names
            self.connectorsChanged.emit()

    @Slot()
    def refreshAudioSources(self):
        sources = discover_audio_sources()
        known = {item["id"] for item in sources}
        selection_changed = False
        if self._audio_source and self._audio_source not in known:
            self._audio_source = ""
            selection_changed = True
        if sources != self._audio_sources:
            self._audio_sources = sources
            selection_changed = True
        if selection_changed:
            self._refresh_audio_control()
            self.audioSourcesChanged.emit()

    def _selected_audio_source(self):
        return next((item for item in self._audio_sources
                     if item["id"] == self._audio_source), None)

    def _emit_audio_meter_sample(self, source, level_db, status):
        """Thread-safe handoff from AudioLevelMeter into Qt's GUI thread."""
        self.audioMeterSample.emit(str(source), float(level_db), str(status))

    @Slot(str, float, str)
    def _apply_audio_meter_sample(self, source, level_db, status):
        if source != self._audio_source:
            return                         # sample from a just-replaced mic
        level_db = max(AudioLevelMeter.FLOOR_DB, min(0.0, float(level_db)))
        if (round(level_db, 1), status) == (round(self._audio_level_db, 1),
                                            self._audio_meter_status):
            return
        self._audio_level_db = level_db
        self._audio_meter_status = status or (
            "Speak normally; aim for peaks around −12 to −6 dB")
        self.audioControlChanged.emit()

    def _refresh_audio_control(self):
        """Bind gain/meter only to a selected physical microphone."""
        self._audio_meter.stop()
        source = self._selected_audio_source()
        self._audio_level_db = AudioLevelMeter.FLOOR_DB
        if source is None or source.get("kind") == "none":
            self._audio_meter_status = "Choose a microphone to show its live level"
        elif source.get("kind") != "microphone":
            self._audio_meter_status = (
                "System-audio monitors are read-only; adjust the originating app instead")
        else:
            try:
                self._audio_gain = source_volume_percent(
                    self._audio_source, run=self._audio_run)
                self._audio_meter_status = self._audio_meter.start(self._audio_source)
            except (OSError, ValueError, RuntimeError) as exc:
                self._audio_meter_status = f"Microphone control unavailable: {exc}"
        self.audioControlChanged.emit()

    @Slot(str)
    def setRecordingAudioSource(self, source_id):
        """Choose audio for the *next* recording; active files stay coherent."""
        source_id = str(source_id or "")
        if source_id not in {item["id"] for item in self._audio_sources}:
            self._set_action_error("Selected audio input is no longer available")
            return
        if source_id == self._audio_source:
            return
        self._audio_source = source_id
        self._refresh_audio_control()
        self.audioSourcesChanged.emit()

    @Slot(int)
    def setRecordingAudioGain(self, percent):
        source = self._selected_audio_source()
        if source is None or source.get("kind") != "microphone":
            self._set_action_error("Choose a microphone before adjusting input gain")
            return
        try:
            self._audio_gain = set_source_volume_percent(
                self._audio_source, percent, run=self._audio_run)
        except (OSError, ValueError, RuntimeError) as exc:
            self._set_action_error(f"Could not set microphone gain: {exc}")
            return
        self.audioControlChanged.emit()

    @Slot(int)
    def setRecordingRate(self, rate):
        rate = int(rate)
        if rate not in (0, 30, 60):
            self._set_action_error("Recording rate must be Auto, 30 FPS, or 60 FPS")
            return
        if rate == self._recording_rate:
            return
        self._recording_rate = rate
        self.recordingFpsChanged.emit()

    def _workout_setter(self, name):
        setter = getattr(self.processor, name, None)
        if setter is None:
            self._set_action_error(
                "Workout analysis is unavailable: no inference loop is running.")
            return None
        return setter

    @Slot(bool)
    def setWorkoutEnabled(self, enabled):
        setter = self._workout_setter("set_workout_enabled")
        if setter is None:
            return
        setter(bool(enabled))
        self._workout_enabled = bool(enabled)
        self.workoutChanged.emit()

    @Slot(str)
    def setWorkoutExercise(self, exercise):
        setter = self._workout_setter("set_workout_exercise")
        if setter is None:
            return
        try:
            setter(str(exercise))
        except ValueError as exc:
            self._set_action_error(str(exc))
            return
        self._workout_exercise = str(exercise)
        self.workoutChanged.emit()

    @Slot()
    def resetWorkout(self):
        setter = self._workout_setter("reset_workout")
        if setter is None:
            return
        setter()
        self._workout_reps = 0
        self._workout_phase = "find rest"
        self.workoutChanged.emit()

    @Slot()
    def refreshModels(self):
        """Re-verify the model manifest against what is actually on disk."""
        options, paths = self._scan_models()
        if options != self._model_options:
            self._model_options, self._model_paths = options, paths
            self.modelsChanged.emit()

    def _scan_models(self):
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text())
        repo_root = Path(__file__).resolve().parents[1]
        options, paths = [], {}
        for model in manifest["models"]:
            path = repo_root / model["file"]
            if path.is_file() and file_sha256(path) == model["sha256"]:
                option = dict(model)
                option["path"] = str(path)
                options.append(option)
                paths[model["id"]] = path
        return options, paths

    @Property(str, notify=statusChanged)
    def status(self):
        return self._status

    @Property(str, notify=sourceChanged)
    def sourceLabel(self):
        return self._source

    @Property("QVariantList", notify=rotationChanged)
    def rotationOptions(self):
        return [
            {"id": degrees,
             "label": ("0° — landscape" if degrees == 0
                       else f"{degrees}° clockwise")}
            for degrees in ROTATION_DEGREES
        ]

    @Property(int, notify=rotationChanged)
    def rotationIndex(self):
        return ROTATION_DEGREES.index(self._rotation_degrees)

    @Property(int, notify=rotationChanged)
    def rotationDegrees(self):
        return self._rotation_degrees

    @Property(int, notify=sequenceChanged)
    def sequence(self):
        return self._sequence

    @Property(str, notify=errorChanged)
    def lastError(self):
        """What the operator sees. An action error outranks a pipeline error."""
        return self._action_error or self._pipeline_error

    @Property(bool, notify=errorChanged)
    def errorDismissable(self):
        """Only an action error can be dismissed; a pipeline error is live state."""
        return bool(self._action_error)

    @Property(str, notify=previewChanged)
    def previewSource(self):
        # preview_url already carries the token in its query string, so the
        # cache-buster joins with '&'. Main.qml then appends "&retry=".
        return f"{self.preview_url}&t={self._preview_tick}"

    @Property(bool, notify=previewStaleChanged)
    def previewStale(self):
        """True when the frame on screen is old enough to not be live video."""
        return self._preview_stale

    @Property(bool, notify=obsChanged)
    def obsEnabled(self):
        return self._obs_enabled

    @Property(bool, notify=eventsChanged)
    def eventsEnabled(self):
        return self._events_enabled

    @Property(int, notify=sceneCountsChanged)
    def objectCount(self):
        return self._object_count

    @Property(int, notify=sceneCountsChanged)
    def faceCount(self):
        return self._face_count

    @Property(int, notify=sceneCountsChanged)
    def gestureCount(self):
        return self._gesture_count

    @Property(int, notify=sceneCountsChanged)
    def poseCount(self):
        return self._pose_count

    @Property("QVariantList", notify=audioSourcesChanged)
    def audioSources(self):
        return list(self._audio_sources)

    @Property(int, notify=audioSourcesChanged)
    def audioSourceIndex(self):
        for index, item in enumerate(self._audio_sources):
            if item["id"] == self._audio_source:
                return index
        return 0

    @Property(str, notify=audioSourcesChanged)
    def recordingAudioLabel(self):
        for item in self._audio_sources:
            if item["id"] == self._audio_source:
                return item["label"]
        return "No audio (video only)"

    @Property(bool, notify=audioControlChanged)
    def recordingMicrophoneSelected(self):
        source = self._selected_audio_source()
        return bool(source and source.get("kind") == "microphone")

    @Property(int, notify=audioControlChanged)
    def recordingAudioGain(self):
        return self._audio_gain

    @Property(float, notify=audioControlChanged)
    def recordingAudioLevelDb(self):
        return self._audio_level_db

    @Property(float, notify=audioControlChanged)
    def recordingAudioLevel(self):
        # Map a practical -60..0 dBFS meter onto Qt's 0..1 ProgressBar range.
        return max(0.0, min(1.0,
                            (self._audio_level_db - AudioLevelMeter.FLOOR_DB)
                            / -AudioLevelMeter.FLOOR_DB))

    @Property(str, notify=audioControlChanged)
    def recordingAudioMeterStatus(self):
        return self._audio_meter_status

    @Property("QVariantList", notify=workoutChanged)
    def workoutExercises(self):
        return [{"id": item.id, "label": item.label} for item in EXERCISES]

    @Property(bool, notify=workoutChanged)
    def workoutEnabled(self):
        return self._workout_enabled

    @Property(int, notify=workoutChanged)
    def workoutReps(self):
        return self._workout_reps

    @Property(str, notify=workoutChanged)
    def workoutPhase(self):
        return self._workout_phase

    @Property(float, notify=workoutChanged)
    def workoutAngle(self):
        return self._workout_angle

    @Property(float, notify=workoutChanged)
    def workoutProgress(self):
        return self._workout_progress

    @Property(str, notify=workoutChanged)
    def workoutFeedback(self):
        return self._workout_feedback

    @Property(str, notify=workoutChanged)
    def workoutFilter(self):
        return self._workout_filter

    @Property(int, notify=workoutChanged)
    def workoutExerciseIndex(self):
        for index, item in enumerate(EXERCISES):
            if item.id == self._workout_exercise:
                return index
        return 0

    @Property(bool, notify=recordingChanged)
    def recordingEnabled(self):
        """Whether this GUI process owns a recording output.

        Recording is another output of the already-running pipeline. It never
        opens a second camera, so enabling it cannot create a preview feedback
        loop or fight the live view for a device handle.
        """
        return self._recorder is not None

    @Property(str, notify=recordingChanged)
    def recordingStatus(self):
        return self._recording_status

    @Property("QVariantList", notify=recordingFpsChanged)
    def recordingRateOptions(self):
        return [
            {"id": 0, "label": "Auto (DroidCam 60 / webcam 30)"},
            {"id": 30, "label": "30 FPS"},
            {"id": 60, "label": "60 FPS"},
        ]

    @Property(int, notify=recordingFpsChanged)
    def recordingRateIndex(self):
        return {0: 0, 30: 1, 60: 2}.get(self._recording_rate, 0)

    @Property(int, notify=recordingFpsChanged)
    def recordingFps(self):
        if self._recording_rate:
            return self._recording_rate
        source = self.pipeline.state().source
        return 60 if source.kind == "droidcam" else 30

    @Property(str, notify=overlayChanged)
    def overlayProfile(self):
        return self._overlay

    @Property(str, notify=gestureChanged)
    def lastGesture(self):
        return self._last_gesture

    @Property("QVariantList", notify=rulesChanged)
    def rules(self):
        return [{
            "id": rule.id,
            "gesture": rule.gesture,
            "actor": rule.actor,
            "connector": rule.connector,
            "action": rule.action,
            "risk": rule.risk,
            "dryRun": rule.dry_run,
            "executable": self.executor.supports(rule.connector, rule.action),
        } for rule in self._rules]

    @Property(str, notify=decisionChanged)
    def lastDecision(self):
        return self._last_decision

    @Property("QVariantList", notify=connectorsChanged)
    def executableConnectors(self):
        """Connectors that have a real executor. Everything else is declared only."""
        return list(self._connector_names)

    @Property(float, notify=performanceChanged)
    def captureFps(self):
        return self._capture_fps

    @Property(float, notify=performanceChanged)
    def inferenceFps(self):
        return self._inference_fps

    @Property(float, notify=performanceChanged)
    def inferenceMs(self):
        return self._inference_ms

    @Property(str, notify=performanceChanged)
    def modelSummary(self):
        return self._model_summary

    @Property(float, notify=performanceChanged)
    def latestInferenceMs(self):
        """The most recent cycle, not the 30-cycle average `inferenceMs` is."""
        return self._latest_inference_ms

    @Property(float, notify=performanceChanged)
    def modelInferenceMs(self):
        """Time inside the object model itself, excluding worker round-trip."""
        return self._model_inference_ms

    # ---- per-stage perception control --------------------------------------

    def _absorb_stage_metrics(self, metrics):
        """Keep the per-stage measurements this poll used to throw away.

        processor.py has published each stage's cost, refresh flag and rate
        every cycle since the stages were split apart. The poll read
        `inference_ms` and `object_model` out of that dict and dropped the
        rest, so the single ~120 ms number on screen could not be attributed
        to any stage — and the knobs behind those numbers were unreachable.
        """
        if "object_stage_ms" not in metrics:
            return                      # no inference loop (smoke-test shell)
        rates = {
            "object": float(metrics.get("object_detect_every",
                                        self._stage_rate["object"])),
            "face": float(metrics.get("face_refresh_hz",
                                      self._stage_rate["face"])),
            "gesture": float(metrics.get("gesture_refresh_hz",
                                         self._stage_rate["gesture"])),
            "pose": float(metrics.get("pose_refresh_hz",
                                      self._stage_rate["pose"])),
        }
        enabled = {stage: bool(metrics.get(f"{stage}_enabled", True))
                   for stage in STAGE_IDS}
        stage_ms = {stage: float(metrics.get(f"{stage}_stage_ms", 0.0))
                    for stage in STAGE_IDS}
        refreshed = {stage: bool(metrics.get(f"{stage}_refreshed", False))
                     for stage in STAGE_IDS}
        if (self._stage_snapshot(rates, enabled, stage_ms, refreshed)
                == self._stage_snapshot(self._stage_rate, self._stage_enabled,
                                        self._stage_ms,
                                        self._stage_refreshed)):
            return
        self._stage_rate, self._stage_enabled = rates, enabled
        self._stage_ms, self._stage_refreshed = stage_ms, refreshed
        self.stagesChanged.emit()

    @staticmethod
    def _stage_snapshot(rates, enabled, stage_ms, refreshed):
        """Comparable form. Costs are float noise below 0.1 ms; don't churn."""
        return (
            {stage: round(value, 2) for stage, value in rates.items()},
            dict(enabled),
            {stage: round(value, 1) for stage, value in stage_ms.items()},
            dict(refreshed),
        )

    def _stage_hz(self, stage):
        """How often this stage actually refreshes, in Hz.

        Face and gesture are gated by a target frequency, and that target is
        their refresh rate. The object stage runs once per inference cycle, so
        its refresh rate is the measured cycle rate — which already accounts
        for the frame divisor and for whatever the camera can deliver.
        """
        if stage == "object":
            return self._inference_fps
        return self._stage_rate[stage]

    def _degradation(self):
        """(stage id, warning) for the shush degradation, or (None, "")."""
        warning = shush_degradation(self._stage_enabled["object"],
                                    self._stage_enabled["face"],
                                    self._stage_rate["face"])
        if not warning:
            return None, ""
        if not self._stage_enabled["face"]:
            return "face", warning
        if not self._stage_enabled["object"]:
            return "object", warning
        return "face", warning

    @Property("QVariantList", notify=stageSetChanged)
    def stageIds(self):
        """The stage set, which is fixed — see stageSetChanged for why."""
        return list(STAGE_IDS)

    @Property("QVariantList", notify=stagesChanged)
    def stageStats(self):
        """One row per stage: what it costs, how often it runs, and its knob."""
        source, warning = self._degradation()
        stats = []
        for spec in PERCEPTION_STAGES:
            stage = spec["id"]
            row = dict(spec)
            row.update({
                "ms": round(self._stage_ms[stage], 1),
                "hz": round(self._stage_hz(stage), 2),
                "enabled": self._stage_enabled[stage],
                "refreshed": self._stage_refreshed[stage],
                "rate": self._stage_rate[stage],
                "warning": warning if stage == source else "",
            })
            stats.append(row)
        return stats

    @Property(str, notify=stagesChanged)
    def shushWarning(self):
        """Visible whenever these knobs have rebound Shush onto the lights."""
        return self._degradation()[1]

    @Property(bool, notify=stagesChanged)
    def shushDegraded(self):
        return bool(self._degradation()[1])

    def _stage_setter(self, stage, name):
        """The processor's setter, or None with the reason already on screen."""
        if stage not in STAGE_IDS:
            self._set_action_error(f"Unknown perception stage: {stage}")
            return None
        setter = getattr(self.processor, name, None)
        if setter is None:
            # Refusing beats accepting. Showing a stage as switched off while
            # it is still running would be a worse lie than declining the click.
            self._set_action_error(
                "Perception stages cannot be controlled: "
                "no inference loop is running.")
            return None
        return setter

    @Slot(str, bool)
    def setStageEnabled(self, stage, enabled):
        setter = self._stage_setter(stage, "set_stage_enabled")
        if setter is None:
            return
        setter(stage, bool(enabled))
        self._stage_enabled[stage] = bool(enabled)
        self.stagesChanged.emit()

    @Slot(str, float)
    def setStageRate(self, stage, value):
        setter = self._stage_setter(stage, STAGE_RATE_SETTERS.get(stage, ""))
        if setter is None:
            return
        value = self._clamp_rate(stage, value)
        setter(int(value) if STAGE_BY_ID[stage]["rateKind"] == "every"
               else value)
        self._stage_rate[stage] = value
        self.stagesChanged.emit()

    @staticmethod
    def _clamp_rate(stage, value):
        """Hold the knob inside the range the panel advertises."""
        spec = STAGE_BY_ID[stage]
        value = max(spec["rateMin"], min(spec["rateMax"], float(value)))
        if spec["rateKind"] == "every":
            value = float(round(value))
        return value

    @Property("QVariantList", notify=modelsChanged)
    def modelOptions(self):
        return self._model_options

    @Property(str, notify=computeChanged)
    def computeDevice(self):
        """The inference device this machine actually has, not the author's."""
        return self._compute_device

    @Property(str, notify=imageChanged)
    def imageWarning(self):
        return self._image_warning

    @Property(float, notify=imageChanged)
    def imageMean(self):
        return self._image_mean

    @Property(bool, notify=exposureCapabilityChanged)
    def manualExposureSupported(self):
        """Whether the *current* camera may be driven in manual exposure."""
        return self._manual_exposure_supported

    @Property(str, notify=exposureCapabilityChanged)
    def exposureNotice(self):
        """Why manual exposure is unavailable, or empty when it is available."""
        return self._manual_exposure_notice

    @Property(bool, notify=cameraTuningChanged)
    def autoExposure(self):
        return self._auto_exposure

    @Slot(bool, int, int, int, int)
    def setCameraTuning(self, auto_exposure, exposure, brightness, contrast, gamma):
        self._auto_exposure = (bool(auto_exposure) or
                               not self._manual_exposure_supported)
        self._exposure = int(exposure)
        self._brightness = int(brightness)
        self._contrast = int(contrast)
        self._gamma = int(gamma)
        self.pipeline.set_camera_controls(
            auto_exposure=self._auto_exposure,
            exposure=self._exposure,
            brightness=self._brightness,
            contrast=self._contrast,
            gamma=self._gamma,
        )
        self.cameraTuningChanged.emit()

    @Slot(int)
    def setRotation(self, degrees):
        """Set the display/capture orientation for the next frames.

        Rotation happens before local perception, so landmark and detection
        coordinates remain true to the portrait/landscape video. An active
        recording holds its geometry intentionally; changing it mid-file would
        force an unsafe resize, so finish that take first.
        """
        try:
            degrees = int(degrees)
            if degrees not in ROTATION_DEGREES:
                raise ValueError
        except (TypeError, ValueError):
            self._set_action_error(
                "Rotation must be 0°, 90°, 180°, or 270° clockwise")
            return
        if degrees == self._rotation_degrees:
            return
        if self.recordingEnabled:
            self._set_action_error(
                "Stop recording before changing orientation; each MP4 keeps one frame size.")
            return
        self.pipeline.set_rotation(degrees)
        if self._obs is not None:
            # pyvirtualcam fixes its frame dimensions when it opens. Reopen
            # the virtual camera for a quarter-turn instead of stretching a
            # portrait frame back into the old landscape output.
            self.pipeline.remove_output(self._obs)
            self._obs = ObsVirtualCameraOutput(PROFILES[self._overlay])
            self.pipeline.add_output(self._obs)
        self._rotation_degrees = degrees
        self.rotationChanged.emit()

    @Slot(str)
    def setObjectModel(self, model_id):
        path = self._model_paths.get(model_id)
        switch = getattr(self.processor, "set_object_model", None)
        if path is None or switch is None:
            self._set_action_error(f"Object model is not available: {model_id}")
            return
        switch(path)

    @Property("QVariantList", notify=webcamsChanged)
    def webcams(self):
        return self._webcams

    @Property("QVariantList", notify=droidCamsChanged)
    def droidCams(self):
        return self._droid_cams

    @Property(bool, notify=droidScanChanged)
    def droidScanActive(self):
        return self._droid_scan_active

    @Property(str, notify=droidScanChanged)
    def droidScanStatus(self):
        return self._droid_scan_status

    @Property("QVariantList", notify=connectorsChanged)
    def connectorNames(self):
        return list(CONNECTORS)

    @Slot(str, result="QVariantList")
    def connectorActions(self, connector):
        return list(CONNECTORS.get(connector, {}))

    @Property("QVariantList", notify=gesturesChanged)
    def gestureNames(self):
        """The shared catalog vocabulary — rules must pick from this list."""
        return list(GESTURE_IDS)

    @Property("QVariantList", notify=actorsChanged)
    def actorNames(self):
        """'*' (anyone) plus every enrolled identity under known_faces/.

        Re-scanned while the app runs. It used to be read once at construction,
        so somebody enrolled during the session could never be picked as a rule
        actor — on the very page you would enrol them from.
        """
        return list(self._actor_names)

    @Slot(str)
    def useWebcam(self, index_text):
        try:
            index = int(index_text) if index_text.strip() else None
        except ValueError:
            self._set_action_error(
                "Webcam index must be a number or blank for auto")
            return
        self.pipeline.switch_source(SourceSpec.from_mapping({
            "id": "webcam", "name": "Webcam", "type": "webcam", "index": index,
        }))

    @Slot(int)
    def useWebcamIndex(self, index):
        selected = next((device for device in self._webcams
                         if device["index"] == index), None)
        name = selected["name"] if selected else "Webcam"
        self.pipeline.switch_source(SourceSpec.from_mapping({
            "id": "webcam", "name": name, "type": "webcam", "index": index,
            "device_path": selected.get("stable_path") if selected else None,
        }))

    @Slot()
    def refreshWebcams(self):
        self._webcams = [device.as_dict() for device in discover_webcams()
                         if device.kind != "virtual"]
        self.webcamsChanged.emit()

    @Slot(str)
    def useDroidCam(self, url):
        if not url.strip():
            self._set_action_error("DroidCam URL is required")
            return
        self.pipeline.switch_source(SourceSpec.from_mapping({
            "id": "droidcam", "name": "DroidCam", "type": "droidcam",
            "url": url.strip(),
        }))

    @Property(str, notify=droidScanChanged)
    def scanPlanTarget(self):
        """The networks a scan would touch, readable before anyone clicks Scan.

        Shown on the Sources page on purpose. This program opens sockets on the
        operator's own home network; which network that is should never be
        something they have to read the source to find out.
        """
        try:
            plan = scan_plan()
        except Exception as exc:            # noqa: BLE001 - never break the UI
            return f"Scan unavailable: {exc}"
        if not plan.networks:
            return ("No scannable network found — set "
                    "ACESVISION_SCAN_INTERFACES or ACESVISION_SCAN_NETWORKS")
        return plan.summary()

    @Slot()
    def scanDroidCams(self):
        if self._droid_scan_active:
            return
        try:
            plan = scan_plan()
        except Exception as exc:            # noqa: BLE001 - never break the UI
            # A refused override lands here. Say so rather than scanning
            # something else instead.
            self._set_action_error(f"DroidCam scan refused: {exc}")
            self._droid_scan_status = f"Scan refused: {exc}"
            self.droidScanChanged.emit()
            return
        if not plan.networks:
            # Distinct from "scanned and found nothing", which is what this used
            # to say on every standard Linux host.
            self._droid_scan_status = (
                "No scannable network found on this machine. Set "
                "ACESVISION_SCAN_INTERFACES or enter a URL below.")
            self.droidScanChanged.emit()
            return

        networks = [str(network) for network in plan.networks]
        self._droid_scan_active = True
        self._droid_scan_status = f"Scanning {plan.summary()} on port 4747"
        self.droidScanChanged.emit()

        def worker():
            try:
                devices = [device.as_dict() for device in scan_droidcam(networks)]
                payload = json.dumps({"devices": devices, "networks": networks})
            except Exception as exc:
                payload = json.dumps({"error": str(exc), "devices": [],
                                      "networks": networks})
            self.droidScanFinished.emit(payload)

        threading.Thread(target=worker, daemon=True,
                         name="droidcam-discovery").start()

    @Slot(str)
    def _apply_droid_scan(self, payload):
        result = json.loads(payload)
        self._droid_cams = result["devices"]
        self._droid_scan_active = False
        scanned = ", ".join(result.get("networks") or []) or "the local network"
        if result.get("error"):
            self._droid_scan_status = "Scan failed: " + result["error"]
        elif self._droid_cams:
            count = len(self._droid_cams)
            self._droid_scan_status = (f"Found {count} possible DroidCam "
                                       f"device(s) on {scanned}")
        else:
            self._droid_scan_status = (f"Scanned {scanned} — nothing answered on "
                                       f"port 4747. You can still enter a URL.")
        self.droidCamsChanged.emit()
        self.droidScanChanged.emit()

    @Slot(bool)
    def setObsEnabled(self, enabled):
        if enabled == self._obs_enabled:
            return
        if enabled:
            self._obs = ObsVirtualCameraOutput(PROFILES[self._overlay])
            self.pipeline.add_output(self._obs)
        elif self._obs is not None:
            self.pipeline.remove_output(self._obs)
            self._obs = None
        self._obs_enabled = enabled
        self.obsChanged.emit()

    @Slot(bool)
    def setRecordingEnabled(self, enabled):
        """Start or finalise a local MP4 using the live GUI pipeline.

        ``RecordingOutput`` is added as a lossless worker, unlike the preview.
        That deliberately trades a visible capture-rate dip for a recording
        with accounted-for continuity rather than silently dropping time.
        """
        enabled = bool(enabled)
        if enabled == self.recordingEnabled:
            return
        if enabled:
            source = self.pipeline.state().source
            try:
                path = self._recording_path_factory("", source.id)
                kwargs = {"profile": PROFILES[self._overlay],
                          "fps": self.recordingFps}
                if self._audio_source:
                    kwargs.update(audio_source=self._audio_source,
                                  audio_label=self.recordingAudioLabel)
                recorder = self._recording_factory(path, **kwargs)
                self.pipeline.add_output(recorder, lossless=True)
            except (OSError, ValueError, RecordingError) as exc:
                self._recording_status = "Recording could not start"
                self._set_action_error(f"Recording could not start: {exc}")
                self.recordingChanged.emit()
                return
            self._recorder = recorder
            suffix = (f" with {self.recordingAudioLabel}"
                      if self._audio_source else " (video only)")
            self._recording_status = (f"Recording to {recorder.path} at "
                                      f"{recorder.fps} FPS{suffix}")
            self.recordingChanged.emit()
            return

        recorder, self._recorder = self._recorder, None
        self.pipeline.remove_output(recorder)
        if recorder.error:
            self._recording_status = f"Recording failed: {recorder.error}"
            self._set_action_error(self._recording_status)
        elif recorder.frames_written:
            self._recording_status = f"Saved {recorder.describe()}"
        else:
            self._recording_status = "Recording stopped before the first frame"
        self.recordingChanged.emit()

    @Slot(bool)
    def setEventsEnabled(self, enabled):
        self.gestures.set_enabled(enabled)
        self._events_enabled = enabled
        self.eventsChanged.emit()

    @Property(bool, notify=overlayProfilesChanged)
    def customOverlayReady(self):
        """True once this session has an applied custom profile to select.

        Overlay Studio needs it to render a fifth card. Without one, "Apply
        custom" set the active profile to an id no card knew about and every
        card read "Select".
        """
        return self._custom_overlay is not None

    @Slot(str)
    def setOverlayProfile(self, profile_id):
        if profile_id == "custom" and self._custom_overlay is not None:
            PROFILES["custom"] = self._custom_overlay
        if profile_id not in PROFILES:
            return
        if profile_id != "clean":
            self._overlay_before_clean = profile_id
        self._overlay = profile_id
        profile = PROFILES[profile_id]
        self.latest.set_profile(profile)
        if self._obs is not None:
            self._obs.set_profile(profile)
        if self._recorder is not None:
            self._recorder.set_profile(profile)
        self.overlayChanged.emit()

    @Slot()
    def toggleCleanOverlay(self):
        """Toggle visible annotations without interrupting the camera stream.

        ``clean`` is a compositor profile, not a capture switch: the raw
        camera, recognition pipeline and gesture event output stay live. The
        second gesture restores the exact profile selected before clean.
        """
        if self._overlay == "clean":
            self.setOverlayProfile(self._overlay_before_clean)
        else:
            self._overlay_before_clean = self._overlay
            self.setOverlayProfile("clean")

    @Slot(bool, bool, bool, int, float, str, str, str, str, bool, bool)
    def applyOverlayStyle(self, show_objects, show_faces, show_gestures,
                          line_width, font_scale, object_hex, known_hex,
                          unknown_hex, gesture_hex, show_landmarks=True,
                          show_pose=True):
        try:
            profile = OverlayProfile(
                id="custom",
                show_objects=show_objects,
                show_faces=show_faces,
                show_gestures=show_gestures,
                show_landmarks=show_landmarks,
                show_pose=show_pose,
                line_width=max(1, min(8, int(line_width))),
                font_scale=max(0.3, min(2.0, float(font_scale))),
                known_colour=self._hex_to_bgr(known_hex),
                unknown_colour=self._hex_to_bgr(unknown_hex),
                gesture_colour=self._hex_to_bgr(gesture_hex),
                object_colour=self._hex_to_bgr(object_hex),
            )
        except ValueError as exc:
            self._set_action_error(str(exc))
            return
        first_custom = self._custom_overlay is None
        PROFILES["custom"] = profile
        self._custom_overlay = profile
        self._overlay = "custom"
        self.latest.set_profile(profile)
        if self._obs is not None:
            self._obs.set_profile(profile)
        if self._recorder is not None:
            self._recorder.set_profile(profile)
        if first_custom:
            self.overlayProfilesChanged.emit()
        self.overlayChanged.emit()

    @staticmethod
    def _hex_to_bgr(value):
        text = value.strip().lstrip("#")
        if len(text) != 6:
            raise ValueError("Overlay colours must use six-digit hex values")
        try:
            red, green, blue = (int(text[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError as exc:
            raise ValueError("Overlay colour contains invalid hex digits") from exc
        return blue, green, red

    @Slot(str, str, str, str)
    def addRule(self, gesture, actor, connector, action):
        # Rule.create validates gesture and actor against the shared catalog and
        # the enrolled identities; unknown values are rejected here rather than
        # silently persisted as a rule that can never match.
        try:
            rule = Rule.create(gesture.strip(), connector, action,
                               actor=actor.strip() or ANY_ACTOR)
        except ValueError as exc:
            self._set_action_error(str(exc))
            return
        self._rules.append(rule)
        self.rule_engine.rules = list(self._rules)
        self._save_rules()
        self.rulesChanged.emit()

    @Slot(str, bool)
    def setRuleDryRun(self, rule_id, dry_run):
        """Arm or disarm one rule. This is the whole mechanism, per rule.

        Nothing calls it at start-up and nothing arms a rule as a side effect
        of anything else — the founder's saved rules stay dry-run until this is
        invoked on a named rule id.
        """
        updated = []
        changed = False
        for rule in self._rules:
            if rule.id == rule_id and rule.dry_run != bool(dry_run):
                updated.append(dataclass_replace(rule, dry_run=bool(dry_run)))
                changed = True
            else:
                updated.append(rule)
        if not changed:
            return
        self._rules = updated
        self.rule_engine.rules = list(self._rules)
        if not dry_run:
            # Arming this named rule was the deliberate gesture opt-in. Do not
            # make the operator find a separate, reset-on-restart output gate.
            self.setEventsEnabled(True)
        self._save_rules()
        self.rulesChanged.emit()

    @Slot(str)
    def removeRule(self, rule_id):
        self._rules = [rule for rule in self._rules if rule.id != rule_id]
        self.rule_engine.rules = list(self._rules)
        self._save_rules()
        self.rulesChanged.emit()

    def _save_rules(self):
        try:
            self.rule_store.save(self._rules)
        except OSError as exc:
            self._set_action_error(f"Rules were not saved: {exc}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="AcesVision desktop GUI")
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument("--smoke-test", action="store_true")
    # The same three knobs the headless runner takes. They are starting
    # positions only — the Perception stages panel drives them live from here.
    parser.add_argument("--detect-every", type=int, default=DEFAULT_DETECT_EVERY,
                        help="submit one captured frame in every N to inference")
    parser.add_argument("--face-hz", type=float, default=DEFAULT_FACE_HZ,
                        help="face stage refresh rate")
    parser.add_argument("--gesture-hz", type=float, default=DEFAULT_GESTURE_HZ,
                        help="gesture stage refresh rate")
    args = parser.parse_args(argv)

    app = QGuiApplication(sys.argv[:1])
    # Parent the backend to the application so C++ owns its lifetime. Without
    # an owner, Python could collect it while QML still held the raw pointer
    # behind the `vision` context property.
    backend = VisionBackend(preview_port=args.preview_port,
                            initialize_models=not args.smoke_test,
                            load_saved_rules=not args.smoke_test,
                            detect_every=args.detect_every,
                            face_hz=args.face_hz,
                            gesture_hz=args.gesture_hz,
                            parent=app)
    # Deliberately unparented: Python owns the engine, so dropping the last
    # reference below destroys the QML tree at a point we choose.
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("vision", backend)
    engine.load(QUrl.fromLocalFile(str(QML_PATH)))
    if not engine.rootObjects():
        return 1
    app.aboutToQuit.connect(backend.stop)
    if args.smoke_test:
        QTimer.singleShot(50, app.quit)
    else:
        backend.start()
    try:
        return app.exec()
    finally:
        # Tear the QML object tree down FIRST, while `vision` is still alive.
        # Freeing the backend first made every binding on every page re-evaluate
        # against a null object and print "Cannot read property 'X' of null" —
        # roughly 45 lines of noise on every clean exit.
        del engine


if __name__ == "__main__":
    raise SystemExit(main())
