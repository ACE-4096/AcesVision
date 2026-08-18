"""Qt 6 and QML desktop shell for AcesVision."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import replace as dataclass_replace
from pathlib import Path

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from gesture_catalog import ANY_ACTOR, GESTURE_IDS

from .connectors import default_registry
from .contracts import SceneFrame, SourceSpec
from .events import GestureEventOutput
from .discovery import discover_webcams, preferred_webcam, scan_droidcam
from .outputs import LatestFrameOutput, ObsVirtualCameraOutput
from .overlay import MINIMAL, PROFILES, OverlayProfile
from .pipeline import VisionPipeline
from .policy import CONNECTORS, Rule, RuleEngine, RuleStore, known_actors
from .preview import PreviewServer
from .processor import FaceGestureProcessor
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

# The refresh timer runs at 100 ms. Filesystem and registry re-scans are far
# too coarse for that, so they ride a slower tick.
SLOW_REFRESH_TICKS = 20


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

    gestureFromWorker = Signal(str)
    droidScanFinished = Signal(str)

    def __init__(self, preview_port=8765, initialize_models=True,
                 load_saved_rules=True, executor=None, parent=None,
                 clock=time.monotonic):
        super().__init__(parent)
        self._clock = clock
        self._status = "starting"
        discovered_webcams = discover_webcams()
        default_webcam = preferred_webcam(discovered_webcams)
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
        self._last_frame_at = None
        self._preview_stale = False
        self._slow_tick = 0
        self._obs_enabled = False
        self._events_enabled = False
        self._overlay = "minimal"
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
        self._model_summary = "Models warming up"
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
        self.executor = default_registry() if executor is None else executor
        self.rule_engine = RuleEngine(self._rules, executor=self.executor)
        self._connector_names = list(self.executor.names())
        self._actor_names = [ANY_ACTOR] + known_actors()
        self._obs = None
        self._webcams = [device.as_dict() for device in discovered_webcams]
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
        self.gestures = GestureEventOutput(self._receive_gesture)
        processor = FaceGestureProcessor() if initialize_models else (
            lambda frame, src, sequence, captured_at:
            SceneFrame(src, sequence, captured_at, frame)
        )
        self.processor = processor
        self.pipeline = VisionPipeline(source, processor, [self.latest, self.gestures])
        self.preview = PreviewServer(self.latest, self.pipeline, port=preview_port)
        self.preview_url = f"http://127.0.0.1:{preview_port}/latest.jpg"

        self.gestureFromWorker.connect(self._set_gesture)
        self.droidScanFinished.connect(self._apply_droid_scan)
        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._refresh)

    def start(self):
        self.pipeline.start()
        self.preview.start()
        self._poll.start()

    def stop(self):
        self._poll.stop()
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
        if state.status != self._status:
            self._status = state.status
            self.statusChanged.emit()
        label = state.source.safe_label()
        if label != self._source:
            self._source = label
            self._apply_exposure_capability(label)
            self.sourceChanged.emit()
        if state.sequence != self._sequence:
            self._sequence = state.sequence
            self._preview_tick += 1
            self._last_frame_at = now
            self.sequenceChanged.emit()
            self.previewChanged.emit()
        self._update_preview_staleness(now)
        if state.last_error != self._pipeline_error:
            visible = self.lastError
            self._pipeline_error = state.last_error
            if self.lastError != visible:
                self.errorChanged.emit()
        self._slow_tick += 1
        if self._slow_tick >= SLOW_REFRESH_TICKS:
            self._slow_tick = 0
            self._refresh_slow_capabilities()
        metrics = state.metrics
        capture_fps = float(metrics.get("capture_fps", 0.0))
        inference_fps = float(metrics.get("inference_fps", 0.0))
        inference_ms = float(metrics.get("inference_ms", 0.0))
        model_summary = str(metrics.get("object_model", "Models warming up"))
        performance = (round(capture_fps, 1), round(inference_fps, 1),
                       round(inference_ms, 1), model_summary)
        previous = (round(self._capture_fps, 1), round(self._inference_fps, 1),
                    round(self._inference_ms, 1), self._model_summary)
        if performance != previous:
            self._capture_fps, self._inference_fps = capture_fps, inference_fps
            self._inference_ms, self._model_summary = inference_ms, model_summary
            self.performanceChanged.emit()
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
        return f"{self.preview_url}?t={self._preview_tick}"

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
        self._webcams = [device.as_dict() for device in discover_webcams()]
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

    @Slot()
    def scanDroidCams(self):
        if self._droid_scan_active:
            return
        self._droid_scan_active = True
        self._droid_scan_status = "Scanning private local networks on port 4747"
        self.droidScanChanged.emit()

        def worker():
            try:
                devices = [device.as_dict() for device in scan_droidcam()]
                payload = json.dumps({"devices": devices})
            except Exception as exc:
                payload = json.dumps({"error": str(exc), "devices": []})
            self.droidScanFinished.emit(payload)

        threading.Thread(target=worker, daemon=True,
                         name="droidcam-discovery").start()

    @Slot(str)
    def _apply_droid_scan(self, payload):
        result = json.loads(payload)
        self._droid_cams = result["devices"]
        self._droid_scan_active = False
        if result.get("error"):
            self._droid_scan_status = "Scan failed: " + result["error"]
        elif self._droid_cams:
            count = len(self._droid_cams)
            self._droid_scan_status = f"Found {count} possible DroidCam device(s)"
        else:
            self._droid_scan_status = "No DroidCam devices found. You can still enter a URL."
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
        self._overlay = profile_id
        profile = PROFILES[profile_id]
        self.latest.set_profile(profile)
        if self._obs is not None:
            self._obs.set_profile(profile)
        self.overlayChanged.emit()

    @Slot(bool, bool, bool, int, float, str, str, str, str)
    def applyOverlayStyle(self, show_objects, show_faces, show_gestures,
                          line_width, font_scale, object_hex, known_hex,
                          unknown_hex, gesture_hex):
        try:
            profile = OverlayProfile(
                id="custom",
                show_objects=show_objects,
                show_faces=show_faces,
                show_gestures=show_gestures,
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
    args = parser.parse_args(argv)

    app = QGuiApplication(sys.argv[:1])
    # Parent the backend to the application so C++ owns its lifetime. Without
    # an owner, Python could collect it while QML still held the raw pointer
    # behind the `vision` context property.
    backend = VisionBackend(preview_port=args.preview_port,
                            initialize_models=not args.smoke_test,
                            load_saved_rules=not args.smoke_test,
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
