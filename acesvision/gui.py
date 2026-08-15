"""Qt 6 and QML desktop shell for AcesVision."""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from .contracts import SceneFrame, SourceSpec
from .events import GestureEventOutput
from .discovery import discover_webcams, preferred_webcam, scan_droidcam
from .outputs import LatestFrameOutput, ObsVirtualCameraOutput
from .overlay import MINIMAL, PROFILES, OverlayProfile
from .pipeline import VisionPipeline
from .policy import CONNECTORS, Rule, RuleEngine, RuleStore
from .preview import PreviewServer
from .processor import FaceGestureProcessor
from .perception import file_sha256

QML_PATH = Path(__file__).parent / "qml" / "Main.qml"
MODEL_MANIFEST_PATH = Path(__file__).parent / "model_manifest.json"


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

    gestureFromWorker = Signal(str)
    droidScanFinished = Signal(str)

    def __init__(self, preview_port=8765, initialize_models=True,
                 load_saved_rules=True, parent=None):
        super().__init__(parent)
        self._status = "starting"
        discovered_webcams = discover_webcams()
        default_webcam = preferred_webcam(discovered_webcams)
        self._source = default_webcam.label if default_webcam else "Webcam (auto)"
        self._sequence = 0
        self._error = ""
        self._preview_tick = 0
        self._obs_enabled = False
        self._events_enabled = False
        self._overlay = "minimal"
        self._last_gesture = "No gesture events yet"
        self._last_decision = "No rule decisions yet"
        self._capture_fps = 0.0
        self._inference_fps = 0.0
        self._inference_ms = 0.0
        self._model_summary = "Models warming up"
        self._image_warning = ""
        self._image_mean = 0.0
        self._auto_exposure = True
        # This Sunplus monitor webcam advertises manual exposure, accepts the
        # control, then returns an almost uniform black frame. Keep the backend
        # guard even if an older QML client asks for manual mode.
        self._manual_exposure_supported = False
        self._exposure = 166
        self._brightness = 0
        self._contrast = 0
        self._gamma = 100
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text())
        repo_root = Path(__file__).resolve().parents[1]
        self._model_options = []
        self._model_paths = {}
        for model in manifest["models"]:
            path = repo_root / model["file"]
            if path.is_file() and file_sha256(path) == model["sha256"]:
                option = dict(model)
                option["path"] = str(path)
                self._model_options.append(option)
                self._model_paths[model["id"]] = path
        self.rule_store = RuleStore()
        try:
            self._rules = self.rule_store.load() if load_saved_rules else []
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self._rules = []
            self._last_decision = f"Saved rules not loaded: {exc}"
        self.rule_engine = RuleEngine(self._rules, dry_run=True)
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

    def _receive_gesture(self, event):
        decisions = self.rule_engine.evaluate(event)
        if decisions:
            decision = decisions[-1]
            self._last_decision = (
                f"{decision.outcome}: {decision.connector}.{decision.action}, "
                f"{decision.reason}"
            )
            self.decisionChanged.emit()
        self.gestureFromWorker.emit(json.dumps(event, sort_keys=True))

    @Slot(str)
    def _set_gesture(self, event_json):
        event = json.loads(event_json)
        actor = event.get("actor") or "unknown actor"
        self._last_gesture = f"{event['gesture']} by {actor} (dry-run)"
        self.gestureChanged.emit()

    @Slot()
    def _refresh(self):
        state = self.pipeline.state()
        if state.status != self._status:
            self._status = state.status
            self.statusChanged.emit()
        label = state.source.safe_label()
        if label != self._source:
            self._source = label
            self.sourceChanged.emit()
        if state.sequence != self._sequence:
            self._sequence = state.sequence
            self._preview_tick += 1
            self.sequenceChanged.emit()
            self.previewChanged.emit()
        if state.last_error != self._error:
            self._error = state.last_error
            self.errorChanged.emit()
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
        if (image_warning, round(image_mean, 1)) != (
                self._image_warning, round(self._image_mean, 1)):
            self._image_warning = image_warning
            self._image_mean = image_mean
            self.imageChanged.emit()

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
        return self._error

    @Property(str, notify=previewChanged)
    def previewSource(self):
        return f"{self.preview_url}?t={self._preview_tick}"

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
        } for rule in self._rules]

    @Property(str, notify=decisionChanged)
    def lastDecision(self):
        return self._last_decision

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

    @Property("QVariantList", constant=True)
    def modelOptions(self):
        return self._model_options

    @Property(str, notify=imageChanged)
    def imageWarning(self):
        return self._image_warning

    @Property(float, notify=imageChanged)
    def imageMean(self):
        return self._image_mean

    @Property(bool, constant=True)
    def manualExposureSupported(self):
        return self._manual_exposure_supported

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

    @Slot(str)
    def setObjectModel(self, model_id):
        path = self._model_paths.get(model_id)
        switch = getattr(self.processor, "set_object_model", None)
        if path is None or switch is None:
            self._error = f"Object model is not available: {model_id}"
            self.errorChanged.emit()
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

    @Property("QVariantList", constant=True)
    def connectorNames(self):
        return list(CONNECTORS)

    @Slot(str, result="QVariantList")
    def connectorActions(self, connector):
        return list(CONNECTORS.get(connector, {}))

    @Slot(str)
    def useWebcam(self, index_text):
        try:
            index = int(index_text) if index_text.strip() else None
        except ValueError:
            self._error = "Webcam index must be a number or blank for auto"
            self.errorChanged.emit()
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
            self._error = "DroidCam URL is required"
            self.errorChanged.emit()
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

    @Slot(str)
    def setOverlayProfile(self, profile_id):
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
            self._error = str(exc)
            self.errorChanged.emit()
            return
        PROFILES["custom"] = profile
        self._overlay = "custom"
        self.latest.set_profile(profile)
        if self._obs is not None:
            self._obs.set_profile(profile)
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
        try:
            rule = Rule.create(gesture.strip(), connector, action,
                               actor=actor.strip() or "*")
        except ValueError as exc:
            self._error = str(exc)
            self.errorChanged.emit()
            return
        if not rule.gesture:
            self._error = "Gesture is required"
            self.errorChanged.emit()
            return
        self._rules.append(rule)
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
            self._error = f"Rules were not saved: {exc}"
            self.errorChanged.emit()


def main(argv=None):
    parser = argparse.ArgumentParser(description="AcesVision desktop GUI")
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args(argv)

    app = QGuiApplication(sys.argv[:1])
    backend = VisionBackend(preview_port=args.preview_port,
                            initialize_models=not args.smoke_test,
                            load_saved_rules=not args.smoke_test)
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
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
