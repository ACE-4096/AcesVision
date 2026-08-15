import errno
import os
import tempfile
import threading
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import camera
import gesture_catalog
from acesvision.contracts import SceneFrame, SourceSpec
from acesvision.events import GestureEventOutput
from acesvision.discovery import (
    WebcamDevice,
    _stable_v4l_path,
    discover_webcams,
    local_scan_networks,
    preferred_webcam,
    scan_droidcam,
)
from acesvision.outputs import CallbackOutput
from acesvision.overlay import CLEAN, OverlayProfile, render
from acesvision.pipeline import VisionPipeline
from acesvision.perception import Detection, file_sha256
from acesvision.policy import (
    CONNECTORS,
    Rule,
    RuleEngine,
    RuleStore,
    validate_actor,
    validate_gesture,
)
from acesvision.processor import FaceGestureProcessor
from acesvision.sources import open_source

Face = namedtuple("Face", "x y w h name conf known")
Gesture = namedtuple("Gesture", "name score x y w h")


class FakeCapture:
    def __init__(self, reads, opened=True):
        self.reads = list(reads)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if self.reads:
            return self.reads.pop(0)
        return False, None

    def release(self):
        self.released = True

    def set(self, *_):
        return True


class SourceTests(unittest.TestCase):
    def test_webcam_index(self):
        source = SourceSpec.from_mapping({
            "id": "monitor", "name": "Monitor", "type": "webcam", "index": 9,
        })
        self.assertEqual((source.kind, source.index), ("webcam", 9))

    def test_droidcam_host_uses_standard_url(self):
        source = SourceSpec.from_mapping({
            "id": "phone", "type": "droidcam", "host": "192.168.1.20",
        })
        self.assertEqual(source.url, "http://192.168.1.20:4747/video")
        self.assertFalse(source.trusted_device)

    def test_safe_label_redacts_credentials_and_query(self):
        source = SourceSpec.from_mapping({
            "type": "droidcam",
            "url": "http://user:secret@phone.local:4747/video?token=hidden",
        })
        label = source.safe_label()
        self.assertNotIn("secret", label)
        self.assertNotIn("token", label)
        self.assertIn("phone.local:4747/video", label)

    def test_failed_network_capture_is_released(self):
        capture = FakeCapture([], opened=False)
        factory = Mock(return_value=capture)
        source = SourceSpec.from_mapping({
            "type": "droidcam", "url": "http://phone:4747/video",
        })
        self.assertIsNone(open_source(source, capture_factory=factory))
        self.assertTrue(capture.released)


class DiscoveryTests(unittest.TestCase):
    def test_webcam_inventory_hides_metadata_and_labels_ir(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, interface_index, name in (
                (5, 0, "USB FHD Webcam"),
                (7, 1, "USB FHD Webcam"),
                (8, 0, "USB FHD Webcam: IR Camer"),
            ):
                node = root / f"video{number}"
                node.mkdir()
                (node / "index").write_text(str(interface_index))
                (node / "name").write_text(name)
            devices = discover_webcams(root)
        self.assertEqual([device.index for device in devices], [5, 8])
        self.assertEqual([device.kind for device in devices], ["colour", "infrared"])
        self.assertIn("Camera 5", devices[0].label)
        self.assertIn("/dev/video5", devices[0].label)

    def test_preferred_webcam_chooses_colour_not_ir_or_virtual(self):
        devices = [
            WebcamDevice(3, "OBS", "/dev/video3", "virtual", "OBS"),
            WebcamDevice(8, "IR", "/dev/video8", "infrared", "IR"),
            WebcamDevice(5, "Webcam", "/dev/video5", "colour", "Webcam"),
        ]
        self.assertEqual(preferred_webcam(devices).index, 5)

    def test_stable_v4l_path_survives_transient_index_selection(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            devices = root / "dev"
            links = root / "by-path"
            devices.mkdir()
            links.mkdir()
            node = devices / "video8"
            node.touch()
            link = links / "usb-port-video-index0"
            link.symlink_to(node)
            selected = _stable_v4l_path(node, roots=(links,))
        self.assertEqual(selected, str(link))

    def test_scan_networks_only_include_private_non_loopback_addresses(self):
        networks = local_scan_networks(["192.168.68.21", "127.0.0.1", "8.8.8.8"])
        self.assertEqual([str(network) for network in networks], ["192.168.68.0/24"])

    def test_droidcam_scan_is_bounded_and_returns_standard_url(self):
        class Connection:
            def close(self):
                pass

        attempts = []

        def connector(address, timeout):
            attempts.append((address, timeout))
            if address[0] == "192.168.68.2":
                return Connection()
            raise ConnectionRefusedError

        found = scan_droidcam(["192.168.68.0/30"], connector=connector,
                              max_workers=2)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(found[0].url, "http://192.168.68.2:4747/video")

    def test_droidcam_scan_refuses_broad_subnets(self):
        connector = Mock()
        self.assertEqual(scan_droidcam(["192.168.0.0/16"], connector=connector), [])
        connector.assert_not_called()


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.zeros((100, 120, 3), dtype=np.uint8)
        self.source = SourceSpec.from_mapping({"type": "webcam"})
        self.scene = SceneFrame(
            self.source, 1, 1.0, self.raw,
            faces=[Face(10, 20, 30, 40, "Toby", 0.2, True)],
            gestures=[Gesture("Victory", 0.9, 60, 20, 20, 30)],
        )

    def test_clean_profile_returns_unannotated_copy(self):
        output = render(self.scene, CLEAN)
        self.assertTrue(np.array_equal(output, self.raw))
        self.assertIsNot(output, self.raw)

    def test_profile_colours_are_output_specific(self):
        red = OverlayProfile(known_colour=(0, 0, 255), show_gestures=False)
        green = OverlayProfile(known_colour=(0, 255, 0), show_gestures=False)
        red_frame = render(self.scene, red)
        green_frame = render(self.scene, green)
        self.assertFalse(np.array_equal(red_frame, green_frame))
        self.assertTrue(np.array_equal(self.raw, np.zeros_like(self.raw)))


class GestureEventTests(unittest.TestCase):
    def test_hold_emits_typed_non_authorized_event(self):
        events = []
        clock = Mock(return_value=10.0)
        output = GestureEventOutput(events.append, hold_frames=2, clock=clock)
        output.set_enabled(True)
        source = SourceSpec.from_mapping({"id": "monitor", "type": "webcam"})
        face = Face(0, 0, 10, 10, "Toby", 0.2, True)
        gesture = Gesture("Victory", 0.95, 0, 0, 10, 10)
        for sequence in range(2):
            output.publish(SceneFrame(source, sequence, float(sequence), np.zeros((2, 2, 3)),
                                      faces=[face], gestures=[gesture]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["actor"], "Toby")
        self.assertEqual(events[0]["gesture"], "Victory")
        self.assertFalse(events[0]["security_authorized"])
        self.assertEqual(events[0]["liveness_state"], "not_evaluated")

    def test_ambiguous_actor_is_unknown(self):
        events = []
        output = GestureEventOutput(events.append, hold_frames=1,
                                    clock=Mock(return_value=10.0))
        output.set_enabled(True)
        source = SourceSpec.from_mapping({"type": "webcam"})
        faces = [Face(0, 0, 5, 5, "A", 0.2, True),
                 Face(6, 0, 5, 5, "B", 0.2, True)]
        gesture = Gesture("Open_Palm", 0.9, 0, 0, 5, 5)
        output.publish(SceneFrame(source, 0, 0.0, np.zeros((2, 2, 3)),
                                  faces=faces, gestures=[gesture]))
        self.assertIsNone(events[0]["actor"])


class AsyncProcessorTests(unittest.TestCase):
    def test_capture_returns_without_waiting_for_inference(self):
        release = threading.Event()

        class ObjectDetector:
            def detect(self, _frame):
                release.wait(1.0)
                return ([Detection(0, 0, 10, 10, "person", 0.9, 4)],
                        {"inference": 4.0}, "test:yolo")

            def close(self):
                release.set()

        face = Face(1, 2, 3, 4, "Toby", 0.2, True)
        gesture = Gesture("Victory", 0.9, 1, 1, 3, 3)
        processor = FaceGestureProcessor(
            face_detector=Mock(return_value=[face]),
            gesture_detector=Mock(detect=Mock(return_value=[gesture])),
            object_detector=ObjectDetector(),
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        started = time.monotonic()
        initial = processor(frame, source, 0, time.monotonic())
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.1)
        self.assertEqual(initial.metadata["inference_status"], "warming_up")
        release.set()
        deadline = time.monotonic() + 1.0
        scene = initial
        while time.monotonic() < deadline:
            scene = processor(frame, source, 1, time.monotonic())
            if scene.objects:
                break
            time.sleep(0.01)
        processor.close()
        self.assertEqual(scene.objects[0].track_id, 4)
        self.assertEqual(scene.faces[0].x, 1)
        self.assertEqual(scene.metadata["object_model"], "test:yolo")

    def test_face_inference_is_limited_to_person_crops(self):
        class ObjectDetector:
            def detect(self, _frame):
                return ([Detection(10, 20, 30, 40, "person", 0.9, 1),
                         Detection(0, 0, 5, 5, "cat", 0.8, 2)], {}, "fake")

            def close(self):
                pass

        face_detector = Mock(return_value=[Face(2, 3, 4, 5, None, 0.8, False)])
        processor = FaceGestureProcessor(
            face_detector=face_detector,
            gesture_detector=Mock(detect=Mock(return_value=[])),
            object_detector=ObjectDetector(),
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processor(frame, source, 0, time.monotonic())
        deadline = time.monotonic() + 1.0
        scene = None
        while time.monotonic() < deadline:
            scene = processor(frame, source, 1, time.monotonic())
            if scene.faces:
                break
            time.sleep(0.01)
        processor.close()
        self.assertEqual(face_detector.call_args.args[0].shape[:2], (40, 30))
        self.assertEqual((scene.faces[0].x, scene.faces[0].y), (12, 23))

    def test_object_model_can_switch_without_restarting_capture(self):
        created = []

        class ObjectDetector:
            def __init__(self, model):
                self.model = model
                self.closed = False
                created.append(self)

            def detect(self, _frame):
                return [], {}, self.model

            def close(self):
                self.closed = True

        first = ObjectDetector("first")
        processor = FaceGestureProcessor(
            face_detector=Mock(return_value=[]),
            gesture_detector=Mock(detect=Mock(return_value=[])),
            object_detector=first,
            object_detector_factory=ObjectDetector,
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        processor(frame, source, 0, time.monotonic())
        processor.set_object_model("second")
        deadline = time.monotonic() + 1.0
        scene = None
        while time.monotonic() < deadline:
            scene = processor(frame, source, 1, time.monotonic())
            if scene.metadata.get("object_model") == "second":
                break
            time.sleep(0.01)
        processor.close()
        self.assertTrue(first.closed)
        self.assertEqual(scene.metadata["object_model"], "second")
        self.assertEqual(len(created), 2)

    def test_model_checksum_helper(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"acesvision")
            self.assertEqual(
                file_sha256(path),
                "2bba274e7725caa167d59af051764d1ebf98a7294323d6c5efe4fd8467894252",
            )

    def test_face_results_are_cached_between_scheduled_refreshes(self):
        class ObjectDetector:
            def detect(self, _frame):
                return [Detection(0, 0, 10, 10, "person", 0.9, 7)], {}, "fake"

            def close(self):
                pass

        face_detector = Mock(return_value=[Face(1, 1, 3, 3, "Toby", 0.2, True)])
        processor = FaceGestureProcessor(
            face_detector=face_detector,
            gesture_detector=Mock(detect=Mock(return_value=[])),
            object_detector=ObjectDetector(),
            face_hz=0.1,
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        deadline = time.monotonic() + 1.0
        sequence = 0
        scene = None
        while time.monotonic() < deadline:
            scene = processor(frame, source, sequence, time.monotonic())
            sequence += 1
            if scene.faces:
                break
            time.sleep(0.01)
        for _ in range(5):
            processor(frame, source, sequence, time.monotonic())
            sequence += 1
            time.sleep(0.01)
        processor.close()
        self.assertEqual(face_detector.call_count, 1)
        self.assertEqual(scene.faces[0].name, "Toby")


class PipelineTests(unittest.TestCase):
    def test_camera_controls_are_clamped_and_applied(self):
        cap = FakeCapture([])
        cap.set = Mock(return_value=True)
        controls = {
            "auto_exposure": False,
            "exposure": 333,
            "brightness": 12,
            "contrast": 40,
            "gamma": 180,
        }
        VisionPipeline._apply_camera_controls(cap, controls)
        values = [call.args[1] for call in cap.set.call_args_list]
        self.assertIn(333, values)
        self.assertIn(12, values)
        self.assertIn(40, values)
        self.assertIn(180, values)

    def test_nearly_black_uniform_frame_reports_privacy_warning(self):
        frame = np.full((20, 20, 3), 15, dtype=np.uint8)
        quality = VisionPipeline._frame_quality(frame)
        self.assertIn("privacy", quality["image_warning"])
        self.assertLess(quality["image_std"], 1)

    def test_black_manual_frame_recommends_auto_exposure(self):
        frame = np.full((20, 20, 3), 2, dtype=np.uint8)
        quality = VisionPipeline._frame_quality(frame, {
            "auto_exposure": False, "brightness": 0,
        })
        self.assertIn("Automatic exposure", quality["image_warning"])

    def test_minimum_brightness_recommends_reset(self):
        frame = np.full((20, 20, 3), 2, dtype=np.uint8)
        quality = VisionPipeline._frame_quality(frame, {
            "auto_exposure": False, "brightness": -64,
        })
        self.assertIn("Reset", quality["image_warning"])

    def test_reconnects_and_fans_out(self):
        first = FakeCapture([(False, None)])
        second = FakeCapture([(True, "raw")])
        captures = [first, second]
        delivered = []
        complete = threading.Event()
        source = SourceSpec.from_mapping({"type": "webcam", "index": 9})

        def opener(_):
            return captures.pop(0) if captures else None

        def processor(frame, src, sequence, captured_at):
            return SceneFrame(src, sequence, captured_at, f"processed:{frame}")

        pipeline = VisionPipeline(
            source,
            processor,
            [CallbackOutput(lambda scene: delivered.append(("a", scene.raw))),
             CallbackOutput(lambda scene: (delivered.append(("b", scene.raw)),
                                           complete.set()))],
            opener=opener,
            retry_min_s=0.001,
            retry_max_s=0.002,
        )
        pipeline.start()
        self.assertTrue(complete.wait(1.0))
        pipeline.stop()
        pipeline.join(1.0)

        self.assertCountEqual(delivered, [
            ("a", "processed:raw"), ("b", "processed:raw"),
        ])
        self.assertTrue(first.released)
        self.assertFalse(pipeline.is_alive())

    def test_switches_from_webcam_to_droidcam_without_restart(self):
        webcam = SourceSpec.from_mapping({"id": "webcam", "type": "webcam"})
        phone = SourceSpec.from_mapping({
            "id": "phone", "type": "droidcam", "url": "http://phone:4747/video",
        })
        webcam_capture = FakeCapture([(True, "webcam-frame")])
        phone_capture = FakeCapture([(True, "phone-frame")])
        opened = []
        delivered = threading.Event()

        def opener(source):
            opened.append(source.id)
            return webcam_capture if source.id == "webcam" else phone_capture

        def processor(frame, source, sequence, captured_at):
            scene = SceneFrame(source, sequence, captured_at, frame)
            if source.id == "webcam":
                pipeline.switch_source(phone)
            return scene

        def receive(scene):
            if scene.source.id == "phone":
                delivered.set()

        pipeline = VisionPipeline(
            webcam, processor, [CallbackOutput(receive)], opener=opener,
            retry_min_s=0.001, retry_max_s=0.002,
        )
        pipeline.start()
        self.assertTrue(delivered.wait(1.0))
        pipeline.stop()
        pipeline.join(1.0)

        self.assertEqual(opened[:2], ["webcam", "phone"])
        self.assertTrue(webcam_capture.released)


class PolicyTests(unittest.TestCase):
    def test_convenience_action_is_dry_run(self):
        rule = Rule.create("Victory", "acergb", "next_theme")
        decision = RuleEngine([rule]).evaluate({
            "gesture": "Victory", "actor": None, "source": "webcam",
        })[0]
        self.assertEqual(decision.outcome, "dry_run")

    def test_personal_action_requires_associated_actor(self):
        rule = Rule.create("Open_Palm", "kde", "next_desktop")
        decision = RuleEngine([rule]).evaluate({
            "gesture": "Open_Palm", "actor": None, "source": "webcam",
        })[0]
        self.assertEqual(decision.outcome, "blocked")
        self.assertIn("actor", decision.reason)

    def test_unknown_connector_and_shell_are_rejected(self):
        with self.assertRaises(ValueError):
            Rule.create("Victory", "shell", "run")

    def test_rule_store_round_trip_is_versioned_and_dry_run(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            rule = Rule.create("Victory", "mpris", "next", actor="Toby")
            store = RuleStore(path)
            store.save([rule])
            loaded = store.load()
            self.assertEqual(loaded, [rule])
            text = path.read_text()
            self.assertIn('"version": 1', text)
            self.assertIn('"dry_run": true', text)


class GuiStyleTests(unittest.TestCase):
    def test_hex_colour_converts_to_opencv_bgr(self):
        from acesvision.gui import VisionBackend

        self.assertEqual(VisionBackend._hex_to_bgr("#12abef"), (239, 171, 18))

    def test_invalid_hex_colour_is_rejected(self):
        from acesvision.gui import VisionBackend

        with self.assertRaises(ValueError):
            VisionBackend._hex_to_bgr("purple")

    def test_monitor_webcam_manual_exposure_is_guarded(self):
        from acesvision.gui import VisionBackend

        backend = VisionBackend(initialize_models=False, load_saved_rules=False)
        backend.setCameraTuning(False, 251, 0, 0, 100)
        self.assertTrue(backend._auto_exposure)
        self.assertTrue(backend.pipeline._camera_controls["auto_exposure"])


Landmark = namedtuple("Landmark", "x y")


def hand_landmarks(extended=("middle",), wrist=(0.5, 0.9)):
    """21 landmarks with the named fingers extended away from the wrist.

    Distances are measured from the wrist, so this mirrors the orientation-
    agnostic test in gesture_catalog._extended.
    """
    points = [Landmark(*wrist)] * 21
    points = list(points)
    for finger, (tip, pip) in gesture_catalog._FINGERS.items():
        pip_distance = 0.10
        tip_distance = 0.20 if finger in extended else 0.05
        points[pip] = Landmark(wrist[0], wrist[1] - pip_distance)
        points[tip] = Landmark(wrist[0], wrist[1] - tip_distance)
    return points


def pointing_hand(tip, wrist=(0.5, 0.95), extended="index"):
    """21 landmarks for a one-finger hand with that fingertip at ``tip``.

    pip and mcp sit on the wrist->tip line at 1/2 and 1/4 of the way, so the
    named finger reads as extended (tip farther from the wrist than the pip)
    and, whenever ``tip`` is above the wrist, as pointing up (tip above pip
    above mcp). Every other finger is curled in toward the wrist.

    Coordinates are MediaPipe-normalised 0..1, like the real landmarks.
    """
    def along(fraction):
        return Landmark(wrist[0] + (tip[0] - wrist[0]) * fraction,
                        wrist[1] + (tip[1] - wrist[1]) * fraction)

    points = [Landmark(*wrist) for _ in range(21)]
    for finger, (tip_index, pip_index) in gesture_catalog._FINGERS.items():
        if finger == extended:
            points[tip_index] = Landmark(*tip)
            points[pip_index] = along(0.5)
        else:
            points[pip_index] = along(0.20)
            points[tip_index] = along(0.10)   # curled: nearer the wrist
    points[gesture_catalog._INDEX_MCP] = along(0.25)
    return points


def open_hand(tip, wrist=(0.5, 0.95)):
    """All four fingers extended toward ``tip`` — an open palm, not a shush."""
    def along(fraction):
        return Landmark(wrist[0] + (tip[0] - wrist[0]) * fraction,
                        wrist[1] + (tip[1] - wrist[1]) * fraction)

    points = [Landmark(*wrist) for _ in range(21)]
    for tip_index, pip_index in gesture_catalog._FINGERS.values():
        points[tip_index] = Landmark(*tip)
        points[pip_index] = along(0.5)
    points[gesture_catalog._INDEX_MCP] = along(0.25)
    return points


class FakeDevice:
    """Stand-in for discovery.WebcamDevice with only the fields camera.py uses."""

    def __init__(self, index, kind, path=None, name="Fake Cam"):
        self.index = index
        self.kind = kind
        self.path = path or f"/dev/video{index}"
        self.name = name


class GestureCatalogTests(unittest.TestCase):
    def test_vocabulary_carries_the_ported_middle_finger(self):
        self.assertIn("Middle_Finger", gesture_catalog.GESTURE_IDS)
        spec = gesture_catalog.gesture_by_id("Middle_Finger")
        self.assertFalse(spec.builtin)   # landmark-derived, not a model label
        # The seven MediaPipe built-ins are all present and marked as built-in.
        builtins = [s.id for s in gesture_catalog.GESTURES if s.builtin]
        self.assertEqual(len(builtins), 7)

    def test_loose_names_normalise_onto_the_emitted_spelling(self):
        # The exact live-rules bug: "open_palm" never matched "Open_Palm".
        self.assertEqual(gesture_catalog.normalise_gesture("open_palm"), "Open_Palm")
        self.assertEqual(gesture_catalog.normalise_gesture("middle finger"),
                         "Middle_Finger")
        self.assertEqual(gesture_catalog.normalise_gesture("ILOVEYOU"), "ILoveYou")

    def test_unknown_gesture_is_named_and_rejected(self):
        self.assertIsNone(gesture_catalog.normalise_gesture("wave"))
        self.assertFalse(gesture_catalog.is_known_gesture("wave"))
        with self.assertRaises(ValueError) as caught:
            gesture_catalog.require_gesture("wave")
        self.assertIn("wave", str(caught.exception))
        self.assertIn("Open_Palm", str(caught.exception))   # lists the vocabulary

    def test_action_catalog_matches_acergb(self):
        self.assertEqual(len(gesture_catalog.ACTION_CATALOG), 16)
        for action_id in ("lights_next_theme", "device_gradient", "media_playpause",
                          "volume_mute", "custom"):
            self.assertEqual(gesture_catalog.require_action(action_id), action_id)
        with self.assertRaises(ValueError):
            gesture_catalog.require_action("launch_missiles")

    def test_connector_bindings_all_resolve_in_the_policy_table(self):
        # Guards the two vocabularies against drifting apart again.
        for action_id, (connector, action) in \
                gesture_catalog.CONNECTOR_BINDINGS.items():
            self.assertIn(action_id, gesture_catalog.ACTION_IDS)
            self.assertIn(connector, CONNECTORS, action_id)
            self.assertIn(action, CONNECTORS[connector], action_id)

    def test_middle_finger_pose_needs_one_finger_up_and_three_curled(self):
        self.assertTrue(gesture_catalog.is_middle_finger(hand_landmarks(["middle"])))
        self.assertFalse(gesture_catalog.is_middle_finger(
            hand_landmarks(["index", "middle", "ring", "pinky"])))   # open palm
        self.assertFalse(gesture_catalog.is_middle_finger(hand_landmarks(["index"])))
        self.assertFalse(gesture_catalog.is_middle_finger([]))
        self.assertFalse(gesture_catalog.is_middle_finger(None))

    def test_classify_hands_prefers_the_custom_pose_over_the_model_label(self):
        from gestures import classify_hands

        Category = namedtuple("Category", "category_name score")
        landmarks = [hand_landmarks(["middle"])]
        # The model calls it Pointing_Up; the landmark pose must win.
        rows = classify_hands(landmarks, [[Category("Pointing_Up", 0.9)]],
                              100, 100, 0.5)
        self.assertEqual([row.name for row in rows], ["Middle_Finger"])
        self.assertEqual(rows[0].score, 1.0)

        # Still detected when the model returned no category for that hand.
        rows = classify_hands(landmarks, [], 100, 100, 0.5)
        self.assertEqual([row.name for row in rows], ["Middle_Finger"])

        # Built-in labels still pass through, and low scores are still dropped.
        open_palm = [hand_landmarks(["index", "middle", "ring", "pinky"])]
        rows = classify_hands(open_palm, [[Category("Open_Palm", 0.9)]],
                              100, 100, 0.5)
        self.assertEqual([row.name for row in rows], ["Open_Palm"])
        rows = classify_hands(open_palm, [[Category("Open_Palm", 0.1)]],
                              100, 100, 0.5)
        self.assertEqual(rows, [])


# One 1000x1000 frame and one face box, shared by every shush test so the
# arithmetic below is checkable by hand. The face spans x 400..600, y 300..560,
# so its mouth point is (500, 300 + 0.72*260) = (500, 487.2) in pixels. Because
# the frame is 1000 wide and tall, a normalised landmark of 0.487 is pixel 487.
FRAME_W = FRAME_H = 1000
FACE_BOX = Face(400, 300, 200, 260, "Toby", 0.2, True)

AT_LIPS = (0.500, 0.490)          # dx 0.00, dy  +0.01  -> inside
AT_CEILING = (0.500, 0.150)       # dx 0.00, dy  -1.30  -> the plain point up
NEAR_EDGE_ABOVE = (0.500, 0.360)  # dx 0.00, dy  -0.49  -> just inside
PAST_EDGE_ABOVE = (0.500, 0.350)  # dx 0.00, dy  -0.53  -> just outside
NEAR_EDGE_SIDE = (0.405, 0.487)   # dx -0.48, dy  0.00  -> just inside
PAST_EDGE_SIDE = (0.390, 0.487)   # dx -0.55, dy  0.00  -> just outside


class ShushGestureTests(unittest.TestCase):
    """The founder's shush: index finger up, held to the lips.

    MediaPipe already labels that hand Pointing_Up, and Pointing_Up runs
    `ledctl next-theme` in automations.json. Everything here exists to keep one
    shush from also cycling the lighting themes.
    """

    def test_shush_is_registered_in_the_shared_vocabulary(self):
        self.assertIn("Shush", gesture_catalog.GESTURE_IDS)
        spec = gesture_catalog.gesture_by_id("Shush")
        self.assertFalse(spec.builtin)   # landmark-derived, not a model label
        self.assertEqual(len(gesture_catalog.GESTURES), 9)
        # The seven MediaPipe built-ins are untouched by the addition.
        self.assertEqual(
            len([s for s in gesture_catalog.GESTURES if s.builtin]), 7)
        # The QML combo box and the applet read this same tuple.
        self.assertEqual(gesture_catalog.catalog_json()["gestures"][-1],
                         {"id": "Shush", "label": "Shush (finger to lips)",
                          "builtin": False})

    def test_the_previously_unfireable_rule_name_now_resolves(self):
        # rules.json carried gesture "shush" against no such gesture. It is a
        # real name now, and the loose spelling normalises onto the emitted one.
        self.assertEqual(gesture_catalog.normalise_gesture("shush"), "Shush")
        self.assertEqual(gesture_catalog.normalise_gesture("SHUSH"), "Shush")
        self.assertTrue(gesture_catalog.is_known_gesture("shush"))
        self.assertEqual(validate_gesture(" shush "), "Shush")

    def test_a_shush_rule_binds_to_the_mute_action(self):
        rule = Rule.create("shush", "pipewire", "mute", actor="*")
        self.assertEqual(rule.gesture, "Shush")
        self.assertEqual((rule.connector, rule.action), ("pipewire", "mute"))
        # Muting is a PipeWire action, not an AceRGB one.
        self.assertEqual(gesture_catalog.CONNECTOR_BINDINGS["volume_mute"],
                         ("pipewire", "mute"))

    def test_finger_at_the_lips_is_a_shush(self):
        hand = pointing_hand(AT_LIPS)
        self.assertTrue(gesture_catalog.is_shush(hand, [FACE_BOX],
                                                 FRAME_W, FRAME_H))

    def test_the_same_hand_pointed_at_the_ceiling_is_not(self):
        # Identical pose, fingertip above the head. This is the whole reason
        # the face box is a term in the geometry.
        hand = pointing_hand(AT_CEILING)
        self.assertFalse(gesture_catalog.is_shush(hand, [FACE_BOX],
                                                  FRAME_W, FRAME_H))

    def test_no_face_means_no_shush(self):
        hand = pointing_hand(AT_LIPS)
        for faces in ([], None):
            self.assertFalse(gesture_catalog.is_shush(hand, faces,
                                                      FRAME_W, FRAME_H))

    def test_proximity_boundary_on_both_sides(self):
        # Nearest-miss pairs straddling the 0.5 normalised radius.
        for tip in (NEAR_EDGE_ABOVE, NEAR_EDGE_SIDE):
            self.assertTrue(
                gesture_catalog.is_shush(pointing_hand(tip), [FACE_BOX],
                                         FRAME_W, FRAME_H), tip)
        for tip in (PAST_EDGE_ABOVE, PAST_EDGE_SIDE):
            self.assertFalse(
                gesture_catalog.is_shush(pointing_hand(tip), [FACE_BOX],
                                         FRAME_W, FRAME_H), tip)

    def test_proximity_scales_with_the_face_box(self):
        # The same fingertip, a person standing twice as far away: the face box
        # halves, so what was at the lips is now well outside the mouth region.
        tip = (0.500, 0.400)
        near = Face(400, 300, 200, 260, "Toby", 0.2, True)
        far = Face(475, 400, 50, 65, "Toby", 0.2, True)
        self.assertTrue(gesture_catalog.is_shush(pointing_hand(tip), [near],
                                                 FRAME_W, FRAME_H))
        self.assertFalse(gesture_catalog.is_shush(pointing_hand(tip), [far],
                                                  FRAME_W, FRAME_H))

    def test_any_face_in_frame_can_anchor_the_shush(self):
        other = Face(10, 10, 100, 130, None, 0.9, False)
        self.assertTrue(gesture_catalog.is_shush(pointing_hand(AT_LIPS),
                                                 [other, FACE_BOX],
                                                 FRAME_W, FRAME_H))

    def test_degenerate_face_boxes_are_ignored_not_divided_by(self):
        empty = Face(500, 480, 0, 0, None, 0.9, False)
        self.assertFalse(gesture_catalog.is_shush(pointing_hand(AT_LIPS),
                                                  [empty], FRAME_W, FRAME_H))

    def test_extended_is_not_the_same_as_pointing_up(self):
        # Fingertip exactly at the lips, but the hand comes from above so the
        # finger points down. _extended is orientation-agnostic and would
        # accept this; the shush must not.
        hand = pointing_hand(AT_LIPS, wrist=(0.5, 0.20))
        self.assertFalse(gesture_catalog.is_shush(hand, [FACE_BOX],
                                                  FRAME_W, FRAME_H))

    def test_other_hand_shapes_at_the_lips_are_not_a_shush(self):
        self.assertFalse(gesture_catalog.is_shush(open_hand(AT_LIPS),
                                                  [FACE_BOX], FRAME_W, FRAME_H))
        self.assertFalse(
            gesture_catalog.is_shush(pointing_hand(AT_LIPS, extended="middle"),
                                     [FACE_BOX], FRAME_W, FRAME_H))
        self.assertFalse(gesture_catalog.is_shush([], [FACE_BOX],
                                                  FRAME_W, FRAME_H))
        self.assertFalse(gesture_catalog.is_shush(None, [FACE_BOX],
                                                  FRAME_W, FRAME_H))

    def test_the_real_engine_face_type_drives_the_geometry(self):
        # The stand-in above mirrors engine.Face; prove the real one binds, so
        # a field rename in engine.py cannot pass this suite silently.
        from engine import Face as EngineFace

        face = EngineFace(*FACE_BOX)
        self.assertTrue(gesture_catalog.is_shush(pointing_hand(AT_LIPS),
                                                 [face], FRAME_W, FRAME_H))

    def test_shush_takes_strict_precedence_over_pointing_up(self):
        """The critical case. A shush must emit Shush and must NOT emit
        Pointing_Up, or every shush also fires `ledctl next-theme`."""
        from gestures import classify_hands

        Category = namedtuple("Category", "category_name score")
        model_says_pointing_up = [[Category("Pointing_Up", 0.95)]]
        hands = [pointing_hand(AT_LIPS)]

        rows = classify_hands(hands, model_says_pointing_up, FRAME_W, FRAME_H,
                              0.5, faces=[FACE_BOX])
        names = [row.name for row in rows]
        self.assertEqual(names, ["Shush"])
        self.assertNotIn("Pointing_Up", names)
        self.assertEqual(rows[0].score, 1.0)

    def test_pointing_up_still_reaches_the_theme_binding_away_from_the_face(self):
        from gestures import classify_hands

        Category = namedtuple("Category", "category_name score")
        rows = classify_hands([pointing_hand(AT_CEILING)],
                              [[Category("Pointing_Up", 0.95)]],
                              FRAME_W, FRAME_H, 0.5, faces=[FACE_BOX])
        self.assertEqual([row.name for row in rows], ["Pointing_Up"])

        # And with no faces at all — the detector must not start swallowing
        # Pointing_Up just because face detection went quiet.
        rows = classify_hands([pointing_hand(AT_LIPS)],
                              [[Category("Pointing_Up", 0.95)]],
                              FRAME_W, FRAME_H, 0.5, faces=None)
        self.assertEqual([row.name for row in rows], ["Pointing_Up"])

    def test_shush_is_detected_even_when_the_model_returned_no_category(self):
        from gestures import classify_hands

        rows = classify_hands([pointing_hand(AT_LIPS)], [], FRAME_W, FRAME_H,
                              0.5, faces=[FACE_BOX])
        self.assertEqual([row.name for row in rows], ["Shush"])

    def test_the_processor_hands_face_boxes_to_the_gesture_detector(self):
        """Without this wiring the geometry above can never fire in the app."""
        class ObjectDetector:
            def detect(self, _frame):
                return ([Detection(0, 0, 100, 100, "person", 0.9, 1)], {}, "fake")

            def close(self):
                pass

        face = Face(10, 10, 20, 26, "Toby", 0.2, True)
        gesture_detector = Mock(detect=Mock(return_value=[]))
        processor = FaceGestureProcessor(
            face_detector=Mock(return_value=[face]),
            gesture_detector=gesture_detector,
            object_detector=ObjectDetector(),
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processor(frame, source, 0, time.monotonic())
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            processor(frame, source, 1, time.monotonic())
            if gesture_detector.detect.call_args is not None:
                break
            time.sleep(0.01)
        processor.close()
        passed = gesture_detector.detect.call_args.kwargs["faces"]
        self.assertEqual([f.name for f in passed], ["Toby"])


class CameraDiscoveryTests(unittest.TestCase):
    """camera.py must own no device list of its own — discovery.py is the SSoT."""

    def test_no_hard_coded_candidate_list_survives(self):
        self.assertFalse(hasattr(camera, "CANDIDATES"))

    def test_candidates_come_from_discovery_colour_first_ir_last(self):
        devices = [
            FakeDevice(20, "virtual", name="OBS Virtual Camera"),
            FakeDevice(3, "infrared", name="IR Camera"),
            FakeDevice(1, "colour", name="UVC WebCam"),
            FakeDevice(5, "camera", name="Unclassified"),
        ]
        ordered = camera.candidate_devices(devices)
        self.assertEqual([d.path for d in ordered],
                         ["/dev/video1", "/dev/video5", "/dev/video3"])

    def test_infrared_node_is_reachable(self):
        # The old CANDIDATES list omitted video3 entirely, which made the IR
        # sensor unreachable and blocked the colour+IR liveness design.
        ordered = camera.candidate_devices([FakeDevice(3, "infrared")])
        self.assertEqual([d.path for d in ordered], ["/dev/video3"])

    def test_metadata_and_virtual_nodes_are_never_probed(self):
        # Metadata nodes are already filtered by discovery (interface index != 0);
        # camera.py additionally drops loopback/virtual outputs.
        probed = []

        def opener(device, manual_exposure=False):
            probed.append(device)
            return _fake_colour_capture(), _colour_frame()

        cap = camera.open_camera(
            devices=[FakeDevice(20, "virtual"), FakeDevice(1, "colour")],
            opener=opener,
            status=lambda device: camera.DEVICE_AVAILABLE,
        )
        self.assertIsNotNone(cap)
        self.assertEqual(probed, ["/dev/video1"])


class CameraStatusTests(unittest.TestCase):
    def test_status_distinguishes_missing_busy_and_available(self):
        def missing(path, flags):
            raise FileNotFoundError(path)

        def busy(path, flags):
            raise OSError(errno.EBUSY, "Device or resource busy")

        closed = []
        self.assertEqual(camera.device_status(9, opener=missing),
                         camera.DEVICE_MISSING)
        self.assertEqual(camera.device_status(1, opener=busy), camera.DEVICE_BUSY)
        self.assertEqual(
            camera.device_status(1, opener=lambda path, flags: 7,
                                 closer=closed.append),
            camera.DEVICE_AVAILABLE)
        self.assertEqual(closed, [7])   # the probe never leaks a descriptor

    def test_holders_are_read_from_proc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "video-node"
            device.write_text("")
            proc = root / "proc"
            for pid, name, target in ((4242, "obs", device),
                                      (4243, "bash", root / "other")):
                if not target.exists():
                    target.write_text("")
                fds = proc / str(pid) / "fd"
                fds.mkdir(parents=True)
                (proc / str(pid) / "comm").write_text(name + "\n")
                (fds / "3").symlink_to(target)
            holders = camera.device_holders(str(device), proc_root=str(proc))
        self.assertEqual(holders, [(4242, "obs")])

    def test_a_busy_device_is_not_reported_as_missing(self):
        # The whole point: "held by OBS" and "does not exist" need different fixes.
        with patch.object(camera, "device_status",
                          return_value=camera.DEVICE_AVAILABLE), \
             patch.object(camera, "device_holders", return_value=[(99, "obs")]):
            status, detail = camera.explain_failure("/dev/video1")
        self.assertEqual(status, camera.DEVICE_BUSY)
        self.assertIn("obs", detail)
        self.assertIn("/dev/video1", detail)


def _colour_frame():
    frame = np.zeros((8, 8, 3), dtype="uint8")
    frame[:, :, 2] = 200          # strong red channel -> chroma above CHROMA_MIN
    return frame


def _grey_frame():
    return np.full((8, 8, 3), 120, dtype="uint8")


def _fake_colour_capture():
    return FakeCapture([(True, "frame")])


class CameraOpenTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("FACE_ID_CAM", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["FACE_ID_CAM"] = self._saved

    def _opener(self, frames):
        """frames: {device -> frame or None}."""
        def opener(device, manual_exposure=False):
            frame = frames.get(device)
            if frame is None:
                return None, None
            return FakeCapture([(True, "frame")]), frame
        return opener

    def test_colour_is_preferred_over_infrared(self):
        cap = camera.open_camera(
            devices=[FakeDevice(1, "colour"), FakeDevice(3, "infrared")],
            opener=self._opener({"/dev/video1": _colour_frame(),
                                 "/dev/video3": _grey_frame()}),
            status=lambda device: camera.DEVICE_AVAILABLE)
        self.assertIsNotNone(cap)

    def test_busy_colour_falls_back_to_infrared_and_says_why(self):
        with patch("builtins.print") as printed:
            cap = camera.open_camera(
                devices=[FakeDevice(1, "colour"), FakeDevice(3, "infrared")],
                opener=self._opener({"/dev/video3": _grey_frame()}),
                status=lambda device: camera.DEVICE_AVAILABLE,
                explain=lambda device: (camera.DEVICE_BUSY,
                                        f"{device} is held by obs (pid 7)"))
        self.assertIsNotNone(cap)
        message = " ".join(str(call) for call in printed.call_args_list)
        self.assertIn("held by obs", message)

    def test_all_devices_busy_raises_busy_not_not_found(self):
        with self.assertRaises(camera.CameraBusyError) as caught:
            camera.open_camera(
                devices=[FakeDevice(1, "colour")],
                opener=self._opener({}),
                status=lambda device: camera.DEVICE_AVAILABLE,
                explain=lambda device: (camera.DEVICE_BUSY,
                                        f"{device} is held by obs (pid 7)"))
        self.assertIn("held by obs", str(caught.exception))
        self.assertNotIn("No working camera found", str(caught.exception))

    def test_no_devices_at_all_raises_not_found(self):
        with self.assertRaises(camera.CameraNotFoundError):
            camera.open_camera(devices=[], opener=self._opener({}),
                               status=lambda device: camera.DEVICE_AVAILABLE)

    def test_forced_busy_device_raises_instead_of_substituting_another(self):
        # An explicit choice that is merely contended must not be silently
        # swapped for a different camera.
        with self.assertRaises(camera.CameraBusyError):
            camera.open_camera(
                preferred="/dev/video1",
                devices=[FakeDevice(3, "infrared")],
                opener=self._opener({"/dev/video3": _grey_frame()}),
                status=lambda device: camera.DEVICE_AVAILABLE,
                explain=lambda device: (camera.DEVICE_BUSY,
                                        f"{device} is held by obs (pid 7)"))

    def test_forced_missing_device_falls_back_to_discovery(self):
        # Previously a stale path raised with no fallback and sources.py turned
        # that into None: total silent failure.
        cap = camera.open_camera(
            preferred="/dev/video9",
            devices=[FakeDevice(1, "colour")],
            opener=self._opener({"/dev/video1": _colour_frame()}),
            status=lambda device: (camera.DEVICE_MISSING
                                   if device == "/dev/video9"
                                   else camera.DEVICE_AVAILABLE))
        self.assertIsNotNone(cap)

    def test_face_id_cam_override_still_works(self):
        os.environ["FACE_ID_CAM"] = "3"
        opened = []

        def opener(device, manual_exposure=False):
            opened.append(device)
            return FakeCapture([(True, "frame")]), _grey_frame()

        cap = camera.open_camera(devices=[FakeDevice(1, "colour")], opener=opener,
                                 status=lambda device: camera.DEVICE_AVAILABLE)
        self.assertIsNotNone(cap)
        self.assertEqual(opened, [3])   # honoured exactly, no colour filtering

    def test_unparsable_override_is_ignored_not_fatal(self):
        os.environ["FACE_ID_CAM"] = "not-a-device"
        cap = camera.open_camera(
            devices=[FakeDevice(1, "colour")],
            opener=self._opener({"/dev/video1": _colour_frame()}),
            status=lambda device: camera.DEVICE_AVAILABLE)
        self.assertIsNotNone(cap)


class SourceFailureReportingTests(unittest.TestCase):
    def test_busy_camera_reason_is_reported_not_swallowed(self):
        source = SourceSpec.from_mapping({"id": "webcam", "type": "webcam"})
        reasons = []

        def raiser(device):
            raise camera.CameraBusyError("/dev/video1 is held by obs (pid 7)")

        self.assertIsNone(open_source(source, webcam_opener=raiser,
                                      on_error=reasons.append))
        self.assertEqual(len(reasons), 1)
        self.assertIn("held by obs", reasons[0])

    def test_pipeline_surfaces_the_specific_reason(self):
        source = SourceSpec.from_mapping({"id": "webcam", "type": "webcam"})

        def fake_open_source(spec, on_error=None):
            if on_error:
                on_error("/dev/video1 is held by obs (pid 7)")
            return None

        with patch("acesvision.pipeline.open_source", fake_open_source):
            pipeline = VisionPipeline(source, lambda *a: None, [],
                                      retry_min_s=0.001, retry_max_s=0.002)
            pipeline.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if "held by obs" in pipeline.state().last_error:
                    break
                time.sleep(0.01)
            pipeline.stop()
            pipeline.join(1.0)
        self.assertIn("held by obs", pipeline.state().last_error)


class RuleVocabularyTests(unittest.TestCase):
    def test_unknown_gesture_is_rejected_at_creation(self):
        # A rule naming a gesture nothing emits could never fire and never said
        # so. "finger_snap" is the standing example: a literal snap is too fast
        # and too subtle to detect from video (gestures.py module docstring).
        # "shush" used to sit here — it is a real pose now, see ShushGestureTests.
        with self.assertRaises(ValueError):
            Rule.create("finger_snap", "pipewire", "mute")

    def test_gesture_case_is_normalised_to_the_emitted_spelling(self):
        rule = Rule.create("open_palm", "mpris", "play_pause")
        self.assertEqual(rule.gesture, "Open_Palm")
        self.assertEqual(validate_gesture(" victory "), "Victory")

    def test_actor_is_validated_against_enrolled_identities(self):
        self.assertEqual(validate_actor("toby", actors=["Toby"]), "Toby")
        self.assertEqual(validate_actor("", actors=["Toby"]), "*")
        self.assertEqual(validate_actor("*", actors=["Toby"]), "*")
        with self.assertRaises(ValueError) as caught:
            validate_actor("mallory", actors=["Toby"])
        self.assertIn("Toby", str(caught.exception))

    def test_normalised_rule_actually_fires(self):
        # Red-then-green: the raw strings produce no decision, the normalised
        # rule produces one.
        event = {"gesture": "Open_Palm", "actor": "Toby", "source": "webcam"}
        unfireable = Rule(id="x", gesture="open_palm", connector="mpris",
                          action="play_pause", actor="toby")
        self.assertEqual(RuleEngine([unfireable]).evaluate(event), [])

        repaired = Rule.create("open_palm", "mpris", "play_pause", actor="toby")
        decisions = RuleEngine([repaired]).evaluate(event)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].outcome, "dry_run")

    def test_store_repairs_legacy_rules_and_quarantines_unfireable_ones(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps({"version": 1, "dry_run": True, "rules": [
                {"id": "a", "gesture": "finger_snap", "connector": "pipewire",
                 "action": "mute", "actor": "toby", "source": "*",
                 "enabled": True, "require_liveness": False,
                 "require_confirmation": False},
                {"id": "b", "gesture": "open_palm", "connector": "mpris",
                 "action": "play_pause", "actor": "toby", "source": "*",
                 "enabled": True, "require_liveness": False,
                 "require_confirmation": False},
            ]}))
            store = RuleStore(path)
            rules = store.load(strict=False, actors=["Toby"])
            self.assertEqual([r.id for r in rules], ["b"])
            self.assertEqual(rules[0].gesture, "Open_Palm")
            self.assertEqual(rules[0].actor, "Toby")
            self.assertEqual([raw["id"] for raw, _ in store.rejected], ["a"])
            with self.assertRaises(ValueError):
                store.load(strict=True, actors=["Toby"])


class HeadlessRunnerTests(unittest.TestCase):
    def test_gesture_events_are_enabled_by_default(self):
        from acesvision.__main__ import build_gesture_output, build_parser

        args = build_parser().parse_args([])
        output = build_gesture_output(args, callback=lambda event: None)
        self.assertTrue(output.enabled)   # used to be False, forever

    def test_no_events_flag_and_tuning_are_honoured(self):
        from acesvision.__main__ import build_gesture_output, build_parser

        args = build_parser().parse_args(["--no-events"])
        self.assertFalse(build_gesture_output(args).enabled)

        args = build_parser().parse_args(["--hold-frames", "3", "--cooldown-s", "0.5"])
        output = build_gesture_output(args, callback=lambda event: None)
        self.assertEqual((output.hold_frames, output.cooldown_s), (3, 0.5))

    def test_enabled_output_emits_without_an_external_toggle(self):
        events = []
        output = GestureEventOutput(events.append, hold_frames=1,
                                    clock=Mock(return_value=1.0), enabled=True)
        source = SourceSpec.from_mapping({"type": "webcam"})
        output.publish(SceneFrame(source, 0, 0.0, np.zeros((2, 2, 3)),
                                  gestures=[Gesture("Victory", 0.9, 0, 0, 5, 5)]))
        self.assertEqual(len(events), 1)

    def test_default_stays_disabled_for_the_gui_toggle(self):
        events = []
        output = GestureEventOutput(events.append, hold_frames=1,
                                    clock=Mock(return_value=1.0))
        source = SourceSpec.from_mapping({"type": "webcam"})
        output.publish(SceneFrame(source, 0, 0.0, np.zeros((2, 2, 3)),
                                  gestures=[Gesture("Victory", 0.9, 0, 0, 5, 5)]))
        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()
