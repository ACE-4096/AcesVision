import threading
import time
import unittest
from collections import namedtuple
from unittest.mock import Mock

import numpy as np

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
from acesvision.policy import CONNECTORS, Rule, RuleEngine, RuleStore
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
        self.assertIsNone(gesture_catalog.normalise_gesture("shush"))
        self.assertFalse(gesture_catalog.is_known_gesture("shush"))
        with self.assertRaises(ValueError) as caught:
            gesture_catalog.require_gesture("shush")
        self.assertIn("shush", str(caught.exception))
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

if __name__ == "__main__":
    unittest.main()
