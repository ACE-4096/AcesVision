#!/usr/bin/env python3
"""Guided live-camera verification of the gesture vocabulary. OBSERVATION ONLY.

Everything gesture-related in this repo is currently proven against *synthetic*
landmarks (test_acesvision.py builds hands out of stub points). No gesture has
ever been observed coming off a real camera. This script is the instrument that
closes that gap: it walks the founder through every gesture in
``gesture_catalog.GESTURES`` on a timer, records what the real pipeline saw for
every frame, and writes a pasteable report.

    .venv/bin/python verify_gestures_live.py

Safety, non-negotiable and enforced by a test:
  * no rule is loaded, no connector is constructed, no action is dispatched.
    This module never imports acesvision.policy or acesvision.connectors and
    never reads ~/.config/acesvision/rules.json. The lights cannot move.
  * the only side effects are files under ./verification/ (gitignored — the
    annotated frames contain the founder's face and must never be committed).

The headline measurement is the Shush occlusion test. A shush is an index
finger held at the lips, which is the *same hand* MediaPipe labels
``Pointing_Up``; only the face box tells them apart (gesture_catalog.is_shush).
A finger across the lips can occlude the mouth, and if YuNet drops the face box
at that instant ``is_shush`` returns False and the frame degrades to
``Pointing_Up`` — which automations.json binds to ``ledctl next-theme``. A
failed shush would therefore cycle the founder's lighting themes. So the report
counts, explicitly, every frame where ``Pointing_Up`` was emitted during a
shush attempt. That count is the defect metric; any leak fails the gesture.

Exit codes: 0 verdict PASS · 1 verdict FAIL · 2 camera busy · 3 no camera
· 4 setup problem (missing model, unreadable device).
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from gesture_catalog import GESTURES, SHUSH

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = REPO_ROOT / "verification"

DEFAULT_HOLD_S = 5.0
DEFAULT_READY_S = 3.0
DEFAULT_MIN_RATE = 0.60
DEFAULT_FRAMES_PER_GESTURE = 3

# MediaPipe's label for an index finger held up. In a shush attempt this is the
# failure mode, not a neighbour: automations.json binds it to `next-theme`.
LEAKY_GESTURE = "Pointing_Up"

EXIT_PASS, EXIT_FAIL, EXIT_BUSY, EXIT_NO_CAMERA, EXIT_SETUP = 0, 1, 2, 3, 4

# One line of plain instruction per gesture. Kept complete by a drift test.
INSTRUCTIONS = {
    "Closed_Fist": "Make a fist, palm towards the camera.",
    "Open_Palm": "Open hand, fingers spread, palm towards the camera.",
    "Pointing_Up": ("Index finger straight up, other fingers curled. Hold it "
                    "high and well AWAY from your face."),
    "Thumb_Up": "Thumbs up, fist closed, thumb clearly vertical.",
    "Thumb_Down": "Thumbs down, fist closed, thumb clearly pointing down.",
    "Victory": "Index and middle finger up in a V, other fingers curled.",
    "ILoveYou": "Thumb, index and little finger extended; middle and ring down.",
    "Middle_Finger": "Middle finger up, the other three curled.",
    "Shush": ("Index finger up and pressed against your lips, as if shushing "
              "someone. Keep your face square to the camera."),
}


# ---------------------------------------------------------------------------
# Pure recording + scoring. No camera, no MediaPipe, no I/O — this half is what
# the unit tests drive with synthetic frames.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameObservation:
    """What one captured frame produced."""

    detected: tuple[str, ...] = ()        # names classify_hands returned
    raw_categories: tuple[str, ...] = ()  # MediaPipe's own labels, pre-override
    face_present: bool = False
    face_count: int = 0
    hands: int = 0
    at_s: float = 0.0


@dataclass(frozen=True)
class AttemptScore:
    """The scored result of one gesture hold."""

    gesture: str
    frames: int
    hits: int
    detection_rate: float
    face_rate: float
    misclassified: tuple[tuple[str, int], ...]
    raw_labels: tuple[tuple[str, int], ...]
    leak_frames: int
    passed: bool
    reason: str

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"


@dataclass
class RunMeta:
    """Everything the report needs about the run itself."""

    started_at: datetime = field(default_factory=datetime.now)
    camera_label: str = "unknown"
    camera_path: str = "unknown"
    resolution: str = "unknown"
    face_engine: str = "unknown"
    hold_s: float = DEFAULT_HOLD_S
    min_rate: float = DEFAULT_MIN_RATE
    read_failures: int = 0
    frames_dir: str | None = None


def _rate(part: int, whole: int) -> float:
    return (float(part) / float(whole)) if whole else 0.0


def score_attempt(gesture, observations, min_rate=DEFAULT_MIN_RATE):
    """Score one hold. Zero frames scores an honest zero, never a pass.

    ``leak_frames`` counts frames where ``Pointing_Up`` was emitted while some
    *other* gesture was being attempted. It is recorded for every gesture but
    only decides the verdict for Shush, where it is the documented defect:
    Pointing_Up is bound to next-theme, so a leaked frame is a lighting change
    the founder did not ask for.
    """
    observations = list(observations)
    frames = len(observations)
    hits = sum(1 for item in observations if gesture in item.detected)
    faces = sum(1 for item in observations if item.face_present)
    misses = Counter(name for item in observations
                     for name in dict.fromkeys(item.detected) if name != gesture)
    raw = Counter(name for item in observations
                  for name in dict.fromkeys(item.raw_categories))
    leak = 0
    if gesture != LEAKY_GESTURE:
        leak = sum(1 for item in observations if LEAKY_GESTURE in item.detected)

    detection_rate = _rate(hits, frames)
    face_rate = _rate(faces, frames)

    if frames == 0:
        passed, reason = False, "no frames captured"
    elif gesture == SHUSH and leak:
        passed = False
        reason = (f"{leak} frame(s) emitted {LEAKY_GESTURE} during the shush — "
                  f"each one would have fired next-theme")
    elif detection_rate < min_rate:
        passed = False
        reason = (f"detected in {detection_rate:.0%} of frames, "
                  f"below the {min_rate:.0%} bar")
    else:
        passed = True
        reason = f"detected in {detection_rate:.0%} of frames"
        if gesture == SHUSH:
            reason += f"; no {LEAKY_GESTURE} leaked; face box held {face_rate:.0%}"

    return AttemptScore(
        gesture=gesture,
        frames=frames,
        hits=hits,
        detection_rate=detection_rate,
        face_rate=face_rate,
        misclassified=_ranked(misses),
        raw_labels=_ranked(raw),
        leak_frames=leak,
        passed=passed,
        reason=reason,
    )


def _ranked(counter):
    return tuple(sorted(counter.items(), key=lambda pair: (-pair[1], pair[0])))


def overall_verdict(scores):
    """(passed, one-line summary) across every scored gesture."""
    scores = list(scores)
    if not scores:
        return False, "FAIL — nothing was measured"
    passed = [score for score in scores if score.passed]
    if len(passed) == len(scores):
        return True, f"PASS — {len(passed)}/{len(scores)} gestures verified live"
    failed = ", ".join(score.gesture for score in scores if not score.passed)
    return False, (f"FAIL — {len(passed)}/{len(scores)} gestures verified live; "
                   f"failed: {failed}")


def _cell(pairs, limit=3):
    if not pairs:
        return "—"
    return ", ".join(f"{name} x{count}" for name, count in pairs[:limit])


def render_report(scores, meta):
    """The whole report as markdown. Terse enough to paste into a message."""
    scores = list(scores)
    passed, summary = overall_verdict(scores)
    stamp = meta.started_at.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Live gesture verification — {stamp}",
        "",
        f"**Verdict: {summary}**",
        "",
        f"- Camera: `{meta.camera_path}` — {meta.camera_label} @ {meta.resolution}",
        f"- Face engine: `{meta.face_engine}` · gesture model: "
        f"`models/gesture_recognizer.task`",
        f"- Hold: {meta.hold_s:g}s per gesture · pass bar: "
        f"{meta.min_rate:.0%} of frames · dropped reads: {meta.read_failures}",
        "- Observation only: no rule loaded, no connector armed, no light touched.",
    ]
    if meta.frames_dir:
        lines.append(f"- Annotated frames: `{meta.frames_dir}` (gitignored)")
    lines += [
        "",
        "| Gesture | Frames | Hit | Rate | Face box | MediaPipe said | "
        "Also detected | Verdict |",
        "|---|---:|---:|---:|---:|---|---|---|",
    ]
    for score in scores:
        lines.append(
            f"| `{score.gesture}` | {score.frames} | {score.hits} | "
            f"{score.detection_rate:.0%} | {score.face_rate:.0%} | "
            f"{_cell(score.raw_labels)} | {_cell(score.misclassified)} | "
            f"**{score.verdict}** |"
        )
    lines += ["", "## Per-gesture verdicts", ""]
    for score in scores:
        lines.append(f"- **{score.gesture} — {score.verdict}**: {score.reason}")

    shush = next((score for score in scores if score.gesture == SHUSH), None)
    lines += ["", "## Shush occlusion test (headline)", ""]
    if shush is None:
        lines.append("Not measured in this run.")
    else:
        lines += [
            f"- Frames in the hold: **{shush.frames}**",
            f"- Face box survived: **{shush.face_rate:.0%}** "
            f"({sum_faces(shush)} frames)",
            f"- `Shush` detected: **{shush.detection_rate:.0%}** "
            f"({shush.hits} frames)",
            f"- **`{LEAKY_GESTURE}` leaked: {shush.leak_frames} frame(s)** "
            f"— the defect metric. `{LEAKY_GESTURE}` is bound to `next-theme`, "
            f"so every leaked frame is a lighting change nobody asked for.",
            "",
            f"**{shush.verdict}** — {shush.reason}",
        ]
    lines += [
        "",
        "## Caveats",
        "",
        "- Faces here are detected on the full frame by `engine.build_detector()`. "
        "The GUI/headless runtime instead computes faces only inside YOLO person "
        "boxes and only at `face_hz` (2 Hz default) — "
        "`acesvision/processor.py:139-153`. Shush in the real runtime can "
        "therefore only be *worse* than this measurement, never better.",
        "- Rates are per frame. The runtime additionally requires "
        "`hold_frames` consecutive frames (default 6) before an event fires — "
        "`acesvision/events.py:43`.",
        "- A gesture the founder held badly and a gesture the model cannot see "
        "look identical from here. Inspect the annotated frames before blaming "
        "the code.",
        "",
    ]
    return "\n".join(lines)


def sum_faces(score):
    return int(round(score.face_rate * score.frames))


def report_path(out_dir, when, exists=Path.exists):
    """Timestamped path; never silently overwrites an earlier run's report."""
    out_dir = Path(out_dir)
    candidate = out_dir / f"live-gestures-{when.strftime('%Y-%m-%d')}.md"
    if exists(candidate):
        candidate = (out_dir /
                     f"live-gestures-{when.strftime('%Y-%m-%d-%H%M%S')}.md")
    return candidate


def write_report(text, out_dir, when):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = report_path(out_dir, when)
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Live capture. Imports cv2/MediaPipe lazily so the pure half above stays
# importable (and testable) on a machine with no camera stack at all.
# ---------------------------------------------------------------------------

class FrameSaver:
    """A handful of annotated JPEGs per gesture — two samples plus one defect."""

    def __init__(self, directory, per_gesture=DEFAULT_FRAMES_PER_GESTURE,
                 hold_s=DEFAULT_HOLD_S):
        self.directory = Path(directory)
        self.per_gesture = int(per_gesture)
        self.hold_s = float(hold_s)
        self._saved = Counter()
        self._tags = {}

    def wanted(self, gesture, observation, at_s):
        """Save early, late, and the first defective frame. Nothing else.

        Each tag is claimed independently, so catching a defect early does not
        cost the run its ordinary sample frames.
        """
        if self._saved[gesture] >= self.per_gesture:
            return None
        used = self._tags.get(gesture, frozenset())
        if ("defect" not in used
                and any(name != gesture for name in observation.detected)):
            return "defect"
        if "early" not in used and at_s >= 0.25 * self.hold_s:
            return "early"
        if "late" not in used and at_s >= 0.70 * self.hold_s:
            return "late"
        return None

    def _record(self, gesture, tag):
        self._saved[gesture] += 1
        self._tags[gesture] = self._tags.get(gesture, frozenset()) | {tag}

    def save(self, gesture, tag, frame, faces, gestures, index):
        import cv2

        from acesvision.contracts import SceneFrame, SourceSpec
        from acesvision.overlay import MINIMAL, render

        self.directory.mkdir(parents=True, exist_ok=True)
        scene = SceneFrame(
            source=SourceSpec(id="verify", name="Verification", kind="webcam"),
            sequence=index, captured_at=time.monotonic(), raw=frame,
            faces=list(faces), gestures=list(gestures),
        )
        path = self.directory / f"{index:02d}-{gesture}-{tag}.jpg"
        cv2.imwrite(str(path), render(scene, MINIMAL))
        self._record(gesture, tag)
        return path


def choose_camera(preferred=None):
    """(device_argument, label, path) using discovery — never a blind probe."""
    from acesvision.discovery import discover_webcams, preferred_webcam

    devices = discover_webcams()
    if preferred is not None:
        chosen = next((device for device in devices
                       if device.path == preferred or str(device.index) == str(preferred)),
                      None)
        if chosen is not None:
            return chosen.path, chosen.name, chosen.path
        return preferred, "forced device", str(preferred)
    chosen = preferred_webcam(devices)
    if chosen is None:
        return None, "no device", "none"
    return chosen.path, chosen.name, chosen.path


def describe_holders(path):
    """Who is holding the device — the fuser -v answer, without shelling out."""
    import camera as camera_module

    holders = camera_module.device_holders(path)
    if not holders:
        return ("no same-user process holds it (other users are invisible in "
                "/proc, so this is weak evidence, not proof)")
    return ", ".join(f"{name} (pid {pid})" for pid, name in holders)


def countdown(gesture_id, label, seconds, out=print, sleep=time.sleep):
    out("")
    out(f"=== {gesture_id} — {label} ===")
    out(f"    {INSTRUCTIONS.get(gesture_id, 'Hold the gesture.')}")
    for remaining in range(int(seconds), 0, -1):
        out(f"    ready in {remaining}...")
        sleep(1.0)


def capture_attempt(cap, gesture_id, detector, face_detector, hold_s,
                    saver=None, index=0, out=print):
    """Hold one gesture for hold_s and record every frame. Returns (obs, reads)."""
    from gestures import classify_hands

    observations = []
    read_failures = 0
    consecutive_failures = 0
    started = time.monotonic()
    out(f">>> HOLD {gesture_id} for {hold_s:g}s — go")
    while True:
        at_s = time.monotonic() - started
        if at_s >= hold_s:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            read_failures += 1
            consecutive_failures += 1
            if consecutive_failures >= 30:
                out(f"    [warn] {consecutive_failures} consecutive read "
                    f"failures — giving up on {gesture_id}")
                break
            continue
        consecutive_failures = 0
        try:
            faces = face_detector(frame)
        except Exception as exc:                      # never abandon the run
            out(f"    [warn] face detection failed: {exc}")
            faces = []
        result, width, height = detector.recognize(frame)
        classified = classify_hands(result.hand_landmarks, result.gestures,
                                    width, height, detector.min_score,
                                    faces=faces)
        raw = tuple(categories[0].category_name
                    for categories in (result.gestures or []) if categories)
        observation = FrameObservation(
            detected=tuple(item.name for item in classified),
            raw_categories=raw,
            face_present=bool(faces),
            face_count=len(faces),
            hands=len(result.hand_landmarks or []),
            at_s=at_s,
        )
        observations.append(observation)
        if saver is not None:
            tag = saver.wanted(gesture_id, observation, at_s)
            if tag:
                saver.save(gesture_id, tag, frame, faces, classified, index)
    return observations, read_failures


def build_parser():
    parser = argparse.ArgumentParser(
        description="Guided live verification of every gesture. Observation "
                    "only — arms nothing, changes no lights.")
    parser.add_argument("--hold-s", type=float, default=DEFAULT_HOLD_S,
                        help="seconds to hold each gesture (default 5)")
    parser.add_argument("--ready-s", type=float, default=DEFAULT_READY_S,
                        help="countdown before each gesture (default 3)")
    parser.add_argument("--min-rate", type=float, default=DEFAULT_MIN_RATE,
                        help="detection rate needed to pass (default 0.6)")
    parser.add_argument("--gestures", default="",
                        help="comma-separated subset (default: all)")
    parser.add_argument("--camera", default=None,
                        help="device index or /dev path (default: discovery)")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--frames-per-gesture", type=int,
                        default=DEFAULT_FRAMES_PER_GESTURE)
    parser.add_argument("--no-frames", action="store_true",
                        help="do not save annotated JPEGs")
    return parser


def selected_gestures(raw):
    """The GESTURES subset named by --gestures, in catalog order."""
    if not raw.strip():
        return list(GESTURES)
    from gesture_catalog import require_gesture

    wanted = {require_gesture(name) for name in raw.split(",") if name.strip()}
    return [spec for spec in GESTURES if spec.id in wanted]


def main(argv=None, out=print):
    args = build_parser().parse_args(argv)
    specs = selected_gestures(args.gestures)

    import camera as camera_module

    device, label, path = choose_camera(args.camera)
    if device is None:
        out("[camera] no V4L2 capture device found. Check `ls /dev/video*`.")
        return EXIT_NO_CAMERA

    out(f"[camera] opening {path} — {label}")
    try:
        cap = camera_module.open_camera(device, manual_exposure=False)
    except camera_module.CameraBusyError as exc:
        out(f"[camera] BUSY: {exc}")
        out(f"[camera] held by: {describe_holders(path)}")
        out("[camera] free the device (close OBS / stop acergb-gesture) "
            "and run this again. Nothing was measured.")
        return EXIT_BUSY
    except camera_module.CameraNotFoundError as exc:
        out(f"[camera] {exc}")
        return EXIT_NO_CAMERA

    meta = RunMeta(camera_label=label, camera_path=path,
                   hold_s=args.hold_s, min_rate=args.min_rate)
    scores = []
    try:
        import engine
        from gestures import GestureDetector

        try:
            face_detector = engine.build_detector()
            detector = GestureDetector()
        except (FileNotFoundError, OSError) as exc:
            out(f"[setup] {exc}")
            return EXIT_SETUP
        meta.face_engine = getattr(face_detector, "engine", "unknown")

        ok, frame = cap.read()
        if ok and frame is not None:
            height, width = frame.shape[:2]
            meta.resolution = f"{width}x{height}"

        saver = None
        if not args.no_frames:
            saver = FrameSaver(Path(args.out_dir) / "frames",
                               per_gesture=args.frames_per_gesture,
                               hold_s=args.hold_s)
            meta.frames_dir = str(Path(args.out_dir) / "frames")

        out("")
        out(f"[run] {len(specs)} gestures x {args.hold_s:g}s. "
            f"Nothing is armed: no rules are loaded and no connector exists in "
            f"this process, so your lights cannot move.")
        for index, spec in enumerate(specs, start=1):
            countdown(spec.id, f"{index}/{len(specs)}  {spec.label}",
                      args.ready_s, out=out)
            observations, failures = capture_attempt(
                cap, spec.id, detector, face_detector, args.hold_s,
                saver=saver, index=index, out=out)
            meta.read_failures += failures
            score = score_attempt(spec.id, observations, min_rate=args.min_rate)
            scores.append(score)
            out(f"    {score.verdict}: {score.reason} "
                f"({score.frames} frames, face box {score.face_rate:.0%})")
        detector.close()
    finally:
        cap.release()

    text = render_report(scores, meta)
    destination = write_report(text, args.out_dir, meta.started_at)
    passed, summary = overall_verdict(scores)
    out("")
    out(summary)
    out(f"[report] {destination}")
    return EXIT_PASS if passed else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
