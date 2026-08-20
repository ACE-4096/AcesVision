"""Record the overlaid feed to a playable MP4, with a structured sidecar.

Why a subprocess and not ``cv2.VideoWriter``
--------------------------------------------
The pip ``opencv-python`` wheel bundles a GPL-stripped FFmpeg. On this machine
it reports ``FFMPEG: YES`` and ``VA: NO``, and asking it for ``avc1`` falls
through to ``h264_v4l2m2m``, finds no device, and returns a writer whose
``isOpened()`` is ``False`` — a recorder that silently produces nothing. Even
where it does open, ``VideoWriter`` exposes no CRF, no preset, no
``+faststart`` and no VAAPI, so the file it writes is the wrong file.

The system FFmpeg has ``libx264`` *and* ``h264_vaapi``. So this module pipes
raw BGR into ``ffmpeg`` on stdin and lets it do the encoding it is good at.

``yuv420p`` and ``+faststart`` are not tuning knobs. Browsers and every social
encoder reject or mangle 4:4:4 H.264, and without ``+faststart`` the moov atom
lands at the end of the file, so the clip will not begin playing until it has
been fetched in full.

Constant frame rate, on purpose
-------------------------------
Capture is variable: a webcam that drifts between 24 and 30 fps, a network
source that stalls, an inference stage that applies backpressure. Writing those
timings straight out gives a VFR MP4, and VFR MP4 confuses editors (audio drift
on the timeline) and social encoders (re-encoded to a guessed rate). So this
holds its own frame clock at the target fps and fits capture to it: a slot with
no new frame repeats the last one, a frame that arrives inside a slot already
filled is dropped from the *video*.

Nothing is lost by that drop, because it never reaches the file's job. The
sidecar still carries the frame — its true ``captured_at``, its sequence and
its inference results — so a downstream clip pipeline can still key off the
real capture time even where the video shows a duplicate.

What "lossless" means here
--------------------------
This output must be added with ``pipeline.add_output(recorder, lossless=True)``.
On the default drop-old mailbox the encoder falling behind for 200 ms silently
removes 200 ms from the middle of the recording. With backpressure the capture
loop waits instead, capture fps dips where the operator can see it, and the
file stays continuous. If the queue overflows anyway the worker calls
``note_dropped`` and the count ends up in the sidecar and in pipeline metrics.

Its own smoother
----------------
Like every other rendering output, this one constructs its **own**
``SceneSmoother`` and never shares it. Each output sees a different subset of
frames through its mailbox, and a shared smoother would advance its clock
against frames this recorder never received.

The sidecar records the **unsmoothed** scene, matching
``LatestFrameOutput.snapshot``: the eased boxes are the picture, the raw
detections are the record.

Where recordings go
-------------------
``$ACESVISION_RECORDINGS`` or ``~/Videos/AcesVision``. Never inside the
repository — this is an AGPL project that gets published, and a stray MP4 of
somebody's living room in a git history cannot be taken back. That is enforced
here (:func:`refuse_repo_path`) rather than left to ``.gitignore``, because a
``.gitignore`` rule only stops the commit, not the write.

Privacy note: the sidecar carries the *names* of enrolled people, because that
is the inference result and the point of the record. It carries no embeddings.
It is exactly as sensitive as the video beside it — keep them together, and
delete them together.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from datetime import datetime
from pathlib import Path

import cv2

from .contracts import SceneFrame
from .overlay import BROADCAST, OverlayProfile, render
from .smoothing import SceneSmoother

#: Sidecar schema tag. Versioned separately from the scene contract, because a
#: consumer of these files does not otherwise care which contract wrote them.
SIDECAR_SCHEMA = "acesvision.recording/1"

#: Where recordings land when nothing says otherwise.
DEFAULT_RECORDINGS_DIR = Path("~/Videos/AcesVision")

#: The environment variable that overrides it.
RECORDINGS_ENV = "ACESVISION_RECORDINGS"

#: Default constant frame rate. 30 is what every social encoder wants and what
#: a 24-30 fps webcam fits into with the fewest repeats.
DEFAULT_FPS = 30

#: x264 quality. 20 is visually near-transparent for this material and still
#: leaves a file small enough to move around.
DEFAULT_CRF = 20

#: veryfast, not faster: this runs beside YOLO on the same CPU, and the frame
#: budget it has to fit inside is 1/fps, not "as fast as possible".
DEFAULT_PRESET = "veryfast"

#: The VAAPI render node ``--record-hw`` uses.
DEFAULT_VAAPI_DEVICE = "/dev/dri/renderD128"

#: h264_vaapi has no CRF; qp 22 is the closest constant-quality equivalent.
DEFAULT_VAAPI_QP = 22

#: How long ``close()`` waits for ffmpeg to flush and write the moov atom
#: before it stops being patient. A wedged encoder must not hang shutdown, but
#: cutting it short leaves an unplayable file, so this is generous.
FINALISE_TIMEOUT_S = 20.0

#: Guards against a float ``elapsed * fps`` landing at 0.9999999 on the slot
#: boundary and costing a frame every time capture and record rates match.
_SLOT_EPSILON = 1e-9

#: The repository this module is part of. Recordings are refused anywhere
#: underneath it.
REPO_ROOT = Path(__file__).resolve().parent.parent


class RecordingError(RuntimeError):
    """ffmpeg could not be started, or died while being fed."""


def refuse_repo_path(path: Path) -> Path:
    """Return ``path`` resolved, or raise if it is inside the repository.

    Called on the directory *and* on the final file path, because an
    ``$ACESVISION_RECORDINGS`` pointing at the checkout and a ``--record``
    naming a file in it are the same mistake arriving by different routes.
    """
    resolved = Path(path).expanduser().resolve()
    if resolved == REPO_ROOT or REPO_ROOT in resolved.parents:
        raise ValueError(
            f"refusing to write a recording inside the repository: {resolved}. "
            f"Recordings hold faces and rooms; this checkout gets published. "
            f"Set {RECORDINGS_ENV} to somewhere outside {REPO_ROOT}."
        )
    return resolved


def recordings_dir(environ=None) -> Path:
    """The directory recordings are written to. Not created here."""
    environ = os.environ if environ is None else environ
    raw = (environ.get(RECORDINGS_ENV) or "").strip()
    return refuse_repo_path(Path(raw) if raw else DEFAULT_RECORDINGS_DIR)


def recording_name(source_id: str, now: datetime | None = None) -> str:
    """``acesvision-YYYYmmdd-HHMMSS-<source-id>.mp4``.

    Local time, not UTC: this name is read by the person who was in the room,
    and they know what time they pressed record.
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "-"
                   for c in (source_id or "source")).strip("-") or "source"
    return f"acesvision-{stamp}-{safe}.mp4"


def resolve_path(requested, source_id, now=None, environ=None) -> Path:
    """Turn a ``--record`` argument into the file to write.

    Nothing, or an empty string, or a directory: the default name inside the
    recordings directory. Anything else: taken as the file itself.
    """
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.is_dir() or str(requested).endswith(os.sep):
            return refuse_repo_path(candidate / recording_name(source_id, now))
        return refuse_repo_path(candidate)
    return refuse_repo_path(recordings_dir(environ) /
                            recording_name(source_id, now))


def software_command(path, width, height, fps, *, crf=DEFAULT_CRF,
                     preset=DEFAULT_PRESET) -> list[str]:
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", str(preset), "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(path),
    ]


def hardware_command(path, width, height, fps, *, qp=DEFAULT_VAAPI_QP,
                     device=DEFAULT_VAAPI_DEVICE) -> list[str]:
    """VAAPI. Opt-in, because the same GPU is running YOLO.

    The reason to want this is not that it is faster end to end — measure it —
    but that it hands the CPU back to inference. Where the GPU is the busy one,
    it is a loss.
    """
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-vaapi_device", str(device),
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an",
        "-vf", "format=nv12,hwupload",
        "-c:v", "h264_vaapi", "-qp", str(qp),
        "-movflags", "+faststart",
        str(path),
    ]


def _spawn(command):
    """Start ffmpeg in its own session, deliberately.

    Ctrl-C signals the whole foreground process **group**, so without this the
    terminal's SIGINT reaches ffmpeg directly and races the orderly shutdown.
    Measured: ffmpeg does still write the trailer on SIGINT, so the file was
    playable — but it exited 255, which this module then reported as a failed
    recording, and a green run that reports an error is as bad as a red one
    that does not. Worse, ffmpeg's own signal handling stops it reading stdin
    immediately, so frames already in flight are cut rather than encoded.

    Detached, exactly one thing finalises the file: ``close()`` shuts stdin,
    ffmpeg sees EOF, writes the moov atom and exits 0.
    """
    return subprocess.Popen(command, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE,
                            start_new_session=True)


def _jsonable(value):
    """Last-resort encoder. numpy scalars arrive here from the detectors."""
    item = getattr(value, "item", None)
    if item is not None:
        try:
            return item()
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        try:
            return tolist()
        except (TypeError, ValueError):
            pass
    return str(value)


def _record_of(item):
    """One detection as a plain dict, whatever namedtuple shape it arrived in.

    Deliberately shape-agnostic: ``Detection``, ``Face`` and ``Gesture`` have
    different fields, they gain fields over time, and a hand-written field list
    here would quietly stop recording the new ones.
    """
    as_dict = getattr(item, "_asdict", None)
    if as_dict is not None:
        return dict(as_dict())
    if hasattr(item, "__dict__"):
        return dict(vars(item))
    return {"value": item}


class RecordingOutput:
    """A ``FrameOutput`` that writes the overlaid feed to an MP4.

    Add it with ``lossless=True``. See the module docstring.
    """

    def __init__(self, path, *, fps: int = DEFAULT_FPS,
                 profile: OverlayProfile = BROADCAST,
                 hardware: bool = False, crf: int = DEFAULT_CRF,
                 preset: str = DEFAULT_PRESET, qp: int = DEFAULT_VAAPI_QP,
                 vaapi_device: str = DEFAULT_VAAPI_DEVICE,
                 spawn=None, sidecar: bool = True):
        self.path = Path(path)
        self.sidecar_path = self.path.with_name(self.path.name + ".json")
        self.fps = max(1, int(fps))
        self.profile = profile
        self.hardware = bool(hardware)
        self.crf, self.preset = crf, preset
        self.qp, self.vaapi_device = qp, vaapi_device
        self.error = ""
        self.command: list[str] = []
        self._spawn = spawn or _spawn
        self._write_sidecar = bool(sidecar)
        self._smoother = SceneSmoother()

        self._process = None
        self._size: tuple[int, int] | None = None
        self._closed = False
        self._started_at: float | None = None
        self._started_wall = ""
        self._source = None

        # frames_written counts what went down the pipe, so it is what the MP4
        # duration is made of; scenes_published counts what the pipeline gave
        # this output. They differ by design and both are reported.
        self.frames_written = 0
        self.frames_duplicated = 0
        self.scenes_published = 0
        self.scenes_dropped_early = 0
        self.frames_dropped_by_backpressure = 0
        self._frames: list[dict] = []
        # note_dropped runs on the capture thread; everything else runs on the
        # worker thread. Only this one counter crosses.
        self._drop_lock = threading.Lock()

    # ---- the FrameOutput protocol -----------------------------------------

    def publish(self, scene: SceneFrame) -> None:
        if self._closed:
            return
        frame = render(self._smoother.apply(scene), self.profile)
        if self._process is None:
            self._start(frame, scene)
        if self._size is not None and frame.shape[1::-1] != self._size:
            # A source switch mid-recording. Resize rather than restart: the
            # stream's geometry is fixed at the first frame and a changed one
            # would corrupt every frame after it.
            frame = cv2.resize(frame, self._size)

        repeats = self._slots_due(scene.captured_at)
        if repeats <= 0:
            # Arrived inside a slot already filled. Not lost — the sidecar
            # still records it, keyed to the video frame that covers it.
            self.scenes_dropped_early += 1
        self._note_scene(scene, repeats)
        self.scenes_published += 1
        if repeats <= 0:
            return
        self._feed(frame, repeats)

    def close(self) -> None:
        """Finalise. Idempotent, and the only place the moov atom gets written.

        ``pipeline._OutputWorker.run`` calls this from a ``finally`` in both
        mailbox modes, so a Ctrl-C that stops the pipeline lands here too.
        """
        if self._closed:
            return
        self._closed = True
        process, self._process = self._process, None
        if process is not None:
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
            except OSError:
                pass                    # already dead; wait() has the reason
            try:
                process.wait(timeout=FINALISE_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._fail("ffmpeg did not exit; the file may be unplayable")
                process.kill()
                process.wait(timeout=5.0)
            else:
                if process.returncode:
                    self._fail(f"ffmpeg exited {process.returncode}: "
                               f"{self._stderr(process)}")
        if self._started_at is not None and self._write_sidecar:
            self._save_sidecar()

    # ---- the optional loss hook -------------------------------------------

    def note_dropped(self, count: int = 1) -> None:
        """Called by a lossless ``_OutputWorker`` when its queue overflowed.

        These frames never reach ``publish`` at all — nothing here can see
        them — so the worker has to say so, and the number goes in the sidecar
        beside the frames that did arrive.
        """
        with self._drop_lock:
            self.frames_dropped_by_backpressure += int(count)

    # ---- the frame clock ---------------------------------------------------

    def _slots_due(self, captured_at: float) -> int:
        """How many video frames this scene owes the file. 0 means too early.

        Video frame ``i`` presents at ``i / fps`` from the first frame. When a
        scene arrives at ``elapsed``, every slot whose start time has already
        passed must exist, so the number that should exist is
        ``floor(elapsed * fps) + 1``. Anything still missing is filled with
        this frame — that is the duplication that keeps the rate constant.
        """
        elapsed = max(0.0, captured_at - self._started_at)
        due = int(elapsed * self.fps + _SLOT_EPSILON) + 1
        return due - self.frames_written

    # ---- internals ---------------------------------------------------------

    def _start(self, frame, scene) -> None:
        height, width = frame.shape[:2]
        self._size = (width, height)
        self._started_at = scene.captured_at
        self._started_wall = datetime.now().isoformat(timespec="seconds")
        self._source = scene.source
        builder = hardware_command if self.hardware else software_command
        kwargs = ({"qp": self.qp, "device": self.vaapi_device} if self.hardware
                  else {"crf": self.crf, "preset": self.preset})
        self.command = builder(self.path, width, height, self.fps, **kwargs)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._process = self._spawn(self.command)
        except (OSError, ValueError) as exc:
            self._closed = True
            raise RecordingError(f"could not start ffmpeg: {exc}") from exc

    def _feed(self, frame, repeats: int) -> None:
        payload = frame.tobytes()
        stdin = self._process.stdin
        try:
            for _ in range(repeats):
                stdin.write(payload)
        except (BrokenPipeError, OSError, ValueError) as exc:
            # Deliberately not counted as written. Inflating frames_written
            # here would inflate duration_s, and a sidecar that claims seconds
            # the file does not contain is worse than one that reports an
            # error. close() reads ffmpeg's own stderr and appends the reason.
            self._fail(f"ffmpeg stopped accepting frames: {exc}")
            self.close()
            raise RecordingError(self.error) from exc
        self.frames_written += repeats
        self.frames_duplicated += repeats - 1

    def _note_scene(self, scene: SceneFrame, repeats: int) -> None:
        self._frames.append({
            "sequence": scene.sequence,
            "captured_at": scene.captured_at,
            "offset_s": round(scene.captured_at - self._started_at, 6),
            # The first video frame this scene is carried by, and how many it
            # fills. Where repeats is 0 the scene arrived inside a slot already
            # filled and shares the previous frame, so this points back at it
            # and video_frames is 0 — the scene is still in the record, and the
            # record still says exactly where in the video to find it.
            "video_frame": (max(0, self.frames_written - 1) if repeats <= 0
                            else self.frames_written),
            "video_frames": max(0, repeats),
            "objects": [_record_of(item) for item in scene.objects],
            "faces": [_record_of(item) for item in scene.faces],
            "gestures": [_record_of(item) for item in scene.gestures],
            "metadata": dict(scene.metadata),
        })

    def _fail(self, message: str) -> None:
        """Accumulate, do not overwrite.

        A broken pipe and the non-zero exit that explains it are two halves of
        one diagnosis, and first-one-wins would keep the useless half.
        """
        if not message or message in self.error:
            return
        self.error = f"{self.error}; {message}" if self.error else message

    @staticmethod
    def _stderr(process) -> str:
        stream = getattr(process, "stderr", None)
        if stream is None:
            return ""
        try:
            return (stream.read() or b"").decode("utf-8", "replace").strip()
        except (OSError, ValueError, AttributeError):
            return ""

    def summary(self) -> dict:
        with self._drop_lock:
            dropped = self.frames_dropped_by_backpressure
        source = self._source
        width, height = self._size or (0, 0)
        return {
            "schema": SIDECAR_SCHEMA,
            "video": self.path.name,
            "source": ({"id": source.id, "name": source.name,
                        "kind": source.kind} if source is not None else None),
            "started_wall": self._started_wall,
            "started_monotonic": self._started_at,
            "fps": self.fps,
            "width": width,
            "height": height,
            "encoder": {
                "hardware": self.hardware,
                "command": list(self.command),
            },
            "frames_written": self.frames_written,
            "frames_duplicated": self.frames_duplicated,
            "duration_s": round(self.frames_written / self.fps, 6),
            "scenes_published": self.scenes_published,
            "scenes_dropped_early": self.scenes_dropped_early,
            # The one that means something is wrong. Zero is the healthy value;
            # anything else is footage that does not exist anywhere.
            "frames_dropped_by_backpressure": dropped,
            "error": self.error,
        }

    def _save_sidecar(self) -> None:
        document = self.summary()
        document["frames"] = self._frames
        try:
            self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.sidecar_path, "w") as handle:
                json.dump(document, handle, default=_jsonable,
                          separators=(",", ":"))
        except OSError as exc:
            self._fail(f"could not write the sidecar {self.sidecar_path}: {exc}")

    def describe(self) -> str:
        seconds = self.frames_written / self.fps
        parts = [f"{self.path}", f"{self.frames_written} frames",
                 f"{seconds:.1f}s at {self.fps} fps"]
        if self.frames_duplicated:
            parts.append(f"{self.frames_duplicated} duplicated")
        if self.scenes_dropped_early:
            parts.append(f"{self.scenes_dropped_early} scenes ahead of the clock")
        with self._drop_lock:
            dropped = self.frames_dropped_by_backpressure
        if dropped:
            parts.append(f"!! {dropped} frames LOST to backpressure")
        if self.error:
            parts.append(f"!! {self.error}")
        return ", ".join(parts)
