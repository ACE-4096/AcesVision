"""Local YOLO worker adapter and shared perception records.

The worker is a *subprocess of this repo* (``acesvision.yolo_worker``), not of
any other project. It exists to keep Ultralytics and the GPU runtime off the
Qt/face-recognition import path, not to borrow a foreign virtualenv — the
default interpreter is therefore the one already running AcesVision.

Wire protocol (version 2). Every exchange is frame-tagged so that a reply which
arrives after its request timed out can be identified and dropped instead of
being handed back as the answer to a *later* frame:

    request   >II  (sequence, payload_length) + JPEG bytes
    reply     >I   (payload_length) + JSON, always carrying "seq"
    handshake one reply with sequence 0, sent once the worker has validated the
              device and loaded the model: {"seq":0,"ready":true,...} or
              {"seq":0,"ready":false,"error":...} followed by worker exit.

Protocol version 1 had no sequence field. A ``select`` timeout left the pending
reply in the pipe and the worker alive, so every later reply was one frame
stale, for ever, while ``inference_status`` still read "live" — boxes drawn on
the wrong frame with nothing raised. The sequence tag is what makes that
recoverable rather than permanent.
"""
from __future__ import annotations

import json
import hashlib
import os
import select
import struct
import subprocess
import sys
import threading
import time
from collections import deque, namedtuple
from pathlib import Path

import cv2


Detection = namedtuple("Detection", "x y w h label score track_id")

PROTOCOL_VERSION = 2

REPO_ROOT = Path(__file__).resolve().parents[1]

# Defaults are read from the environment when they are *used*, not when this
# module is imported. Importing acesvision early — the GUI does — otherwise
# froze the configuration before anything had a chance to set it.


def default_worker_python(env=None):
    """The interpreter that runs the YOLO worker.

    The one already running AcesVision, unless ``ACESVISION_YOLO_PYTHON`` says
    otherwise. There is deliberately no path to any other repository here: the
    worker is this repo's own module and needs this repo's own dependencies.
    """
    env = os.environ if env is None else env
    return Path(env.get("ACESVISION_YOLO_PYTHON") or sys.executable)


def default_model(env=None):
    env = os.environ if env is None else env
    return Path(env.get("ACESVISION_YOLO_MODEL") or (REPO_ROOT / "yolo26n.pt"))


def default_device(env=None):
    """'auto' resolves in the worker to GPU 0 when one exists, else CPU.

    An explicit value ("0", "1", "cuda:0", "cpu") is validated at worker
    startup and refused if it does not exist.
    """
    env = os.environ if env is None else env
    return env.get("ACESVISION_YOLO_DEVICE") or "auto"


def default_hsa_gfx_version(env=None):
    """gfx1030 is what an RX 6600 (gfx1032) needs ROCm to pretend to be.

    Meaningless — and misleading in a crash log — on a host with no AMD GPU.
    """
    env = os.environ if env is None else env
    return env.get("ACESVISION_HSA_GFX_VERSION") or "10.3.0"


DEFAULT_WORKER_PYTHON = default_worker_python()
DEFAULT_MODEL = default_model()
DEFAULT_DEVICE = default_device()
DEFAULT_HSA_OVERRIDE = default_hsa_gfx_version()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def amd_rocm_present(root="/"):
    """True when this host actually has the AMD GPU stack loaded."""
    base = Path(root)
    return (base / "sys/module/amdgpu").exists() or (base / "opt/rocm").exists()


def rocm_env_overrides(env, device, present=None, gfx_version=None):
    """The ROCm-specific environment additions for ``device``, if any.

    Returns a dict to merge into the worker environment. Empty unless this is
    an AMD host *and* the worker is going to touch a GPU. An inherited or
    caller-supplied ``HSA_OVERRIDE_GFX_VERSION`` always wins, including an
    explicit empty value, which suppresses the override entirely.
    """
    gfx_version = default_hsa_gfx_version() if gfx_version is None else gfx_version
    if "HSA_OVERRIDE_GFX_VERSION" in env:
        return {}
    if str(device).strip().lower() == "cpu":
        return {}
    if not gfx_version:
        return {}
    present = amd_rocm_present() if present is None else present
    if not present:
        return {}
    return {"HSA_OVERRIDE_GFX_VERSION": str(gfx_version)}


class WorkerDeviceError(RuntimeError):
    """The worker refused to start because the requested device is not there."""


class YoloSubprocessDetector:
    """Run Ultralytics and the GPU runtime outside the Qt/face-recognition env."""

    def __init__(self, model=None, python=None,
                 tracker="bytetrack.yaml", device=None, timeout_s=20.0,
                 jpeg_quality=82, startup_timeout_s=90.0,
                 hsa_gfx_version=None, rocm_present=None,
                 clock=time.monotonic):
        self.model = default_model() if model is None else Path(model)
        self.python = default_worker_python() if python is None else Path(python)
        self.tracker = tracker
        self.device = str(default_device() if device is None else device)
        self.timeout_s = float(timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.jpeg_quality = int(jpeg_quality)
        self.hsa_gfx_version = (default_hsa_gfx_version()
                                if hsa_gfx_version is None else hsa_gfx_version)
        self.rocm_present = rocm_present
        self.clock = clock
        self._process = None
        self._stderr_lines = deque(maxlen=20)
        self._stderr_thread = None
        self._sequence = 0
        #: Replies dropped because they belonged to an already-abandoned frame.
        #: Non-zero means a timeout happened and the pipe re-synchronised.
        self.stale_replies_discarded = 0
        self.resolved_device = None
        self.model_id = f"ultralytics:{self.model.stem}"

    def _start(self):
        if not self.python.is_file():
            raise RuntimeError(f"YOLO worker Python not found: {self.python}")
        if not self.model.is_file():
            raise RuntimeError(
                f"YOLO model is not installed: {self.model}. "
                "AcesVision never downloads models automatically."
            )
        env = os.environ.copy()
        env.update(rocm_env_overrides(env, self.device,
                                      present=self.rocm_present,
                                      gfx_version=self.hsa_gfx_version))
        env["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            [str(self.python), "-m", "acesvision.yolo_worker",
             "--model", str(self.model), "--tracker", self.tracker,
             "--device", self.device],
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Unbuffered on purpose. select() reports on the file descriptor,
            # so a BufferedReader that has already pulled the next reply out of
            # the pipe makes select() say "nothing to read" while a complete
            # message sits in Python's buffer. That is invisible until
            # something reads two replies in one call — which is exactly what
            # discarding a stale frame does.
            bufsize=0,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="yolo-worker-log"
        )
        self._stderr_thread.start()
        self._handshake()

    def _handshake(self):
        """Block until the worker confirms the device and the loaded model.

        Doing this here rather than letting the first frame discover it is the
        whole point of the check: ``device='9'`` used to start cleanly and
        return perfectly normal detections from some other device.
        """
        process = self._process
        try:
            reply = self._await_reply(process, 0, self.clock() + self.startup_timeout_s)
        except TimeoutError as exc:
            self.close()
            raise RuntimeError(self._worker_error(
                f"YOLO worker did not become ready within "
                f"{self.startup_timeout_s:g}s")) from exc
        except (BrokenPipeError, EOFError) as exc:
            self.close()
            raise RuntimeError(
                self._worker_error("YOLO worker exited before it was ready")) from exc
        if not reply.get("ready"):
            error = reply.get("error") or "worker reported not ready"
            self.close()
            raise WorkerDeviceError(f"YOLO worker refused to start: {error}")
        self.resolved_device = reply.get("device")
        self.model_id = reply.get("model", self.model_id)
        return reply

    def _drain_stderr(self):
        process = self._process
        if process is None or process.stderr is None:
            return
        for line in iter(process.stderr.readline, b""):
            self._stderr_lines.append(line.decode(errors="replace").strip())

    def detect(self, frame):
        if self._process is None:
            self._start()
        process = self._process
        if process.poll() is not None:
            raise RuntimeError(self._worker_error("YOLO worker stopped"))
        ok, encoded = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("Could not encode frame for YOLO worker")
        payload = encoded.tobytes()
        self._sequence += 1
        sequence = self._sequence
        try:
            self._write_all(process.stdin, struct.pack(">II", sequence, len(payload)))
            self._write_all(process.stdin, payload)
            process.stdin.flush()
            result = self._await_reply(process, sequence,
                                       self.clock() + self.timeout_s)
        except (BrokenPipeError, EOFError) as exc:
            raise RuntimeError(self._worker_error("YOLO worker connection lost")) from exc
        objects = [Detection(**item) for item in result.get("objects", [])]
        return objects, result.get("timings", {}), result.get("model", self.model_id)

    def _await_reply(self, process, sequence, deadline):
        """Read replies until the one tagged ``sequence`` arrives.

        Anything older belongs to a frame whose ``detect`` already raised
        ``TimeoutError``; it is counted and dropped. Without this the pipe stays
        one reply behind for the rest of the process's life.
        """
        while True:
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise TimeoutError(
                    f"YOLO worker did not respond within {self.timeout_s:g}s "
                    f"(frame {sequence})"
                )
            ready, _, _ = select.select([process.stdout], [], [], remaining)
            if not ready:
                raise TimeoutError(
                    f"YOLO worker did not respond within {self.timeout_s:g}s "
                    f"(frame {sequence})"
                )
            size = struct.unpack(">I", self._read_exact(process.stdout, 4))[0]
            result = json.loads(self._read_exact(process.stdout, size))
            reply_sequence = result.get("seq")
            if reply_sequence == sequence:
                return result
            if reply_sequence is None:
                raise RuntimeError(
                    "YOLO worker reply carried no frame tag — worker and adapter "
                    f"disagree on the wire protocol (expected v{PROTOCOL_VERSION})"
                )
            self.stale_replies_discarded += 1

    @staticmethod
    def _write_all(stream, data):
        """Write every byte. Raw pipe writes are allowed to be partial."""
        view = memoryview(data)
        while view:
            written = stream.write(view)
            if not written:
                raise BrokenPipeError("YOLO worker stdin accepted no bytes")
            view = view[written:]

    @staticmethod
    def _read_exact(stream, size):
        chunks = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _worker_error(self, prefix):
        detail = " | ".join(line for line in self._stderr_lines if line)
        return f"{prefix}: {detail}" if detail else prefix

    def close(self):
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)
