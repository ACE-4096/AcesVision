"""Local YOLO worker adapter and shared perception records."""
from __future__ import annotations

import json
import hashlib
import os
import select
import struct
import subprocess
import threading
from collections import deque, namedtuple
from pathlib import Path

import cv2


Detection = namedtuple("Detection", "x y w h label score track_id")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER_PYTHON = Path(
    os.environ.get(
        "ACESVISION_YOLO_PYTHON",
        "/home/ace4096/Documents/git_private/cv-worker/.venv/bin/python",
    )
)
DEFAULT_MODEL = Path(os.environ.get("ACESVISION_YOLO_MODEL",
                                    str(REPO_ROOT / "yolo26n.pt")))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class YoloSubprocessDetector:
    """Run Ultralytics and ROCm outside the Qt/face-recognition environment."""

    def __init__(self, model=DEFAULT_MODEL, python=DEFAULT_WORKER_PYTHON,
                 tracker="bytetrack.yaml", device="0", timeout_s=20.0,
                 jpeg_quality=82):
        self.model = Path(model)
        self.python = Path(python)
        self.tracker = tracker
        self.device = str(device)
        self.timeout_s = float(timeout_s)
        self.jpeg_quality = int(jpeg_quality)
        self._process = None
        self._stderr_lines = deque(maxlen=20)
        self._stderr_thread = None
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
        env.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
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
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="yolo-worker-log"
        )
        self._stderr_thread.start()

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
        try:
            process.stdin.write(struct.pack(">I", len(payload)))
            process.stdin.write(payload)
            process.stdin.flush()
            ready, _, _ = select.select([process.stdout], [], [], self.timeout_s)
            if not ready:
                raise TimeoutError(
                    f"YOLO worker did not respond within {self.timeout_s:g}s"
                )
            size = struct.unpack(">I", self._read_exact(process.stdout, 4))[0]
            result = json.loads(self._read_exact(process.stdout, size))
        except (BrokenPipeError, EOFError) as exc:
            raise RuntimeError(self._worker_error("YOLO worker connection lost")) from exc
        objects = [Detection(**item) for item in result.get("objects", [])]
        return objects, result.get("timings", {}), result.get("model", self.model_id)

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
