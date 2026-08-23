"""Local PipeWire/Pulse input discovery for opt-in AcesVision recordings."""
from __future__ import annotations

import array
import math
import subprocess
import sys
import threading


def discover_audio_sources(run=None):
    """Return safe UI records for currently visible Pulse-compatible sources.

    PipeWire exposes its Pulse compatibility server through ``pactl``.  The
    actual identifier is retained for FFmpeg, while the UI gets a human label.
    Failure is non-fatal: video recording remains available without audio.
    """
    run = run or subprocess.run
    fallback = [{"id": "", "label": "No audio (video only)", "kind": "none"}]
    try:
        completed = run(["pactl", "list", "short", "sources"],
                        capture_output=True, text=True, check=False)
    except OSError:
        return fallback
    if completed.returncode:
        return fallback
    rows = list(fallback)
    for line in completed.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) < 2 or not fields[1]:
            continue
        source = fields[1]
        monitor = source.endswith(".monitor")
        friendly = source.replace("alsa_input.", "").replace("alsa_output.", "")
        friendly = friendly.replace(".monitor", "").replace("_", " ")
        rows.append({
            "id": source,
            "kind": "monitor" if monitor else "microphone",
            "label": ("System audio — " if monitor else "Microphone — ") + friendly,
        })
    return rows


def source_volume_percent(source, run=None):
    """Read a Pulse source's first channel volume as a UI-safe percentage."""
    if not source:
        raise ValueError("choose a microphone before reading its gain")
    run = run or subprocess.run
    try:
        completed = run(["pactl", "get-source-volume", str(source)],
                        capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"pactl is unavailable: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "unknown pactl error").strip()
        raise RuntimeError(detail)
    for token in completed.stdout.replace("/", " ").split():
        if token.endswith("%") and token[:-1].isdigit():
            return max(0, min(150, int(token[:-1])))
    raise RuntimeError("Pulse did not report a source volume")


def set_source_volume_percent(source, percent, run=None):
    """Set one local microphone source, capped at a deliberate +50% boost."""
    if not source:
        raise ValueError("choose a microphone before setting its gain")
    percent = max(0, min(150, int(percent)))
    run = run or subprocess.run
    try:
        completed = run(["pactl", "set-source-volume", str(source), f"{percent}%"],
                        capture_output=True, text=True, check=False)
    except OSError as exc:
        raise RuntimeError(f"pactl is unavailable: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "unknown pactl error").strip()
        raise RuntimeError(detail)
    return source_volume_percent(source, run=run)


class AudioLevelMeter:
    """A small local Pulse capture used only to visualise selected mic level.

    ``parec`` is a second *reader* of the microphone, never the recorder.
    That means a meter failure cannot steal, reshape, or silently mute audio in
    a take. The callback is deliberately plain Python; the Qt backend forwards
    it through a thread-safe signal before touching the UI.
    """

    RATE = 48_000
    CHUNK_BYTES = 4_800             # 50 ms mono s16le
    FLOOR_DB = -60.0

    def __init__(self, on_sample, popen=None):
        self._on_sample = on_sample
        self._popen = popen or subprocess.Popen
        self._process = None
        self._thread = None
        self._stopping = threading.Event()
        self._source = ""

    def start(self, source):
        self.stop()
        if not source:
            return "Choose a microphone to meter it"
        command = [
            "parec", "--raw", "--format=s16le", f"--rate={self.RATE}",
            "--channels=1", "--process-time-msec=50", "--device", str(source),
            "--client-name=AcesVision-meter",
        ]
        try:
            process = self._popen(command, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, bufsize=0)
        except OSError as exc:
            return f"Meter unavailable: {exc}"
        self._source = str(source)
        self._process = process
        stopping = threading.Event()
        self._stopping = stopping
        self._thread = threading.Thread(target=self._read,
                                        args=(process, self._source, stopping),
                                        daemon=True,
                                        name="acesvision-audio-meter")
        self._thread.start()
        return "Speak normally; aim for peaks around −12 to −6 dB"

    def stop(self):
        self._stopping.set()
        process, self._process = self._process, None
        if process is not None:
            try:
                process.terminate()
            except OSError:
                pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._source = ""

    def _read(self, process, source, stopping):
        stream = getattr(process, "stdout", None)
        if stream is None:
            self._on_sample(source, self.FLOOR_DB, "Meter stream unavailable")
            return
        while not stopping.is_set():
            try:
                payload = stream.read(self.CHUNK_BYTES)
            except OSError:
                payload = b""
            if not payload:
                if not stopping.is_set():
                    self._on_sample(source, self.FLOOR_DB,
                                    "Meter stopped; check the microphone")
                return
            samples = array.array("h")
            samples.frombytes(payload[:len(payload) - len(payload) % 2])
            if sys.byteorder != "little":
                samples.byteswap()
            if not samples:
                continue
            rms = math.sqrt(sum(value * value for value in samples) / len(samples))
            db = self.FLOOR_DB if rms <= 0 else 20.0 * math.log10(rms / 32768.0)
            self._on_sample(source, max(self.FLOOR_DB, min(0.0, db)), "")
