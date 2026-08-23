"""Local PipeWire/Pulse input discovery for opt-in AcesVision recordings."""
from __future__ import annotations

import subprocess


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
