"""Binary stdio worker for Ultralytics YOLO tracking.

Speaks protocol version 2 (see ``acesvision.perception``): every request is
frame-tagged and every reply echoes the tag, so a reply that arrives after its
request timed out is identifiable as stale instead of silently becoming the
answer to the next frame.

The device is resolved and *validated* before the model loads, and the result
is reported in the handshake. Ultralytics accepts a nonexistent device index
without complaint and quietly runs somewhere else, which made a misconfigured
host look perfectly healthy.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

PROTOCOL_VERSION = 2


def _read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            return None
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_message(stream, payload):
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    stream.write(struct.pack(">I", len(encoded)))
    stream.write(encoded)
    stream.flush()


def resolve_device(requested, available=None, accelerator="cuda"):
    """Map a requested device onto a validated Ultralytics device string.

    ``available`` is the accelerator device count (``torch.cuda.device_count()``
    covers ROCm too — ROCm builds report AMD GPUs through the cuda API). It is
    injected so the rule is testable without a GPU.

    Raises ValueError for anything that does not exist. Silence here is what
    let ``device='9'`` return normal-looking detections from device 0.
    """
    requested = "auto" if requested is None else str(requested).strip()
    available = 0 if available is None else int(available)
    lowered = requested.lower()

    if lowered in ("", "auto"):
        return "0" if available > 0 else "cpu"
    if lowered == "cpu":
        return "cpu"

    index_text = lowered.split(":", 1)[1] if lowered.startswith(f"{accelerator}:") else lowered
    if not index_text.isdigit():
        raise ValueError(
            f"unsupported device {requested!r}: use 'auto', 'cpu', an index "
            f"like '0', or '{accelerator}:0'"
        )
    index = int(index_text)
    if available <= 0:
        raise ValueError(
            f"device {requested!r} was requested but this host has no "
            f"{accelerator} device — set ACESVISION_YOLO_DEVICE=cpu (or 'auto')"
        )
    if index >= available:
        raise ValueError(
            f"device {requested!r} does not exist — this host has {available} "
            f"{accelerator} device(s), valid indices 0..{available - 1}"
        )
    return str(index)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)

    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO

    try:
        device = resolve_device(
            args.device,
            available=torch.cuda.device_count() if torch.cuda.is_available() else 0,
        )
    except ValueError as exc:
        _write_message(protocol_out, {"seq": 0, "ready": False, "error": str(exc),
                                      "protocol": PROTOCOL_VERSION})
        return 2

    model = YOLO(args.model)
    model_id = f"ultralytics:{Path(args.model).stem}"
    _write_message(protocol_out, {
        "seq": 0, "ready": True, "device": device, "model": model_id,
        "protocol": PROTOCOL_VERSION,
    })

    protocol_in = sys.stdin.buffer
    while True:
        header = _read_exact(protocol_in, 8)
        if header is None:
            break
        sequence, size = struct.unpack(">II", header)
        if size > 32 * 1024 * 1024:
            raise ValueError("frame payload exceeds 32 MiB")
        encoded = _read_exact(protocol_in, size)
        if encoded is None:
            break
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            result = {"seq": sequence, "error": "invalid JPEG", "objects": []}
        else:
            tracked = model.track(
                frame, persist=True, tracker=args.tracker, device=device,
                verbose=False,
            )[0]
            objects = []
            boxes = tracked.boxes
            if boxes is not None:
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else []
                xyxy = boxes.xyxy.int().cpu().tolist()
                confs = boxes.conf.cpu().tolist()
                classes = boxes.cls.int().cpu().tolist()
                for index, (coords, score, class_id) in enumerate(
                        zip(xyxy, confs, classes)):
                    x1, y1, x2, y2 = coords
                    objects.append({
                        "x": x1, "y": y1, "w": max(0, x2 - x1),
                        "h": max(0, y2 - y1),
                        "label": tracked.names[int(class_id)],
                        "score": float(score),
                        "track_id": ids[index] if index < len(ids) else None,
                    })
            result = {
                "seq": sequence,
                "objects": objects,
                "timings": {key: float(value) for key, value in tracked.speed.items()},
                "model": model_id,
            }
        _write_message(protocol_out, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
