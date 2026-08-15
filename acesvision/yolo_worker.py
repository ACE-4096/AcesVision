"""Binary stdio worker for Ultralytics YOLO tracking on the ROCm environment."""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path


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


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="0")
    args = parser.parse_args(argv)

    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(args.model)
    protocol_in = sys.stdin.buffer
    while True:
        header = _read_exact(protocol_in, 4)
        if header is None:
            break
        size = struct.unpack(">I", header)[0]
        if size > 32 * 1024 * 1024:
            raise ValueError("frame payload exceeds 32 MiB")
        encoded = _read_exact(protocol_in, size)
        if encoded is None:
            break
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            result = {"error": "invalid JPEG", "objects": []}
        else:
            tracked = model.track(
                frame, persist=True, tracker=args.tracker, device=args.device,
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
                "objects": objects,
                "timings": {key: float(value) for key, value in tracked.speed.items()},
                "model": f"ultralytics:{Path(args.model).stem}",
            }
        payload = json.dumps(result, separators=(",", ":")).encode()
        protocol_out.write(struct.pack(">I", len(payload)))
        protocol_out.write(payload)
        protocol_out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
