"""Headless Phase 1 runner. Gesture actions remain dry-run."""
from __future__ import annotations

import argparse
import json
import os

from .contracts import SourceSpec
from .events import GestureEventOutput
from .outputs import LatestFrameOutput, ObsVirtualCameraOutput
from .overlay import BROADCAST, MINIMAL
from .pipeline import VisionPipeline
from .preview import PreviewServer
from .processor import FaceGestureProcessor
from .perception import DEFAULT_MODEL, YoloSubprocessDetector


def source_from_args(args):
    if args.source == "webcam":
        return SourceSpec.from_mapping({
            "id": "webcam", "name": "Webcam", "type": "webcam",
            "index": args.camera,
        })
    if not args.url:
        raise SystemExit(f"--url is required for --source {args.source}")
    return SourceSpec.from_mapping({
        "id": args.source, "name": args.source.title(),
        "type": args.source, "url": args.url,
    })


def main():
    parser = argparse.ArgumentParser(description="AcesVision local runtime")
    parser.add_argument("--source", choices=["webcam", "droidcam", "network"],
                        default="webcam")
    parser.add_argument("--camera", type=int, default=None)
    parser.add_argument("--url")
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument("--detect-every", type=int,
                        default=int(os.environ.get("FACE_ID_DETECT_EVERY", "1")))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--obs", action="store_true")
    parser.add_argument("--obs-device", default=os.environ.get("FACE_ID_VCAM"))
    args = parser.parse_args()

    source = source_from_args(args)
    latest = LatestFrameOutput(MINIMAL)
    outputs = [latest, GestureEventOutput(
        lambda event: print("[gesture dry-run] " + json.dumps(event, sort_keys=True))
    )]
    if args.obs:
        outputs.append(ObsVirtualCameraOutput(BROADCAST, device=args.obs_device))

    pipeline = VisionPipeline(
        source, FaceGestureProcessor(
            detect_every=args.detect_every,
            object_detector=YoloSubprocessDetector(model=args.model),
        ), outputs
    )
    preview = PreviewServer(latest, pipeline, port=args.preview_port)

    print(f"[source] {source.safe_label()}")
    print(f"[preview] http://127.0.0.1:{args.preview_port}")
    print(f"[obs] {'enabled' if args.obs else 'disabled'}")
    print("[actions] dry-run only")
    pipeline.start()
    preview.start()
    try:
        while pipeline.is_alive():
            pipeline.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[stop]")
    finally:
        pipeline.stop()
        pipeline.join(timeout=5.0)
        preview.stop()


if __name__ == "__main__":
    main()
