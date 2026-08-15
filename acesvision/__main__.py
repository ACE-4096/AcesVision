"""Headless runner. No Qt anywhere on this path — including the connector.

Rules are evaluated and dispatched here exactly as they are in the GUI, through
the same RuleEngine and the same connector registry. Each rule's own ``dry_run``
decides whether anything actually happens; the default is True, so a rule file
written before per-rule arming existed produces dry-run decisions only.
"""
from __future__ import annotations

import argparse
import json
import os

from .connectors import default_registry
from .contracts import SourceSpec
from .events import GestureEventOutput
from .policy import RuleEngine, RuleStore
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


def build_parser():
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
    parser.add_argument("--no-events", action="store_true",
                        help="suppress gesture events (they are on by default)")
    parser.add_argument("--hold-frames", type=int, default=6,
                        help="frames a gesture must persist before it fires")
    parser.add_argument("--cooldown-s", type=float, default=1.5,
                        help="minimum seconds between gesture events")
    parser.add_argument("--no-rules", action="store_true",
                        help="observe gestures without evaluating saved rules")
    return parser


def build_rule_engine(args, store=None, executor=None):
    """RuleEngine over the saved rules, wired to the connector registry.

    Non-strict load: a rule that cannot be repaired is quarantined and named
    rather than blanking the set. Returns ``(engine, rejected)``.
    """
    if args.no_rules:
        return RuleEngine([]), []
    store = store or RuleStore()
    rules = store.load(strict=False)
    executor = default_registry() if executor is None else executor
    return RuleEngine(rules, executor=executor), list(store.rejected)


def build_gesture_output(args, callback=None, engine=None):
    """Gesture output for the headless runner — enabled unless --no-events.

    This runner is the documented headless entry point (README.md:20-24) and it
    used to emit nothing, ever: GestureEventOutput defaults to disabled and
    nothing here turned it on.
    """
    return GestureEventOutput(
        callback or _printing_callback(engine),
        hold_frames=args.hold_frames,
        cooldown_s=args.cooldown_s,
        enabled=not args.no_events,
    )


def _printing_callback(engine=None):
    """Print the event, then every decision it produced. Failures are loud."""
    def emit(event):
        print("[gesture] " + json.dumps(event, sort_keys=True))
        for decision in (engine.evaluate(event) if engine else []):
            marker = "!!" if decision.outcome == "failed" else "  "
            print(f"[decision]{marker} {decision.describe()}")
    return emit


def main():
    args = build_parser().parse_args()

    source = source_from_args(args)
    latest = LatestFrameOutput(MINIMAL)
    engine, rejected = build_rule_engine(args)
    gestures = build_gesture_output(args, engine=engine)
    outputs = [latest, gestures]
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
    print(f"[events] {'enabled' if gestures.enabled else 'disabled'} "
          f"(hold {gestures.hold_frames} frames, cooldown {gestures.cooldown_s:g}s)")
    armed = [rule for rule in engine.rules if not rule.dry_run]
    print(f"[rules] {len(engine.rules)} loaded, {len(armed)} armed "
          f"({', '.join(f'{r.connector}.{r.action}' for r in armed) or 'none'})")
    for raw, reason in rejected:
        print(f"[rules] skipped {raw.get('gesture', '?')!r}: {reason}")
    print(f"[connectors] {', '.join(default_registry().names())}")
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
