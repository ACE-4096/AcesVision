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
from .discovery import scan_droidcam, scan_plan
from .emitter import SCHEMA_GESTURE, EventBus, GestureEmitter, PublishFilter
from .events import GestureEventOutput
from .policy import RuleEngine, RuleStore
from .outputs import LatestFrameOutput, ObsVirtualCameraOutput
from .overlay import BROADCAST, MINIMAL
from .pipeline import VisionPipeline
from .server import (
    BindRefused,
    VisionServer,
    is_loopback,
    load_or_create_token,
    resolve_bind,
)
from .processor import FaceGestureProcessor
from .perception import default_device, default_model, YoloSubprocessDetector


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
    # --preview-port is the old spelling. The server is no longer only a
    # preview, but a flag in somebody's shell history should not stop working.
    parser.add_argument("--port", "--preview-port", dest="port", type=int,
                        default=8765, help="HTTP port for the preview and the "
                                           "event API")
    parser.add_argument("--bind", default="127.0.0.1",
                        help="address to bind (default 127.0.0.1; anything "
                             "else needs --allow-remote)")
    parser.add_argument("--allow-remote", action="store_true",
                        help="permit a non-loopback --bind. Identity "
                             "publishing is forced off when you do.")
    parser.add_argument("--no-emit", action="store_true",
                        help="serve the event API but publish nothing to it")
    parser.add_argument("--print-token", action="store_true",
                        help="print the emitter token and the subscriber URL, "
                             "then exit")
    parser.add_argument("--list-networks", action="store_true",
                        help="print which interfaces DroidCam discovery would "
                             "scan, and why every other one was skipped, then "
                             "exit. Sends no packets.")
    parser.add_argument("--scan-droidcam", action="store_true",
                        help="print the scan plan, run the bounded DroidCam "
                             "scan on it, list what answered, then exit")
    parser.add_argument("--detect-every", type=int,
                        default=int(os.environ.get("FACE_ID_DETECT_EVERY", "1")))
    parser.add_argument("--model", default=str(default_model()))
    parser.add_argument("--device", default=default_device(),
                        help="YOLO device: 'auto', 'cpu', or an index like '0' "
                             "(default from ACESVISION_YOLO_DEVICE). A device "
                             "that does not exist is refused at startup.")
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


def build_gesture_output(args, callback=None, engine=None, emitter=None):
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
        emitter=emitter,
    )


def build_emitter(args, bus=None, publish_filter=None):
    """The event emitter, and the bind address it is allowed to serve on.

    Returns ``(emitter, host)``. A non-loopback bind is refused outright unless
    ``--allow-remote``, and when it is allowed, identity publishing is forced
    off: naming an enrolled person to a LAN socket is a different decision from
    naming them to another process on the same machine, and a config-file
    default does not get to make it.
    """
    host = resolve_bind(args.bind, args.allow_remote)
    publish_filter = (PublishFilter.load() if publish_filter is None
                      else publish_filter)
    if not is_loopback(host):
        publish_filter = publish_filter.without_identity()
    emitter = GestureEmitter(bus if bus is not None else EventBus(),
                             publish_filter=publish_filter,
                             publishing=not args.no_emit)
    return emitter, host


def report_networks(scan=False, planner=scan_plan, scanner=scan_droidcam,
                    write=print):
    """Show what DroidCam discovery would scan, and optionally scan it.

    The plan is printed either way. A program that opens sockets on somebody's
    home network owes them a way to see which network, from a terminal, without
    starting a GUI and without sending a packet first.

    Returns the process exit code.
    """
    try:
        plan = planner()
    except ValueError as exc:           # a refused or malformed override
        write(f"[networks] {exc}")
        return 2
    write("[networks] " + plan.describe().replace("\n", "\n[networks] "))
    if not plan.networks:
        write("[networks] nothing to scan — no interface on this host "
              "qualifies. Name one explicitly to override.")
        return 1
    if not scan:
        return 0
    devices = scanner(plan.networks)
    for device in devices:
        write(f"[droidcam] {device.label}  ->  {device.url}")
    if not devices:
        write("[droidcam] scan finished, nothing answered on port 4747")
    return 0


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

    if args.list_networks or args.scan_droidcam:
        # Answered before anything else starts: no camera, no server, no token.
        raise SystemExit(report_networks(scan=args.scan_droidcam))

    token, created = load_or_create_token()
    if args.print_token:
        # Opt-in on purpose. Printing a bearer token on every start would put it
        # in every log and scrollback that ever captured this process.
        print(token)
        print(f"http://127.0.0.1:{args.port}/api/events?token={token}")
        return

    try:
        emitter, host = build_emitter(args)
    except BindRefused as exc:
        raise SystemExit(f"[bind] {exc}")

    source = source_from_args(args)
    latest = LatestFrameOutput(MINIMAL)
    engine, rejected = build_rule_engine(args)
    gestures = build_gesture_output(args, engine=engine, emitter=emitter)
    outputs = [latest, gestures]
    if args.obs:
        outputs.append(ObsVirtualCameraOutput(BROADCAST, device=args.obs_device))

    pipeline = VisionPipeline(
        source, FaceGestureProcessor(
            detect_every=args.detect_every,
            object_detector=YoloSubprocessDetector(model=args.model,
                                                   device=args.device),
        ), outputs
    )
    server = VisionServer(latest, pipeline, host=host, port=args.port,
                          bus=emitter.bus, emitter=emitter, token=token)

    print(f"[source] {source.safe_label()}")
    print(f"[server] http://{host}:{args.port} "
          f"(token required{'; generated on first run' if created else ''} — "
          f"run with --print-token)")
    print(f"[yolo] model {args.model} on device {args.device!r}")
    print(f"[obs] {'enabled' if args.obs else 'disabled'}")
    print(f"[events] {'enabled' if gestures.enabled else 'disabled'} "
          f"(hold {gestures.hold_frames} frames, cooldown {gestures.cooldown_s:g}s)")
    print(f"[emitter] {'publishing' if emitter.publishing else 'not publishing'} "
          f"{SCHEMA_GESTURE} as instance {emitter.identity.instance} "
          f"(catalog v{emitter.catalog.version} {emitter.catalog.sha256[:12]})")
    publish_filter = emitter.filter
    print("[publish] identity "
          f"{'on' if publish_filter.publish_identity else 'off'}, "
          f"source label {'on' if publish_filter.publish_source_label else 'off'}, "
          f"min confidence {publish_filter.min_confidence:g}, gestures "
          f"{'all' if publish_filter.gestures is None else ', '.join(publish_filter.gestures) or 'none'}")
    for name, reason in publish_filter.rejected:
        print(f"[publish] ignoring allowlist entry {name!r}: {reason}")
    armed = [rule for rule in engine.rules if not rule.dry_run]
    print(f"[rules] {len(engine.rules)} loaded, {len(armed)} armed "
          f"({', '.join(f'{r.connector}.{r.action}' for r in armed) or 'none'})")
    for raw, reason in rejected:
        print(f"[rules] skipped {raw.get('gesture', '?')!r}: {reason}")
    print(f"[connectors] {', '.join(default_registry().names())}")
    pipeline.start()
    server.start()
    try:
        while pipeline.is_alive():
            pipeline.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[stop]")
    finally:
        pipeline.stop()
        pipeline.join(timeout=5.0)
        server.stop()


if __name__ == "__main__":
    main()
