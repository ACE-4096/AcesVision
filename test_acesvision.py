import errno
import hashlib
import io
import hmac
import http.client
import inspect
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import stat
import struct
import sys
import tempfile
import threading
import time
import unittest
import uuid
from collections import namedtuple
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import camera
import gesture_catalog
import verify_gestures_live as verify
from acesvision import connectors
from acesvision.catalog import (
    CATALOG,
    CatalogError,
    GestureCatalog,
    canonical_json,
    catalog_sha256,
    load_catalog,
)
from acesvision.contracts import SceneFrame, SourceSpec
from acesvision import server as server_module
from acesvision.emitter import (
    SCHEMA_GESTURE,
    SCHEMA_HELLO,
    EmitterIdentity,
    EventBus,
    GestureEmitter,
    PublishFilter,
    Subscription,
    TooManySubscribers,
    harden_directory,
    identity_state,
    write_secret_file,
)
from acesvision.server import (
    BindRefused,
    VisionServer,
    host_header_ok,
    is_loopback,
    keepalive_frame,
    load_or_create_token,
    parse_since,
    presented_token,
    resolve_bind,
    sse_frame,
    token_matches,
)
from acesvision.discovery import (
    DEFAULT_PROBE_TIMEOUT_S,
    DROIDCAM_PORT,
    SCAN_INTERFACES_ENV,
    SCAN_NETWORKS_ENV,
    DroidCamDevice,
    NetworkInterface,
    NoScannableNetwork,
    ScanPlan,
    WebcamDevice,
    _stable_v4l_path,
    discover_webcams,
    interface_exclusion_reason,
    local_scan_networks,
    parse_scan_networks,
    preferred_webcam,
    read_interface_table,
    scan_droidcam,
    scan_plan,
)
from acesvision.outputs import (
    CallbackOutput,
    LatestFrameOutput,
    ObsVirtualCameraOutput,
)
from acesvision.overlay import CLEAN, MINIMAL, OverlayProfile, render
from acesvision.smoothing import (
    FADE_IN_S,
    FADE_OUT_S,
    SceneSmoother,
    SmoothedItem,
    ease_factor,
    fade_out_s,
)
from acesvision.pipeline import (
    DROP_OLD_JOIN_S,
    LOSSLESS_QUEUE_FRAMES,
    PipelineState,
    VisionPipeline,
    _OutputWorker,
)
from acesvision.recording import (
    DEFAULT_FPS,
    RECORDINGS_ENV,
    REPO_ROOT,
    SIDECAR_SCHEMA,
    RecordingError,
    RecordingOutput,
    hardware_command,
    recording_name,
    recordings_dir,
    refuse_repo_path,
    resolve_path,
    software_command,
)
from acesvision import perception
from acesvision.events import (
    GestureEventOutput,
    attribute_actor,
    select_gesture,
)
from acesvision.perception import (
    Detection,
    default_device,
    default_worker_python,
    WorkerDeviceError,
    YoloSubprocessDetector,
    file_sha256,
    rocm_env_overrides,
)
from acesvision.yolo_worker import resolve_device
from acesvision.policy import (
    CONNECTORS,
    Rule,
    RuleEngine,
    RuleStore,
    validate_actor,
    validate_gesture,
)
from acesvision.processor import FaceGestureProcessor
from acesvision.sources import LatestFrameReader, open_source

Face = namedtuple("Face", "x y w h name conf known")
Gesture = namedtuple("Gesture", "name score x y w h")


class FakeCapture:
    def __init__(self, reads, opened=True):
        self.reads = list(reads)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened and not self.released

    def read(self):
        if self.reads:
            return self.reads.pop(0)
        return False, None

    def release(self):
        self.released = True

    def set(self, *_):
        return True


class SourceTests(unittest.TestCase):
    def test_webcam_index(self):
        source = SourceSpec.from_mapping({
            "id": "monitor", "name": "Monitor", "type": "webcam", "index": 9,
        })
        self.assertEqual((source.kind, source.index), ("webcam", 9))

    def test_droidcam_host_uses_standard_url(self):
        source = SourceSpec.from_mapping({
            "id": "phone", "type": "droidcam", "host": "192.168.1.20",
        })
        self.assertEqual(source.url, "http://192.168.1.20:4747/video")
        self.assertFalse(source.trusted_device)

    def test_safe_label_redacts_credentials_and_query(self):
        source = SourceSpec.from_mapping({
            "type": "droidcam",
            "url": "http://user:secret@phone.local:4747/video?token=hidden",
        })
        label = source.safe_label()
        self.assertNotIn("secret", label)
        self.assertNotIn("token", label)
        self.assertIn("phone.local:4747/video", label)

    def test_failed_network_capture_is_released(self):
        capture = FakeCapture([], opened=False)
        factory = Mock(return_value=capture)
        source = SourceSpec.from_mapping({
            "type": "droidcam", "url": "http://phone:4747/video",
        })
        self.assertIsNone(open_source(source, capture_factory=factory))
        self.assertTrue(capture.released)


class DiscoveryTests(unittest.TestCase):
    def test_webcam_inventory_hides_metadata_and_labels_ir(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number, interface_index, name in (
                (5, 0, "USB FHD Webcam"),
                (7, 1, "USB FHD Webcam"),
                (8, 0, "USB FHD Webcam: IR Camer"),
            ):
                node = root / f"video{number}"
                node.mkdir()
                (node / "index").write_text(str(interface_index))
                (node / "name").write_text(name)
            devices = discover_webcams(root)
        self.assertEqual([device.index for device in devices], [5, 8])
        self.assertEqual([device.kind for device in devices], ["colour", "infrared"])
        self.assertIn("Camera 5", devices[0].label)
        self.assertIn("/dev/video5", devices[0].label)

    def test_preferred_webcam_chooses_colour_not_ir_or_virtual(self):
        devices = [
            WebcamDevice(3, "OBS", "/dev/video3", "virtual", "OBS"),
            WebcamDevice(8, "IR", "/dev/video8", "infrared", "IR"),
            WebcamDevice(5, "Webcam", "/dev/video5", "colour", "Webcam"),
        ]
        self.assertEqual(preferred_webcam(devices).index, 5)

    def test_stable_v4l_path_survives_transient_index_selection(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            devices = root / "dev"
            links = root / "by-path"
            devices.mkdir()
            links.mkdir()
            node = devices / "video8"
            node.touch()
            link = links / "usb-port-video-index0"
            link.symlink_to(node)
            selected = _stable_v4l_path(node, roots=(links,))
        self.assertEqual(selected, str(link))

    def test_scan_networks_only_include_private_non_loopback_addresses(self):
        networks = local_scan_networks(["192.168.68.21", "127.0.0.1", "8.8.8.8"])
        self.assertEqual([str(network) for network in networks], ["192.168.68.0/24"])

    def test_droidcam_scan_is_bounded_and_returns_standard_url(self):
        class Connection:
            def close(self):
                pass

        attempts = []

        def connector(address, timeout):
            attempts.append((address, timeout))
            if address[0] == "192.168.68.2":
                return Connection()
            raise ConnectionRefusedError

        found = scan_droidcam(["192.168.68.0/30"], connector=connector,
                              max_workers=2)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(found[0].url, "http://192.168.68.2:4747/video")

    def test_droidcam_scan_refuses_broad_subnets(self):
        connector = Mock()
        self.assertEqual(scan_droidcam(["192.168.0.0/16"], connector=connector), [])
        connector.assert_not_called()

    def test_droidcam_scan_stops_at_its_deadline(self):
        """A blocked probe must not hold the GUI's scan open forever."""
        release = threading.Event()
        self.addCleanup(release.set)

        def connector(address, timeout):
            release.wait(30)
            raise ConnectionRefusedError

        started = time.monotonic()
        found = scan_droidcam(["192.168.68.0/29"], connector=connector,
                              max_workers=2, deadline_s=0.1)
        self.assertEqual(found, [])
        self.assertLess(time.monotonic() - started, 5.0)


# The interface table of the development host, verbatim, as /sys/class/net and
# SIOCGIFADDR report it: one real LAN, two VPNs, two libvirt guest networks and
# two docker networks. Auto-discovery must pick exactly one of the seven.
# Scanning any of the other six would mean port-scanning VPN peers, virtual
# machines and containers that never asked to be probed.
HOST_INTERFACES = (
    # name, ARP type, flags, operstate, physical, wireless, address, netmask
    ("br-1f526b924fe3", 1, 0x1003, "up", False, False,
     "172.18.0.1", "255.255.0.0"),
    ("docker0", 1, 0x1003, "down", False, False, "172.17.0.1", "255.255.0.0"),
    ("enp8s0", 1, 0x1003, "up", True, False, "192.168.68.107", "255.255.255.0"),
    ("tailscale0", 65534, 0x1091, "unknown", False, False,
     "100.114.53.112", "255.255.255.255"),
    ("tun0", 65534, 0x1091, "unknown", False, False, "10.8.0.1", "255.255.255.0"),
    ("virbr-acesops", 1, 0x1003, "up", False, False,
     "10.171.71.1", "255.255.255.0"),
    ("virbr0", 1, 0x1003, "up", False, False, "192.168.122.1", "255.255.255.0"),
)


class InterfaceAcquisitionTests(unittest.TestCase):
    """The half of discovery that talks to the OS, which had no coverage at all.

    Every earlier test injected ``addresses=`` and so exercised the filtering
    rules only. The acquisition path — the part that decides *which* addresses
    those are — was ``getaddrinfo(gethostname())``, returned 127.0.1.1 on any
    Debian or Ubuntu host, and therefore found nothing, ever. These tests run
    that half against a fake interface table instead of a real kernel.
    """

    def fake_net(self, entries=HOST_INTERFACES):
        """A /sys/class/net tree plus the matching address lookup."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        # The hardware target lives outside the tree: /sys/class/net holds
        # interfaces and nothing else.
        root = Path(directory.name) / "net"
        root.mkdir()
        hardware = Path(directory.name) / "pci0000:00"
        hardware.mkdir()
        addresses = {}
        for (name, arp, flags, operstate, physical, wireless,
             address, netmask) in entries:
            node = root / name
            node.mkdir()
            (node / "type").write_text(f"{arp}\n")
            (node / "flags").write_text(f"{flags:#x}\n")
            (node / "operstate").write_text(f"{operstate}\n")
            if physical:
                # The kernel puts a symlink to the backing PCI/USB device here.
                # Bridges, tunnels and veth pairs have none.
                (node / "device").symlink_to(hardware)
            if wireless:
                (node / "wireless").mkdir()
            addresses[name] = (address, netmask)
        return root, (lambda name: addresses.get(name, (None, None)))

    def plan(self, entries=HOST_INTERFACES, env=None):
        root, lookup = self.fake_net(entries)
        return scan_plan(sysfs=root, address_lookup=lookup,
                         env={} if env is None else env)

    def test_only_the_real_lan_is_selected_from_a_real_interface_table(self):
        plan = self.plan()
        self.assertEqual([interface.name for interface in plan.selected],
                         ["enp8s0"])
        self.assertEqual([str(network) for network in plan.networks],
                         ["192.168.68.0/24"])

    def test_every_vpn_container_and_hypervisor_network_is_skipped_with_a_reason(self):
        plan = self.plan()
        skipped = dict(plan.excluded)
        self.assertEqual(sorted(skipped), [
            "br-1f526b924fe3", "docker0", "tailscale0", "tun0",
            "virbr-acesops", "virbr0",
        ])
        self.assertIn("container network", skipped["docker0"])
        self.assertIn("container network", skipped["br-1f526b924fe3"])
        self.assertIn("VPN overlay", skipped["tailscale0"])
        self.assertIn("tunnel", skipped["tun0"])
        self.assertIn("hypervisor guest network", skipped["virbr0"])
        self.assertIn("hypervisor guest network", skipped["virbr-acesops"])

    def test_local_scan_networks_asks_the_interfaces_not_the_hostname(self):
        """The reproducer. /etc/hosts maps the hostname to 127.0.1.1 here."""
        root, lookup = self.fake_net()
        with patch("acesvision.discovery.socket.getaddrinfo",
                   side_effect=AssertionError("hostname resolution consulted")), \
                patch("acesvision.discovery.socket.gethostname",
                      side_effect=AssertionError("hostname consulted")):
            networks = local_scan_networks(sysfs=root, address_lookup=lookup,
                                           env={})
        self.assertEqual([str(network) for network in networks],
                         ["192.168.68.0/24"])

    def test_a_wireless_interface_is_selected(self):
        wireless = (("wlp3s0", 1, 0x1003, "up", True, True,
                     "192.168.1.50", "255.255.255.0"),)
        plan = self.plan(HOST_INTERFACES + wireless)
        self.assertEqual([interface.name for interface in plan.selected],
                         ["enp8s0", "wlp3s0"])
        self.assertEqual([interface.kind for interface in plan.selected],
                         ["ethernet", "wireless"])
        self.assertIn(ipaddress.ip_network("192.168.1.0/24"), plan.networks)

    def test_a_wide_prefix_is_capped_to_a_24_never_swept(self):
        wide = (("eno1", 1, 0x1003, "up", True, False,
                 "172.20.5.9", "255.255.0.0"),)
        plan = self.plan(wide)
        self.assertEqual([str(network) for network in plan.networks],
                         ["172.20.5.0/24"])
        self.assertEqual(plan.networks[0].num_addresses, 256)

    def test_a_prefix_narrower_than_a_24_is_honoured_as_it_stands(self):
        narrow = (("eno1", 1, 0x1003, "up", True, False,
                   "192.168.9.70", "255.255.255.192"),)
        plan = self.plan(narrow)
        self.assertEqual([str(network) for network in plan.networks],
                         ["192.168.9.64/26"])

    def test_a_public_address_on_a_physical_interface_is_never_returned(self):
        public = (("eno1", 1, 0x1003, "up", True, False,
                   "93.184.216.34", "255.255.255.0"),)
        plan = self.plan(public)
        self.assertEqual(plan.networks, ())
        self.assertEqual(dict(plan.excluded)["eno1"], "public address")

    def test_a_downed_interface_is_not_scanned(self):
        unplugged = (("eno1", 1, 0x1003, "down", True, False,
                      "192.168.4.4", "255.255.255.0"),)
        plan = self.plan(unplugged)
        self.assertEqual(plan.networks, ())
        self.assertIn("down", dict(plan.excluded)["eno1"])

    def test_finding_nothing_fails_loudly_instead_of_returning_empty(self):
        """The original failure mode, now an exception with the reasons in it.

        Silently returning [] made a host that cannot be scanned look exactly
        like a network with no phone on it. They are different answers.
        """
        root, lookup = self.fake_net(
            [entry for entry in HOST_INTERFACES if entry[0] != "enp8s0"])
        with self.assertRaises(NoScannableNetwork) as caught:
            local_scan_networks(sysfs=root, address_lookup=lookup, env={})
        message = str(caught.exception)
        self.assertIn("docker0", message)
        self.assertIn("tailscale0", message)
        self.assertIn(SCAN_INTERFACES_ENV, message)

    def test_the_scan_plan_names_what_it_will_touch_and_what_it_skipped(self):
        report = self.plan().describe()
        self.assertIn("192.168.68.0/24", report)
        for name, *_ in HOST_INTERFACES:
            self.assertIn(name, report)
        self.assertIn(SCAN_NETWORKS_ENV, report)

    def test_discovery_does_not_depend_on_hostname_resolution_on_this_host(self):
        """The original bug, reproduced against the real machine.

        On Debian and Ubuntu /etc/hosts maps the hostname to 127.0.1.1. The old
        acquisition path asked ``getaddrinfo(gethostname())``, got exactly that,
        discarded it as loopback and answered [] — on every such host, forever.
        Rig hostname resolution to give that same answer and discovery must
        still find the LAN this machine is actually plugged into.
        """
        if not any(interface_exclusion_reason(interface) is None
                   for interface in read_interface_table()):
            self.skipTest("this machine has no scannable LAN interface")
        loopback = [(2, 1, 6, "", ("127.0.1.1", 0))]
        with patch("acesvision.discovery.socket.getaddrinfo",
                   return_value=loopback), \
                patch("acesvision.discovery.socket.gethostname",
                      return_value="aced"):
            networks = local_scan_networks(env={})
        self.assertTrue(networks, "discovery found no network on a host that "
                                  "has one")
        self.assertTrue(all(network.is_private for network in networks))
        self.assertTrue(all(network.prefixlen >= 24 for network in networks))

    def test_the_real_interface_table_of_this_host_is_readable(self):
        """One test that does touch the real kernel — read only, no packets."""
        interfaces = {interface.name: interface
                      for interface in read_interface_table()}
        if not interfaces:                  # not Linux, or no /sys
            self.skipTest("no /sys/class/net on this machine")
        self.assertIn("lo", interfaces)
        self.assertEqual(interfaces["lo"].kind, "loopback")
        self.assertEqual(interface_exclusion_reason(interfaces["lo"]),
                         "loopback (name matches lo)")


class ScanOverrideTests(unittest.TestCase):
    """The operator's override. Silent scanning of the wrong network is worse
    than finding nothing, so the choice has to be theirs to take."""

    def interfaces(self):
        return [
            NetworkInterface("enp8s0", "192.168.68.107", "255.255.255.0",
                             kind="ethernet", is_up=True, is_physical=True),
            NetworkInterface("tun0", "10.8.0.1", "255.255.255.0",
                             kind="tunnel", is_up=True, is_point_to_point=True),
        ]

    def test_named_interfaces_win_over_auto_detection(self):
        plan = scan_plan(interfaces=self.interfaces(),
                         env={SCAN_INTERFACES_ENV: "tun0"})
        self.assertEqual([interface.name for interface in plan.selected], ["tun0"])
        self.assertEqual([str(network) for network in plan.networks],
                         ["10.8.0.0/24"])
        self.assertIn(SCAN_INTERFACES_ENV, dict(plan.excluded)["enp8s0"])

    def test_named_networks_win_over_the_interfaces_entirely(self):
        def explode(name):
            raise AssertionError("interfaces enumerated despite an override")

        plan = scan_plan(sysfs=Path("/nonexistent"), address_lookup=explode,
                         env={SCAN_NETWORKS_ENV: "192.168.5.0/24, 10.3.4.0/24",
                              SCAN_INTERFACES_ENV: "enp8s0"})
        self.assertEqual([str(network) for network in plan.networks],
                         ["10.3.4.0/24", "192.168.5.0/24"])
        self.assertEqual(plan.origin, SCAN_NETWORKS_ENV)

    def test_an_override_still_cannot_sweep_a_16(self):
        with self.assertRaises(ValueError) as caught:
            parse_scan_networks("192.168.0.0/16")
        self.assertIn("/24", str(caught.exception))

    def test_an_override_still_cannot_reach_a_public_network(self):
        with self.assertRaises(ValueError) as caught:
            parse_scan_networks("93.184.216.0/24")
        self.assertIn("private", str(caught.exception))

    def test_an_override_naming_no_such_interface_is_refused_not_ignored(self):
        with self.assertRaises(ValueError) as caught:
            scan_plan(interfaces=self.interfaces(),
                      env={SCAN_INTERFACES_ENV: "wlan9"})
        self.assertIn("wlan9", str(caught.exception))

    def test_a_named_interface_with_a_public_address_is_still_refused(self):
        public = [NetworkInterface("enp8s0", "93.184.216.34", "255.255.255.0",
                                   kind="ethernet", is_up=True, is_physical=True)]
        plan = scan_plan(interfaces=public, env={SCAN_INTERFACES_ENV: "enp8s0"})
        self.assertEqual(plan.networks, ())
        self.assertEqual(dict(plan.excluded)["enp8s0"], "public address")


class NetworkReportCliTests(unittest.TestCase):
    """`--list-networks` — the scan is inspectable from a terminal, no GUI."""

    def plan(self):
        return ScanPlan(
            networks=(ipaddress.ip_network("192.168.68.0/24"),),
            selected=(NetworkInterface("enp8s0", "192.168.68.107",
                                       "255.255.255.0", kind="ethernet",
                                       is_up=True, is_physical=True),),
            excluded=(("docker0", "container network (name matches docker*)"),),
        )

    def report(self, **kwargs):
        from acesvision.__main__ import report_networks

        lines = []
        kwargs.setdefault("planner", self.plan)
        kwargs.setdefault("scanner", Mock(side_effect=AssertionError("scanned")))
        code = report_networks(write=lines.append, **kwargs)
        return code, "\n".join(lines)

    def test_the_flags_are_accepted(self):
        from acesvision.__main__ import build_parser

        args = build_parser().parse_args(["--list-networks"])
        self.assertTrue(args.list_networks)
        self.assertFalse(build_parser().parse_args([]).scan_droidcam)

    def test_listing_prints_the_plan_and_sends_nothing(self):
        code, text = self.report()
        self.assertEqual(code, 0)
        self.assertIn("192.168.68.0/24", text)
        self.assertIn("enp8s0", text)
        self.assertIn("docker0", text)
        self.assertIn(SCAN_INTERFACES_ENV, text)

    def test_scanning_probes_only_the_planned_networks(self):
        scanned = []

        def scanner(networks, port=DROIDCAM_PORT,
                    timeout_s=DEFAULT_PROBE_TIMEOUT_S):
            scanned.append(list(networks))
            return [DroidCamDevice("192.168.68.31", 4747,
                                   "http://192.168.68.31:4747/video", "phone")]

        code, text = self.report(scan=True, scanner=scanner)
        self.assertEqual(code, 0)
        self.assertEqual([str(network) for network in scanned[0]],
                         ["192.168.68.0/24"])
        self.assertIn("http://192.168.68.31:4747/video", text)

    def test_the_probe_budget_reaches_the_scanner(self):
        """--scan-timeout and --scan-port are settings, not decoration.

        The scan that lost the phone was a scan whose probe budget nothing
        could reach. A flag that parses but never arrives is the same bug.
        """
        seen = {}

        def scanner(networks, port, timeout_s):
            seen.update(port=port, timeout_s=timeout_s)
            return []

        code, text = self.report(scan=True, scanner=scanner, timeout_s=2.5,
                                 port=4848)
        self.assertEqual(code, 0)
        self.assertEqual(seen, {"port": 4848, "timeout_s": 2.5})
        # An empty result names the budget it was given, so "nothing answered"
        # can be told apart from "nothing answered in time".
        self.assertIn("within 2.5s per host", text)
        self.assertIn("--scan-timeout 5", text)

    def test_the_probe_budget_flags_default_to_the_measured_values(self):
        from acesvision.__main__ import build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.scan_timeout, DEFAULT_PROBE_TIMEOUT_S)
        self.assertEqual(args.scan_port, DROIDCAM_PORT)
        args = build_parser().parse_args(["--scan-timeout", "3", "--scan-port",
                                          "5000"])
        self.assertEqual((args.scan_timeout, args.scan_port), (3.0, 5000))

    def test_nothing_to_scan_exits_nonzero_and_says_so(self):
        code, text = self.report(planner=ScanPlan)
        self.assertEqual(code, 1)
        self.assertIn("nothing to scan", text)

    def test_a_refused_override_exits_nonzero_with_the_reason(self):
        def planner():
            raise ValueError("ACESVISION_SCAN_NETWORKS: '10.0.0.0/8' is wider")

        code, text = self.report(planner=planner)
        self.assertEqual(code, 2)
        self.assertIn("wider", text)


class DroidCamCliEndToEndTests(unittest.TestCase):
    """`--scan-droidcam` from argv to stdout, over a real socket.

    Every earlier DroidCam test called ``scan_droidcam`` directly with an
    injected connector. That is the mockable half, and it has now hidden two
    faults in a row: first an acquisition step no test ever ran, then a probe
    timeout no test ever timed. Both were reported as "the Sources page finds
    nothing", and both passed 491 tests on the way out.

    So this drives the real entry point — ``main()``, real ``sys.argv``, real
    ``socket.create_connection`` — against a listener that is really
    listening. The only injected thing is *which* network to scan, because a
    test may not scan somebody's LAN, and that choice is what the scan-plan
    tests above already cover on their own.
    """

    def listener(self):
        """A real TCP listener on loopback. Returns its port."""
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        self.addCleanup(server.close)
        stop = threading.Event()
        self.addCleanup(stop.set)

        def accept_loop():
            server.settimeout(0.1)
            while not stop.is_set():
                try:
                    server.accept()[0].close()
                except (TimeoutError, OSError):
                    continue

        thread = threading.Thread(target=accept_loop, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2.0)
        return server.getsockname()[1]

    def run_cli(self, *argv):
        import contextlib
        import io

        from acesvision.__main__ import main

        plan = ScanPlan(networks=(ipaddress.ip_network("127.0.0.1/32"),),
                        selected=(NetworkInterface("test0", "127.0.0.1",
                                                   "255.255.255.255"),))
        out = io.StringIO()
        with patch("acesvision.__main__.scan_plan", lambda: plan), \
                patch.object(sys, "argv", ["acesvision", *argv]), \
                contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as exit_code:
                main()
        return exit_code.exception.code, out.getvalue()

    def test_a_listening_host_survives_the_whole_cli_path(self):
        port = self.listener()
        code, text = self.run_cli("--scan-droidcam", "--scan-port", str(port),
                                  "--scan-timeout", "2")
        self.assertEqual(code, 0)
        self.assertIn(f"http://127.0.0.1:{port}/video", text)
        self.assertNotIn("nothing answered", text)

    def test_a_silent_host_is_reported_as_silent_with_its_budget(self):
        # Port 1 on loopback: nothing is there, and the kernel refuses at once.
        code, text = self.run_cli("--scan-droidcam", "--scan-port", "1",
                                  "--scan-timeout", "2")
        self.assertEqual(code, 0)
        self.assertIn("nothing answered on port 1", text)
        self.assertIn("within 2s per host", text)

    def test_listing_alone_opens_no_socket(self):
        port = self.listener()
        code, text = self.run_cli("--list-networks", "--scan-port", str(port))
        self.assertEqual(code, 0)
        self.assertIn("127.0.0.1/32", text)
        self.assertNotIn("[droidcam]", text)


class ProbeBudgetTests(unittest.TestCase):
    """The probe timeout, sized against a real phone rather than a guess.

    Measured on the development LAN, timing how long the phone took to give a
    *definitive* TCP answer: ten single probes came back at a median of 211 ms
    and a maximum of 335 ms, and six full /24 sweeps at 99-252 ms. One of those
    sixteen measurements was inside the old 0.12 s budget. The delay is Wi-Fi
    radio power-saving, not distance and not load, so it is a property of every
    phone this program is meant to find.
    """

    def slow_connector(self, delay_s):
        """A connector with a real socket's timeout semantics.

        It answers after ``delay_s`` — unless that would overrun the timeout it
        was handed, which is precisely what a socket does and precisely what
        the old 0.12 s budget kept triggering.
        """
        def connector(address, timeout):
            if delay_s > timeout:
                raise TimeoutError("timed out")
            time.sleep(delay_s)
            return Mock(close=Mock())
        return connector

    def test_the_default_budget_tolerates_a_phone_that_is_waking_its_radio(self):
        found = scan_droidcam(["192.168.68.0/30"],
                              connector=self.slow_connector(0.335))
        self.assertEqual([device.host for device in found],
                         ["192.168.68.1", "192.168.68.2"])

    def test_the_old_budget_is_what_lost_that_phone(self):
        """The red half of the fix, kept as the reason the default moved."""
        found = scan_droidcam(["192.168.68.0/30"], timeout_s=0.12,
                              connector=self.slow_connector(0.335))
        self.assertEqual(found, [])

    def test_the_default_is_not_quietly_narrowed_again(self):
        self.assertGreaterEqual(DEFAULT_PROBE_TIMEOUT_S, 0.5)
        # Just under Linux's 1 s initial SYN retransmit: one SYN per probe.
        self.assertLessEqual(DEFAULT_PROBE_TIMEOUT_S, 1.0)


class OverlayTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.zeros((100, 120, 3), dtype=np.uint8)
        self.source = SourceSpec.from_mapping({"type": "webcam"})
        self.scene = SceneFrame(
            self.source, 1, 1.0, self.raw,
            faces=[Face(10, 20, 30, 40, "Toby", 0.2, True)],
            gestures=[Gesture("Victory", 0.9, 60, 20, 20, 30)],
        )

    def test_clean_profile_returns_unannotated_copy(self):
        output = render(self.scene, CLEAN)
        self.assertTrue(np.array_equal(output, self.raw))
        self.assertIsNot(output, self.raw)

    def test_profile_colours_are_output_specific(self):
        red = OverlayProfile(known_colour=(0, 0, 255), show_gestures=False)
        green = OverlayProfile(known_colour=(0, 255, 0), show_gestures=False)
        red_frame = render(self.scene, red)
        green_frame = render(self.scene, green)
        self.assertFalse(np.array_equal(red_frame, green_frame))
        self.assertTrue(np.array_equal(self.raw, np.zeros_like(self.raw)))


class FakeClock:
    """The injectable clock the smoother is written against, driven by hand.

    Same idiom as ``GestureEventOutput``'s ``clock``: nothing in the smoother
    reads a real clock, so every test below is exact rather than approximate.
    """

    def __init__(self, start=1000.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)
        return self.now


class SceneSmootherTests(unittest.TestCase):
    """The temporal filter that sits between the detectors and the renderer."""

    # The operating point this filter was specified against: 60 fps capture
    # against 40 fps inference, so roughly every third captured frame carries
    # a fresh detection and the two either side repeat it verbatim. That
    # repeat is the flicker. (Not a claim about any particular camera — the
    # USB webcam in the README is camera-limited well below this, which is the
    # identity case and is tested as such.)
    CAPTURE_DT = 1.0 / 60.0
    INFERENCE_FPS = 40.0

    def setUp(self):
        self.clock = FakeClock()
        self.smoother = SceneSmoother(clock=self.clock)
        self.source = SourceSpec.from_mapping({"type": "webcam"})
        self.raw = np.zeros((100, 200, 3), dtype=np.uint8)
        self.sequence = 0

    def scene(self, objects=(), faces=(), gestures=(), fps=None, **flags):
        metadata = {"inference_fps": self.INFERENCE_FPS if fps is None else fps}
        metadata.update(flags)
        self.sequence += 1
        return SceneFrame(self.source, self.sequence, float(self.sequence),
                          self.raw, objects=list(objects), faces=list(faces),
                          gestures=list(gestures), metadata=metadata)

    def step(self, objects=(), faces=(), gestures=(), dt=None, **kwargs):
        """One captured frame later, through this smoother."""
        self.clock.advance(self.CAPTURE_DT if dt is None else dt)
        return self.smoother.apply(self.scene(objects, faces, gestures, **kwargs))

    # ---- the ease ---------------------------------------------------------

    def test_the_ease_converges_on_the_new_box_and_never_passes_it(self):
        """Critically damped, so lag — never overshoot.

        On a recording there is no on-screen reference truth: a box a fraction
        of a frame behind reads as tracking, one that overshoots and snaps
        back reads as broken.
        """
        detection = Detection(10, 10, 20, 20, "person", 0.9, 1)
        self.step([detection])
        moved = detection._replace(x=110)
        travel = [self.step([moved]).objects[0].x for _ in range(12)]

        self.assertGreater(travel[0], 10.0)   # it moved,
        self.assertLess(travel[0], 110.0)     # but not the whole way at once.
        for previous, current in zip(travel, travel[1:]):
            self.assertGreaterEqual(current, previous)
            self.assertLessEqual(current, 110.0)
        self.assertAlmostEqual(travel[-1], 110.0, places=3)

    def test_inference_at_or_above_the_capture_rate_is_the_identity(self):
        """No repeated frames, nothing to smooth. Falls out of the arithmetic
        rather than out of a special case."""
        detection = Detection(10, 10, 20, 20, "person", 0.9, 1)
        for fps in (60.0, 90.0):
            with self.subTest(inference_fps=fps):
                smoother = SceneSmoother(clock=self.clock)
                smoother.apply(self.scene([detection], fps=fps))
                self.clock.advance(self.CAPTURE_DT)
                moved = detection._replace(x=110, y=55, w=44)
                item = smoother.apply(self.scene([moved], fps=fps)).objects[0]
                self.assertEqual((item.x, item.y, item.w, item.h),
                                 (110.0, 55.0, 44.0, 20.0))
                self.assertEqual(item.alpha, 1.0)

    def test_the_ease_factor_is_bounded_and_degenerates_to_identity(self):
        self.assertEqual(ease_factor(self.CAPTURE_DT, self.CAPTURE_DT), 1.0)
        self.assertEqual(ease_factor(self.CAPTURE_DT, 1.0 / 120.0), 1.0)
        # No measured inference rate yet: smooth nothing, invent nothing.
        self.assertEqual(ease_factor(self.CAPTURE_DT, 0.0), 1.0)
        # No time passed: nothing moves.
        self.assertEqual(ease_factor(0.0, 0.025), 0.0)
        for interval in (0.02, 0.025, 0.05, 0.1, 0.5):
            with self.subTest(interval=interval):
                factor = ease_factor(self.CAPTURE_DT, interval)
                self.assertGreater(factor, 0.0)
                self.assertLessEqual(factor, 1.0)
        # Slower inference, longer stale window, gentler ease.
        self.assertLess(ease_factor(self.CAPTURE_DT, 0.1),
                        ease_factor(self.CAPTURE_DT, 0.05))

    def test_the_first_frame_a_smoother_sees_is_drawn_in_full(self):
        """A still — /latest.jpg, a snapshot — must not come out half
        transparent because the stream had only just started. There is no
        mid-stream for a box to have appeared out of on frame one."""
        detection = Detection(10, 10, 20, 20, "person", 0.9, 1)
        item = self.smoother.apply(self.scene([detection])).objects[0]
        self.assertEqual(item.alpha, 1.0)
        self.assertEqual((item.x, item.y), (10.0, 10.0))

    # ---- fades ------------------------------------------------------------

    def test_a_new_track_fades_in_at_its_true_position(self):
        """Never eased in from nowhere: a box sliding in from the origin is a
        claim the detector never made."""
        self.step([])
        detection = Detection(10, 10, 20, 20, "person", 0.9, 7)
        born = self.step([detection]).objects[0]
        self.assertEqual((born.x, born.y, born.w, born.h), (10.0, 10.0, 20.0, 20.0))
        self.assertAlmostEqual(born.alpha, self.CAPTURE_DT / FADE_IN_S, places=6)
        self.assertLess(born.alpha, 1.0)

        alphas = [born.alpha]
        for _ in range(9):   # 9 more frames at 60 fps is 150 ms
            alphas.append(self.step([detection]).objects[0].alpha)
        for previous, current in zip(alphas, alphas[1:]):
            self.assertGreaterEqual(current, previous)
        self.assertEqual(alphas[-1], 1.0)

    def test_a_lost_track_is_held_and_faded_capped_at_two_inference_intervals(self):
        """A ghost must not outlive the evidence for it by more than two
        chances to be re-detected. At 10 fps that cap (0.2 s) binds before the
        0.25 s default does."""
        self.assertAlmostEqual(fade_out_s(0.1), 0.2)      # cap binds
        self.assertEqual(fade_out_s(0.5), FADE_OUT_S)     # default binds
        self.assertEqual(fade_out_s(0.0), 0.0)            # no rate, no ghost

        detection = Detection(10, 10, 20, 20, "person", 0.9, 3)
        self.step([detection], fps=10.0)
        self.assertEqual(self.step([detection], fps=10.0).objects[0].alpha, 1.0)

        scene = self.step([], fps=10.0)
        held = scene.objects[0]
        self.assertEqual((held.x, held.y), (10.0, 10.0))   # held, not moved
        self.assertLess(held.alpha, 1.0)

        elapsed = self.CAPTURE_DT
        for _ in range(200):
            if not scene.objects:
                break
            scene = self.step([], fps=10.0)
            elapsed += self.CAPTURE_DT
        self.assertEqual(scene.objects, [])
        self.assertLessEqual(elapsed, 0.2 + self.CAPTURE_DT)
        self.assertGreater(elapsed, 0.2 - 2 * self.CAPTURE_DT)

    def test_a_slow_inference_rate_does_not_stretch_the_ghost_past_the_default(self):
        detection = Detection(10, 10, 20, 20, "person", 0.9, 4)
        self.step([detection], fps=2.0)
        self.step([detection], fps=2.0)
        scene, elapsed = self.step([], fps=2.0), self.CAPTURE_DT
        for _ in range(200):
            if not scene.objects:
                break
            scene = self.step([], fps=2.0)
            elapsed += self.CAPTURE_DT
        self.assertEqual(scene.objects, [])
        self.assertLessEqual(elapsed, FADE_OUT_S + self.CAPTURE_DT)

    # ---- the stage switches ----------------------------------------------

    def test_switching_a_stage_off_clears_it_instantly_instead_of_fading(self):
        """The trap this filter had to avoid.

        ``FaceGestureProcessor.set_stage_enabled`` clears a disabled stage's
        results rather than freezing them, deliberately, and
        ``test_disabling_the_face_stage_skips_it_and_clears_the_boxes`` pins
        that. A fade-out applied to that clear would put the boxes the
        operator has just switched off back on screen for up to a quarter of a
        second — which looks exactly like the stage controls not working.
        """
        detection = Detection(10, 10, 20, 20, "person", 0.9, 1)
        face = Face(10, 20, 30, 40, "Toby", 0.2, True)
        gesture = Gesture("Victory", 0.9, 60, 20, 20, 30)
        self.step([detection], [face], [gesture])
        scene = self.step([detection], [face], [gesture])
        self.assertEqual((len(scene.objects), len(scene.faces),
                          len(scene.gestures)), (1, 1, 1))

        off = dict(object_enabled=False, face_enabled=False,
                   gesture_enabled=False)
        scene = self.step(**off)
        self.assertEqual((scene.objects, scene.faces, scene.gestures),
                         ([], [], []))
        # Still gone on the next frame: no state was kept to come back from.
        scene = self.step(**off)
        self.assertEqual((scene.objects, scene.faces, scene.gestures),
                         ([], [], []))

        # One stage going off does not disturb one that is still on.
        scene = self.step([detection], face_enabled=False)
        self.assertEqual(len(scene.objects), 1)
        self.assertEqual(scene.faces, [])

    def test_boxes_that_merely_vanish_do_still_fade(self):
        """The contrast that makes the instant clear a decision rather than an
        accident: identical empty lists, opposite behaviour, and the only
        difference is the enable flag the processor publishes."""
        detection = Detection(10, 10, 20, 20, "person", 0.9, 1)
        self.step([detection])
        self.step([detection])
        faded = self.step([]).objects
        self.assertEqual(len(faded), 1)
        self.assertLess(faded[0].alpha, 1.0)
        self.assertGreater(faded[0].alpha, 0.0)

    # ---- per-stage treatment ---------------------------------------------

    def test_faces_move_the_moment_the_detector_does_but_objects_ease(self):
        """``Face`` has no track_id and refreshes at 2 Hz. Easing a face box
        over a 500 ms interval walks it off the face it is naming, which is
        worse than a still box — so faces fade, and only fade."""
        face = Face(10, 20, 30, 40, "Toby", 0.2, True)
        detection = Detection(10, 20, 30, 40, "person", 0.9, 1)
        self.step([detection], [face])
        scene = self.step([detection._replace(x=20)], [face._replace(x=20)])
        self.assertEqual(scene.faces[0].x, 20.0)
        self.assertEqual(scene.faces[0].name, "Toby")
        self.assertGreater(scene.objects[0].x, 10.0)
        self.assertLess(scene.objects[0].x, 20.0)

    def test_a_face_that_leaves_fades_rather_than_disappearing(self):
        face = Face(10, 20, 30, 40, "Toby", 0.2, True)
        self.step(faces=[face])
        self.step(faces=[face])
        scene = self.step(faces=[])
        self.assertEqual(len(scene.faces), 1)
        self.assertEqual((scene.faces[0].x, scene.faces[0].y), (10.0, 20.0))
        self.assertLess(scene.faces[0].alpha, 1.0)

    def test_gestures_fade_but_are_never_debounced(self):
        """``GestureEventOutput`` already owns hold and cooldown. A second
        definition of 'the gesture is on' living here would eventually
        disagree with the one that fires the automations, so a gesture is
        drawn the frame it arrives and named the frame it changes."""
        gesture = Gesture("Victory", 0.9, 60, 20, 20, 30)
        self.step(gestures=[gesture])
        scene = self.step(gestures=[gesture._replace(x=65)])
        self.assertEqual(scene.gestures[0].x, 65.0)

        changed = self.step(gestures=[Gesture("Thumb_Up", 0.9, 60, 20, 20, 30)])
        self.assertIn("Thumb_Up", [item.name for item in changed.gestures])
        self.assertGreater(
            [item.alpha for item in changed.gestures
             if item.name == "Thumb_Up"][0], 0.0)

    def test_an_object_with_no_track_id_is_passed_through_untouched(self):
        """Two untracked boxes cannot be told apart between frames, and
        guessing which is which is the one thing this module refuses to do."""
        untracked = Detection(10, 10, 20, 20, "cup", 0.5, None)
        self.step([untracked])
        scene = self.step([untracked._replace(x=90)])
        self.assertEqual(scene.objects[0].x, 90.0)
        self.assertEqual(scene.objects[0].alpha, 1.0)
        self.assertEqual(self.step([]).objects, [])   # and leaves no ghost

    # ---- ownership and purity --------------------------------------------

    def test_two_smoothers_fed_different_frames_do_not_interfere(self):
        """``pipeline._OutputWorker`` is a one-slot drop-old mailbox, so every
        output sees a *different subset* of frames. A shared smoother would be
        raced across those threads and would advance its clock against frames
        a given output never received."""
        fast = SceneSmoother(clock=self.clock)
        slow = SceneSmoother(clock=self.clock)
        start = Detection(10, 10, 20, 20, "person", 0.9, 1)
        moved = start._replace(x=110)
        first = self.scene([start])
        fast.apply(first)
        slow.apply(first)

        for _ in range(3):        # the fast output receives all three...
            self.clock.advance(self.CAPTURE_DT)
            fast_scene = fast.apply(self.scene([moved]))
        slow_scene = slow.apply(self.scene([moved]))   # ...the slow one, one.

        # The slow output had one long gap, longer than an inference interval,
        # so there was nothing stale to smooth and it snapped. The fast one is
        # still easing across the frames it actually received.
        self.assertEqual(slow_scene.objects[0].x, 110.0)
        self.assertLess(fast_scene.objects[0].x, 110.0)

        # And no track crosses between them.
        self.clock.advance(self.CAPTURE_DT)
        slow.apply(self.scene([moved, Detection(0, 0, 5, 5, "cup", 0.5, 99)]))
        fast_scene = fast.apply(self.scene([moved]))
        self.assertEqual([item.track_id for item in fast_scene.objects], [1])

    def test_switching_camera_forgets_the_previous_camera_s_boxes(self):
        detection = Detection(10, 10, 20, 20, "person", 0.9, 1)
        self.step([detection])
        self.step([detection])
        other = SourceSpec.from_mapping({"id": "other", "type": "webcam"})
        self.clock.advance(self.CAPTURE_DT)
        scene = self.smoother.apply(
            SceneFrame(other, 9, 9.0, self.raw, metadata={"inference_fps": 40.0}))
        self.assertEqual(scene.objects, [])

    def test_apply_returns_a_new_frame_and_leaves_the_scene_alone(self):
        detection = Detection(10, 10, 20, 20, "person", 0.9, 1)
        scene = self.scene([detection])
        out = self.smoother.apply(scene)
        self.assertIsNot(out, scene)
        self.assertIsNot(out.objects, scene.objects)
        self.assertEqual(scene.objects, [detection])
        self.assertIs(out.raw, scene.raw)
        self.assertIs(out.metadata, scene.metadata)
        self.assertEqual(out.sequence, scene.sequence)
        self.assertEqual(out.contract_version, scene.contract_version)

    def test_the_proxy_carries_everything_the_renderer_reads(self):
        detection = Detection(10, 10, 20, 20, "person", 0.87, 42)
        item = self.smoother.apply(self.scene([detection])).objects[0]
        self.assertEqual(item.label, "person")
        self.assertEqual(item.track_id, 42)
        self.assertEqual(item.score, 0.87)
        self.assertEqual(item.alpha, 1.0)
        face = self.smoother.apply(
            self.scene(faces=[Face(1, 2, 3, 4, "Toby", 0.2, True)])).faces[0]
        self.assertEqual((face.name, face.conf, face.known), ("Toby", 0.2, True))


class OverlayAlphaTests(unittest.TestCase):
    """The overlay's only concession to time — and it stays a pure function.

    ``render`` holds no cross-frame state: the alpha arrives *on the item*,
    from ``smoothing.SceneSmoother``, which is the single place that knows
    anything about previous frames.
    """

    def setUp(self):
        self.raw = np.zeros((100, 200, 3), dtype=np.uint8)
        self.source = SourceSpec.from_mapping({"type": "webcam"})
        self.profile = OverlayProfile(show_faces=False, show_gestures=False)
        self.detection = Detection(20, 40, 60, 30, "person", 0.9, 1)

    def scene(self, item):
        return SceneFrame(self.source, 1, 1.0, self.raw, objects=[item],
                          metadata={})

    def faded(self, alpha):
        return SmoothedItem(self.detection, 20.0, 40.0, 60.0, 30.0, alpha)

    def test_an_item_without_an_alpha_draws_exactly_as_it_always_did(self):
        """The ~100 existing render tests pass namedtuples with no alpha at
        all. Their path must be byte-identical to the alpha == 1 path."""
        plain = render(self.scene(self.detection), self.profile)
        wrapped = render(self.scene(self.faded(1.0)), self.profile)
        self.assertTrue(np.array_equal(plain, wrapped))

    def test_a_half_faded_box_is_composited_not_drawn_flat(self):
        full = render(self.scene(self.faded(1.0)), self.profile).astype(float)
        half = render(self.scene(self.faded(0.5)), self.profile).astype(float)
        self.assertGreater(full.sum(), 0.0)
        self.assertFalse(np.array_equal(full, half))
        self.assertAlmostEqual(half.sum() / full.sum(), 0.5, places=2)
        # Never brighter than the opaque draw anywhere: a fade only removes.
        self.assertTrue(np.all(half <= full + 1.0))

    def test_a_fully_faded_box_draws_nothing_at_all(self):
        output = render(self.scene(self.faded(0.0)), self.profile)
        self.assertTrue(np.array_equal(output, self.raw))
        self.assertIsNot(output, self.raw)

    def test_render_holds_no_state_between_calls(self):
        opaque, half = self.detection, self.faded(0.4)
        frames = [render(self.scene(item), self.profile).tobytes()
                  for item in (opaque, half, opaque, half)]
        self.assertEqual(frames[0], frames[2])
        self.assertEqual(frames[1], frames[3])
        self.assertNotEqual(frames[0], frames[1])
        # And the scene's own frame is never written to.
        self.assertTrue(np.array_equal(self.raw, np.zeros_like(self.raw)))


class OutputSmootherOwnershipTests(unittest.TestCase):
    """Every rendering output owns its own smoother. None of them share one."""

    def setUp(self):
        self.source = SourceSpec.from_mapping({"type": "webcam"})
        self.raw = np.zeros((60, 200, 3), dtype=np.uint8)

    def scene(self, sequence, objects):
        return SceneFrame(self.source, sequence, float(sequence), self.raw,
                          objects=list(objects),
                          metadata={"inference_fps": 40.0})

    def test_each_rendering_output_owns_its_own_smoother(self):
        first, second = LatestFrameOutput(), LatestFrameOutput()
        self.assertIsNot(first._smoother, second._smoother)
        self.assertIsNot(first._smoother, ObsVirtualCameraOutput()._smoother)

    def test_the_preview_feed_is_eased_while_the_kept_scene_is_not(self):
        """The JPEG is the picture and is eased. The scene kept for callers is
        the record of what was actually detected, and is handed back as it
        arrived."""
        import cv2

        output = LatestFrameOutput(MINIMAL)
        # Substitute the clock, not the ownership: the output still holds one
        # smoother of its own, this one just does not read wall time.
        clock = FakeClock()
        output._smoother = SceneSmoother(clock=clock)
        start = Detection(5, 5, 20, 20, "person", 0.9, 1)
        moved = start._replace(x=150)
        output.publish(self.scene(0, [start]))
        clock.advance(1.0 / 60.0)
        second = self.scene(1, [moved])
        output.publish(second)

        kept, eased = output.snapshot()
        self.assertIs(kept, second)
        self.assertTrue(eased)
        unsmoothed = cv2.imencode(
            ".jpg", render(second, MINIMAL),
            [cv2.IMWRITE_JPEG_QUALITY, output.jpeg_quality])[1].tobytes()
        self.assertNotEqual(eased, unsmoothed)


class GestureEventTests(unittest.TestCase):
    def test_hold_emits_typed_non_authorized_event(self):
        events = []
        clock = Mock(return_value=10.0)
        output = GestureEventOutput(events.append, hold_frames=2, clock=clock)
        output.set_enabled(True)
        source = SourceSpec.from_mapping({"id": "monitor", "type": "webcam"})
        face = Face(0, 0, 10, 10, "Toby", 0.2, True)
        gesture = Gesture("Victory", 0.95, 0, 0, 10, 10)
        for sequence in range(2):
            output.publish(SceneFrame(source, sequence, float(sequence), np.zeros((2, 2, 3)),
                                      faces=[face], gestures=[gesture]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["actor"], "Toby")
        self.assertEqual(events[0]["gesture"], "Victory")
        self.assertFalse(events[0]["security_authorized"])
        self.assertEqual(events[0]["liveness_state"], "not_evaluated")

    def test_several_known_faces_attribute_to_the_gesturing_hand(self):
        """Two known faces used to give actor=None, which silently killed every
        actor-scoped rule. The nearest enrolled face now takes the gesture, and
        the event says the attribution was made under ambiguity."""
        events = []
        output = GestureEventOutput(events.append, hold_frames=1,
                                    clock=Mock(return_value=10.0))
        output.set_enabled(True)
        source = SourceSpec.from_mapping({"type": "webcam"})
        faces = [Face(0, 0, 5, 5, "A", 0.2, True),
                 Face(6, 0, 5, 5, "B", 0.2, True)]
        gesture = Gesture("Open_Palm", 0.9, 0, 0, 5, 5)
        output.publish(SceneFrame(source, 0, 0.0, np.zeros((2, 2, 3)),
                                  faces=faces, gestures=[gesture]))
        self.assertEqual(events[0]["actor"], "A")
        self.assertEqual(events[0]["actor_attribution"], "nearest")
        self.assertEqual(events[0]["actor_candidates"], ["A", "B"])


class AsyncProcessorTests(unittest.TestCase):
    def test_capture_returns_without_waiting_for_inference(self):
        release = threading.Event()

        class ObjectDetector:
            def detect(self, _frame):
                release.wait(1.0)
                return ([Detection(0, 0, 10, 10, "person", 0.9, 4)],
                        {"inference": 4.0}, "test:yolo")

            def close(self):
                release.set()

        face = Face(1, 2, 3, 4, "Toby", 0.2, True)
        gesture = Gesture("Victory", 0.9, 1, 1, 3, 3)
        processor = FaceGestureProcessor(
            face_detector=Mock(return_value=[face]),
            gesture_detector=Mock(detect=Mock(return_value=[gesture])),
            object_detector=ObjectDetector(),
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        started = time.monotonic()
        initial = processor(frame, source, 0, time.monotonic())
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.1)
        self.assertEqual(initial.metadata["inference_status"], "warming_up")
        release.set()
        deadline = time.monotonic() + 1.0
        scene = initial
        while time.monotonic() < deadline:
            scene = processor(frame, source, 1, time.monotonic())
            if scene.objects:
                break
            time.sleep(0.01)
        processor.close()
        self.assertEqual(scene.objects[0].track_id, 4)
        self.assertEqual(scene.faces[0].x, 1)
        self.assertEqual(scene.metadata["object_model"], "test:yolo")

    def test_face_inference_is_limited_to_person_crops(self):
        class ObjectDetector:
            def detect(self, _frame):
                return ([Detection(10, 20, 30, 40, "person", 0.9, 1),
                         Detection(0, 0, 5, 5, "cat", 0.8, 2)], {}, "fake")

            def close(self):
                pass

        face_detector = Mock(return_value=[Face(2, 3, 4, 5, None, 0.8, False)])
        processor = FaceGestureProcessor(
            face_detector=face_detector,
            gesture_detector=Mock(detect=Mock(return_value=[])),
            object_detector=ObjectDetector(),
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processor(frame, source, 0, time.monotonic())
        deadline = time.monotonic() + 1.0
        scene = None
        while time.monotonic() < deadline:
            scene = processor(frame, source, 1, time.monotonic())
            if scene.faces:
                break
            time.sleep(0.01)
        processor.close()
        self.assertEqual(face_detector.call_args.args[0].shape[:2], (40, 30))
        self.assertEqual((scene.faces[0].x, scene.faces[0].y), (12, 23))

    def test_object_model_can_switch_without_restarting_capture(self):
        created = []

        class ObjectDetector:
            def __init__(self, model):
                self.model = model
                self.closed = False
                created.append(self)

            def detect(self, _frame):
                return [], {}, self.model

            def close(self):
                self.closed = True

        first = ObjectDetector("first")
        processor = FaceGestureProcessor(
            face_detector=Mock(return_value=[]),
            gesture_detector=Mock(detect=Mock(return_value=[])),
            object_detector=first,
            object_detector_factory=ObjectDetector,
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        processor(frame, source, 0, time.monotonic())
        processor.set_object_model("second")
        deadline = time.monotonic() + 1.0
        scene = None
        while time.monotonic() < deadline:
            scene = processor(frame, source, 1, time.monotonic())
            if scene.metadata.get("object_model") == "second":
                break
            time.sleep(0.01)
        processor.close()
        self.assertTrue(first.closed)
        self.assertEqual(scene.metadata["object_model"], "second")
        self.assertEqual(len(created), 2)

    def test_model_checksum_helper(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"acesvision")
            self.assertEqual(
                file_sha256(path),
                "2bba274e7725caa167d59af051764d1ebf98a7294323d6c5efe4fd8467894252",
            )

    def test_face_results_are_cached_between_scheduled_refreshes(self):
        class ObjectDetector:
            def detect(self, _frame):
                return [Detection(0, 0, 10, 10, "person", 0.9, 7)], {}, "fake"

            def close(self):
                pass

        face_detector = Mock(return_value=[Face(1, 1, 3, 3, "Toby", 0.2, True)])
        processor = FaceGestureProcessor(
            face_detector=face_detector,
            gesture_detector=Mock(detect=Mock(return_value=[])),
            object_detector=ObjectDetector(),
            face_hz=0.1,
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((20, 20, 3), dtype=np.uint8)
        deadline = time.monotonic() + 1.0
        sequence = 0
        scene = None
        while time.monotonic() < deadline:
            scene = processor(frame, source, sequence, time.monotonic())
            sequence += 1
            if scene.faces:
                break
            time.sleep(0.01)
        for _ in range(5):
            processor(frame, source, sequence, time.monotonic())
            sequence += 1
            time.sleep(0.01)
        processor.close()
        self.assertEqual(face_detector.call_count, 1)
        self.assertEqual(scene.faces[0].name, "Toby")


class CountingObjectDetector:
    """Object detector that reports how often the object stage actually ran."""

    def __init__(self, detections=(), model="test:yolo"):
        self.detections = list(detections)
        self.model = model
        self.calls = 0
        self.closed = False

    def detect(self, _frame):
        self.calls += 1
        return list(self.detections), {"inference": 1.0}, self.model

    def close(self):
        self.closed = True


PERSON = Detection(0, 0, 20, 20, "person", 0.9, 1)


def drive(processor, source, frame, sequence, cycles=1, timeout_s=1.0):
    """Feed frames until the inference loop has published `cycles` new ones.

    The loop is asynchronous, so a test that submits one frame and reads the
    metrics immediately is reading the *previous* cycle. Returns the sequence
    number to carry on from.
    """
    started = processor.metrics()["inference_sequence"]
    seen = 0
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        processor(frame, source, sequence, time.monotonic())
        sequence += 1
        published = processor.metrics()["inference_sequence"]
        if published != started:
            started = published
            seen += 1
            if seen >= cycles:
                return sequence
        time.sleep(0.005)
    raise AssertionError(f"the inference loop published {seen}/{cycles} cycles")


class StageControlTests(unittest.TestCase):
    """The three perception knobs were constructor-only. Now they are live.

    Every stage cost the GUI wants to show is produced by this loop, and every
    knob that moves those costs is read by it on every cycle. So each setter is
    tested against a *running* loop, not against the attribute it writes.
    """

    def setUp(self):
        self.source = SourceSpec.from_mapping({"type": "webcam"})
        self.frame = np.zeros((40, 40, 3), dtype=np.uint8)
        self.objects = CountingObjectDetector([PERSON])
        self.faces = Mock(return_value=[Face(1, 2, 3, 4, "Toby", 0.2, True)])
        self.gestures = Mock(detect=Mock(
            return_value=[Gesture("Shush", 1.0, 1, 1, 3, 3)]))

    def processor(self, **kwargs):
        # Both refresh rates default to "due every cycle" here on purpose. At
        # the shipped 2 Hz and 15 Hz a stage would not be due again inside the
        # few milliseconds a test cycle takes, so "the stage stopped running"
        # would assert green whether or not disabling it did anything.
        kwargs.setdefault("face_hz", 1000.0)
        kwargs.setdefault("gesture_hz", 1000.0)
        processor = FaceGestureProcessor(
            face_detector=self.faces,
            gesture_detector=self.gestures,
            object_detector=self.objects,
            **kwargs)
        self.addCleanup(processor.close)
        return processor

    def assertRanEveryCycle(self, counter, processor, sequence, cycles=3):
        """Prove the stage was live before a test claims disabling stopped it."""
        before = counter()
        sequence = drive(processor, self.source, self.frame, sequence,
                         cycles=cycles)
        self.assertGreaterEqual(counter() - before, cycles,
                                "the stage was not running every cycle, so "
                                "'it stopped' would prove nothing")
        return sequence

    def test_the_frame_divisor_changes_on_the_running_loop(self):
        processor = self.processor()
        self.assertEqual(processor.detect_every, 1)
        processor.set_detect_every(4)
        self.assertEqual(processor.detect_every, 4)
        self.assertEqual(processor.metrics()["object_detect_every"], 4)
        # Sequences 1..3 are not multiples of 4, so none of them may reach
        # inference; sequence 4 must.
        before = self.objects.calls
        for sequence in (1, 2, 3):
            processor(self.frame, self.source, sequence, time.monotonic())
        time.sleep(0.05)
        self.assertEqual(self.objects.calls, before)
        drive(processor, self.source, self.frame, 4)
        self.assertGreater(self.objects.calls, before)

    def test_a_divisor_below_one_cannot_stall_the_loop(self):
        processor = self.processor()
        processor.set_detect_every(0)
        self.assertEqual(processor.detect_every, 1)

    def test_the_face_refresh_rate_changes_on_the_running_loop(self):
        # Starts effectively never: one refresh on the first person, then no
        # scheduled refresh for ten seconds.
        processor = self.processor(face_hz=0.1)
        sequence = drive(processor, self.source, self.frame, 0, cycles=3)
        self.assertEqual(self.faces.call_count, 1)
        processor.set_face_hz(1000.0)
        drive(processor, self.source, self.frame, sequence, cycles=3)
        self.assertGreater(self.faces.call_count, 1)
        self.assertGreater(processor.metrics()["face_refresh_hz"], 100.0)

    def test_the_gesture_refresh_rate_changes_on_the_running_loop(self):
        processor = self.processor(gesture_hz=0.1)
        sequence = drive(processor, self.source, self.frame, 0, cycles=3)
        self.assertEqual(self.gestures.detect.call_count, 1)
        processor.set_gesture_hz(1000.0)
        drive(processor, self.source, self.frame, sequence, cycles=3)
        self.assertGreater(self.gestures.detect.call_count, 1)

    def test_a_rate_of_zero_is_clamped_instead_of_dividing_by_zero(self):
        processor = self.processor()
        processor.set_face_hz(0.0)
        processor.set_gesture_hz(-5.0)
        self.assertEqual(processor.metrics()["face_refresh_hz"], 0.1)
        self.assertEqual(processor.metrics()["gesture_refresh_hz"], 0.1)

    def test_disabling_the_object_stage_skips_yolo(self):
        processor = self.processor()
        sequence = self.assertRanEveryCycle(
            lambda: self.objects.calls, processor, 0)
        processor.set_stage_enabled("object", False)
        ran = self.objects.calls
        sequence = drive(processor, self.source, self.frame, sequence, cycles=3)
        self.assertEqual(self.objects.calls, ran)
        metrics = processor.metrics()
        self.assertFalse(metrics["object_enabled"])
        self.assertFalse(metrics["object_refreshed"])
        self.assertEqual(metrics["object_stage_ms"], 0.0)
        # And the metadata stops naming a model that is not being run.
        self.assertEqual(metrics["object_model"], "stage disabled")

    def test_disabling_the_face_stage_skips_it_and_clears_the_boxes(self):
        processor = self.processor()
        sequence = self.assertRanEveryCycle(
            lambda: self.faces.call_count, processor, 0)
        scene = processor(self.frame, self.source, sequence, time.monotonic())
        self.assertTrue(scene.faces)
        processor.set_stage_enabled("face", False)
        ran = self.faces.call_count
        sequence = drive(processor, self.source, self.frame, sequence + 1,
                         cycles=3)
        scene = processor(self.frame, self.source, sequence, time.monotonic())
        self.assertEqual(self.faces.call_count, ran)
        # Cleared, not frozen: a stale box that keeps rendering is
        # indistinguishable from a live one.
        self.assertEqual(scene.faces, [])
        self.assertFalse(processor.metrics()["face_enabled"])
        self.assertEqual(processor.metrics()["face_stage_ms"], 0.0)

    def test_disabling_the_gesture_stage_skips_mediapipe(self):
        processor = self.processor()
        sequence = self.assertRanEveryCycle(
            lambda: self.gestures.detect.call_count, processor, 0)
        processor.set_stage_enabled("gesture", False)
        ran = self.gestures.detect.call_count
        sequence = drive(processor, self.source, self.frame, sequence, cycles=3)
        scene = processor(self.frame, self.source, sequence, time.monotonic())
        self.assertEqual(self.gestures.detect.call_count, ran)
        self.assertEqual(scene.gestures, [])
        self.assertFalse(processor.metrics()["gesture_enabled"])

    def test_a_disabled_object_stage_starves_the_face_stage(self):
        """Faces are only searched for inside person boxes. Say so in metrics.

        This is the coupling the GUI warning exists for: switching objects off
        takes the face stage down with it, one step removed.
        """
        processor = self.processor()
        sequence = self.assertRanEveryCycle(
            lambda: self.faces.call_count, processor, 0)
        processor.set_stage_enabled("object", False)
        ran = self.faces.call_count
        sequence = drive(processor, self.source, self.frame, sequence, cycles=3)
        scene = processor(self.frame, self.source, sequence, time.monotonic())
        self.assertEqual(self.faces.call_count, ran)
        self.assertEqual(scene.faces, [])
        # The face stage itself is still *enabled*; it simply has nowhere to
        # look. The metadata must not conflate the two.
        self.assertTrue(processor.metrics()["face_enabled"])

    def test_a_stage_comes_back_when_it_is_switched_on_again(self):
        processor = self.processor()
        drive(processor, self.source, self.frame, 0)
        processor.set_stage_enabled("gesture", False)
        sequence = drive(processor, self.source, self.frame, 1, cycles=2)
        ran = self.gestures.detect.call_count
        processor.set_stage_enabled("gesture", True)
        drive(processor, self.source, self.frame, sequence, cycles=3)
        self.assertGreater(self.gestures.detect.call_count, ran)

    def test_an_unknown_stage_is_refused_rather_than_silently_ignored(self):
        processor = self.processor()
        with self.assertRaises(ValueError):
            processor.set_stage_enabled("faces", False)
        with self.assertRaises(ValueError):
            processor.stage_enabled("everything")

    def test_the_knobs_stay_coherent_under_concurrent_set_and_read(self):
        """Reconfiguring under load must not tear a cycle or lose an update.

        The inference loop reads all four knobs every cycle. This drives the
        capture side, the setter side and the metrics side at once and asserts
        that no cycle ever observed a value nobody set, that no stage blew up
        mid-reconfiguration, and that the last write survives.
        """
        processor = self.processor()
        rates = (0.5, 2.0, 8.0)
        divisors = (1, 2, 3)
        observed_hz, statuses = [], []
        failures = []
        stop = threading.Event()

        def guarded(work):
            def run():
                try:
                    work()
                except Exception as exc:              # noqa: BLE001
                    failures.append(exc)
            return run

        def writer():
            while not stop.is_set():
                for index, hz in enumerate(rates):
                    processor.set_face_hz(hz)
                    processor.set_gesture_hz(hz * 2)
                    processor.set_detect_every(divisors[index])
                    processor.set_stage_enabled("face", index != 1)
                    processor.set_stage_enabled("object", index != 2)

        def reader():
            while not stop.is_set():
                metrics = processor.metrics()
                observed_hz.append(round(metrics["face_refresh_hz"], 6))
                statuses.append(metrics["inference_status"])

        def feeder():
            sequence = 0
            while not stop.is_set():
                processor(self.frame, self.source, sequence, time.monotonic())
                sequence += 1

        threads = [threading.Thread(target=guarded(work), daemon=True)
                   for work in (writer, reader, feeder)]
        for thread in threads:
            thread.start()
        time.sleep(0.6)
        stop.set()
        for thread in threads:
            thread.join(timeout=5.0)
        self.assertEqual(failures, [])
        self.assertGreater(len(observed_hz), 50)
        self.assertEqual(set(observed_hz) - set(rates), set(),
                         "a cycle published a rate nobody set")
        self.assertNotIn("degraded", statuses,
                         "a stage failed while the knobs were being moved")
        # Last write wins: settle the loop and read it back.
        processor.set_face_hz(3.0)
        processor.set_stage_enabled("face", True)
        processor.set_stage_enabled("object", True)
        drive(processor, self.source, self.frame, 0, cycles=2)
        metrics = processor.metrics()
        self.assertEqual(round(metrics["face_refresh_hz"], 6), 3.0)
        self.assertTrue(metrics["face_enabled"])
        self.assertTrue(metrics["object_enabled"])


class PipelineTests(unittest.TestCase):
    def test_camera_controls_are_clamped_and_applied(self):
        cap = FakeCapture([])
        cap.set = Mock(return_value=True)
        controls = {
            "auto_exposure": False,
            "exposure": 333,
            "brightness": 12,
            "contrast": 40,
            "gamma": 180,
        }
        VisionPipeline._apply_camera_controls(cap, controls)
        values = [call.args[1] for call in cap.set.call_args_list]
        self.assertIn(333, values)
        self.assertIn(12, values)
        self.assertIn(40, values)
        self.assertIn(180, values)

    def test_nearly_black_uniform_frame_reports_privacy_warning(self):
        frame = np.full((20, 20, 3), 15, dtype=np.uint8)
        quality = VisionPipeline._frame_quality(frame)
        self.assertIn("privacy", quality["image_warning"])
        self.assertLess(quality["image_std"], 1)

    def test_black_manual_frame_recommends_auto_exposure(self):
        frame = np.full((20, 20, 3), 2, dtype=np.uint8)
        quality = VisionPipeline._frame_quality(frame, {
            "auto_exposure": False, "brightness": 0,
        })
        self.assertIn("Automatic exposure", quality["image_warning"])

    def test_minimum_brightness_recommends_reset(self):
        frame = np.full((20, 20, 3), 2, dtype=np.uint8)
        quality = VisionPipeline._frame_quality(frame, {
            "auto_exposure": False, "brightness": -64,
        })
        self.assertIn("Reset", quality["image_warning"])

    def test_reconnects_and_fans_out(self):
        first = FakeCapture([(False, None)])
        second = FakeCapture([(True, "raw")])
        captures = [first, second]
        delivered = []
        complete = threading.Event()
        source = SourceSpec.from_mapping({"type": "webcam", "index": 9})

        def opener(_):
            return captures.pop(0) if captures else None

        def processor(frame, src, sequence, captured_at):
            return SceneFrame(src, sequence, captured_at, f"processed:{frame}")

        pipeline = VisionPipeline(
            source,
            processor,
            [CallbackOutput(lambda scene: delivered.append(("a", scene.raw))),
             CallbackOutput(lambda scene: (delivered.append(("b", scene.raw)),
                                           complete.set()))],
            opener=opener,
            retry_min_s=0.001,
            retry_max_s=0.002,
        )
        pipeline.start()
        self.assertTrue(complete.wait(1.0))
        pipeline.stop()
        pipeline.join(1.0)

        self.assertCountEqual(delivered, [
            ("a", "processed:raw"), ("b", "processed:raw"),
        ])
        self.assertTrue(first.released)
        self.assertFalse(pipeline.is_alive())

    def test_switches_from_webcam_to_droidcam_without_restart(self):
        webcam = SourceSpec.from_mapping({"id": "webcam", "type": "webcam"})
        phone = SourceSpec.from_mapping({
            "id": "phone", "type": "droidcam", "url": "http://phone:4747/video",
        })
        webcam_capture = FakeCapture([(True, "webcam-frame")])
        phone_capture = FakeCapture([(True, "phone-frame")])
        opened = []
        delivered = threading.Event()

        def opener(source):
            opened.append(source.id)
            return webcam_capture if source.id == "webcam" else phone_capture

        def processor(frame, source, sequence, captured_at):
            scene = SceneFrame(source, sequence, captured_at, frame)
            if source.id == "webcam":
                pipeline.switch_source(phone)
            return scene

        def receive(scene):
            if scene.source.id == "phone":
                delivered.set()

        pipeline = VisionPipeline(
            webcam, processor, [CallbackOutput(receive)], opener=opener,
            retry_min_s=0.001, retry_max_s=0.002,
        )
        pipeline.start()
        self.assertTrue(delivered.wait(1.0))
        pipeline.stop()
        pipeline.join(1.0)

        self.assertEqual(opened[:2], ["webcam", "phone"])
        self.assertTrue(webcam_capture.released)


class ScriptedCapture:
    """A capture that hands out a scripted list of frames, then parks.

    Parking rather than returning ``(False, None)`` is what a live network
    source does between frames, and it is the state the reader has to be
    correct in: the producer is idle, not finished.
    """

    def __init__(self, frames, *, fail_at_end=False):
        self._frames = list(frames)
        self._fail_at_end = fail_at_end
        self.reads = 0
        self.released = False
        self.exhausted = threading.Event()
        self._resume = threading.Event()
        self._lock = threading.Lock()

    def resume(self, frames=()):
        with self._lock:
            self._frames.extend(frames)
        self.exhausted.clear()
        self._resume.set()

    def isOpened(self):                     # noqa: N802 - cv2 spelling
        return not self.released

    def read(self):
        while True:
            with self._lock:
                if self._frames:
                    self.reads += 1
                    return True, self._frames.pop(0)
            if self._fail_at_end:
                return False, None
            self.exhausted.set()
            self._resume.clear()
            if self.released:
                return False, None
            self._resume.wait(2.0)
            if self.released:
                return False, None

    def release(self):
        self.released = True
        self._resume.set()

    def set(self, *_):
        return True

    def get(self, *_):
        return 0.0


class LatestFrameReaderTests(unittest.TestCase):
    """Decode off the capture loop, keep the newest frame, drop the rest.

    The fault this closes is not throughput, it is age. Read in order from the
    capture loop, a network source hands the pipeline every frame it ever sent,
    including the ones from several seconds ago, and face recognition on a
    frame from several seconds ago is wrong rather than slow.
    """

    def reader(self, capture, **kwargs):
        reader = LatestFrameReader(capture, **kwargs)
        self.addCleanup(reader.release)
        return reader

    def test_the_producer_never_waits_for_the_consumer(self):
        """Ten frames arrive while the consumer has not called read() once."""
        capture = ScriptedCapture(range(10))
        reader = self.reader(capture)
        self.assertTrue(capture.exhausted.wait(2.0),
                        "the reader thread stalled behind an idle consumer")
        self.assertEqual(capture.reads, 10)

    def test_the_consumer_gets_the_newest_frame_and_the_rest_are_counted(self):
        capture = ScriptedCapture(range(10))
        reader = self.reader(capture)
        self.assertTrue(capture.exhausted.wait(2.0))

        ok, frame = reader.read()
        self.assertTrue(ok)
        self.assertEqual(frame, 9)          # newest, not the nine before it
        self.assertEqual(reader.dropped, 9)

    def test_a_waiting_frame_is_handed_over_without_waiting(self):
        capture = ScriptedCapture([1])
        reader = self.reader(capture)
        self.assertTrue(capture.exhausted.wait(2.0))

        started = time.monotonic()
        ok, frame = reader.read()
        self.assertEqual((ok, frame), (True, 1))
        self.assertLess(time.monotonic() - started, 0.05)

    def test_a_frame_is_never_delivered_twice(self):
        capture = ScriptedCapture([1])
        reader = self.reader(capture, read_timeout_s=0.05)
        self.assertEqual(reader.read(), (True, 1))
        # Nothing new has arrived; the old frame must not be served again as
        # if it were current.
        self.assertEqual(reader.read(), (False, None))

        capture.resume([2])
        self.assertEqual(reader.read(), (True, 2))

    def test_a_dead_source_is_reported_as_not_ok(self):
        capture = ScriptedCapture([], fail_at_end=True)
        reader = self.reader(capture)
        self.assertEqual(reader.read(), (False, None))
        self.assertFalse(reader.isOpened())

    def test_a_source_that_goes_quiet_gives_up_at_its_timeout(self):
        """The signal VisionPipeline already reads as 'reconnect'."""
        capture = ScriptedCapture([])
        reader = self.reader(capture, read_timeout_s=0.05)
        started = time.monotonic()
        self.assertEqual(reader.read(), (False, None))
        self.assertLess(time.monotonic() - started, 2.0)

    def test_release_stops_the_thread_and_releases_the_capture(self):
        capture = ScriptedCapture([])
        reader = LatestFrameReader(capture)
        reader.release()
        self.assertTrue(capture.released)
        self.assertFalse(reader._thread.is_alive())
        self.assertFalse(reader.isOpened())

    def test_control_calls_reach_the_capture_underneath(self):
        capture = ScriptedCapture([])
        capture.set = Mock(return_value=True)
        reader = self.reader(capture)
        reader.set(cv2_prop := 38, 1)
        capture.set.assert_called_once_with(cv2_prop, 1)


class ThreadedSourceWiringTests(unittest.TestCase):
    """open_source and VisionPipeline, with the reader actually in the path."""

    def test_network_sources_are_wrapped_and_webcams_are_not(self):
        capture = ScriptedCapture([])
        phone = SourceSpec.from_mapping({
            "type": "droidcam", "url": "http://phone:4747/video",
        })
        opened = open_source(phone, capture_factory=lambda *_: capture)
        self.addCleanup(opened.release)
        self.assertIsInstance(opened, LatestFrameReader)

        webcam = SourceSpec.from_mapping({"type": "webcam", "index": 0})
        # A webcam takes camera-control set() calls from the capture loop while
        # it runs, and OpenCV is not safe to set() and read() at once. That is
        # why the reader stops at the network boundary.
        self.assertIs(open_source(webcam, webcam_opener=lambda _: capture),
                      capture)

    def test_a_failed_network_open_is_still_none_not_a_reader(self):
        capture = FakeCapture([], opened=False)
        phone = SourceSpec.from_mapping({
            "type": "droidcam", "url": "http://phone:4747/video",
        })
        self.assertIsNone(open_source(phone, capture_factory=lambda *_: capture))
        self.assertTrue(capture.released)

    def test_switching_sources_through_the_reader_releases_the_old_one(self):
        """The switch contract of test_switches_from_webcam_to_droidcam...,
        re-run with real LatestFrameReaders in the capture slot."""
        first = SourceSpec.from_mapping({
            "id": "phone", "type": "droidcam", "url": "http://one:4747/video",
        })
        second = SourceSpec.from_mapping({
            "id": "tablet", "type": "droidcam", "url": "http://two:4747/video",
        })
        captures = {"phone": ScriptedCapture(["one"] * 50),
                    "tablet": ScriptedCapture(["two"] * 50)}
        opened = []
        delivered = threading.Event()

        def opener(source):
            opened.append(source.id)
            return open_source(source,
                               capture_factory=lambda *_: captures[source.id])

        def processor(frame, source, sequence, captured_at):
            if source.id == "phone":
                pipeline.switch_source(second)
            return SceneFrame(source, sequence, captured_at, frame)

        def receive(scene):
            if scene.source.id == "tablet":
                delivered.set()

        pipeline = VisionPipeline(first, processor, [CallbackOutput(receive)],
                                  opener=opener, retry_min_s=0.001,
                                  retry_max_s=0.002)
        pipeline.start()
        self.addCleanup(pipeline.join, 2.0)
        self.addCleanup(pipeline.stop)
        self.assertTrue(delivered.wait(5.0), "the switched source never arrived")
        pipeline.stop()
        pipeline.join(2.0)

        self.assertEqual(opened[:2], ["phone", "tablet"])
        self.assertTrue(captures["phone"].released)


class PolicyTests(unittest.TestCase):
    def test_convenience_action_is_dry_run(self):
        rule = Rule.create("Victory", "acergb", "next_theme")
        decision = RuleEngine([rule]).evaluate({
            "gesture": "Victory", "actor": None, "source": "webcam",
        })[0]
        self.assertEqual(decision.outcome, "dry_run")

    def test_personal_action_requires_associated_actor(self):
        rule = Rule.create("Open_Palm", "kde", "next_desktop")
        decision = RuleEngine([rule]).evaluate({
            "gesture": "Open_Palm", "actor": None, "source": "webcam",
        })[0]
        self.assertEqual(decision.outcome, "blocked")
        self.assertIn("actor", decision.reason)

    def test_unknown_connector_and_shell_are_rejected(self):
        with self.assertRaises(ValueError):
            Rule.create("Victory", "shell", "run")

    def test_rule_store_round_trip_is_versioned_and_dry_run(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            rule = Rule.create("Victory", "mpris", "next", actor="Toby")
            store = RuleStore(path)
            store.save([rule])
            loaded = store.load()
            self.assertEqual(loaded, [rule])
            text = path.read_text()
            self.assertIn('"version": 1', text)
            # dry_run lives on the rule now, not on the file. Asserted against
            # the parsed structure so this cannot pass on the old global key.
            payload = json.loads(text)
            self.assertNotIn("dry_run", payload)
            self.assertIs(payload["rules"][0]["dry_run"], True)


class GuiStyleTests(unittest.TestCase):
    def test_hex_colour_converts_to_opencv_bgr(self):
        from acesvision.gui import VisionBackend

        self.assertEqual(VisionBackend._hex_to_bgr("#12abef"), (239, 171, 18))

    def test_invalid_hex_colour_is_rejected(self):
        from acesvision.gui import VisionBackend

        with self.assertRaises(ValueError):
            VisionBackend._hex_to_bgr("purple")

    def test_manual_exposure_is_allowed_until_a_camera_proves_otherwise(self):
        """Manual exposure used to be hardcoded off for every user, forever.

        One monitor webcam returning black frames is not a reason to grey the
        control out on hardware that handles it fine.
        """
        from acesvision.gui import VisionBackend

        backend = VisionBackend(initialize_models=False, load_saved_rules=False)
        self.assertTrue(backend.manualExposureSupported)
        self.assertEqual(backend.exposureNotice, "")
        backend.setCameraTuning(False, 251, 0, 0, 100)
        self.assertFalse(backend._auto_exposure)
        self.assertFalse(backend.pipeline._camera_controls["auto_exposure"])


Landmark = namedtuple("Landmark", "x y")


def hand_landmarks(extended=("middle",), wrist=(0.5, 0.9)):
    """21 landmarks with the named fingers extended away from the wrist.

    Distances are measured from the wrist, so this mirrors the orientation-
    agnostic test in gesture_catalog._extended.
    """
    points = [Landmark(*wrist)] * 21
    points = list(points)
    for finger, (tip, pip) in gesture_catalog._FINGERS.items():
        pip_distance = 0.10
        tip_distance = 0.20 if finger in extended else 0.05
        points[pip] = Landmark(wrist[0], wrist[1] - pip_distance)
        points[tip] = Landmark(wrist[0], wrist[1] - tip_distance)
    return points


def pointing_hand(tip, wrist=(0.5, 0.95), extended="index"):
    """21 landmarks for a one-finger hand with that fingertip at ``tip``.

    pip and mcp sit on the wrist->tip line at 1/2 and 1/4 of the way, so the
    named finger reads as extended (tip farther from the wrist than the pip)
    and, whenever ``tip`` is above the wrist, as pointing up (tip above pip
    above mcp). Every other finger is curled in toward the wrist.

    Coordinates are MediaPipe-normalised 0..1, like the real landmarks.
    """
    def along(fraction):
        return Landmark(wrist[0] + (tip[0] - wrist[0]) * fraction,
                        wrist[1] + (tip[1] - wrist[1]) * fraction)

    points = [Landmark(*wrist) for _ in range(21)]
    for finger, (tip_index, pip_index) in gesture_catalog._FINGERS.items():
        if finger == extended:
            points[tip_index] = Landmark(*tip)
            points[pip_index] = along(0.5)
        else:
            points[pip_index] = along(0.20)
            points[tip_index] = along(0.10)   # curled: nearer the wrist
    points[gesture_catalog._INDEX_MCP] = along(0.25)
    return points


def open_hand(tip, wrist=(0.5, 0.95)):
    """All four fingers extended toward ``tip`` — an open palm, not a shush."""
    def along(fraction):
        return Landmark(wrist[0] + (tip[0] - wrist[0]) * fraction,
                        wrist[1] + (tip[1] - wrist[1]) * fraction)

    points = [Landmark(*wrist) for _ in range(21)]
    for tip_index, pip_index in gesture_catalog._FINGERS.values():
        points[tip_index] = Landmark(*tip)
        points[pip_index] = along(0.5)
    points[gesture_catalog._INDEX_MCP] = along(0.25)
    return points


class FakeDevice:
    """Stand-in for discovery.WebcamDevice with only the fields camera.py uses."""

    def __init__(self, index, kind, path=None, name="Fake Cam"):
        self.index = index
        self.kind = kind
        self.path = path or f"/dev/video{index}"
        self.name = name


class GestureCatalogTests(unittest.TestCase):
    def test_vocabulary_carries_the_ported_middle_finger(self):
        self.assertIn("Middle_Finger", gesture_catalog.GESTURE_IDS)
        spec = gesture_catalog.gesture_by_id("Middle_Finger")
        self.assertFalse(spec.builtin)   # landmark-derived, not a model label
        # The seven MediaPipe built-ins are all present and marked as built-in.
        builtins = [s.id for s in gesture_catalog.GESTURES if s.builtin]
        self.assertEqual(len(builtins), 7)

    def test_loose_names_normalise_onto_the_emitted_spelling(self):
        # The exact live-rules bug: "open_palm" never matched "Open_Palm".
        self.assertEqual(gesture_catalog.normalise_gesture("open_palm"), "Open_Palm")
        self.assertEqual(gesture_catalog.normalise_gesture("middle finger"),
                         "Middle_Finger")
        self.assertEqual(gesture_catalog.normalise_gesture("ILOVEYOU"), "ILoveYou")

    def test_unknown_gesture_is_named_and_rejected(self):
        self.assertIsNone(gesture_catalog.normalise_gesture("wave"))
        self.assertFalse(gesture_catalog.is_known_gesture("wave"))
        with self.assertRaises(ValueError) as caught:
            gesture_catalog.require_gesture("wave")
        self.assertIn("wave", str(caught.exception))
        self.assertIn("Open_Palm", str(caught.exception))   # lists the vocabulary

    def test_action_catalog_matches_acergb(self):
        self.assertEqual(len(gesture_catalog.ACTION_CATALOG), 16)
        for action_id in ("lights_next_theme", "device_gradient", "media_playpause",
                          "volume_mute", "custom"):
            self.assertEqual(gesture_catalog.require_action(action_id), action_id)
        with self.assertRaises(ValueError):
            gesture_catalog.require_action("launch_missiles")

    def test_connector_bindings_all_resolve_in_the_policy_table(self):
        # Guards the two vocabularies against drifting apart again.
        for action_id, (connector, action) in \
                gesture_catalog.CONNECTOR_BINDINGS.items():
            self.assertIn(action_id, gesture_catalog.ACTION_IDS)
            self.assertIn(connector, CONNECTORS, action_id)
            self.assertIn(action, CONNECTORS[connector], action_id)

    def test_middle_finger_pose_needs_one_finger_up_and_three_curled(self):
        self.assertTrue(gesture_catalog.is_middle_finger(hand_landmarks(["middle"])))
        self.assertFalse(gesture_catalog.is_middle_finger(
            hand_landmarks(["index", "middle", "ring", "pinky"])))   # open palm
        self.assertFalse(gesture_catalog.is_middle_finger(hand_landmarks(["index"])))
        self.assertFalse(gesture_catalog.is_middle_finger([]))
        self.assertFalse(gesture_catalog.is_middle_finger(None))

    def test_classify_hands_prefers_the_custom_pose_over_the_model_label(self):
        from gestures import classify_hands

        Category = namedtuple("Category", "category_name score")
        landmarks = [hand_landmarks(["middle"])]
        # The model calls it Pointing_Up; the landmark pose must win.
        rows = classify_hands(landmarks, [[Category("Pointing_Up", 0.9)]],
                              100, 100, 0.5)
        self.assertEqual([row.name for row in rows], ["Middle_Finger"])
        self.assertEqual(rows[0].score, 1.0)

        # Still detected when the model returned no category for that hand.
        rows = classify_hands(landmarks, [], 100, 100, 0.5)
        self.assertEqual([row.name for row in rows], ["Middle_Finger"])

        # Built-in labels still pass through, and low scores are still dropped.
        open_palm = [hand_landmarks(["index", "middle", "ring", "pinky"])]
        rows = classify_hands(open_palm, [[Category("Open_Palm", 0.9)]],
                              100, 100, 0.5)
        self.assertEqual([row.name for row in rows], ["Open_Palm"])
        rows = classify_hands(open_palm, [[Category("Open_Palm", 0.1)]],
                              100, 100, 0.5)
        self.assertEqual(rows, [])


# One 1000x1000 frame and one face box, shared by every shush test so the
# arithmetic below is checkable by hand. The face spans x 400..600, y 300..560,
# so its mouth point is (500, 300 + 0.72*260) = (500, 487.2) in pixels. Because
# the frame is 1000 wide and tall, a normalised landmark of 0.487 is pixel 487.
FRAME_W = FRAME_H = 1000
FACE_BOX = Face(400, 300, 200, 260, "Toby", 0.2, True)

AT_LIPS = (0.500, 0.490)          # dx 0.00, dy  +0.01  -> inside
AT_CEILING = (0.500, 0.150)       # dx 0.00, dy  -1.30  -> the plain point up
NEAR_EDGE_ABOVE = (0.500, 0.360)  # dx 0.00, dy  -0.49  -> just inside
PAST_EDGE_ABOVE = (0.500, 0.350)  # dx 0.00, dy  -0.53  -> just outside
NEAR_EDGE_SIDE = (0.405, 0.487)   # dx -0.48, dy  0.00  -> just inside
PAST_EDGE_SIDE = (0.390, 0.487)   # dx -0.55, dy  0.00  -> just outside


class ShushGestureTests(unittest.TestCase):
    """The founder's shush: index finger up, held to the lips.

    MediaPipe already labels that hand Pointing_Up, and Pointing_Up runs
    `ledctl next-theme` in automations.json. Everything here exists to keep one
    shush from also cycling the lighting themes.
    """

    def test_shush_is_registered_in_the_shared_vocabulary(self):
        self.assertIn("Shush", gesture_catalog.GESTURE_IDS)
        spec = gesture_catalog.gesture_by_id("Shush")
        self.assertFalse(spec.builtin)   # landmark-derived, not a model label
        self.assertEqual(len(gesture_catalog.GESTURES), 9)
        # The seven MediaPipe built-ins are untouched by the addition.
        self.assertEqual(
            len([s for s in gesture_catalog.GESTURES if s.builtin]), 7)
        # The QML combo box and the applet read this same tuple.
        self.assertEqual(gesture_catalog.catalog_json()["gestures"][-1],
                         {"id": "Shush", "label": "Shush (finger to lips)",
                          "builtin": False})

    def test_the_previously_unfireable_rule_name_now_resolves(self):
        # rules.json carried gesture "shush" against no such gesture. It is a
        # real name now, and the loose spelling normalises onto the emitted one.
        self.assertEqual(gesture_catalog.normalise_gesture("shush"), "Shush")
        self.assertEqual(gesture_catalog.normalise_gesture("SHUSH"), "Shush")
        self.assertTrue(gesture_catalog.is_known_gesture("shush"))
        self.assertEqual(validate_gesture(" shush "), "Shush")

    def test_a_shush_rule_binds_to_the_mute_action(self):
        rule = Rule.create("shush", "pipewire", "mute", actor="*")
        self.assertEqual(rule.gesture, "Shush")
        self.assertEqual((rule.connector, rule.action), ("pipewire", "mute"))
        # Muting is a PipeWire action, not an AceRGB one.
        self.assertEqual(gesture_catalog.CONNECTOR_BINDINGS["volume_mute"],
                         ("pipewire", "mute"))

    def test_finger_at_the_lips_is_a_shush(self):
        hand = pointing_hand(AT_LIPS)
        self.assertTrue(gesture_catalog.is_shush(hand, [FACE_BOX],
                                                 FRAME_W, FRAME_H))

    def test_the_same_hand_pointed_at_the_ceiling_is_not(self):
        # Identical pose, fingertip above the head. This is the whole reason
        # the face box is a term in the geometry.
        hand = pointing_hand(AT_CEILING)
        self.assertFalse(gesture_catalog.is_shush(hand, [FACE_BOX],
                                                  FRAME_W, FRAME_H))

    def test_no_face_means_no_shush(self):
        hand = pointing_hand(AT_LIPS)
        for faces in ([], None):
            self.assertFalse(gesture_catalog.is_shush(hand, faces,
                                                      FRAME_W, FRAME_H))

    def test_proximity_boundary_on_both_sides(self):
        # Nearest-miss pairs straddling the 0.5 normalised radius.
        for tip in (NEAR_EDGE_ABOVE, NEAR_EDGE_SIDE):
            self.assertTrue(
                gesture_catalog.is_shush(pointing_hand(tip), [FACE_BOX],
                                         FRAME_W, FRAME_H), tip)
        for tip in (PAST_EDGE_ABOVE, PAST_EDGE_SIDE):
            self.assertFalse(
                gesture_catalog.is_shush(pointing_hand(tip), [FACE_BOX],
                                         FRAME_W, FRAME_H), tip)

    def test_proximity_scales_with_the_face_box(self):
        # The same fingertip, a person standing twice as far away: the face box
        # halves, so what was at the lips is now well outside the mouth region.
        tip = (0.500, 0.400)
        near = Face(400, 300, 200, 260, "Toby", 0.2, True)
        far = Face(475, 400, 50, 65, "Toby", 0.2, True)
        self.assertTrue(gesture_catalog.is_shush(pointing_hand(tip), [near],
                                                 FRAME_W, FRAME_H))
        self.assertFalse(gesture_catalog.is_shush(pointing_hand(tip), [far],
                                                  FRAME_W, FRAME_H))

    def test_any_face_in_frame_can_anchor_the_shush(self):
        other = Face(10, 10, 100, 130, None, 0.9, False)
        self.assertTrue(gesture_catalog.is_shush(pointing_hand(AT_LIPS),
                                                 [other, FACE_BOX],
                                                 FRAME_W, FRAME_H))

    def test_degenerate_face_boxes_are_ignored_not_divided_by(self):
        empty = Face(500, 480, 0, 0, None, 0.9, False)
        self.assertFalse(gesture_catalog.is_shush(pointing_hand(AT_LIPS),
                                                  [empty], FRAME_W, FRAME_H))

    def test_extended_is_not_the_same_as_pointing_up(self):
        # Fingertip exactly at the lips, but the hand comes from above so the
        # finger points down. _extended is orientation-agnostic and would
        # accept this; the shush must not.
        hand = pointing_hand(AT_LIPS, wrist=(0.5, 0.20))
        self.assertFalse(gesture_catalog.is_shush(hand, [FACE_BOX],
                                                  FRAME_W, FRAME_H))

    def test_other_hand_shapes_at_the_lips_are_not_a_shush(self):
        self.assertFalse(gesture_catalog.is_shush(open_hand(AT_LIPS),
                                                  [FACE_BOX], FRAME_W, FRAME_H))
        self.assertFalse(
            gesture_catalog.is_shush(pointing_hand(AT_LIPS, extended="middle"),
                                     [FACE_BOX], FRAME_W, FRAME_H))
        self.assertFalse(gesture_catalog.is_shush([], [FACE_BOX],
                                                  FRAME_W, FRAME_H))
        self.assertFalse(gesture_catalog.is_shush(None, [FACE_BOX],
                                                  FRAME_W, FRAME_H))

    def test_the_real_engine_face_type_drives_the_geometry(self):
        # The stand-in above mirrors engine.Face; prove the real one binds, so
        # a field rename in engine.py cannot pass this suite silently. It has
        # already earned its keep once: engine.Face grew a `metric` field when
        # ArcFace landed, because `conf` stopped meaning "distance, lower is
        # better", and this test is what refused to let that pass unnoticed.
        from engine import Face as EngineFace
        import matching as _matching

        face = EngineFace(*FACE_BOX, metric=_matching.EUCLIDEAN_L2)
        self.assertTrue(gesture_catalog.is_shush(pointing_hand(AT_LIPS),
                                                 [face], FRAME_W, FRAME_H))

    def test_shush_takes_strict_precedence_over_pointing_up(self):
        """The critical case. A shush must emit Shush and must NOT emit
        Pointing_Up, or every shush also fires `ledctl next-theme`."""
        from gestures import classify_hands

        Category = namedtuple("Category", "category_name score")
        model_says_pointing_up = [[Category("Pointing_Up", 0.95)]]
        hands = [pointing_hand(AT_LIPS)]

        rows = classify_hands(hands, model_says_pointing_up, FRAME_W, FRAME_H,
                              0.5, faces=[FACE_BOX])
        names = [row.name for row in rows]
        self.assertEqual(names, ["Shush"])
        self.assertNotIn("Pointing_Up", names)
        self.assertEqual(rows[0].score, 1.0)

    def test_pointing_up_still_reaches_the_theme_binding_away_from_the_face(self):
        from gestures import classify_hands

        Category = namedtuple("Category", "category_name score")
        rows = classify_hands([pointing_hand(AT_CEILING)],
                              [[Category("Pointing_Up", 0.95)]],
                              FRAME_W, FRAME_H, 0.5, faces=[FACE_BOX])
        self.assertEqual([row.name for row in rows], ["Pointing_Up"])

        # And with no faces at all — the detector must not start swallowing
        # Pointing_Up just because face detection went quiet.
        rows = classify_hands([pointing_hand(AT_LIPS)],
                              [[Category("Pointing_Up", 0.95)]],
                              FRAME_W, FRAME_H, 0.5, faces=None)
        self.assertEqual([row.name for row in rows], ["Pointing_Up"])

    def test_shush_is_detected_even_when_the_model_returned_no_category(self):
        from gestures import classify_hands

        rows = classify_hands([pointing_hand(AT_LIPS)], [], FRAME_W, FRAME_H,
                              0.5, faces=[FACE_BOX])
        self.assertEqual([row.name for row in rows], ["Shush"])

    def test_the_processor_hands_face_boxes_to_the_gesture_detector(self):
        """Without this wiring the geometry above can never fire in the app."""
        class ObjectDetector:
            def detect(self, _frame):
                return ([Detection(0, 0, 100, 100, "person", 0.9, 1)], {}, "fake")

            def close(self):
                pass

        face = Face(10, 10, 20, 26, "Toby", 0.2, True)
        gesture_detector = Mock(detect=Mock(return_value=[]))
        processor = FaceGestureProcessor(
            face_detector=Mock(return_value=[face]),
            gesture_detector=gesture_detector,
            object_detector=ObjectDetector(),
        )
        source = SourceSpec.from_mapping({"type": "webcam"})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        processor(frame, source, 0, time.monotonic())
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            processor(frame, source, 1, time.monotonic())
            if gesture_detector.detect.call_args is not None:
                break
            time.sleep(0.01)
        processor.close()
        passed = gesture_detector.detect.call_args.kwargs["faces"]
        self.assertEqual([f.name for f in passed], ["Toby"])


class CameraDiscoveryTests(unittest.TestCase):
    """camera.py must own no device list of its own — discovery.py is the SSoT."""

    def test_no_hard_coded_candidate_list_survives(self):
        self.assertFalse(hasattr(camera, "CANDIDATES"))

    def test_candidates_come_from_discovery_colour_first_ir_last(self):
        devices = [
            FakeDevice(20, "virtual", name="OBS Virtual Camera"),
            FakeDevice(3, "infrared", name="IR Camera"),
            FakeDevice(1, "colour", name="UVC WebCam"),
            FakeDevice(5, "camera", name="Unclassified"),
        ]
        ordered = camera.candidate_devices(devices)
        self.assertEqual([d.path for d in ordered],
                         ["/dev/video1", "/dev/video5", "/dev/video3"])

    def test_infrared_node_is_reachable(self):
        # The old CANDIDATES list omitted video3 entirely, which made the IR
        # sensor unreachable and blocked the colour+IR liveness design.
        ordered = camera.candidate_devices([FakeDevice(3, "infrared")])
        self.assertEqual([d.path for d in ordered], ["/dev/video3"])

    def test_metadata_and_virtual_nodes_are_never_probed(self):
        # Metadata nodes are already filtered by discovery (interface index != 0);
        # camera.py additionally drops loopback/virtual outputs.
        probed = []

        def opener(device, manual_exposure=False):
            probed.append(device)
            return _fake_colour_capture(), _colour_frame()

        cap = camera.open_camera(
            devices=[FakeDevice(20, "virtual"), FakeDevice(1, "colour")],
            opener=opener,
            status=lambda device: camera.DEVICE_AVAILABLE,
        )
        self.assertIsNotNone(cap)
        self.assertEqual(probed, ["/dev/video1"])


class CameraStatusTests(unittest.TestCase):
    def test_status_distinguishes_missing_busy_and_available(self):
        def missing(path, flags):
            raise FileNotFoundError(path)

        def busy(path, flags):
            raise OSError(errno.EBUSY, "Device or resource busy")

        closed = []
        self.assertEqual(camera.device_status(9, opener=missing),
                         camera.DEVICE_MISSING)
        self.assertEqual(camera.device_status(1, opener=busy), camera.DEVICE_BUSY)
        self.assertEqual(
            camera.device_status(1, opener=lambda path, flags: 7,
                                 closer=closed.append),
            camera.DEVICE_AVAILABLE)
        self.assertEqual(closed, [7])   # the probe never leaks a descriptor

    def test_holders_are_read_from_proc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "video-node"
            device.write_text("")
            proc = root / "proc"
            for pid, name, target in ((4242, "obs", device),
                                      (4243, "bash", root / "other")):
                if not target.exists():
                    target.write_text("")
                fds = proc / str(pid) / "fd"
                fds.mkdir(parents=True)
                (proc / str(pid) / "comm").write_text(name + "\n")
                (fds / "3").symlink_to(target)
            holders = camera.device_holders(str(device), proc_root=str(proc))
        self.assertEqual(holders, [(4242, "obs")])

    def test_a_busy_device_is_not_reported_as_missing(self):
        # The whole point: "held by OBS" and "does not exist" need different fixes.
        with patch.object(camera, "device_status",
                          return_value=camera.DEVICE_AVAILABLE), \
             patch.object(camera, "device_holders", return_value=[(99, "obs")]):
            status, detail = camera.explain_failure("/dev/video1")
        self.assertEqual(status, camera.DEVICE_BUSY)
        self.assertIn("obs", detail)
        self.assertIn("/dev/video1", detail)


def _colour_frame():
    frame = np.zeros((8, 8, 3), dtype="uint8")
    frame[:, :, 2] = 200          # strong red channel -> chroma above CHROMA_MIN
    return frame


def _grey_frame():
    return np.full((8, 8, 3), 120, dtype="uint8")


def _fake_colour_capture():
    return FakeCapture([(True, "frame")])


class CameraOpenTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("FACE_ID_CAM", None)

    def tearDown(self):
        if self._saved is not None:
            os.environ["FACE_ID_CAM"] = self._saved

    def _opener(self, frames):
        """frames: {device -> frame or None}."""
        def opener(device, manual_exposure=False):
            frame = frames.get(device)
            if frame is None:
                return None, None
            return FakeCapture([(True, "frame")]), frame
        return opener

    def test_colour_is_preferred_over_infrared(self):
        cap = camera.open_camera(
            devices=[FakeDevice(1, "colour"), FakeDevice(3, "infrared")],
            opener=self._opener({"/dev/video1": _colour_frame(),
                                 "/dev/video3": _grey_frame()}),
            status=lambda device: camera.DEVICE_AVAILABLE)
        self.assertIsNotNone(cap)

    def test_busy_colour_falls_back_to_infrared_and_says_why(self):
        with patch("builtins.print") as printed:
            cap = camera.open_camera(
                devices=[FakeDevice(1, "colour"), FakeDevice(3, "infrared")],
                opener=self._opener({"/dev/video3": _grey_frame()}),
                status=lambda device: camera.DEVICE_AVAILABLE,
                explain=lambda device: (camera.DEVICE_BUSY,
                                        f"{device} is held by obs (pid 7)"))
        self.assertIsNotNone(cap)
        message = " ".join(str(call) for call in printed.call_args_list)
        self.assertIn("held by obs", message)

    def test_all_devices_busy_raises_busy_not_not_found(self):
        with self.assertRaises(camera.CameraBusyError) as caught:
            camera.open_camera(
                devices=[FakeDevice(1, "colour")],
                opener=self._opener({}),
                status=lambda device: camera.DEVICE_AVAILABLE,
                explain=lambda device: (camera.DEVICE_BUSY,
                                        f"{device} is held by obs (pid 7)"))
        self.assertIn("held by obs", str(caught.exception))
        self.assertNotIn("No working camera found", str(caught.exception))

    def test_no_devices_at_all_raises_not_found(self):
        with self.assertRaises(camera.CameraNotFoundError):
            camera.open_camera(devices=[], opener=self._opener({}),
                               status=lambda device: camera.DEVICE_AVAILABLE)

    def test_forced_busy_device_raises_instead_of_substituting_another(self):
        # An explicit choice that is merely contended must not be silently
        # swapped for a different camera.
        with self.assertRaises(camera.CameraBusyError):
            camera.open_camera(
                preferred="/dev/video1",
                devices=[FakeDevice(3, "infrared")],
                opener=self._opener({"/dev/video3": _grey_frame()}),
                status=lambda device: camera.DEVICE_AVAILABLE,
                explain=lambda device: (camera.DEVICE_BUSY,
                                        f"{device} is held by obs (pid 7)"))

    def test_forced_missing_device_falls_back_to_discovery(self):
        # Previously a stale path raised with no fallback and sources.py turned
        # that into None: total silent failure.
        cap = camera.open_camera(
            preferred="/dev/video9",
            devices=[FakeDevice(1, "colour")],
            opener=self._opener({"/dev/video1": _colour_frame()}),
            status=lambda device: (camera.DEVICE_MISSING
                                   if device == "/dev/video9"
                                   else camera.DEVICE_AVAILABLE))
        self.assertIsNotNone(cap)

    def test_face_id_cam_override_still_works(self):
        os.environ["FACE_ID_CAM"] = "3"
        opened = []

        def opener(device, manual_exposure=False):
            opened.append(device)
            return FakeCapture([(True, "frame")]), _grey_frame()

        cap = camera.open_camera(devices=[FakeDevice(1, "colour")], opener=opener,
                                 status=lambda device: camera.DEVICE_AVAILABLE)
        self.assertIsNotNone(cap)
        self.assertEqual(opened, [3])   # honoured exactly, no colour filtering

    def test_unparsable_override_is_ignored_not_fatal(self):
        os.environ["FACE_ID_CAM"] = "not-a-device"
        cap = camera.open_camera(
            devices=[FakeDevice(1, "colour")],
            opener=self._opener({"/dev/video1": _colour_frame()}),
            status=lambda device: camera.DEVICE_AVAILABLE)
        self.assertIsNotNone(cap)


class SourceFailureReportingTests(unittest.TestCase):
    def test_busy_camera_reason_is_reported_not_swallowed(self):
        source = SourceSpec.from_mapping({"id": "webcam", "type": "webcam"})
        reasons = []

        def raiser(device):
            raise camera.CameraBusyError("/dev/video1 is held by obs (pid 7)")

        self.assertIsNone(open_source(source, webcam_opener=raiser,
                                      on_error=reasons.append))
        self.assertEqual(len(reasons), 1)
        self.assertIn("held by obs", reasons[0])

    def test_pipeline_surfaces_the_specific_reason(self):
        source = SourceSpec.from_mapping({"id": "webcam", "type": "webcam"})

        def fake_open_source(spec, on_error=None):
            if on_error:
                on_error("/dev/video1 is held by obs (pid 7)")
            return None

        with patch("acesvision.pipeline.open_source", fake_open_source):
            pipeline = VisionPipeline(source, lambda *a: None, [],
                                      retry_min_s=0.001, retry_max_s=0.002)
            pipeline.start()
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if "held by obs" in pipeline.state().last_error:
                    break
                time.sleep(0.01)
            pipeline.stop()
            pipeline.join(1.0)
        self.assertIn("held by obs", pipeline.state().last_error)


class RuleVocabularyTests(unittest.TestCase):
    def test_unknown_gesture_is_rejected_at_creation(self):
        # A rule naming a gesture nothing emits could never fire and never said
        # so. "finger_snap" is the standing example: a literal snap is too fast
        # and too subtle to detect from video (gestures.py module docstring).
        # "shush" used to sit here — it is a real pose now, see ShushGestureTests.
        with self.assertRaises(ValueError):
            Rule.create("finger_snap", "pipewire", "mute")

    def test_gesture_case_is_normalised_to_the_emitted_spelling(self):
        rule = Rule.create("open_palm", "mpris", "play_pause")
        self.assertEqual(rule.gesture, "Open_Palm")
        self.assertEqual(validate_gesture(" victory "), "Victory")

    def test_actor_is_validated_against_enrolled_identities(self):
        self.assertEqual(validate_actor("toby", actors=["Toby"]), "Toby")
        self.assertEqual(validate_actor("", actors=["Toby"]), "*")
        self.assertEqual(validate_actor("*", actors=["Toby"]), "*")
        with self.assertRaises(ValueError) as caught:
            validate_actor("mallory", actors=["Toby"])
        self.assertIn("Toby", str(caught.exception))

    def test_normalised_rule_actually_fires(self):
        # Red-then-green: the raw strings produce no decision, the normalised
        # rule produces one.
        event = {"gesture": "Open_Palm", "actor": "Toby", "source": "webcam"}
        unfireable = Rule(id="x", gesture="open_palm", connector="mpris",
                          action="play_pause", actor="toby")
        self.assertEqual(RuleEngine([unfireable]).evaluate(event), [])

        repaired = Rule.create("open_palm", "mpris", "play_pause", actor="toby")
        decisions = RuleEngine([repaired]).evaluate(event)
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].outcome, "dry_run")

    def test_store_repairs_legacy_rules_and_quarantines_unfireable_ones(self):
        import json

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps({"version": 1, "dry_run": True, "rules": [
                {"id": "a", "gesture": "finger_snap", "connector": "pipewire",
                 "action": "mute", "actor": "toby", "source": "*",
                 "enabled": True, "require_liveness": False,
                 "require_confirmation": False},
                {"id": "b", "gesture": "open_palm", "connector": "mpris",
                 "action": "play_pause", "actor": "toby", "source": "*",
                 "enabled": True, "require_liveness": False,
                 "require_confirmation": False},
            ]}))
            store = RuleStore(path)
            rules = store.load(strict=False, actors=["Toby"])
            self.assertEqual([r.id for r in rules], ["b"])
            self.assertEqual(rules[0].gesture, "Open_Palm")
            self.assertEqual(rules[0].actor, "Toby")
            self.assertEqual([raw["id"] for raw, _ in store.rejected], ["a"])
            with self.assertRaises(ValueError):
                store.load(strict=True, actors=["Toby"])


class HeadlessRunnerTests(unittest.TestCase):
    def test_gesture_events_are_enabled_by_default(self):
        from acesvision.__main__ import build_gesture_output, build_parser

        args = build_parser().parse_args([])
        output = build_gesture_output(args, callback=lambda event: None)
        self.assertTrue(output.enabled)   # used to be False, forever

    def test_no_events_flag_and_tuning_are_honoured(self):
        from acesvision.__main__ import build_gesture_output, build_parser

        args = build_parser().parse_args(["--no-events"])
        self.assertFalse(build_gesture_output(args).enabled)

        args = build_parser().parse_args(["--hold-frames", "3", "--cooldown-s", "0.5"])
        output = build_gesture_output(args, callback=lambda event: None)
        self.assertEqual((output.hold_frames, output.cooldown_s), (3, 0.5))

    def test_enabled_output_emits_without_an_external_toggle(self):
        events = []
        output = GestureEventOutput(events.append, hold_frames=1,
                                    clock=Mock(return_value=1.0), enabled=True)
        source = SourceSpec.from_mapping({"type": "webcam"})
        output.publish(SceneFrame(source, 0, 0.0, np.zeros((2, 2, 3)),
                                  gestures=[Gesture("Victory", 0.9, 0, 0, 5, 5)]))
        self.assertEqual(len(events), 1)

    def test_default_stays_disabled_for_the_gui_toggle(self):
        events = []
        output = GestureEventOutput(events.append, hold_frames=1,
                                    clock=Mock(return_value=1.0))
        source = SourceSpec.from_mapping({"type": "webcam"})
        output.publish(SceneFrame(source, 0, 0.0, np.zeros((2, 2, 3)),
                                  gestures=[Gesture("Victory", 0.9, 0, 0, 5, 5)]))
        self.assertEqual(events, [])


# ---------------------------------------------------------------------------
# Connector dispatch — the action-execution path that did not exist.
# ---------------------------------------------------------------------------


class FakeSession:
    """Stands in for a D-Bus connection. Records calls, replays a script.

    ``script`` maps a member name to a list of responses consumed in order; the
    last entry repeats. A response that is an Exception instance is raised, so
    every failure mode is expressible without a bus.
    """

    def __init__(self, script):
        self.script = {member: list(responses)
                       for member, responses in script.items()}
        self.calls = []
        self.closed = False

    def call(self, member, signature="", body=(), action=""):
        self.calls.append((member, signature, tuple(body)))
        responses = self.script.get(member)
        if responses is None:
            raise connectors.MethodFailedError(
                f"unscripted member {member}", connector="acergb", action=action)
        response = responses[0] if len(responses) == 1 else responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def members(self):
        return [member for member, _signature, _body in self.calls]

    def close(self):
        self.closed = True


class FakeTransport:
    def __init__(self, script):
        self.session = FakeSession(script)
        self.opened = 0

    def open(self, action=""):
        self.opened += 1
        return self.session


class ExplodingTransport:
    """Raises when opened — how a missing bus or missing library presents."""

    def __init__(self, error):
        self.error = error

    def open(self, action=""):
        raise self.error


def ticking_clock(step=0.1):
    state = {"now": 0.0}

    def clock():
        now = state["now"]
        state["now"] = now + step
        return now
    return clock


def acergb(script, **kwargs):
    transport = FakeTransport(script)
    kwargs.setdefault("sleep", lambda _seconds: None)
    return connectors.AceRgbConnector(transport=transport, **kwargs), transport


class ConnectorDispatchTests(unittest.TestCase):
    def test_next_theme_polls_is_ready_before_calling_next_theme(self):
        # IsReady is the readiness signal. Readiness is never inferred from the
        # NextTheme string, so IsReady must come first, every time.
        connector, transport = acergb({
            "IsReady": [(True,)], "NextTheme": [("purple",)],
        })
        detail = connector.execute("next_theme")
        self.assertEqual(transport.session.members(), ["IsReady", "NextTheme"])
        self.assertIn("purple", detail)
        self.assertTrue(transport.session.closed)

    def test_off_is_confirmed_by_reading_state_back(self):
        connector, transport = acergb({
            "IsReady": [(True,)], "Off": [()], "CurrentTheme": [("",)],
        })
        detail = connector.execute("off")
        self.assertEqual(transport.session.members(),
                         ["IsReady", "Off", "CurrentTheme"])
        self.assertIn("accepted", detail)

    def test_brightness_sends_a_typed_double(self):
        connector, transport = acergb({
            "IsReady": [(True,)], "SetBrightness": [()],
        })
        connector.execute("brightness", {"level": 0.4})
        self.assertEqual(transport.session.calls[-1], ("SetBrightness", "d", (0.4,)))

    def test_brightness_rejects_out_of_range_and_non_numeric_levels(self):
        for level in (1.5, -0.1, "bright", None):
            connector, transport = acergb({
                "IsReady": [(True,)], "SetBrightness": [()],
            })
            with self.assertRaises(connectors.InvalidParameterError):
                connector.execute("brightness", {"level": level})
            self.assertNotIn("SetBrightness", transport.session.members())

    def test_unsupported_action_never_touches_the_bus(self):
        connector, transport = acergb({"IsReady": [(True,)]})
        with self.assertRaises(connectors.UnsupportedActionError):
            connector.execute("explode")
        self.assertEqual(transport.opened, 0)

    def test_registry_dispatch_returns_a_result_not_an_exception(self):
        connector, _transport = acergb({
            "IsReady": [(True,)], "NextTheme": [("rainbow",)],
        })
        registry = connectors.ConnectorRegistry([connector])
        result = registry.dispatch("acergb", "next_theme")
        self.assertTrue(result.ok)
        self.assertIsNone(result.error_kind)
        self.assertIn("rainbow", result.detail)

    def test_unwired_connector_reports_loudly_instead_of_doing_nothing(self):
        # pipewire.mute is bound by the Shush rule and has no executor yet.
        # Silence here is exactly the failure mode this whole module exists to
        # remove, so it must come back as a named failure.
        registry = connectors.default_registry()
        result = registry.dispatch("pipewire", "mute")
        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "not_implemented")
        self.assertIn("acergb", result.detail)
        self.assertFalse(registry.supports("pipewire", "mute"))
        self.assertTrue(registry.supports("acergb", "next_theme"))

    def test_registry_is_open_to_new_connectors_without_touching_dispatch(self):
        class FakePipewire:
            name = "pipewire"

            @staticmethod
            def actions():
                return ("mute",)

            @staticmethod
            def execute(action, params=None):
                return "muted"

        registry = connectors.default_registry().register(FakePipewire())
        self.assertEqual(registry.names(), ["acergb", "pipewire"])
        self.assertTrue(registry.dispatch("pipewire", "mute").ok)

    def test_connector_actions_match_the_policy_catalog(self):
        # Drift guard: policy.CONNECTORS["acergb"] is the declaration, the
        # connector is the implementation. They must not diverge silently.
        self.assertEqual(sorted(connectors.AceRgbConnector.actions()),
                         sorted(CONNECTORS["acergb"]))

    def test_overlay_connector_is_typed_and_calls_only_its_callback(self):
        calls = []
        connector = connectors.OverlayConnector(lambda: calls.append("toggle"))
        registry = connectors.ConnectorRegistry([connector])
        self.assertEqual(sorted(connector.actions()),
                         sorted(CONNECTORS["overlay"]))
        result = registry.dispatch("overlay", "toggle_clean")
        self.assertTrue(result.ok)
        self.assertEqual(calls, ["toggle"])

    def test_bindings_that_name_acergb_are_all_executable(self):
        registry = connectors.default_registry()
        bound = [(name, pair) for name, pair
                 in gesture_catalog.CONNECTOR_BINDINGS.items()
                 if pair[0] == "acergb"]
        self.assertTrue(bound)
        for name, (connector, action) in bound:
            self.assertTrue(registry.supports(connector, action), name)


class ConnectorErrorModeTests(unittest.TestCase):
    """Every failure mode is distinguishable and none of them look like success.

    The predecessor shelled out ``ledctl theme next``; on 2026-08-15 that binary
    was a stale build that printed usage and exited 0. Loudness is the feature.
    """

    def dispatch(self, connector):
        return connectors.ConnectorRegistry([connector]).dispatch(
            "acergb", "next_theme")

    def test_daemon_absent_is_reported_as_daemon_absent(self):
        absent = connectors.DaemonUnavailableError(
            "org.acergb.Daemon is not on the session bus", connector="acergb")
        connector, transport = acergb({"IsReady": [absent]})
        result = self.dispatch(connector)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "daemon_absent")
        self.assertNotIn("NextTheme", transport.session.members())

    def test_daemon_not_ready_is_polled_and_never_calls_the_method(self):
        connector, transport = acergb(
            {"IsReady": [(False,)], "NextTheme": [("purple",)]},
            ready_timeout_s=0.2, clock=ticking_clock(0.1))
        result = self.dispatch(connector)
        self.assertEqual(result.error_kind, "not_ready")
        self.assertEqual(transport.session.members(), ["IsReady", "IsReady"])

    def test_readiness_that_arrives_late_still_succeeds(self):
        connector, transport = acergb(
            {"IsReady": [(False,), (False,), (True,)],
             "NextTheme": [("warm",)]},
            ready_timeout_s=5.0, clock=ticking_clock(0.1))
        result = self.dispatch(connector)
        self.assertTrue(result.ok)
        self.assertEqual(transport.session.members(),
                         ["IsReady", "IsReady", "IsReady", "NextTheme"])

    def test_method_error_is_reported_as_method_error(self):
        broken = connectors.MethodFailedError(
            "NextTheme failed (org.freedesktop.DBus.Error.UnknownMethod)",
            connector="acergb")
        connector, _transport = acergb({"IsReady": [(True,)], "NextTheme": [broken]})
        self.assertEqual(self.dispatch(connector).error_kind, "method_error")

    def test_timeout_is_reported_as_timeout(self):
        slow = connectors.ConnectorTimeoutError("NextTheme did not answer",
                                                connector="acergb")
        connector, _transport = acergb({"IsReady": [(True,)], "NextTheme": [slow]})
        self.assertEqual(self.dispatch(connector).error_kind, "timeout")

    def test_empty_next_theme_is_a_failure_not_a_success(self):
        # NextTheme returns "" when nothing was applied. The daemon already
        # said IsReady, so this is *not* a readiness problem and must not be
        # reported as one — and it must certainly not read as success.
        connector, _transport = acergb({"IsReady": [(True,)], "NextTheme": [("",)]})
        result = self.dispatch(connector)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "no_effect")
        self.assertNotEqual(result.error_kind, "not_ready")

    def test_missing_bus_is_reported_as_bus_absent(self):
        connector = connectors.AceRgbConnector(transport=ExplodingTransport(
            connectors.BusUnavailableError("no session bus address")))
        self.assertEqual(self.dispatch(connector).error_kind, "bus_absent")

    def test_missing_client_library_is_reported_as_transport_missing(self):
        connector = connectors.AceRgbConnector(transport=ExplodingTransport(
            connectors.TransportUnavailableError("jeepney is not installed")))
        self.assertEqual(self.dispatch(connector).error_kind, "transport_missing")

    def test_an_unexpected_exception_is_surfaced_not_swallowed(self):
        connector = connectors.AceRgbConnector(
            transport=ExplodingTransport(RuntimeError("socket exploded")))
        result = self.dispatch(connector)
        self.assertFalse(result.ok)
        self.assertEqual(result.error_kind, "unexpected")
        self.assertIn("socket exploded", result.detail)

    def test_every_produced_kind_is_declared(self):
        declared = set(connectors.ERROR_KINDS)
        produced = {cls.kind for cls in (
            connectors.TransportUnavailableError,
            connectors.BusUnavailableError,
            connectors.DaemonUnavailableError,
            connectors.DaemonNotReadyError,
            connectors.MethodFailedError,
            connectors.ConnectorTimeoutError,
            connectors.ActionHadNoEffectError,
            connectors.UnsupportedActionError,
            connectors.InvalidParameterError,
        )} | {"not_implemented", "unexpected"}
        self.assertEqual(produced, declared)


class DBusErrorMappingTests(unittest.TestCase):
    """The real jeepney session maps D-Bus error names onto typed failures."""

    def session(self, reply):
        connection = Mock()
        connection.send_and_get_reply.return_value = reply
        address = Mock(bus_name="org.acergb.Daemon")
        return connectors.JeepneySession(connection, address, 1.0,
                                         connector="acergb")

    @staticmethod
    def error_reply(name, text="boom"):
        from jeepney import MessageType

        reply = Mock()
        reply.header.message_type = MessageType.error
        reply.header.fields = {4: name}
        reply.body = (text,)
        return reply

    def test_service_unknown_becomes_daemon_absent(self):
        session = self.session(
            self.error_reply("org.freedesktop.DBus.Error.ServiceUnknown"))
        with self.assertRaises(connectors.DaemonUnavailableError):
            session.call("NextTheme")

    def test_no_reply_becomes_timeout(self):
        session = self.session(
            self.error_reply("org.freedesktop.DBus.Error.NoReply"))
        with self.assertRaises(connectors.ConnectorTimeoutError):
            session.call("NextTheme")

    def test_unknown_method_becomes_method_error(self):
        # The stale-binary class: a daemon that does not export the member.
        # A shell-out printed usage and exited 0; D-Bus cannot.
        session = self.session(
            self.error_reply("org.freedesktop.DBus.Error.UnknownMethod"))
        with self.assertRaises(connectors.MethodFailedError) as caught:
            session.call("NextTheme")
        self.assertIn("UnknownMethod", str(caught.exception))

    def test_socket_timeout_becomes_timeout(self):
        connection = Mock()
        connection.send_and_get_reply.side_effect = TimeoutError("timed out")
        session = connectors.JeepneySession(connection, Mock(), 0.5,
                                            connector="acergb")
        with self.assertRaises(connectors.ConnectorTimeoutError):
            session.call("NextTheme")

    def test_successful_reply_returns_the_body(self):
        from jeepney import MessageType

        reply = Mock()
        reply.header.message_type = MessageType.method_return
        reply.body = ("cool",)
        self.assertEqual(self.session(reply).call("CurrentTheme"), ("cool",))

    def test_absent_bus_address_becomes_bus_absent(self):
        transport = connectors.JeepneyTransport(
            "org.acergb.Daemon", "/org/acergb/Daemon", "org.acergb.Daemon",
            connector="acergb")
        with patch("jeepney.io.blocking.open_dbus_connection",
                   side_effect=KeyError("DBUS_SESSION_BUS_ADDRESS")):
            with self.assertRaises(connectors.BusUnavailableError):
                transport.open()


# ---------------------------------------------------------------------------
# Per-rule dry_run
# ---------------------------------------------------------------------------


class RecordingExecutor:
    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def dispatch(self, connector, action, params=None):
        self.calls.append((connector, action, dict(params or {})))
        return self.result or connectors.ExecutionResult(
            connector, action, True, "did the thing")

    @staticmethod
    def supports(connector, action):
        return True

    @staticmethod
    def names():
        return ["acergb"]


class PerRuleDryRunTests(unittest.TestCase):
    EVENT = {"gesture": "Victory", "actor": "Toby", "source": "webcam"}

    def test_rules_default_to_dry_run(self):
        self.assertTrue(Rule.create("Victory", "acergb", "next_theme").dry_run)

    def test_the_engine_has_no_global_dry_run(self):
        # Regression guard. The global was the bug: the only way to arm one
        # rule was to arm every rule, including any sensitive one added later.
        engine = RuleEngine([])
        self.assertFalse(hasattr(engine, "dry_run"))
        with self.assertRaises(TypeError):
            RuleEngine([], dry_run=False)

    def test_a_dry_run_rule_never_reaches_the_executor(self):
        executor = RecordingExecutor()
        rule = Rule.create("Victory", "acergb", "next_theme")
        decision = RuleEngine([rule], executor=executor).evaluate(self.EVENT)[0]
        self.assertEqual(decision.outcome, "dry_run")
        self.assertFalse(decision.ok)
        self.assertEqual(executor.calls, [])

    def test_an_armed_rule_executes_and_reports_what_happened(self):
        executor = RecordingExecutor()
        rule = Rule.create("Victory", "acergb", "next_theme", dry_run=False)
        decision = RuleEngine([rule], executor=executor).evaluate(self.EVENT)[0]
        self.assertEqual(decision.outcome, "executed")
        self.assertTrue(decision.ok)
        self.assertEqual(decision.reason, "did the thing")
        self.assertEqual(executor.calls, [("acergb", "next_theme", {})])

    def test_a_failed_action_shows_up_in_the_decision(self):
        executor = RecordingExecutor(connectors.ExecutionResult(
            "acergb", "next_theme", False, "daemon is not on the bus",
            error_kind="daemon_absent"))
        rule = Rule.create("Victory", "acergb", "next_theme", dry_run=False)
        decision = RuleEngine([rule], executor=executor).evaluate(self.EVENT)[0]
        self.assertEqual(decision.outcome, "failed")
        self.assertEqual(decision.error_kind, "daemon_absent")
        self.assertFalse(decision.ok)
        self.assertIn("daemon_absent", decision.describe())

    def test_arming_one_rule_leaves_the_others_dry_run(self):
        executor = RecordingExecutor()
        armed = Rule.create("Victory", "acergb", "next_theme", dry_run=False)
        untouched = Rule.create("Victory", "acergb", "off")
        decisions = RuleEngine([armed, untouched],
                               executor=executor).evaluate(self.EVENT)
        self.assertEqual([d.outcome for d in decisions], ["executed", "dry_run"])
        self.assertEqual(executor.calls, [("acergb", "next_theme", {})])

    def test_an_armed_rule_without_an_executor_stops_at_approved(self):
        rule = Rule.create("Victory", "acergb", "next_theme", dry_run=False)
        decision = RuleEngine([rule]).evaluate(self.EVENT)[0]
        self.assertEqual(decision.outcome, "approved")
        self.assertFalse(decision.ok)

    def test_policy_gates_still_win_over_arming(self):
        # Arming does not buy past the deny-by-default gates.
        armed = Rule.create("Open_Palm", "kde", "next_desktop", dry_run=False)
        executor = RecordingExecutor()
        decision = RuleEngine([armed], executor=executor).evaluate({
            "gesture": "Open_Palm", "actor": None, "source": "webcam"})[0]
        self.assertEqual(decision.outcome, "blocked")
        self.assertEqual(executor.calls, [])

    def test_dry_run_is_persisted_per_rule_with_no_global_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            store = RuleStore(path)
            store.save([Rule.create("Victory", "acergb", "next_theme",
                                    dry_run=False),
                        Rule.create("Victory", "acergb", "off")])
            payload = json.loads(path.read_text())
            self.assertNotIn("dry_run", payload)           # no global any more
            self.assertEqual([rule["dry_run"] for rule in payload["rules"]],
                             [False, True])
            self.assertEqual([rule.dry_run for rule in store.load()],
                             [False, True])

    def test_legacy_files_load_as_dry_run(self):
        # The founder's live rules.json predates this field. It must stay
        # dry-run, and the stale global must not arm anything either.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            path.write_text(json.dumps({"version": 1, "dry_run": False, "rules": [
                {"id": "a", "gesture": "Open_Palm", "connector": "mpris",
                 "action": "play_pause", "actor": "*", "source": "*",
                 "enabled": True, "require_liveness": False,
                 "require_confirmation": False},
            ]}))
            rules = RuleStore(path).load()
            self.assertEqual([rule.dry_run for rule in rules], [True])

    def test_the_live_rule_file_is_still_dry_run(self):
        live = Path.home() / ".config" / "acesvision" / "rules.json"
        if not live.exists():
            self.skipTest("no live rules.json on this machine")
        rules = RuleStore(live).load(strict=False)
        armed = [f"{rule.connector}.{rule.action}"
                 for rule in rules if not rule.dry_run]
        self.assertEqual(armed, [], f"live rules were armed: {armed}")


class GuiConnectorTests(unittest.TestCase):
    def backend(self, executor):
        from acesvision.gui import VisionBackend

        return VisionBackend(initialize_models=False, load_saved_rules=False,
                             executor=executor)

    def test_a_failed_action_reaches_the_error_banner(self):
        executor = RecordingExecutor(connectors.ExecutionResult(
            "acergb", "next_theme", False, "daemon is not on the bus",
            error_kind="daemon_absent"))
        backend = self.backend(executor)
        backend._rules = [Rule.create("Victory", "acergb", "next_theme",
                                      dry_run=False)]
        backend.rule_engine.rules = list(backend._rules)
        backend._receive_gesture({"gesture": "Victory", "actor": "Toby",
                                  "source": "webcam"})
        self.assertIn("daemon_absent", backend.lastError)
        self.assertIn("failed", backend.lastDecision)

    def test_a_dry_run_decision_leaves_the_error_banner_clean(self):
        backend = self.backend(RecordingExecutor())
        backend._rules = [Rule.create("Victory", "acergb", "next_theme")]
        backend.rule_engine.rules = list(backend._rules)
        backend._receive_gesture({"gesture": "Victory", "actor": "Toby",
                                  "source": "webcam"})
        self.assertEqual(backend.lastError, "")
        self.assertIn("dry_run", backend.lastDecision)

    def test_arming_is_per_rule_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = self.backend(RecordingExecutor())
            backend.rule_store = RuleStore(Path(directory) / "rules.json")
            first = Rule.create("Victory", "acergb", "next_theme")
            second = Rule.create("Victory", "acergb", "off")
            backend._rules = [first, second]
            backend.setRuleDryRun(first.id, False)
            self.assertEqual([rule["dryRun"] for rule in backend.rules],
                             [False, True])
            self.assertEqual(
                [rule.dry_run for rule in backend.rule_store.load()],
                [False, True])
            self.assertEqual([rule.dry_run for rule in backend.rule_engine.rules],
                             [False, True])

    def test_the_gui_reports_which_connectors_can_actually_execute(self):
        from acesvision.gui import VisionBackend

        backend = VisionBackend(initialize_models=False, load_saved_rules=False)
        self.assertEqual(backend.executableConnectors, ["acergb", "overlay"])
        backend._rules = [Rule.create("Shush", "pipewire", "mute"),
                          Rule.create("Victory", "acergb", "next_theme")]
        self.assertEqual([rule["executable"] for rule in backend.rules],
                         [False, True])

    def test_an_armed_overlay_rule_toggles_clean_and_restores_the_prior_profile(self):
        from acesvision.gui import VisionBackend

        backend = VisionBackend(initialize_models=False, load_saved_rules=False)
        backend.setOverlayProfile("broadcast")
        backend._rules = [Rule.create("Open_Palm", "overlay", "toggle_clean",
                                      dry_run=False)]
        backend.rule_engine.rules = list(backend._rules)

        event = {"gesture": "Open_Palm", "actor": "Toby", "source": "webcam"}
        backend._receive_gesture(event)
        self.assertEqual(backend.overlayProfile, "clean")
        self.assertIn("executed", backend.lastDecision)

        backend._receive_gesture(event)
        self.assertEqual(backend.overlayProfile, "broadcast")


class HeadlessConnectorTests(unittest.TestCase):
    """The headless path must work with no Qt anywhere in it."""

    def test_the_connector_module_imports_without_qt(self):
        import subprocess
        import sys

        script = (
            "import sys\n"
            "sys.modules['PySide6'] = None\n"
            "import acesvision.connectors as c\n"
            "import acesvision.policy, acesvision.__main__\n"
            "assert 'PySide6.QtCore' not in sys.modules\n"
            "print(c.default_registry().names())\n"
        )
        completed = subprocess.run([sys.executable, "-c", script],
                                   cwd=str(Path(__file__).parent),
                                   capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("acergb", completed.stdout)

    def test_the_runner_loads_rules_and_wires_the_registry(self):
        from acesvision.__main__ import build_parser, build_rule_engine

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rules.json"
            store = RuleStore(path)
            store.save([Rule.create("Victory", "acergb", "next_theme")])
            executor = RecordingExecutor()
            engine, rejected = build_rule_engine(
                build_parser().parse_args([]), store=RuleStore(path),
                executor=executor)
            self.assertEqual(rejected, [])
            self.assertIs(engine.executor, executor)
            self.assertEqual([rule.dry_run for rule in engine.rules], [True])

    def test_no_rules_flag_loads_nothing(self):
        from acesvision.__main__ import build_parser, build_rule_engine

        engine, rejected = build_rule_engine(
            build_parser().parse_args(["--no-rules"]))
        self.assertEqual((engine.rules, rejected), ([], []))

    def test_decisions_are_printed_including_failures(self):
        from acesvision.__main__ import _printing_callback

        executor = RecordingExecutor(connectors.ExecutionResult(
            "acergb", "next_theme", False, "daemon is not on the bus",
            error_kind="daemon_absent"))
        engine = RuleEngine(
            [Rule.create("Victory", "acergb", "next_theme", dry_run=False)],
            executor=executor)
        lines = []
        with patch("builtins.print", lines.append):
            _printing_callback(engine)({"gesture": "Victory", "actor": "Toby",
                                        "source": "webcam"})
        self.assertTrue(any("daemon_absent" in line for line in lines))
        self.assertTrue(any(line.startswith("[decision]!!") for line in lines))


def observation(detected=(), raw=(), face=False, at_s=0.0):
    """One synthetic captured frame for the live-verification scorer."""
    return verify.FrameObservation(
        detected=tuple(detected), raw_categories=tuple(raw),
        face_present=bool(face), face_count=1 if face else 0,
        hands=1 if detected else 0, at_s=at_s,
    )


class LiveVerificationScoringTests(unittest.TestCase):
    """Scoring for verify_gestures_live, driven entirely by synthetic frames.

    The harness itself can only be run in front of a real webcam, so the part
    that decides PASS/FAIL is deliberately pure and is proven here instead.
    """

    def test_a_clean_hold_passes_with_the_measured_rate(self):
        score = verify.score_attempt(
            "Thumb_Up", [observation(["Thumb_Up"], ["Thumb_Up"], face=True)] * 10)
        self.assertTrue(score.passed)
        self.assertEqual((score.frames, score.hits), (10, 10))
        self.assertEqual(score.detection_rate, 1.0)
        self.assertEqual(score.face_rate, 1.0)
        self.assertEqual(score.misclassified, ())
        self.assertEqual(score.verdict, "PASS")

    def test_no_frames_scores_an_honest_zero_and_fails(self):
        score = verify.score_attempt("Victory", [])
        self.assertFalse(score.passed)
        self.assertEqual((score.frames, score.hits), (0, 0))
        self.assertEqual((score.detection_rate, score.face_rate), (0.0, 0.0))
        self.assertIn("no frames", score.reason)

    def test_a_hold_below_the_bar_fails_and_names_the_rate(self):
        frames = ([observation(["Open_Palm"])] * 5 + [observation()] * 5)
        score = verify.score_attempt("Open_Palm", frames, min_rate=0.6)
        self.assertFalse(score.passed)
        self.assertEqual(score.detection_rate, 0.5)
        self.assertIn("50%", score.reason)
        self.assertIn("60%", score.reason)
        # The same frames pass against a lower bar — the bar is what decides.
        self.assertTrue(verify.score_attempt("Open_Palm", frames,
                                             min_rate=0.5).passed)

    def test_face_rate_counts_only_frames_that_kept_a_box(self):
        frames = [observation(["Victory"], face=True)] * 3 + \
                 [observation(["Victory"], face=False)] * 1
        score = verify.score_attempt("Victory", frames)
        self.assertEqual(score.face_rate, 0.75)
        self.assertEqual(score.detection_rate, 1.0)

    def test_misclassifications_are_ranked_and_counted_once_per_frame(self):
        frames = [
            observation(["Closed_Fist", "Thumb_Up", "Thumb_Up"]),   # two hands
            observation(["Thumb_Up"]),
            observation(["Victory"]),
        ]
        score = verify.score_attempt("Closed_Fist", frames)
        self.assertEqual(score.misclassified, (("Thumb_Up", 2), ("Victory", 1)))
        self.assertEqual(score.hits, 1)

    def test_raw_mediapipe_labels_are_recorded_alongside_the_verdict(self):
        frames = [observation(["Shush"], ["Pointing_Up"], face=True)] * 4
        score = verify.score_attempt("Shush", frames)
        self.assertEqual(score.raw_labels, (("Pointing_Up", 4),))
        # The model said Pointing_Up on every frame, but the custom pose won,
        # so nothing leaked and nothing would have fired next-theme.
        self.assertEqual(score.leak_frames, 0)
        self.assertTrue(score.passed)

    def test_a_single_leaked_pointing_up_fails_the_shush(self):
        # The defect this whole harness exists to measure: an occluded mouth
        # drops the face box, is_shush returns False, and the frame degrades to
        # Pointing_Up -- which automations.json binds to `next-theme`.
        frames = [observation(["Shush"], ["Pointing_Up"], face=True)] * 19 + \
                 [observation(["Pointing_Up"], ["Pointing_Up"], face=False)]
        score = verify.score_attempt("Shush", frames)
        self.assertEqual(score.leak_frames, 1)
        self.assertEqual(score.detection_rate, 0.95)   # 95% is not good enough
        self.assertFalse(score.passed)
        self.assertIn("next-theme", score.reason)

    def test_leaks_are_recorded_for_other_gestures_but_do_not_fail_them(self):
        frames = [observation(["Victory"])] * 9 + [observation(["Pointing_Up"])]
        score = verify.score_attempt("Victory", frames)
        self.assertEqual(score.leak_frames, 1)
        self.assertTrue(score.passed)

    def test_pointing_up_is_never_its_own_leak(self):
        score = verify.score_attempt(
            "Pointing_Up", [observation(["Pointing_Up"])] * 4)
        self.assertEqual(score.leak_frames, 0)
        self.assertTrue(score.passed)

    def test_overall_verdict_needs_every_gesture(self):
        good = verify.score_attempt("Victory", [observation(["Victory"])] * 4)
        bad = verify.score_attempt("Shush", [])
        self.assertEqual(verify.overall_verdict([good])[0], True)
        passed, summary = verify.overall_verdict([good, bad])
        self.assertFalse(passed)
        self.assertIn("Shush", summary)
        self.assertIn("1/2", summary)
        self.assertEqual(verify.overall_verdict([])[0], False)

    def test_every_catalog_gesture_has_a_prompt(self):
        # Drift guard: a gesture added to the catalog with no instruction would
        # be prompted as a blank line the founder cannot act on.
        for spec in gesture_catalog.GESTURES:
            self.assertIn(spec.id, verify.INSTRUCTIONS)
        self.assertEqual(len(verify.INSTRUCTIONS), len(gesture_catalog.GESTURES))

    def test_gesture_subset_selection_keeps_catalog_order(self):
        chosen = verify.selected_gestures("shush,thumb up")
        self.assertEqual([spec.id for spec in chosen], ["Thumb_Up", "Shush"])
        self.assertEqual(len(verify.selected_gestures("")),
                         len(gesture_catalog.GESTURES))
        with self.assertRaises(ValueError):
            verify.selected_gestures("wave")


class LiveVerificationReportTests(unittest.TestCase):
    def _scores(self):
        return [
            verify.score_attempt("Thumb_Up",
                                 [observation(["Thumb_Up"], ["Thumb_Up"])] * 8),
            verify.score_attempt(
                "Shush",
                [observation(["Shush"], ["Pointing_Up"], face=True)] * 6 +
                [observation(["Pointing_Up"], ["Pointing_Up"])] * 4),
        ]

    def test_report_carries_the_table_the_verdicts_and_the_leak_count(self):
        meta = verify.RunMeta(started_at=datetime(2026, 8, 15, 14, 30),
                              camera_label="Integrated Camera",
                              camera_path="/dev/video1", resolution="1280x720",
                              face_engine="yunet")
        text = verify.render_report(self._scores(), meta)
        self.assertIn("# Live gesture verification — 2026-08-15 14:30:00", text)
        self.assertIn("**Verdict: FAIL", text)
        self.assertIn("/dev/video1", text)
        self.assertIn("| `Thumb_Up` |", text)
        self.assertIn("**PASS**", text)
        self.assertIn("**FAIL**", text)
        self.assertIn("Shush occlusion test", text)
        self.assertIn("`Pointing_Up` leaked: 4 frame(s)", text)
        self.assertIn("Face box survived: **60%** (6 frames)", text)
        self.assertIn("no rule loaded, no connector armed", text)

    def test_report_says_so_when_the_shush_was_not_measured(self):
        text = verify.render_report(
            [verify.score_attempt("Victory", [observation(["Victory"])])],
            verify.RunMeta())
        self.assertIn("Shush occlusion test", text)
        self.assertIn("Not measured", text)
        self.assertIn("**Verdict: PASS", text)

    def test_report_is_written_and_never_overwrites_an_earlier_run(self):
        when = datetime(2026, 8, 15, 14, 30, 5)
        with tempfile.TemporaryDirectory() as directory:
            first = verify.write_report("one", directory, when)
            self.assertEqual(first.name, "live-gestures-2026-08-15.md")
            self.assertEqual(first.read_text(), "one")
            second = verify.write_report("two", directory, when)
            self.assertEqual(second.name, "live-gestures-2026-08-15-143005.md")
            self.assertEqual(first.read_text(), "one")   # still intact
            self.assertEqual(second.read_text(), "two")


class FakeRecognizer:
    """Stand-in for the MediaPipe GestureRecognizer."""

    Result = namedtuple("Result", "hand_landmarks gestures")
    Category = namedtuple("Category", "category_name score")

    def __init__(self, landmarks, label="Pointing_Up", score=0.9):
        self.landmarks = landmarks
        self.label = label
        self.score = score
        self.calls = 0

    def recognize(self, image):
        self.calls += 1
        return self.Result([self.landmarks],
                           [[self.Category(self.label, self.score)]])

    def close(self):
        pass


class FakeGestureDetector:
    """Only what capture_attempt uses: recognize() and min_score."""

    def __init__(self, landmarks, label="Pointing_Up"):
        self.rec = FakeRecognizer(landmarks, label)
        self.min_score = 0.5

    def recognize(self, frame):
        return self.rec.recognize(None), FRAME_W, FRAME_H


class LiveVerificationCaptureTests(unittest.TestCase):
    """The recording loop, exercised with synthetic frames and no camera."""

    def _run(self, face_sequence, reads=4):
        frames = [(True, np.zeros((4, 4, 3), np.uint8)) for _ in range(reads)]
        cap = FakeCapture(frames)
        detector = FakeGestureDetector(pointing_hand(AT_LIPS))
        faces = list(face_sequence)

        def face_detector(frame):
            return faces.pop(0) if faces else []

        # hold_s is long; the run ends on the read-failure guard instead, which
        # makes the frame count deterministic rather than wall-clock dependent.
        return verify.capture_attempt(cap, "Shush", detector, face_detector,
                                      hold_s=60.0, out=lambda *a: None)

    def test_the_same_hand_reads_as_shush_or_as_a_leak_by_the_face_box_alone(self):
        observations, failures = self._run([[FACE_BOX], [FACE_BOX], [], []])
        self.assertEqual(len(observations), 4)
        self.assertEqual([item.detected for item in observations],
                         [("Shush",), ("Shush",), ("Pointing_Up",),
                          ("Pointing_Up",)])
        # MediaPipe called every one of them Pointing_Up.
        self.assertEqual([item.raw_categories for item in observations],
                         [("Pointing_Up",)] * 4)
        self.assertEqual([item.face_present for item in observations],
                         [True, True, False, False])
        self.assertEqual(failures, 30)   # the guard that ends a dead capture

        score = verify.score_attempt("Shush", observations)
        self.assertEqual((score.hits, score.leak_frames), (2, 2))
        self.assertEqual((score.detection_rate, score.face_rate), (0.5, 0.5))
        self.assertFalse(score.passed)

    def test_a_failing_face_detector_never_aborts_the_run(self):
        def exploding(frame):
            raise RuntimeError("yunet fell over")

        cap = FakeCapture([(True, np.zeros((4, 4, 3), np.uint8))])
        detector = FakeGestureDetector(pointing_hand(AT_LIPS))
        observations, _ = verify.capture_attempt(
            cap, "Shush", detector, exploding, hold_s=60.0, out=lambda *a: None)
        self.assertEqual(len(observations), 1)
        self.assertFalse(observations[0].face_present)
        self.assertEqual(observations[0].detected, ("Pointing_Up",))

    def test_frame_saver_keeps_two_samples_plus_the_first_defect(self):
        saver = verify.FrameSaver("/nonexistent", per_gesture=3, hold_s=10.0)
        clean = observation(["Shush"])
        leak = observation(["Pointing_Up"])
        self.assertIsNone(saver.wanted("Shush", clean, 0.5))    # too early
        self.assertEqual(saver.wanted("Shush", leak, 0.5), "defect")
        saver._record("Shush", "defect")
        self.assertIsNone(saver.wanted("Shush", leak, 0.6))     # only the first
        self.assertIsNone(saver.wanted("Shush", clean, 1.0))
        self.assertEqual(saver.wanted("Shush", clean, 3.0), "early")
        saver._record("Shush", "early")
        self.assertEqual(saver.wanted("Shush", clean, 8.0), "late")
        saver._record("Shush", "late")
        self.assertIsNone(saver.wanted("Shush", clean, 9.0))    # capped at 3
        # A second gesture gets its own budget.
        self.assertEqual(saver.wanted("Victory", leak, 3.0), "defect")

    def test_saved_frames_are_annotated_jpegs_on_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            saver = verify.FrameSaver(directory)
            frame = np.zeros((200, 200, 3), np.uint8)
            path = saver.save("Shush", "defect", frame,
                              [Face(20, 20, 60, 80, "Toby", 0.2, True)],
                              [Gesture("Pointing_Up", 0.9, 10, 10, 40, 40)], 9)
            self.assertEqual(path.name, "09-Shush-defect.jpg")
            self.assertTrue(path.exists() and path.stat().st_size > 0)
            # The overlay drew on a copy; the captured frame is untouched.
            self.assertEqual(int(frame.max()), 0)

    def test_a_contended_camera_reports_the_holder_and_exits_without_spinning(self):
        lines = []
        with patch.object(verify, "choose_camera",
                          return_value=("/dev/video1", "Cam", "/dev/video1")), \
                patch.object(camera, "open_camera",
                             side_effect=camera.CameraBusyError("in use")), \
                patch.object(verify, "describe_holders",
                             return_value="obs (pid 4242)"):
            code = verify.main([], out=lines.append)
        self.assertEqual(code, verify.EXIT_BUSY)
        self.assertTrue(any("obs (pid 4242)" in line for line in lines))
        self.assertTrue(any("Nothing was measured" in line for line in lines))


class LiveVerificationSafetyTests(unittest.TestCase):
    """The harness must not be able to move a light, by construction."""

    SOURCE = Path(__file__).with_name("verify_gestures_live.py")

    def test_it_imports_no_rule_or_connector_machinery(self):
        import ast

        tree = ast.parse(self.SOURCE.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                imported.add(module)
                imported.update(f"{module}.{alias.name}" for alias in node.names)
        # Lazy, in-function imports are visible to ast too, so this covers the
        # whole file rather than just the header.
        for banned in ("acesvision.policy", "acesvision.connectors",
                       "acesvision.__main__", "policy", "connectors", "dbus"):
            self.assertNotIn(banned, imported)

    def test_it_names_no_executor_and_no_rules_file(self):
        import ast

        tree = ast.parse(self.SOURCE.read_text())
        used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        used |= {node.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Attribute)}
        for banned in ("RuleEngine", "RuleStore", "default_registry",
                       "AceRgbConnector", "evaluate", "execute", "subprocess",
                       "Popen"):
            self.assertNotIn(banned, used)
        literals = [node.value for node in ast.walk(tree)
                    if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        # The module docstring is allowed to explain what the file must not do.
        literals.remove(ast.get_docstring(tree, clean=False))
        self.assertFalse([text for text in literals if "rules.json" in text])

    def test_the_report_directory_is_gitignored(self):
        # The annotated frames contain the founder's face. They must never be
        # committable, so this is asserted rather than assumed.
        ignore = Path(__file__).with_name(".gitignore").read_text().splitlines()
        self.assertIn("verification/", [line.strip() for line in ignore])


class GestureRecognizeSeamTests(unittest.TestCase):
    def test_detect_is_recognize_plus_classify(self):
        # verify_gestures_live needs the model's own label as well as the
        # classified one; recognize() exists so it does not run inference twice.
        from gestures import GestureDetector

        detector = GestureDetector.__new__(GestureDetector)
        detector.rec = FakeRecognizer(pointing_hand(AT_LIPS))
        detector.min_score = 0.5
        frame = np.zeros((FRAME_H, FRAME_W, 3), np.uint8)

        result, width, height = detector.recognize(frame)
        self.assertEqual((width, height), (FRAME_W, FRAME_H))
        self.assertEqual(len(result.hand_landmarks), 1)

        self.assertEqual([row.name for row in detector.detect(frame)],
                         ["Pointing_Up"])
        self.assertEqual([row.name for row in
                          detector.detect(frame, faces=[FACE_BOX])], ["Shush"])
        self.assertEqual(detector.rec.calls, 3)   # one pass per detect call


# ---------------------------------------------------------------------------
# Worker protocol — the silent one-frame-stale desync, and the device gate.
# ---------------------------------------------------------------------------


class PipeWorker:
    """A pipe-backed stand-in for the YOLO worker subprocess.

    Real os.pipe() file descriptors, because the adapter blocks in
    select.select() on the worker's stdout — a Mock cannot exercise that path.
    """

    def __init__(self):
        to_worker_r, to_worker_w = os.pipe()
        from_worker_r, from_worker_w = os.pipe()
        # Unbuffered, matching Popen(bufsize=0): a buffered reader would hide
        # an arrived reply from select() and the test would pass for the wrong
        # reason (or fail for one).
        self.stdin = os.fdopen(to_worker_w, "wb", 0)         # parent writes
        self.stdout = os.fdopen(from_worker_r, "rb", 0)      # parent reads
        self._requests = os.fdopen(to_worker_r, "rb", 0)     # worker reads
        self._replies = os.fdopen(from_worker_w, "wb", 0)    # worker writes
        self.stderr = None
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0
        return 0

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9

    def reply(self, payload):
        encoded = json.dumps(payload).encode()
        self._replies.write(struct.pack(">I", len(encoded)) + encoded)
        self._replies.flush()

    def read_request(self):
        """(sequence, payload) for one framed request the adapter sent."""
        header = self._read_exact(8)
        sequence, size = struct.unpack(">II", header)
        return sequence, self._read_exact(size)

    def _read_exact(self, size):
        chunks = []
        while size:
            chunk = self._requests.read(size)
            if not chunk:
                raise EOFError
            chunks.append(chunk)
            size -= len(chunk)
        return b"".join(chunks)

    def close(self):
        for handle in (self.stdin, self.stdout, self._requests, self._replies):
            try:
                handle.close()
            except OSError:
                pass


class WorkerProtocolTests(unittest.TestCase):
    """The reply for frame N must never be returned as the answer to frame N+1."""

    def setUp(self):
        self.worker = PipeWorker()
        self.addCleanup(self.worker.close)
        self.frame = np.zeros((8, 8, 3), dtype=np.uint8)

    def _detector(self, **kwargs):
        detector = YoloSubprocessDetector(**kwargs)
        detector._process = self.worker          # skip Popen and the handshake
        return detector

    @staticmethod
    def _objects(label):
        return [{"x": 0, "y": 0, "w": 4, "h": 4, "label": label,
                 "score": 0.9, "track_id": 1}]

    def test_timeout_does_not_desync_the_pipe_for_ever(self):
        """Reproducer for the silent stale-frame bug.

        Frame 1 times out. Its reply then lands in the pipe anyway, because the
        worker is still alive and still working. Frame 2 must get frame 2's
        answer — not frame 1's, and not one frame behind for the rest of the
        process's life.
        """
        detector = self._detector(timeout_s=0.05)

        with self.assertRaises(TimeoutError):
            detector.detect(self.frame)
        first_sequence, _ = self.worker.read_request()

        # The abandoned reply arrives late, followed by the live one.
        self.worker.reply({"seq": first_sequence, "objects": self._objects("couch")})
        self.worker.reply({"seq": first_sequence + 1,
                           "objects": self._objects("person")})

        detector.timeout_s = 2.0
        objects, _, _ = detector.detect(self.frame)
        self.assertEqual([item.label for item in objects], ["person"])
        self.assertEqual(detector.stale_replies_discarded, 1)

    def test_requests_carry_an_increasing_frame_tag(self):
        detector = self._detector(timeout_s=2.0)
        for expected in (1, 2, 3):
            self.worker.reply({"seq": expected, "objects": []})
            detector.detect(self.frame)
            sequence, payload = self.worker.read_request()
            self.assertEqual(sequence, expected)
            self.assertTrue(payload.startswith(b"\xff\xd8"))   # JPEG SOI

    def test_untagged_reply_is_a_loud_protocol_error(self):
        detector = self._detector(timeout_s=2.0)
        self.worker.reply({"objects": self._objects("person")})
        with self.assertRaises(RuntimeError) as caught:
            detector.detect(self.frame)
        self.assertIn("frame tag", str(caught.exception))

    def test_timeout_names_the_frame_it_abandoned(self):
        detector = self._detector(timeout_s=0.05)
        with self.assertRaises(TimeoutError) as caught:
            detector.detect(self.frame)
        self.assertIn("frame 1", str(caught.exception))

    def test_handshake_rejects_a_device_that_does_not_exist(self):
        detector = self._detector()
        self.worker.reply({"seq": 0, "ready": False,
                           "error": "device '9' does not exist"})
        with self.assertRaises(WorkerDeviceError) as caught:
            detector._handshake()
        self.assertIn("device '9'", str(caught.exception))

    def test_handshake_records_the_resolved_device(self):
        detector = self._detector()
        self.worker.reply({"seq": 0, "ready": True, "device": "cpu",
                           "model": "ultralytics:yolo26n"})
        detector._handshake()
        self.assertEqual(detector.resolved_device, "cpu")
        self.assertEqual(detector.model_id, "ultralytics:yolo26n")


class WorkerDeviceResolutionTests(unittest.TestCase):
    """device='9' used to succeed silently and return normal detections."""

    def test_auto_prefers_a_gpu_and_falls_back_to_cpu(self):
        self.assertEqual(resolve_device("auto", available=1), "0")
        self.assertEqual(resolve_device("auto", available=0), "cpu")
        self.assertEqual(resolve_device(None, available=0), "cpu")

    def test_explicit_cpu_is_always_valid(self):
        self.assertEqual(resolve_device("cpu", available=0), "cpu")

    def test_index_beyond_the_device_count_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            resolve_device("9", available=1)
        self.assertIn("valid indices 0..0", str(caught.exception))

    def test_gpu_index_on_a_host_with_no_gpu_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            resolve_device("0", available=0)
        self.assertIn("ACESVISION_YOLO_DEVICE=cpu", str(caught.exception))

    def test_qualified_and_bare_indices_both_resolve(self):
        self.assertEqual(resolve_device("cuda:1", available=2), "1")
        self.assertEqual(resolve_device("1", available=2), "1")

    def test_nonsense_device_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            resolve_device("gpu-please", available=2)


class WorkerEnvironmentTests(unittest.TestCase):
    def test_default_worker_python_is_this_interpreter_not_another_repo(self):
        self.assertEqual(str(default_worker_python({})), sys.executable)
        source = Path(perception.__file__).read_text()
        self.assertNotIn("cv-worker", source)

    def test_worker_python_is_overridable_by_environment(self):
        override = {"ACESVISION_YOLO_PYTHON": "/opt/py/bin/python"}
        self.assertEqual(str(default_worker_python(override)), "/opt/py/bin/python")

    def test_device_default_comes_from_environment(self):
        self.assertEqual(default_device({}), "auto")
        self.assertEqual(default_device({"ACESVISION_YOLO_DEVICE": "cpu"}), "cpu")

    def test_detector_reads_the_environment_when_it_is_built_not_when_imported(self):
        """Importing acesvision early used to freeze the configuration."""
        with patch.dict(os.environ, {"ACESVISION_YOLO_DEVICE": "cpu"}):
            self.assertEqual(YoloSubprocessDetector().device, "cpu")
        self.assertEqual(YoloSubprocessDetector().device, "auto")

    def test_rocm_override_is_set_only_on_an_amd_host(self):
        self.assertEqual(rocm_env_overrides({}, "0", present=True),
                         {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"})
        self.assertEqual(rocm_env_overrides({}, "0", present=False), {})

    def test_rocm_override_is_skipped_for_cpu_inference(self):
        self.assertEqual(rocm_env_overrides({}, "cpu", present=True), {})

    def test_inherited_override_always_wins(self):
        env = {"HSA_OVERRIDE_GFX_VERSION": "11.0.0"}
        self.assertEqual(rocm_env_overrides(env, "0", present=True), {})
        blank = {"HSA_OVERRIDE_GFX_VERSION": ""}
        self.assertEqual(rocm_env_overrides(blank, "0", present=True), {})

    def test_headless_runner_passes_a_device_through(self):
        from acesvision.__main__ import build_parser

        self.assertEqual(build_parser().parse_args([]).device, default_device())
        self.assertEqual(build_parser().parse_args(["--device", "cpu"]).device, "cpu")


# ---------------------------------------------------------------------------
# Gesture attribution — the two silent no-op gates.
# ---------------------------------------------------------------------------


class GestureSelectionTests(unittest.TestCase):
    """GestureDetector defaults to num_hands=2, so 'exactly one' meant never."""

    def test_two_hands_in_frame_still_produce_an_event(self):
        events = []
        output = GestureEventOutput(events.append, hold_frames=1,
                                    clock=Mock(return_value=1.0), enabled=True)
        source = SourceSpec.from_mapping({"type": "webcam"})
        output.publish(SceneFrame(source, 0, 0.0, np.zeros((2, 2, 3)),
                                  gestures=[Gesture("Open_Palm", 0.7, 0, 0, 5, 5),
                                            Gesture("Victory", 0.95, 40, 0, 5, 5)]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["gesture"], "Victory")
        self.assertEqual(events[0]["hands_in_frame"], 2)

    def test_most_confident_gesture_wins(self):
        picked = select_gesture([Gesture("Open_Palm", 0.61, 0, 0, 5, 5),
                                 Gesture("Thumb_Up", 0.99, 0, 0, 5, 5),
                                 Gesture("Victory", 0.72, 0, 0, 5, 5)])
        self.assertEqual(picked.name, "Thumb_Up")

    def test_custom_landmark_poses_keep_their_precedence(self):
        # Middle_Finger and Shush are scored 1.0 by gesture_catalog on purpose.
        picked = select_gesture([Gesture("Pointing_Up", 0.98, 0, 0, 5, 5),
                                 Gesture(gesture_catalog.SHUSH, 1.0, 0, 0, 5, 5)])
        self.assertEqual(picked.name, gesture_catalog.SHUSH)

    def test_no_gestures_selects_nothing(self):
        self.assertIsNone(select_gesture([]))
        self.assertIsNone(select_gesture(None))


class ActorAttributionTests(unittest.TestCase):
    """Two known faces used to mean actor=None and every actor rule going dead."""

    def setUp(self):
        self.source = SourceSpec.from_mapping({"type": "webcam"})

    def _publish(self, faces, gestures):
        events = []
        output = GestureEventOutput(events.append, hold_frames=1,
                                    clock=Mock(return_value=1.0), enabled=True)
        output.publish(SceneFrame(self.source, 0, 0.0, np.zeros((2, 2, 3)),
                                  faces=faces, gestures=gestures))
        return events

    def test_single_known_face_is_attributed_as_unique(self):
        events = self._publish([Face(0, 0, 5, 5, "Toby", 0.2, True)],
                               [Gesture("Victory", 0.9, 0, 0, 5, 5)])
        self.assertEqual(events[0]["actor"], "Toby")
        self.assertEqual(events[0]["actor_attribution"], "unique")

    def test_two_known_faces_attribute_to_the_nearest_hand(self):
        faces = [Face(0, 0, 10, 10, "A", 0.2, True),
                 Face(200, 0, 10, 10, "B", 0.2, True)]
        near_b = self._publish(faces, [Gesture("Open_Palm", 0.9, 195, 0, 10, 10)])
        self.assertEqual(near_b[0]["actor"], "B")
        self.assertEqual(near_b[0]["actor_attribution"], "nearest")
        self.assertEqual(near_b[0]["actor_candidates"], ["A", "B"])

        near_a = self._publish(faces, [Gesture("Open_Palm", 0.9, 0, 0, 10, 10)])
        self.assertEqual(near_a[0]["actor"], "A")

    def test_unknown_faces_are_never_actors(self):
        events = self._publish([Face(0, 0, 5, 5, None, 0.9, False)],
                               [Gesture("Victory", 0.9, 0, 0, 5, 5)])
        self.assertIsNone(events[0]["actor"])
        self.assertEqual(events[0]["actor_attribution"], "none")
        self.assertEqual(events[0]["identity_state"], "unknown")

    def test_ambiguity_without_geometry_is_named_not_silent(self):
        Blind = namedtuple("Blind", "name known")
        actor, attribution, candidates = attribute_actor(
            [Blind("A", True), Blind("B", True)],
            Gesture("Victory", 0.9, 0, 0, 5, 5))
        self.assertIsNone(actor)
        self.assertEqual(attribution, "ambiguous")
        self.assertEqual(candidates, ["A", "B"])

    def test_actor_scoped_rules_match_again_with_two_people_present(self):
        """The reason this mattered: the rule silently stopped firing."""
        rule = Rule.create("Victory", "mpris", "next", actor="Toby",
                           dry_run=True)
        engine = RuleEngine([rule])
        events = self._publish(
            [Face(0, 0, 10, 10, "Toby", 0.2, True),
             Face(300, 0, 10, 10, "Toby", 0.2, True)],
            [Gesture("Victory", 0.9, 0, 0, 10, 10)])
        self.assertEqual(len(engine.evaluate(events[0])), 1)

# ---------------------------------------------------------------------------
# GUI shell defects — error visibility, preview freshness, runtime capability
# detection, and the QML layout itself.
# ---------------------------------------------------------------------------


class FakeClock:
    """Monotonic clock the tests advance by hand."""

    def __init__(self, start=0.0):
        self.now = float(start)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)
        return self.now


def gui_backend(**kwargs):
    from acesvision.gui import VisionBackend

    kwargs.setdefault("initialize_models", False)
    kwargs.setdefault("load_saved_rules", False)
    return VisionBackend(**kwargs)


def pipeline_state(backend, **kwargs):
    """Push a synthetic PipelineState, the way a running capture loop would."""
    backend.pipeline._set_state(**kwargs)


class GuiErrorChannelTests(unittest.TestCase):
    """An error the operator never sees is the same as no error at all."""

    def test_an_action_error_survives_the_refresh_tick(self):
        # The 100 ms tick used to copy pipeline.last_error ("") straight over
        # whatever a slot had just set, so every message died inside one frame.
        backend = gui_backend()
        backend.useDroidCam("   ")
        self.assertIn("DroidCam URL", backend.lastError)
        for _ in range(10):
            backend._refresh()
        self.assertIn("DroidCam URL", backend.lastError)

    def test_a_connector_failure_stays_on_screen(self):
        executor = RecordingExecutor(connectors.ExecutionResult(
            "acergb", "next_theme", False, "daemon is not on the bus",
            error_kind="daemon_absent"))
        backend = gui_backend(executor=executor)
        backend._rules = [Rule.create("Victory", "acergb", "next_theme",
                                      dry_run=False)]
        backend.rule_engine.rules = list(backend._rules)
        backend._receive_gesture({"gesture": "Victory", "actor": "Toby",
                                  "source": "webcam"})
        for _ in range(20):
            backend._refresh()
        self.assertIn("daemon_absent", backend.lastError)
        self.assertTrue(backend.errorDismissable)

    def test_only_the_operator_clears_an_action_error(self):
        backend = gui_backend()
        backend.useDroidCam("")
        backend.dismissError()
        self.assertEqual(backend.lastError, "")
        self.assertFalse(backend.errorDismissable)

    def test_a_newer_action_error_supersedes_the_older_one(self):
        backend = gui_backend()
        backend.useDroidCam("")
        backend.useWebcam("not-a-number")
        self.assertIn("Webcam index", backend.lastError)

    def test_a_pipeline_error_shows_when_nothing_else_is_pending(self):
        backend = gui_backend()
        pipeline_state(backend, error="Webcam is unavailable or in use")
        backend._refresh()
        self.assertIn("unavailable", backend.lastError)
        self.assertFalse(backend.errorDismissable)   # live state, not dismissable

    def test_an_action_error_outranks_a_pipeline_error(self):
        backend = gui_backend()
        pipeline_state(backend, error="Webcam is unavailable or in use")
        backend._refresh()
        backend.useDroidCam("")
        self.assertIn("DroidCam URL", backend.lastError)
        backend.dismissError()
        backend._refresh()
        self.assertIn("unavailable", backend.lastError)

    def test_the_error_signal_fires_only_when_the_visible_text_changes(self):
        backend = gui_backend()
        seen = []
        backend.errorChanged.connect(lambda: seen.append(backend.lastError))
        backend.useDroidCam("")
        pipeline_state(backend, error="something the operator cannot see yet")
        backend._refresh()
        backend._refresh()
        self.assertEqual(len(seen), 1)


class GuiDroidCamScanTests(unittest.TestCase):
    """The Sources page must say which network it is about to scan, and must
    not report an unscannable host in the words it uses for an empty one."""

    def plan(self):
        return ScanPlan(
            networks=(ipaddress.ip_network("192.168.68.0/24"),),
            selected=(NetworkInterface("enp8s0", "192.168.68.107",
                                       "255.255.255.0", kind="ethernet",
                                       is_up=True, is_physical=True),),
            excluded=(("tailscale0", "VPN overlay (name matches tailscale*)"),),
        )

    def test_the_page_names_the_network_before_anything_is_scanned(self):
        backend = gui_backend()
        with patch("acesvision.gui.scan_plan", return_value=self.plan()):
            self.assertEqual(backend.scanPlanTarget, "192.168.68.0/24 (enp8s0)")

    def test_an_unscannable_host_is_not_reported_as_no_device_found(self):
        backend = gui_backend()
        scanner = Mock()
        with patch("acesvision.gui.scan_plan", return_value=ScanPlan()), \
                patch("acesvision.gui.scan_droidcam", scanner):
            backend.scanDroidCams()
        scanner.assert_not_called()
        self.assertFalse(backend.droidScanActive)
        self.assertIn("No scannable network", backend.droidScanStatus)
        self.assertIn(SCAN_INTERFACES_ENV, backend.droidScanStatus)

    def test_a_refused_override_reaches_the_operator(self):
        backend = gui_backend()
        with patch("acesvision.gui.scan_plan",
                   side_effect=ValueError("not a private local network")):
            backend.scanDroidCams()
        self.assertIn("Scan refused", backend.droidScanStatus)
        self.assertIn("private local network", backend.lastError)

    def test_the_scan_probes_exactly_the_planned_networks(self):
        backend = gui_backend()
        probed = []
        finished = threading.Event()

        def scanner(networks):
            probed.append(list(networks))
            finished.set()
            return []

        with patch("acesvision.gui.scan_plan", return_value=self.plan()), \
                patch("acesvision.gui.scan_droidcam", scanner):
            backend.scanDroidCams()
            self.assertTrue(finished.wait(5))
        self.assertEqual(probed, [["192.168.68.0/24"]])
        self.assertIn("192.168.68.0/24", backend.droidScanStatus)

    def test_a_finished_scan_says_which_network_it_covered(self):
        backend = gui_backend()
        backend._apply_droid_scan(json.dumps(
            {"devices": [], "networks": ["192.168.68.0/24"]}))
        self.assertIn("192.168.68.0/24", backend.droidScanStatus)
        self.assertIn("nothing answered", backend.droidScanStatus)

    def test_a_real_listener_reaches_the_sources_page(self):
        """The Sources page button, over a real socket, with nothing mocked
        but the choice of network.

        This is the click the operator actually makes, and it is a second call
        site: the CLI passes the plan's network objects, the GUI passes their
        strings, and neither had ever been run against something that was
        really listening. It shares one probe budget with the CLI, so it also
        pins that the GUI does not quietly carry a narrower one.
        """
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server.bind(("127.0.0.1", DROIDCAM_PORT))
        except OSError:
            server.close()
            self.skipTest(f"port {DROIDCAM_PORT} is in use on this host")
        server.listen(8)
        self.addCleanup(server.close)
        stop = threading.Event()
        self.addCleanup(stop.set)

        def accept_loop():
            server.settimeout(0.1)
            while not stop.is_set():
                try:
                    server.accept()[0].close()
                except (TimeoutError, OSError):
                    continue

        thread = threading.Thread(target=accept_loop, daemon=True)
        thread.start()
        self.addCleanup(thread.join, 2.0)

        backend = gui_backend()
        finished = threading.Event()
        calls, results = [], []

        def observed(*args, **kwargs):
            # A passthrough, not a stub: the real scan runs, over the real
            # socket, with exactly the arguments the GUI chose. Observing here
            # rather than on droidScanFinished is how the sibling tests do it,
            # because a cross-thread Qt signal needs an event loop to arrive.
            calls.append((args, kwargs))
            try:
                results.extend(scan_droidcam(*args, **kwargs))
            finally:
                finished.set()
            return list(results)

        plan = ScanPlan(networks=(ipaddress.ip_network("127.0.0.1/32"),))
        with patch("acesvision.gui.scan_plan", return_value=plan), \
                patch("acesvision.gui.scan_droidcam", observed):
            backend.scanDroidCams()
            self.assertTrue(finished.wait(15), "the scan never came back")

        self.assertEqual([device.url for device in results],
                         [f"http://127.0.0.1:{DROIDCAM_PORT}/video"])
        # No probe budget of its own: the GUI and the CLI share one default, so
        # a fix to that default cannot reach one page and miss the other.
        self.assertNotIn("timeout_s", calls[0][1])
        self.assertEqual(len(calls[0][0]), 1)

        backend._apply_droid_scan(json.dumps({
            "devices": [device.as_dict() for device in results],
            "networks": ["127.0.0.1/32"],
        }))
        self.assertEqual([device["url"] for device in backend.droidCams],
                         [f"http://127.0.0.1:{DROIDCAM_PORT}/video"])
        self.assertIn("Found 1", backend.droidScanStatus)


class GuiPreviewFreshnessTests(unittest.TestCase):
    """A frozen last frame renders exactly like live video. Say which it is."""

    def test_a_stalled_capture_is_reported_as_stale(self):
        clock = FakeClock()
        backend = gui_backend(clock=clock)
        pipeline_state(backend, status="live", sequence=1)
        backend._refresh()
        self.assertFalse(backend.previewStale)
        clock.advance(0.5)
        backend._refresh()
        self.assertFalse(backend.previewStale)
        clock.advance(5.0)
        backend._refresh()
        self.assertTrue(backend.previewStale)

    def test_a_fresh_frame_clears_staleness(self):
        clock = FakeClock()
        backend = gui_backend(clock=clock)
        pipeline_state(backend, status="live", sequence=1)
        backend._refresh()
        clock.advance(5.0)
        backend._refresh()
        self.assertTrue(backend.previewStale)
        pipeline_state(backend, status="live", sequence=2)
        backend._refresh()
        self.assertFalse(backend.previewStale)

    def test_a_runtime_that_never_produced_a_frame_is_not_stale(self):
        clock = FakeClock()
        backend = gui_backend(clock=clock)
        pipeline_state(backend, status="reconnecting")
        clock.advance(60.0)
        backend._refresh()
        self.assertFalse(backend.previewStale)

    def test_a_stopped_runtime_is_not_reported_as_a_stale_feed(self):
        clock = FakeClock()
        backend = gui_backend(clock=clock)
        pipeline_state(backend, status="live", sequence=1)
        backend._refresh()
        clock.advance(30.0)
        pipeline_state(backend, status="stopped")
        backend._refresh()
        self.assertFalse(backend.previewStale)

    def test_the_preview_url_only_moves_when_a_frame_does(self):
        backend = gui_backend()
        first = backend.previewSource
        backend._refresh()
        self.assertEqual(backend.previewSource, first)
        pipeline_state(backend, status="live", sequence=1)
        backend._refresh()
        self.assertNotEqual(backend.previewSource, first)


class GuiExposureCapabilityTests(unittest.TestCase):
    """OSS blocker: one webcam's quirk must not disable a control for everyone."""

    BLACK = {"image_mean": 1.5, "image_std": 0.4,
             "camera_controls": {"auto_exposure": False}}

    def drive_black_frame(self, backend):
        backend.setCameraTuning(False, 251, 0, 0, 100)
        pipeline_state(backend, status="live", sequence=1, metrics=dict(self.BLACK))
        backend._refresh()

    def test_a_black_frame_in_manual_mode_withdraws_manual_exposure(self):
        backend = gui_backend()
        self.assertTrue(backend.manualExposureSupported)
        self.drive_black_frame(backend)
        self.assertFalse(backend.manualExposureSupported)
        self.assertIn("black frame", backend.exposureNotice)

    def test_automatic_exposure_is_restored_for_the_offending_camera(self):
        backend = gui_backend()
        self.drive_black_frame(backend)
        self.assertTrue(backend.autoExposure)
        self.assertTrue(backend.pipeline._camera_controls["auto_exposure"])
        self.assertIn("black frame", backend.lastError)

    def test_a_dark_but_not_flat_frame_keeps_manual_exposure(self):
        backend = gui_backend()
        backend.setCameraTuning(False, 251, 0, 0, 100)
        pipeline_state(backend, status="live", sequence=1,
                       metrics={"image_mean": 20.0, "image_std": 18.0,
                                "camera_controls": {"auto_exposure": False}})
        backend._refresh()
        self.assertTrue(backend.manualExposureSupported)

    def test_a_black_frame_in_automatic_mode_is_not_the_cameras_fault(self):
        backend = gui_backend()
        pipeline_state(backend, status="live", sequence=1,
                       metrics={"image_mean": 1.0, "image_std": 0.1,
                                "camera_controls": {"auto_exposure": True}})
        backend._refresh()
        self.assertTrue(backend.manualExposureSupported)

    def test_the_block_is_per_device_not_global(self):
        backend = gui_backend()
        self.drive_black_frame(backend)
        self.assertFalse(backend.manualExposureSupported)
        backend.pipeline.switch_source(SourceSpec.from_mapping({
            "id": "webcam", "name": "Some Other Camera", "type": "webcam",
            "index": 7,
        }))
        backend._refresh()
        self.assertTrue(backend.manualExposureSupported)
        self.assertEqual(backend.exposureNotice, "")

    def test_the_guard_still_forces_auto_while_a_camera_is_blocked(self):
        backend = gui_backend()
        self.drive_black_frame(backend)
        backend.setCameraTuning(False, 251, 0, 0, 100)
        self.assertTrue(backend._auto_exposure)
        self.assertTrue(backend.pipeline._camera_controls["auto_exposure"])


class GuiOverlayStudioTests(unittest.TestCase):
    def setUp(self):
        from acesvision import overlay

        self.addCleanup(overlay.PROFILES.pop, "custom", None)

    def test_applying_a_custom_style_gives_it_a_selectable_card(self):
        # It used to set overlayProfile to "custom" while the QML card model
        # only knew clean/minimal/broadcast/security, so nothing read "Active".
        backend = gui_backend()
        self.assertFalse(backend.customOverlayReady)
        backend.applyOverlayStyle(True, True, True, 3, 0.8, "#112233",
                                  "#00ff00", "#ff0000", "#0000ff")
        self.assertTrue(backend.customOverlayReady)
        self.assertEqual(backend.overlayProfile, "custom")

    def test_the_custom_card_can_be_reselected_after_switching_away(self):
        backend = gui_backend()
        backend.applyOverlayStyle(True, True, True, 3, 0.8, "#112233",
                                  "#00ff00", "#ff0000", "#0000ff")
        backend.setOverlayProfile("broadcast")
        self.assertEqual(backend.overlayProfile, "broadcast")
        backend.setOverlayProfile("custom")
        self.assertEqual(backend.overlayProfile, "custom")

    def test_a_rejected_custom_style_neither_registers_nor_activates(self):
        backend = gui_backend()
        backend.applyOverlayStyle(True, True, True, 3, 0.8, "not-a-colour",
                                  "#00ff00", "#ff0000", "#0000ff")
        self.assertFalse(backend.customOverlayReady)
        self.assertEqual(backend.overlayProfile, "minimal")
        self.assertIn("hex", backend.lastError)

    def test_the_profile_list_signals_when_the_custom_card_appears(self):
        backend = gui_backend()
        fired = []
        backend.overlayProfilesChanged.connect(lambda: fired.append(True))
        backend.applyOverlayStyle(True, True, True, 3, 0.8, "#112233",
                                  "#00ff00", "#ff0000", "#0000ff")
        backend.applyOverlayStyle(False, True, True, 4, 0.9, "#112233",
                                  "#00ff00", "#ff0000", "#0000ff")
        self.assertEqual(len(fired), 1)   # the card appears once, not per apply


class GuiNotifyingPropertyTests(unittest.TestCase):
    """`constant=True` on anything that can change is a stale-UI bug."""

    LIVE_PROPERTIES = ("actorNames", "executableConnectors", "modelOptions",
                       "manualExposureSupported", "connectorNames",
                       "gestureNames", "computeDevice", "stageIds",
                       "stageStats", "shushWarning", "shushDegraded",
                       "latestInferenceMs", "modelInferenceMs")

    def test_none_of_the_live_properties_are_declared_constant(self):
        backend = gui_backend()
        meta = backend.metaObject()
        without_notify = []
        for name in self.LIVE_PROPERTIES:
            index = meta.indexOfProperty(name)
            self.assertNotEqual(index, -1, name)
            if not meta.property(index).hasNotifySignal():
                without_notify.append(name)
        self.assertEqual(without_notify, [])

    def test_a_newly_enrolled_person_appears_without_a_restart(self):
        backend = gui_backend()
        fired = []
        backend.actorsChanged.connect(lambda: fired.append(True))
        with patch("acesvision.gui.known_actors", return_value=["Toby", "Ada"]):
            backend.refreshActors()
        self.assertIn("Ada", backend.actorNames)
        self.assertIn(gesture_catalog.ANY_ACTOR, backend.actorNames)
        self.assertEqual(len(fired), 1)

    def test_re_scanning_the_same_people_does_not_churn_the_ui(self):
        backend = gui_backend()
        fired = []
        backend.actorsChanged.connect(lambda: fired.append(True))
        backend.refreshActors()
        backend.refreshActors()
        self.assertEqual(fired, [])

    def test_a_connector_daemon_registered_later_reaches_the_sidebar(self):
        class GrowingExecutor(RecordingExecutor):
            wired = ["acergb"]

            def names(self):
                return list(self.wired)

        executor = GrowingExecutor()
        backend = gui_backend(executor=executor)
        self.assertEqual(backend.executableConnectors, ["acergb"])
        fired = []
        backend.connectorsChanged.connect(lambda: fired.append(True))
        executor.wired = ["acergb", "pipewire"]
        backend.refreshConnectors()
        self.assertEqual(backend.executableConnectors, ["acergb", "pipewire"])
        self.assertEqual(len(fired), 1)

    def test_the_slow_tick_picks_up_new_people_and_connectors(self):
        from acesvision import gui as gui_module

        backend = gui_backend()
        with patch("acesvision.gui.known_actors", return_value=["Ada"]):
            for _ in range(gui_module.SLOW_REFRESH_TICKS):
                backend._refresh()
        self.assertIn("Ada", backend.actorNames)

    def test_rescanning_models_reports_what_is_on_disk(self):
        backend = gui_backend()
        fired = []
        backend.modelsChanged.connect(lambda: fired.append(True))
        backend.refreshModels()
        self.assertEqual(fired, [])          # nothing changed on disk
        backend._model_options = []
        backend.refreshModels()
        self.assertEqual(len(fired), 1)


class ComputeDeviceTests(unittest.TestCase):
    """OSS blocker: the UI named the author's GPU."""

    def device(self, torch_module):
        from acesvision.gui import describe_compute_device

        return describe_compute_device(torch_module)

    def test_no_torch_means_no_hardware_claim(self):
        from acesvision.gui import describe_compute_device

        with patch.dict("sys.modules", {"torch": None}):
            self.assertEqual(describe_compute_device(), "this machine")

    def test_a_rocm_build_reports_rocm_and_the_real_card(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "AMD Radeon RX 7900 XTX"
        torch.version.hip = "6.2.0"
        self.assertEqual(self.device(torch),
                         "AMD Radeon RX 7900 XTX via ROCm")

    def test_a_cuda_build_reports_cuda(self):
        torch = Mock()
        torch.cuda.is_available.return_value = True
        torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4070"
        torch.version.hip = None
        self.assertEqual(self.device(torch), "NVIDIA GeForce RTX 4070 via CUDA")

    def test_a_cpu_only_build_says_cpu(self):
        torch = Mock()
        torch.cuda.is_available.return_value = False
        self.assertEqual(self.device(torch), "the CPU on this machine")

    def test_a_broken_torch_never_breaks_the_ui(self):
        torch = Mock()
        torch.cuda.is_available.side_effect = RuntimeError("no driver")
        self.assertEqual(self.device(torch), "this machine")


class RecordingProcessor:
    """Stands in for FaceGestureProcessor: records knob calls, serves metrics.

    Publishes the measured stage costs from the profiling run that motivated
    the panel — object 12.3 ms, face 78.0 ms, gesture 19.6 ms, ~120 ms a cycle.
    """

    def __init__(self, **overrides):
        self.calls = []
        self.metrics_dict = {
            "object_model": "test:yolo",
            "inference_status": "live",
            "inference_fps": 8.3,
            "inference_ms": 120.0,
            "latest_inference_ms": 121.0,
            "model_inference_ms": 9.9,
            "object_stage_ms": 12.3,
            "object_refreshed": True,
            "object_detect_every": 1,
            "object_enabled": True,
            "face_stage_ms": 78.0,
            "face_refreshed": True,
            "face_refresh_hz": 2.0,
            "face_enabled": True,
            "gesture_stage_ms": 19.6,
            "gesture_refreshed": True,
            "gesture_refresh_hz": 15.0,
            "gesture_enabled": True,
        }
        self.metrics_dict.update(overrides)

    def __call__(self, frame, source, sequence, captured_at):
        return SceneFrame(source, sequence, captured_at, frame)

    def metrics(self):
        return dict(self.metrics_dict)

    def set_stage_enabled(self, stage, enabled):
        self.calls.append(("set_stage_enabled", stage, bool(enabled)))
        self.metrics_dict[f"{stage}_enabled"] = bool(enabled)

    def set_detect_every(self, frames):
        self.calls.append(("set_detect_every", frames))
        self.metrics_dict["object_detect_every"] = frames

    def set_face_hz(self, hz):
        self.calls.append(("set_face_hz", hz))
        self.metrics_dict["face_refresh_hz"] = hz

    def set_gesture_hz(self, hz):
        self.calls.append(("set_gesture_hz", hz))
        self.metrics_dict["gesture_refresh_hz"] = hz

    def close(self):
        pass


class StagePanelTestCase(unittest.TestCase):
    def controlled_backend(self, **overrides):
        processor = RecordingProcessor(**overrides)
        return gui_backend(processor=processor), processor

    @staticmethod
    def poll(backend, processor):
        """One 100 ms tick, carrying what the inference loop is publishing."""
        pipeline_state(backend, status="live",
                       metrics=dict(processor.metrics(), capture_fps=14.9))
        backend._refresh()

    @staticmethod
    def rows(backend):
        return {row["id"]: row for row in backend.stageStats}


class GuiStageControlTests(StagePanelTestCase):
    """The GUI computed all of this and then threw it away.

    processor.py has published per-stage cost, refresh and rate every cycle.
    The poll kept `inference_ms` and `object_model` and dropped the rest, so
    one 120 ms number stood in for three stages — and the constructor took no
    knobs at all, so the desktop app could not reach the controls the headless
    runner already had.
    """

    def test_the_poll_keeps_the_per_stage_measurements(self):
        backend, processor = self.controlled_backend()
        self.poll(backend, processor)
        self.assertEqual({stage: row["ms"] for stage, row in
                          self.rows(backend).items()},
                         {"object": 12.3, "face": 78.0, "gesture": 19.6})
        # And the two cycle-level numbers that were also being dropped.
        self.assertEqual(backend.latestInferenceMs, 121.0)
        self.assertEqual(backend.modelInferenceMs, 9.9)

    def test_stage_stats_carry_the_documented_shape(self):
        backend, processor = self.controlled_backend()
        self.poll(backend, processor)
        stats = backend.stageStats
        self.assertEqual([row["id"] for row in stats], list(backend.stageIds))
        for row in stats:
            self.assertTrue(
                {"id", "label", "ms", "hz", "enabled", "refreshed"}.issubset(row),
                row)
            self.assertIsInstance(row["label"], str)
            self.assertIsInstance(row["enabled"], bool)
            self.assertIsInstance(row["refreshed"], bool)

    def test_stage_stats_notify_when_the_measurements_move(self):
        backend, processor = self.controlled_backend()
        fired = []
        backend.stagesChanged.connect(lambda: fired.append(True))
        self.poll(backend, processor)
        self.assertEqual(len(fired), 1)
        self.poll(backend, processor)
        self.assertEqual(len(fired), 1, "an unchanged tick churned the panel")
        processor.metrics_dict["face_stage_ms"] = 91.4
        self.poll(backend, processor)
        self.assertEqual(len(fired), 2)
        self.assertEqual(self.rows(backend)["face"]["ms"], 91.4)

    def test_the_stage_set_does_not_churn_with_the_measurements(self):
        # The QML Repeater is driven off stageIds. If that list re-notified on
        # every poll, every switch and slider would be rebuilt ten times a
        # second and become undraggable.
        backend, processor = self.controlled_backend()
        fired = []
        backend.stageSetChanged.connect(lambda: fired.append(True))
        for _ in range(5):
            processor.metrics_dict["face_stage_ms"] += 1.0
            self.poll(backend, processor)
        self.assertEqual(fired, [])

    def test_switching_a_stage_reaches_the_processor(self):
        backend, processor = self.controlled_backend()
        backend.setStageEnabled("gesture", False)
        self.assertIn(("set_stage_enabled", "gesture", False), processor.calls)
        self.assertFalse(self.rows(backend)["gesture"]["enabled"])
        backend.setStageEnabled("gesture", True)
        self.assertTrue(self.rows(backend)["gesture"]["enabled"])

    def test_each_stage_rate_reaches_its_own_setter(self):
        backend, processor = self.controlled_backend()
        backend.setStageRate("object", 3.0)
        backend.setStageRate("face", 4.5)
        backend.setStageRate("gesture", 20.0)
        self.assertEqual(processor.calls, [
            ("set_detect_every", 3),
            ("set_face_hz", 4.5),
            ("set_gesture_hz", 20.0),
        ])
        rates = {stage: row["rate"] for stage, row in self.rows(backend).items()}
        self.assertEqual(rates, {"object": 3.0, "face": 4.5, "gesture": 20.0})

    def test_a_rate_outside_the_advertised_range_is_clamped(self):
        backend, processor = self.controlled_backend()
        backend.setStageRate("face", 400.0)
        backend.setStageRate("object", 0.0)
        self.assertEqual(processor.calls,
                         [("set_face_hz", 10.0), ("set_detect_every", 1)])

    def test_an_unknown_stage_is_named_rather_than_ignored(self):
        backend, _ = self.controlled_backend()
        backend.setStageEnabled("faces", False)
        self.assertIn("Unknown perception stage", backend.lastError)

    def test_stage_control_is_refused_when_no_inference_loop_is_running(self):
        """Better to decline the click than to show a stage off while it runs."""
        backend = gui_backend()                 # smoke shell, no processor
        backend.setStageEnabled("face", False)
        self.assertIn("no inference loop", backend.lastError)
        self.assertTrue(all(row["enabled"] for row in backend.stageStats))

    def test_the_gui_hands_its_own_knobs_to_the_processor(self):
        """gui.py built FaceGestureProcessor() with no arguments at all.

        __main__.py has always passed detect_every, face_hz and gesture_hz, so
        the desktop app was pinned to the defaults with no way to say otherwise.
        """
        with patch("acesvision.gui.FaceGestureProcessor") as factory:
            factory.return_value = RecordingProcessor()
            backend = gui_backend(initialize_models=True, detect_every=3,
                                  face_hz=4.0, gesture_hz=12.0)
        self.assertEqual(factory.call_args.kwargs,
                         {"detect_every": 3, "face_hz": 4.0, "gesture_hz": 12.0})
        self.assertEqual([row["rate"] for row in backend.stageStats],
                         [3.0, 4.0, 12.0])

    def test_the_desktop_entry_point_accepts_the_same_knobs(self):
        import subprocess

        _require_qt()
        environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        environment.pop("DISPLAY", None)
        environment.pop("WAYLAND_DISPLAY", None)
        completed = subprocess.run(
            [sys.executable, "-m", "acesvision.gui", "--smoke-test",
             "--preview-port", "8792", "--detect-every", "2",
             "--face-hz", "3", "--gesture-hz", "20"],
            cwd=str(Path(__file__).parent), env=environment,
            capture_output=True, text=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])


class ShushDegradationTests(unittest.TestCase):
    """Disabling the face stage does not drop a shush. It rebinds it.

    ShushGestureTests already proves the perception half: a finger at the lips
    with no face boxes classifies as Pointing_Up. These tests hold the panel to
    saying so, and hold the message to facts that are checked against the files
    that define them rather than written as prose.
    """

    def warning(self, object_enabled=True, face_enabled=True, face_hz=2.0):
        from acesvision.gui import shush_degradation

        return shush_degradation(object_enabled, face_enabled, face_hz)

    def test_a_healthy_configuration_says_nothing(self):
        self.assertEqual(self.warning(), "")

    def test_switching_the_face_stage_off_warns_about_the_lights(self):
        text = self.warning(face_enabled=False)
        self.assertIn("face stage is switched off", text)
        self.assertIn("Pointing_Up", text)
        self.assertIn("ledctl next-theme", text)
        self.assertIn("lights", text)

    def test_starving_the_face_stage_of_refreshes_warns_the_same_way(self):
        from acesvision.gui import SAFE_FACE_HZ

        self.assertEqual(self.warning(face_hz=SAFE_FACE_HZ), "")
        text = self.warning(face_hz=SAFE_FACE_HZ - 0.1)
        self.assertIn("Pointing_Up", text)
        self.assertIn("ledctl next-theme", text)

    def test_switching_objects_off_warns_because_faces_need_person_boxes(self):
        text = self.warning(object_enabled=False)
        self.assertIn("person boxes", text)
        self.assertIn("ledctl next-theme", text)

    def test_the_warning_states_a_binding_that_actually_exists(self):
        """The gesture and the command are read out of the shipped files.

        A warning that names a binding nobody has is worse than no warning: it
        trains the operator to ignore the panel.
        """
        from acesvision.gui import (SHUSH_FALLBACK_ACTION,
                                    SHUSH_FALLBACK_GESTURE)

        automations = json.loads(
            (Path(__file__).parent / "automations.example.json").read_text())
        bound = [entry.get("command") for entry in automations
                 if entry.get("event") == "gesture"
                 and entry.get("gesture") == SHUSH_FALLBACK_GESTURE]
        self.assertIn(SHUSH_FALLBACK_ACTION, bound)
        self.assertIn(SHUSH_FALLBACK_GESTURE, gesture_catalog.GESTURE_IDS)
        self.assertIn(gesture_catalog.SHUSH, gesture_catalog.GESTURE_IDS)

    def test_the_premise_holds_end_to_end_without_a_face_box(self):
        """No face boxes -> Pointing_Up -> the theme command. All three links."""
        from acesvision.gui import (SHUSH_FALLBACK_ACTION,
                                    SHUSH_FALLBACK_GESTURE)
        from gestures import classify_hands

        Category = namedtuple("Category", "category_name score")
        rows = classify_hands([pointing_hand(AT_LIPS)],
                              [[Category("Pointing_Up", 0.95)]],
                              FRAME_W, FRAME_H, 0.5, faces=[])
        self.assertEqual([row.name for row in rows], [SHUSH_FALLBACK_GESTURE])
        rule = Rule.create(SHUSH_FALLBACK_GESTURE, "acergb", "next_theme")
        self.assertEqual(rule.gesture, SHUSH_FALLBACK_GESTURE)
        self.assertIn("next-theme", SHUSH_FALLBACK_ACTION)


class GuiShushGuardTests(StagePanelTestCase):
    """Putting that one click away without saying so is the failure."""

    def test_switching_the_face_stage_off_raises_the_warning(self):
        backend, processor = self.controlled_backend()
        self.poll(backend, processor)
        self.assertFalse(backend.shushDegraded)
        self.assertEqual(backend.shushWarning, "")
        fired = []
        backend.stagesChanged.connect(lambda: fired.append(True))
        backend.setStageEnabled("face", False)
        self.assertTrue(backend.shushDegraded)
        self.assertIn("ledctl next-theme", backend.shushWarning)
        self.assertEqual(len(fired), 1)

    def test_the_warning_is_attached_to_the_stage_that_caused_it(self):
        backend, processor = self.controlled_backend()
        backend.setStageEnabled("face", False)
        rows = self.rows(backend)
        self.assertIn("Pointing_Up", rows["face"]["warning"])
        self.assertEqual(rows["gesture"]["warning"], "")
        self.assertEqual(rows["object"]["warning"], "")

    def test_switching_objects_off_blames_the_object_stage(self):
        backend, processor = self.controlled_backend()
        backend.setStageEnabled("object", False)
        rows = self.rows(backend)
        self.assertIn("person boxes", rows["object"]["warning"])
        self.assertEqual(rows["face"]["warning"], "")

    def test_dropping_the_face_rate_too_low_warns_without_disabling_anything(self):
        from acesvision.gui import SAFE_FACE_HZ

        backend, processor = self.controlled_backend()
        backend.setStageRate("face", SAFE_FACE_HZ)
        self.assertFalse(backend.shushDegraded)
        backend.setStageRate("face", 0.2)
        self.assertTrue(backend.shushDegraded)
        self.assertIn("0.2 Hz", backend.shushWarning)
        self.assertTrue(self.rows(backend)["face"]["enabled"])

    def test_the_warning_survives_the_refresh_tick(self):
        # The tick rewrites the whole stage snapshot from the inference loop.
        # A guard that a 100 ms timer can erase is not a guard.
        backend, processor = self.controlled_backend()
        backend.setStageEnabled("face", False)
        for _ in range(10):
            self.poll(backend, processor)
        self.assertTrue(backend.shushDegraded)

    def test_turning_the_stage_back_on_clears_it(self):
        backend, processor = self.controlled_backend()
        backend.setStageEnabled("face", False)
        backend.setStageEnabled("face", True)
        self.poll(backend, processor)
        self.assertFalse(backend.shushDegraded)
        self.assertEqual(backend.shushWarning, "")


class QmlSourceTests(unittest.TestCase):
    """Cheap deterministic guards over the shipped QML text."""

    @classmethod
    def setUpClass(cls):
        from acesvision.gui import QML_PATH

        cls.qml = QML_PATH.read_text()

    @staticmethod
    def own_property_lines(text):
        """[(block header, its own property lines)] for every block in the file.

        Own means declared at that block's own brace depth, so a parent is
        never credited with a child's anchors and vice versa.
        """
        blocks, stack = [], []
        for raw in text.split("\n"):
            line = raw.split("//", 1)[0].strip()
            if not line:
                continue
            opens, closes = line.count("{"), line.count("}")
            if opens > closes:
                if stack:
                    stack[-1][1].append(line)
                for _ in range(opens - closes):
                    stack.append((line, []))
            elif closes > opens:
                for _ in range(closes - opens):
                    if stack:
                        blocks.append(stack.pop())
            elif stack:
                stack[-1][1].append(line)
        while stack:
            blocks.append(stack.pop())
        return blocks

    def test_no_element_sets_an_anchor_margin_it_has_no_anchor_for(self):
        """The clipping bug, stated as a rule instead of as a page count.

        Every page root was `ColumnLayout { width: parent.width;
        anchors.margins: 20 }` with no anchor at all, so the gutter was inert
        and titles and cards ran into the window edge. The page gutters now
        come from ScrollView padding, and this holds the whole file — not just
        the six pages that happened to have the bug — to never asking for a
        margin on an edge it never attached.
        """
        offenders = []
        for header, lines in self.own_property_lines(self.qml):
            anchors = [line for line in lines if line.startswith("anchors.")]
            margins = [line for line in anchors
                       if "argin" in line.split(":", 1)[0]]
            attachments = [line for line in anchors if line not in margins]
            if margins and not attachments:
                offenders.append((header, margins))
        self.assertEqual(offenders, [])

    def test_every_page_declares_its_gutter(self):
        # The gutter lives on the ScrollView as padding. A page that forgets it
        # runs its content into the nav rail.
        pages = re.findall(r'objectName: "page(\d+)"', self.qml)
        self.assertEqual(pages, ["0", "1", "2", "3"])
        self.assertEqual(self.qml.count("padding: window.pageMargin"), 4)

    def test_no_design_token_is_declared_and_never_used(self):
        """A palette entry nothing reads is a decision nobody made."""
        # The window's own palette, at its own indent — not the local colour
        # properties the inline components take as parameters.
        declared = re.findall(r"^    property color (\w+):", self.qml, re.M)
        self.assertTrue(declared)
        for name in declared:
            self.assertGreater(self.qml.count("window.%s" % name), 0,
                               f"the {name} colour is declared and never used")

    def test_no_specific_gpu_is_named_in_user_facing_copy(self):
        for banned in ("RX 6600", "ROCm on the", "RTX"):
            self.assertNotIn(banned, self.qml)
        self.assertIn("vision.computeDevice", self.qml)

    def test_no_internal_roadmap_vocabulary_ships_in_the_ui(self):
        for banned in ("migration slice", "migration baseline",
                       "compositor slice", "advanced compositor"):
            self.assertNotIn(banned, self.qml)

    def test_the_error_banner_can_be_dismissed(self):
        self.assertIn("vision.dismissError()", self.qml)
        self.assertIn("vision.errorDismissable", self.qml)

    def test_the_preview_has_an_error_and_a_staleness_branch(self):
        self.assertIn("Image.Error", self.qml)
        self.assertIn("vision.previewStale", self.qml)

    def test_the_overlay_selector_can_render_a_custom_profile(self):
        self.assertIn("vision.customOverlayReady", self.qml)

    def test_every_notifying_property_and_slot_still_has_a_reader(self):
        """The layout rework moved controls; it must not have dropped any.

        Each of these is a property or slot VisionBackend publishes. A control
        that was consolidated away silently would show up here as a backend
        surface nothing reads.
        """
        for binding in ("vision.stageIds", "vision.stageStats",
                        "vision.setStageEnabled(", "vision.setStageRate(",
                        "vision.shushWarning", "vision.shushDegraded",
                        "vision.latestInferenceMs", "vision.modelInferenceMs",
                        "vision.captureFps", "vision.inferenceFps",
                        "vision.inferenceMs", "vision.sourceLabel",
                        "vision.setObjectModel(", "vision.setObsEnabled(",
                        "vision.setEventsEnabled(", "vision.setOverlayProfile(",
                        "vision.applyOverlayStyle(", "vision.setCameraTuning(",
                        "vision.scanDroidCams()", "vision.useDroidCam(",
                        "vision.refreshWebcams()", "vision.useWebcamIndex(",
                        "vision.scanPlanTarget", "vision.refreshActors()",
                        "vision.refreshModels()", "vision.addRule(",
                        "vision.setRuleDryRun(", "vision.removeRule(",
                        "vision.executableConnectors", "vision.lastDecision",
                        "vision.exposureNotice", "vision.imageWarning",
                        "vision.manualExposureSupported"):
            self.assertIn(binding, self.qml, binding)

    def test_the_stage_repeater_is_not_driven_off_the_polled_measurements(self):
        # A Repeater rebuilds every delegate when its model changes, and
        # stageStats changes on the 100 ms poll. Binding the repeater to it
        # would recreate the switches and sliders ten times a second.
        self.assertIn("model: vision.stageIds", self.qml)
        self.assertNotIn("model: vision.stageStats", self.qml)

    def test_the_shush_guard_is_not_filed_inside_a_tab(self):
        """It has to be visible from wherever the operator already is.

        The card sits in the Live page body, above the dock, so no tool tab
        selection can hide it. Its own tests measure that on the rendered tree;
        this one just refuses the shape that would allow it to be buried.
        """
        body = self.qml.split('objectName: "page0Body"', 1)[1]
        card = body.split('objectName: "shushWarningCard"', 1)[0]
        self.assertNotIn('objectName: "toolDock"', card,
                         "the shush card was moved below the tooling dock")
        self.assertNotIn("ToolPanel", card,
                         "the shush card was moved inside a tool panel")

    def test_the_scan_disclosure_travelled_with_the_scan_button(self):
        # Which network this program opens sockets on must never be something
        # the operator has to go looking for. It used to live on the Sources
        # page; the Sources page is now a tool panel, and the disclosure moved
        # with it rather than being dropped in the consolidation.
        self.assertIn("VPN, container and virtual-machine networks are never scanned",
                      self.qml)
        self.assertIn("vision.scanPlanTarget", self.qml)


def _require_qt():
    try:
        import PySide6.QtQuick                       # noqa: F401
    except Exception as exc:                         # noqa: BLE001
        raise unittest.SkipTest(f"PySide6 is not installed: {exc}")


def _run_qml_probe(script):
    """Load Main.qml on the offscreen platform in a clean subprocess."""
    import subprocess

    environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    environment.pop("DISPLAY", None)
    environment.pop("WAYLAND_DISPLAY", None)
    return subprocess.run([sys.executable, "-c", script],
                          cwd=str(Path(__file__).parent), env=environment,
                          capture_output=True, text=True, timeout=300)


#: The window shapes the layout is held to. The last one is the awkward case:
#: tall and narrow, which is where a fixed-width word rail did the most damage —
#: it took 35% of the width and left the video feed a 20 px sliver.
PROBE_SIZES = [(800, 600), (1024, 768), (1280, 800), (1920, 1080), (620, 1000)]

QML_LAYOUT_PROBE = '''
import json, os, sys
from PySide6.QtCore import (QUrl, QTimer, QEventLoop, QPointF,
                            qInstallMessageHandler)
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuick import QQuickItem
from acesvision.contracts import SceneFrame
from acesvision.gui import VisionBackend, QML_PATH

SIZES = %(sizes)s

messages = []
qInstallMessageHandler(lambda kind, ctx, text: messages.append(text))


class FakeProcessor:
    """Enough of FaceGestureProcessor for the panel's slots to be accepted."""

    def __call__(self, frame, source, sequence, captured_at):
        return SceneFrame(source, sequence, captured_at, frame)

    def metrics(self):
        return {}

    def set_stage_enabled(self, stage, enabled):
        pass

    def set_detect_every(self, frames):
        pass

    def set_face_hz(self, hz):
        pass

    def set_gesture_hz(self, hz):
        pass


app = QGuiApplication(sys.argv[:1])
backend = VisionBackend(initialize_models=False, load_saved_rules=False,
                        processor=FakeProcessor())
engine = QQmlApplicationEngine()
engine.rootContext().setContextProperty("vision", backend)
engine.load(QUrl.fromLocalFile(str(QML_PATH)))
window = engine.rootObjects()[0]
root = window.property("contentItem")

def settle(ms=150):
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()

def find(name):
    return window.findChild(QQuickItem, name)

def box(item):
    """Position and size in window coordinates, or None when absent."""
    if item is None:
        return None
    point = item.mapToItem(root, QPointF(0, 0))
    return {"x": point.x(), "y": point.y(),
            "w": item.width(), "h": item.height(),
            "visible": bool(item.isVisible())}

def descends_from(item, name):
    node = item
    while node is not None:
        if node.objectName() == name:
            return True
        node = node.parentItem()
    return False

def find_all(item, prefix, found=None):
    found = [] if found is None else found
    if item is None:
        return found
    for child in item.childItems():
        if child.metaObject().className().startswith(prefix):
            found.append(child)
        find_all(child, prefix, found)
    return found

def measure(width, height):
    window.setWidth(width)
    window.setHeight(height)
    settle(250)
    stack = find("pageStack")
    data = {
        "actual": [window.width(), window.height()],
        "stack": box(stack),
        "rail": box(find("navRail")),
        "statusBar": box(find("statusBar")),
        "nav": [box(find("nav%%d" %% index)) for index in range(4)],
        "pages": [],
        "tools": [],
        "panels": [],
    }
    for index in range(4):
        window.setProperty("currentPage", index)
        settle(170)
        page = find("page%%d" %% index)
        entry = {"page": box(page), "body": box(find("page%%dBody" %% index))}
        children = page.childItems() if page is not None else []
        if children:
            flick = children[0]
            entry["viewport"] = [flick.width(), flick.height()]
            entry["content"] = [flick.property("contentWidth"),
                                flick.property("contentHeight")]
        data["pages"].append(entry)

    window.setProperty("currentPage", 0)
    settle(200)
    data["feed"] = box(find("feedCard"))
    data["metrics"] = box(find("metricStrip"))
    data["dock"] = box(find("toolDock"))
    data["tools"] = [box(find("tool%%d" %% slot)) for slot in range(5)]
    for slot in range(5):
        window.setProperty("currentTool", slot)
        settle(170)
        panel = find("toolPanel%%d" %% slot)
        entry = box(panel)
        if entry is not None:
            children = panel.childItems()
            if children:
                entry["content"] = [children[0].property("contentWidth"),
                                    children[0].property("contentHeight")]
                entry["viewport"] = [children[0].width(), children[0].height()]
        data["panels"].append(entry)

    window.setProperty("currentTool", 0)
    settle(200)
    tuning = find("liveTuning")
    data["tuning"] = None if tuning is None else {
        "h": tuning.height(), "implicit": tuning.property("implicitHeight")}

    # The guard, measured on the page the operator is already on.
    card = find("shushWarningCard")
    text = find("shushWarningText")
    data["shushOnLivePage"] = descends_from(card, "page0")
    data["shushBefore"] = None if card is None else bool(card.isVisible())
    backend.setStageEnabled("face", False)
    settle(220)
    data["shushAfter"] = box(card)
    data["shushText"] = "" if text is None else text.property("text")
    backend.setStageEnabled("face", True)
    settle(220)
    data["shushRestored"] = None if card is None else bool(card.isVisible())
    return data

def measure_stage_controls():
    """The stageIds binding trap, measured on the objects themselves."""
    window.setWidth(1280)
    window.setHeight(800)
    window.setProperty("currentPage", 0)
    window.setProperty("currentTool", 1)
    settle(350)
    panel = find("stagePanel")
    # Controls styled by the Material theme are QML-defined, so they report as
    # "Switch_QMLTYPE_n" rather than QQuickSwitch. The trailing underscore keeps
    # SwitchIndicator and SliderHandle out of the counts.
    sliders = find_all(panel, "Slider_")
    switches = find_all(panel, "Switch_")
    # A tag the QML never sets. It rides the C++ object, so it survives a
    # rebind and does NOT survive a delegate being destroyed and recreated.
    for index, item in enumerate(sliders):
        item.setProperty("probeTag", index)
    before = [item.property("value") for item in sliders]
    # Exactly what the 100 ms poll does: republish the measurements.
    for _ in range(20):
        backend.stagesChanged.emit()
        settle(15)
    after = find_all(panel, "Slider_")
    return {
        "count": len(sliders),
        "switches": [item.property("text") for item in switches],
        "names": [item.objectName() for item in after],
        "tagsAfter": [item.property("probeTag") for item in after],
        "enabledAfter": [bool(item.property("enabled")) for item in after],
        "valuesBefore": before,
        "valuesAfter": [item.property("value") for item in after],
    }

def collect():
    settle(400)
    report = {
        "minimumWidth": window.property("minimumWidth"),
        "minimumHeight": window.property("minimumHeight"),
        "pageCount": window.property("pageCount"),
        "toolCount": window.property("toolCount"),
        "sizes": {},
    }
    for width, height in SIZES:
        report["sizes"]["%%dx%%d" %% (width, height)] = measure(width, height)
    report["stageControls"] = measure_stage_controls()
    report["messages"] = messages
    sys.stdout.write("PROBE" + json.dumps(report) + "\\n")
    sys.stdout.flush()

def run():
    try:
        collect()
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(3)
    os._exit(0)

# Never let a broken probe hang the suite; a stuck probe is a failure.
QTimer.singleShot(240000, lambda: os._exit(4))
QTimer.singleShot(50, run)
app.exec()
''' % {"sizes": repr(PROBE_SIZES)}


class QmlRenderedLayoutTestCase(unittest.TestCase):
    """One offscreen render of the real scene graph, measured at five sizes."""

    report = None

    @classmethod
    def setUpClass(cls):
        # Skip ONLY when Qt itself is missing. A probe that starts and then
        # fails to report is a defect in the UI, not an unavailable toolchain,
        # and must not be allowed to pass as a skip.
        _require_qt()
        if QmlRenderedLayoutTestCase.report is None:
            completed = _run_qml_probe(QML_LAYOUT_PROBE)
            marker = "PROBE"
            if completed.returncode != 0 or marker not in completed.stdout:
                raise AssertionError(
                    "the QML layout probe did not report:\n"
                    + (completed.stderr or "")[-3000:])
            QmlRenderedLayoutTestCase.report = json.loads(
                completed.stdout.split(marker, 1)[1])
        cls.report = QmlRenderedLayoutTestCase.report

    def sizes(self):
        return sorted(self.report["sizes"].items())

    @staticmethod
    def right(rect):
        return rect["x"] + rect["w"]

    @staticmethod
    def bottom(rect):
        return rect["y"] + rect["h"]


class QmlWindowShapeTests(QmlRenderedLayoutTestCase):
    """The app has to be able to take the shapes it is judged in."""

    def test_the_window_may_be_made_as_small_as_the_layout_is_tested_at(self):
        # It refused to go below 980x640, which made two of the five sizes
        # unreachable on any real window manager — the layout could not be put
        # into the shape it was breaking in.
        narrowest = min(width for width, _ in PROBE_SIZES)
        shortest = min(height for _, height in PROBE_SIZES)
        self.assertLessEqual(self.report["minimumWidth"], narrowest)
        self.assertLessEqual(self.report["minimumHeight"], shortest)

    def test_every_requested_size_is_the_size_that_got_measured(self):
        for size, data in self.sizes():
            self.assertEqual("%dx%d" % tuple(data["actual"]), size)

    def test_loading_and_resizing_the_ui_logs_no_qml_warnings(self):
        self.assertEqual(self.report["messages"], [])


class QmlPageGutterTests(QmlRenderedLayoutTestCase):
    def test_every_page_has_a_real_gutter_on_every_size(self):
        for size, data in self.sizes():
            stack = data["stack"]
            self.assertEqual(len(data["pages"]), self.report["pageCount"], size)
            for index, page in enumerate(data["pages"]):
                where = f"{size} page{index}"
                body = page["body"]
                self.assertIsNotNone(body, where)
                self.assertGreaterEqual(body["x"] - stack["x"], 16, where)
                self.assertGreaterEqual(body["y"] - stack["y"], 16, where)
                self.assertLessEqual(
                    self.right(body), self.right(stack) - 16 + 0.5, where)

    def test_no_page_is_wider_than_the_window_it_is_in(self):
        # Sideways scrolling is clipping with extra steps.
        for size, data in self.sizes():
            for index, page in enumerate(data["pages"]):
                self.assertLessEqual(page["content"][0],
                                     page["viewport"][0] + 0.5,
                                     f"{size} page{index}")


class QmlNavRailTests(QmlRenderedLayoutTestCase):
    """The specific complaint: the sidebar in odd window sizes."""

    def test_every_destination_stays_reachable_at_every_size(self):
        for size, data in self.sizes():
            rail = data["rail"]
            self.assertTrue(rail["visible"], size)
            previous_bottom = rail["y"]
            for index, button in enumerate(data["nav"]):
                where = f"{size} nav{index}"
                self.assertIsNotNone(button, where)
                self.assertTrue(button["visible"], where)
                self.assertGreater(button["w"], 0, where)
                self.assertGreater(button["h"], 0, where)
                # Inside the rail, and inside the window.
                self.assertGreaterEqual(button["x"], rail["x"] - 0.5, where)
                self.assertLessEqual(self.right(button),
                                     self.right(rail) + 0.5, where)
                self.assertLessEqual(self.bottom(button),
                                     data["actual"][1] + 0.5, where)
                # Stacked, never on top of each other.
                self.assertGreaterEqual(button["y"], previous_bottom - 0.5, where)
                previous_bottom = self.bottom(button)

    def test_the_rail_never_overlaps_the_page_beside_it(self):
        for size, data in self.sizes():
            self.assertLessEqual(self.right(data["rail"]),
                                 data["stack"]["x"] + 0.5, size)

    def test_the_rail_gives_the_width_back_when_the_window_is_narrow(self):
        # A 220 px word rail is 35% of a 620 px window. Under the breakpoint the
        # rail keeps the icons and hands the rest to the video.
        narrow = self.report["sizes"]["620x1000"]
        wide = self.report["sizes"]["1920x1080"]
        self.assertLess(narrow["rail"]["w"], 80)
        self.assertLess(narrow["rail"]["w"] / narrow["actual"][0], 0.14)
        self.assertGreater(wide["rail"]["w"], narrow["rail"]["w"])

    def test_the_status_the_rail_used_to_carry_is_on_screen_everywhere(self):
        # It was a fixed-height card pinned to the bottom of the rail column,
        # so a short window squeezed it. It is a wrapping bar at the top now.
        for size, data in self.sizes():
            bar = data["statusBar"]
            self.assertTrue(bar["visible"], size)
            self.assertGreater(bar["h"], 20, size)
            self.assertLessEqual(self.bottom(bar), data["actual"][1] + 0.5, size)
            self.assertAlmostEqual(bar["w"], data["actual"][0], delta=1.0)


class QmlToolingDockTests(QmlRenderedLayoutTestCase):
    """The main ask: a suite of tooling below the camera feed."""

    def test_the_feed_the_metrics_and_the_dock_stack_without_overlapping(self):
        for size, data in self.sizes():
            feed, metrics, dock = data["feed"], data["metrics"], data["dock"]
            for name, rect in (("feed", feed), ("metrics", metrics),
                               ("dock", dock)):
                self.assertIsNotNone(rect, f"{size} has no {name}")
                self.assertTrue(rect["visible"], f"{size} {name}")
            self.assertLessEqual(self.bottom(feed), metrics["y"] + 0.5, size)
            self.assertLessEqual(self.bottom(metrics), dock["y"] + 0.5, size)

    def test_the_feed_dominates_instead_of_being_squeezed_by_the_chrome(self):
        for size, data in self.sizes():
            feed, stack = data["feed"], data["stack"]
            # At 620x1000 the old fixed rail plus the beside-feed tuning panel
            # left the video 20 px wide. Nothing may do that again.
            self.assertGreater(feed["w"] / stack["w"], 0.85, size)
            self.assertGreaterEqual(feed["h"], 160, size)
        for size in ("800x600", "1024x768", "1280x800", "1920x1080"):
            data = self.report["sizes"][size]
            self.assertGreaterEqual(data["feed"]["h"], data["dock"]["h"], size)

    def test_every_tool_is_one_click_away_at_every_size(self):
        for size, data in self.sizes():
            dock = data["dock"]
            self.assertEqual(len(data["tools"]), self.report["toolCount"], size)
            for slot, tab in enumerate(data["tools"]):
                where = f"{size} tool{slot}"
                self.assertIsNotNone(tab, where)
                self.assertTrue(tab["visible"], where)
                self.assertGreater(tab["w"], 20, where)
                self.assertGreater(tab["h"], 20, where)
                # Inside the dock, so no tab is off the edge of the strip.
                self.assertGreaterEqual(tab["x"], dock["x"] - 0.5, where)
                self.assertLessEqual(self.right(tab), self.right(dock) + 0.5,
                                     where)
                self.assertLessEqual(self.bottom(tab),
                                     self.bottom(dock) + 0.5, where)

    def test_selecting_a_tool_shows_a_panel_that_fits_the_dock(self):
        for size, data in self.sizes():
            dock = data["dock"]
            for slot, panel in enumerate(data["panels"]):
                where = f"{size} panel{slot}"
                self.assertIsNotNone(panel, where)
                self.assertTrue(panel["visible"], where)
                self.assertGreater(panel["w"], 0, where)
                self.assertGreater(panel["h"], 0, where)
                self.assertGreaterEqual(panel["x"], dock["x"] - 0.5, where)
                self.assertLessEqual(self.right(panel),
                                     self.right(dock) + 0.5, where)
                # A panel scrolls inside the dock; it never runs sideways.
                self.assertLessEqual(panel["content"][0],
                                     panel["viewport"][0] + 0.5, where)

    def test_the_live_page_itself_does_not_scroll(self):
        # The controls under the feed are only "below the feed" if they are on
        # the same screen. The dock scrolls internally instead.
        for size, data in self.sizes():
            page = data["pages"][0]
            self.assertLessEqual(page["content"][1], page["viewport"][1] + 0.5,
                                 f"the live page scrolls at {size}")

    def test_the_image_tuning_panel_is_never_squeezed(self):
        for size, data in self.sizes():
            tuning = data["tuning"]
            self.assertIsNotNone(tuning, f"no liveTuning panel at {size}")
            self.assertGreaterEqual(tuning["h"], tuning["implicit"] - 0.5, size)


class QmlShushGuardTests(QmlRenderedLayoutTestCase):
    """Disabling the face stage rebinds shushing onto the lights. Say so.

    Without a face box MediaPipe labels the same hand Pointing_Up, which
    automations.example.json binds to `ledctl next-theme`. The warning used to
    live on the Models and Security page, which is the one page the operator
    is not on while they are watching the feed.
    """

    def test_the_guard_lives_on_the_live_page_not_in_a_tab(self):
        for size, data in self.sizes():
            self.assertTrue(data["shushOnLivePage"],
                            f"the shush warning is not on the Live page ({size})")

    def test_the_guard_appears_when_the_face_stage_goes_off_at_every_size(self):
        for size, data in self.sizes():
            self.assertFalse(data["shushBefore"], size)
            after = data["shushAfter"]
            self.assertTrue(after["visible"], size)
            # Visible and actually occupying the page, not a zero-height stub.
            self.assertGreater(after["h"], 20, size)
            self.assertIn("ledctl next-theme", data["shushText"], size)
            self.assertFalse(data["shushRestored"], size)

    def test_the_guard_is_inside_the_window_at_every_size(self):
        for size, data in self.sizes():
            after, stack = data["shushAfter"], data["stack"]
            self.assertGreaterEqual(after["x"], stack["x"] - 0.5, size)
            self.assertLessEqual(self.right(after), self.right(stack) + 0.5, size)
            self.assertLessEqual(self.bottom(after), data["actual"][1] + 0.5, size)


class QmlStagePanelTests(QmlRenderedLayoutTestCase):
    """Measure the rendered panel, not the QML text that was meant to build it."""

    def controls(self):
        return self.report["stageControls"]

    def test_every_stage_gets_a_switch_and_a_rate_control(self):
        controls = self.controls()
        self.assertEqual(len(controls["switches"]), 3, controls["switches"])
        self.assertEqual(controls["count"], 3)
        for label in controls["switches"]:
            self.assertTrue(label, "a stage switch rendered with no label")

    def test_the_measurement_poll_does_not_rebuild_the_rate_sliders(self):
        """The stageIds binding trap, checked on the objects themselves.

        A Repeater rebuilds every delegate when its model changes. stageStats
        changes on the 100 ms poll, so a repeater bound to it would destroy and
        recreate these sliders ten times a second and leave them undraggable.
        The probe tags each slider before republishing the measurements twenty
        times; a tag that did not survive means a new object is on screen.
        """
        controls = self.controls()
        self.assertEqual(controls["names"],
                         ["stageRate0", "stageRate1", "stageRate2"])
        self.assertEqual(controls["tagsAfter"], [0, 1, 2],
                         "the rate sliders were rebuilt by the poll")
        self.assertEqual(controls["enabledAfter"], [True, True, True])
        self.assertEqual(controls["valuesAfter"], controls["valuesBefore"])


class QmlTeardownTests(unittest.TestCase):
    """A clean exit must not print a page of null-property warnings."""

    def test_the_smoke_run_exits_without_warnings(self):
        import subprocess

        _require_qt()
        environment = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        environment.pop("DISPLAY", None)
        environment.pop("WAYLAND_DISPLAY", None)
        completed = subprocess.run(
            [sys.executable, "-m", "acesvision.gui", "--smoke-test",
             "--preview-port", "8791"],
            cwd=str(Path(__file__).parent), env=environment,
            capture_output=True, text=True, timeout=180)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertNotIn("of null", completed.stderr)
        self.assertNotIn("TypeError", completed.stderr)
        self.assertEqual(completed.stderr.strip(), "")


# ---------------------------------------------------------------------------
# The gesture catalog as data — gestures.json, its version, and its content hash.
# ---------------------------------------------------------------------------


#: Every catalog version this repo has ever published, and the sha256 of its
#: canonical serialisation. This is the canary: editing gestures.json changes the
#: hash, and the only way to make this table agree again is to bump
#: catalog_version and add a row. An edit that forgets the bump fails here,
#: which is the whole point — a subscriber pinned to version 1 must never be
#: handed a different version-1 vocabulary.
CATALOG_HASHES = {
    1: "e1243aea83118534ba92ae915bafb74101fbc446fab4d055737a731841709a13",
}


class GestureCatalogFileTests(unittest.TestCase):
    def test_the_shipped_catalog_matches_its_pinned_hash(self):
        self.assertIn(
            CATALOG.version, CATALOG_HASHES,
            "gestures.json declares catalog_version "
            f"{CATALOG.version}, which has no pinned hash. Add it to "
            "CATALOG_HASHES.")
        self.assertEqual(
            CATALOG.sha256, CATALOG_HASHES[CATALOG.version],
            "gestures.json changed but catalog_version is still "
            f"{CATALOG.version}. Bump the version and pin the new hash.")

    def test_the_catalog_holds_exactly_the_nine_known_gestures(self):
        self.assertEqual(
            CATALOG.ids,
            ("Closed_Fist", "Open_Palm", "Pointing_Up", "Thumb_Up",
             "Thumb_Down", "Victory", "ILoveYou", "Middle_Finger", "Shush"))
        builtins = [spec.id for spec in CATALOG.gestures if spec.builtin]
        self.assertEqual(len(builtins), 7)            # the MediaPipe labels
        self.assertEqual([spec.id for spec in CATALOG.gestures if not spec.builtin],
                         ["Middle_Finger", "Shush"])  # the landmark poses

    def test_the_hash_describes_content_not_formatting(self):
        """Reformatting the file must not move the hash; a real edit must."""
        document = CATALOG.as_document()
        reordered = {
            "gestures": [dict(reversed(list(entry.items())))
                         for entry in document["gestures"]],
            "catalog_version": document["catalog_version"],
        }
        self.assertEqual(catalog_sha256(reordered), CATALOG.sha256)

        added = CATALOG.as_document()
        added["gestures"].append({"id": "Wave", "label": "Wave", "builtin": True})
        self.assertNotEqual(catalog_sha256(added), CATALOG.sha256)

    def test_a_subscriber_can_recompute_the_hash_from_the_served_payload(self):
        """The wire contract: strip 'sha256', canonicalise, hash, compare."""
        payload = CATALOG.as_payload()
        served_hash = payload.pop("sha256")
        recomputed = hashlib.sha256(
            canonical_json(payload).encode("utf-8")).hexdigest()
        self.assertEqual(recomputed, served_hash)
        self.assertEqual(recomputed, CATALOG.sha256)

    def test_the_file_on_disk_is_what_the_module_loaded(self):
        root = Path(__file__).resolve().parent
        on_disk = json.loads((root / "gestures.json").read_text())
        self.assertEqual(on_disk, CATALOG.as_document())


class GestureCatalogValidationTests(unittest.TestCase):
    """A vocabulary two processes could read differently is refused outright."""

    def good(self, **overrides):
        document = {
            "catalog_version": 1,
            "gestures": [{"id": "Open_Palm", "label": "Open palm",
                          "builtin": True}],
        }
        document.update(overrides)
        return document

    def assert_rejected(self, document, fragment):
        with self.assertRaises(CatalogError) as caught:
            GestureCatalog(document, origin="probe.json")
        self.assertIn(fragment, str(caught.exception))

    def test_a_valid_catalog_loads(self):
        catalog = GestureCatalog(self.good(), origin="probe.json")
        self.assertEqual(catalog.ids, ("Open_Palm",))
        self.assertEqual(catalog.version, 1)

    def test_version_must_be_a_positive_integer(self):
        for bad in (0, -1, "1", 1.0, None, True):
            self.assert_rejected(self.good(catalog_version=bad),
                                 "'catalog_version' must be an integer >= 1")

    def test_gestures_must_be_a_non_empty_array(self):
        self.assert_rejected(self.good(gestures=[]), "non-empty array")
        self.assert_rejected(self.good(gestures={}), "non-empty array")

    def test_every_field_is_required_and_typed(self):
        self.assert_rejected(self.good(gestures=[{"id": "A", "label": "A"}]),
                             "is missing builtin")
        self.assert_rejected(
            self.good(gestures=[{"id": "", "label": "A", "builtin": True}]),
            "non-string or empty 'id'")
        self.assert_rejected(
            self.good(gestures=[{"id": "A", "label": "A", "builtin": "yes"}]),
            "non-boolean 'builtin'")
        self.assert_rejected(self.good(gestures=["Open_Palm"]),
                             "must be an object")

    def test_ids_that_fold_together_are_a_collision(self):
        """'open_palm' and 'Open_Palm' are one gesture wearing two hats: names
        are matched ignoring case and separators, so normalisation would pick
        between them arbitrarily."""
        self.assert_rejected(
            self.good(gestures=[
                {"id": "Open_Palm", "label": "Open palm", "builtin": True},
                {"id": "open palm", "label": "Open palm again", "builtin": True},
            ]),
            "collides with 'Open_Palm'")

    def test_a_missing_or_unparsable_file_names_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "nope.json"
            with self.assertRaises(CatalogError) as caught:
                load_catalog(missing)
            self.assertIn("nope.json", str(caught.exception))

            broken = Path(directory) / "broken.json"
            broken.write_text("{not json")
            with self.assertRaises(CatalogError) as caught:
                load_catalog(broken)
            self.assertIn("not valid JSON", str(caught.exception))

    def test_catalog_error_is_a_value_error(self):
        # Callers already catch ValueError around vocabulary loading.
        self.assertTrue(issubclass(CatalogError, ValueError))


class GestureCatalogReexportTests(unittest.TestCase):
    """gesture_catalog kept its whole public surface when the data moved out."""

    def test_the_vocabulary_functions_are_the_catalog_object(self):
        # Bound methods of the one loaded catalog, not a second copy of the
        # name-folding rule. (Bound methods compare equal, never identical.)
        self.assertEqual(gesture_catalog.normalise_gesture, CATALOG.normalise)
        self.assertEqual(gesture_catalog.require_gesture, CATALOG.require)
        self.assertIs(gesture_catalog.GESTURES, CATALOG.gestures)
        self.assertIs(gesture_catalog.GESTURE_IDS, CATALOG.ids)

    def test_catalog_json_now_carries_the_version_and_hash(self):
        payload = gesture_catalog.catalog_json()
        self.assertEqual(payload["catalog_version"], CATALOG.version)
        self.assertEqual(payload["catalog_sha256"], CATALOG.sha256)
        self.assertEqual([entry["id"] for entry in payload["gestures"]],
                         list(CATALOG.ids))
        self.assertEqual(len(payload["actions"]), 16)

    def test_the_landmark_geometry_stayed_behind(self):
        # Real perception code, not vocabulary — it must not have moved.
        for name in ("is_middle_finger", "is_shush", "fingertip_near_mouth",
                     "MOUTH_CENTRE_Y", "MOUTH_RADIUS"):
            self.assertTrue(hasattr(gesture_catalog, name), name)
        self.assertTrue(gesture_catalog.is_middle_finger(hand_landmarks(["middle"])))

    def test_importing_the_vocabulary_does_not_import_opencv(self):
        """gesture_catalog promises no cv2. Loading the vocabulary through the
        acesvision package must not quietly break that promise — which it would
        if acesvision/__init__ imported the capture pipeline eagerly."""
        import subprocess

        probe = ("import sys, gesture_catalog, acesvision.catalog;"
                 "print(sorted(m for m in ('cv2', 'mediapipe', 'PySide6')"
                 " if m in sys.modules))")
        completed = subprocess.run([sys.executable, "-c", probe],
                                   cwd=str(Path(__file__).parent),
                                   capture_output=True, text=True, timeout=120)
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        self.assertEqual(completed.stdout.strip(), "[]")

    def test_the_lazy_package_export_still_resolves(self):
        import acesvision

        from acesvision.pipeline import VisionPipeline as direct

        self.assertIs(acesvision.VisionPipeline, direct)
        with self.assertRaises(AttributeError):
            acesvision.NoSuchThing


# ---------------------------------------------------------------------------
# The event emitter — bus, wire schema, and the HTTP surface subscribers use.
# ---------------------------------------------------------------------------


TEST_TOKEN = "test-token-Yg7Qx0pM3nV8sLzR2bT5wC1aH4eJ6kD9"

#: An RTSP source with credentials in the URL. Every source assertion below uses
#: it, because "the URL never reaches the wire" is only worth asserting against a
#: URL that would hurt if it did.
CREDENTIALED_RTSP = {
    "id": "desk", "name": "Desk cam", "type": "rtsp",
    "url": "rtsp://admin:hunter2@192.168.68.40:554/Preferred?channel=1",
}


def gesture_event(gesture="Open_Palm", attribution="unique", actor="Toby",
                  confidence=0.91, held_frames=6, captured_at=12.5,
                  candidates=("Toby", "Ana")):
    """The dict GestureEventOutput hands to its callback and to the emitter."""
    return {
        "event": "gesture",
        "gesture": gesture,
        "confidence": confidence,
        "held_frames": held_frames,
        "actor": actor,
        "actor_attribution": attribution,
        "actor_candidates": list(candidates),
        "hands_in_frame": 1,
        "identity_state": "identified" if actor else "unknown",
        "liveness_state": "not_evaluated",
        "source": "desk",
        "captured_at_monotonic": captured_at,
        "security_authorized": False,
    }


def emitter_for(publish_filter=None, publishing=True, source=None):
    source = source or SourceSpec.from_mapping(CREDENTIALED_RTSP)
    bus = EventBus()
    emitter = GestureEmitter(
        bus, identity=EmitterIdentity("acesvision", "instance-1", "9.9.9", "test-host"),
        publish_filter=publish_filter or PublishFilter(), publishing=publishing,
        clock=lambda: 1700000000.5)
    return bus, emitter, source


def scene_for(source, sequence=117, captured_at=12.5):
    return SceneFrame(source, sequence, captured_at, None)


class FakeLatestOutput:
    def __init__(self, jpeg=b"\xff\xd8jpeg"):
        self.jpeg = jpeg

    def snapshot(self):
        return None, self.jpeg


class FakePipeline:
    def __init__(self, source):
        self._state = PipelineState("live", source, 7, "", {"capture_fps": 30.0})

    def state(self):
        return self._state


def parse_sse(raw):
    """``(event, id, data)`` from one raw SSE frame."""
    event_id = event = None
    data = []
    for line in raw.split("\n"):
        field, _, value = line.partition(": ")
        if field == "id":
            event_id = int(value)
        elif field == "event":
            event = value
        elif field == "data":
            data.append(value)
    return event, event_id, (json.loads("\n".join(data)) if data else None)


class SSEStream:
    """An SSE reader that pulls a fixed number of bytes at a time.

    The chunk size is a test parameter on purpose. A frame boundary and a read
    boundary have nothing to do with each other on a real socket, and a stream
    that only survives when they coincide is not a stream.
    """

    def __init__(self, response, chunk=64):
        self.response = response
        self.chunk = chunk
        self.buffer = b""

    def raw_frame(self):
        while b"\n\n" not in self.buffer:
            piece = self.response.read1(self.chunk)
            if not piece:
                raise EOFError("stream closed while waiting for a frame")
            self.buffer += piece
        frame, _, self.buffer = self.buffer.partition(b"\n\n")
        return frame.decode("utf-8")

    def frame(self):
        """The next frame that is not a comment."""
        while True:
            raw = self.raw_frame()
            if not raw.startswith(":"):
                return parse_sse(raw)


class StreamHandle:
    """Closes an SSE stream properly.

    ``HTTPConnection.close()`` alone is not enough. The server answers a stream
    with ``Connection: close``, so ``http.client`` detaches the connection the
    moment the headers are read — ``self.sock`` is already ``None`` and
    ``close()`` is a no-op by the time a test calls it. The socket's file
    descriptor is held open by the *response* object, whose ``makefile`` handle
    keeps it alive. Closing only the connection leaves the server writing into a
    socket nobody will ever read, and its subscriber slot occupied.
    """

    def __init__(self, connection, response):
        self.connection = connection
        self.response = response

    def close(self):
        try:
            self.response.close()
        finally:
            self.connection.close()


class EmitterHarness:
    """A running VisionServer over a real socket, with a fake camera behind it."""

    def __init__(self, token=TEST_TOKEN, publish_filter=None, publishing=True,
                 keepalive_s=15.0, poll_s=0.02, ring_size=256, queue_size=64,
                 max_subscribers=8, host="127.0.0.1"):
        self.source = SourceSpec.from_mapping(CREDENTIALED_RTSP)
        self.bus = EventBus(ring_size=ring_size, queue_size=queue_size,
                            max_subscribers=max_subscribers)
        self.emitter = GestureEmitter(
            self.bus,
            identity=EmitterIdentity("acesvision", "instance-1", "9.9.9", "test-host"),
            publish_filter=publish_filter or PublishFilter(),
            publishing=publishing, clock=lambda: 1700000000.5)
        self.server = VisionServer(
            FakeLatestOutput(), FakePipeline(self.source), host=host, port=0,
            bus=self.bus, emitter=self.emitter, token=token,
            keepalive_s=keepalive_s, poll_s=poll_s)
        self.server.start()
        self.connections = []

    @property
    def port(self):
        return self.server.bound_port

    def stop(self):
        for connection in self.connections:
            try:
                connection.close()
            except OSError:
                pass
        self.server.stop()

    def publish(self, **kwargs):
        sequence = kwargs.pop("sequence", 117)
        return self.emitter.publish_gesture(gesture_event(**kwargs),
                                            scene_for(self.source, sequence))

    def request(self, path, token=TEST_TOKEN, headers=None, stream=False):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        self.connections.append(connection)
        sent = dict(headers or {})
        if token is not None and "Authorization" not in sent:
            sent["Authorization"] = f"Bearer {token}"
        connection.request("GET", path, headers=sent)
        response = connection.getresponse()
        if stream:
            return connection, response
        body = response.read()
        connection.close()
        self.connections.remove(connection)
        return response.status, body

    def json(self, path, **kwargs):
        status, body = self.request(path, **kwargs)
        return status, (json.loads(body) if body else None)

    def stream(self, path="/api/events", chunk=64, **kwargs):
        connection, response = self.request(path, stream=True, **kwargs)
        return (StreamHandle(connection, response), response,
                SSEStream(response, chunk=chunk))


class EmitterHarnessCase(unittest.TestCase):
    harness_kwargs: dict = {}

    def setUp(self):
        self.harness = EmitterHarness(**self.harness_kwargs)
        self.addCleanup(self.harness.stop)


# --- the wire schema -------------------------------------------------------


class GestureEventSchemaTests(unittest.TestCase):
    REQUIRED = (
        "schema", "type", "seq", "emitted_at", "emitter", "catalog", "gesture",
        "confidence", "held_frames", "actor", "identity_state", "liveness_state",
        "security_authorized", "source", "frame_sequence",
        "captured_at_monotonic", "dropped",
    )

    def setUp(self):
        self.bus, self.emitter, self.source = emitter_for()

    def publish(self, **kwargs):
        sequence = kwargs.pop("sequence", 117)
        return self.emitter.publish_gesture(gesture_event(**kwargs),
                                            scene_for(self.source, sequence))

    def test_every_declared_field_is_present_and_nothing_else_is(self):
        event = self.publish()
        self.assertEqual(sorted(event), sorted(self.REQUIRED))

    def test_the_payload_round_trips_through_json_unchanged(self):
        """It has to survive the wire, so nothing in it may be unserialisable."""
        event = self.publish()
        self.assertEqual(json.loads(json.dumps(event, sort_keys=True)), event)

    def test_the_values_are_the_ones_that_were_observed(self):
        event = self.publish(gesture="Victory", confidence=0.77, held_frames=4,
                             captured_at=42.25, sequence=901)
        self.assertEqual(event["schema"], SCHEMA_GESTURE)
        self.assertEqual(event["type"], "gesture")
        self.assertEqual(event["gesture"], "Victory")
        self.assertAlmostEqual(event["confidence"], 0.77)
        self.assertEqual(event["held_frames"], 4)
        self.assertEqual(event["frame_sequence"], 901)
        self.assertEqual(event["captured_at_monotonic"], 42.25)
        self.assertEqual(event["emitted_at"], 1700000000.5)
        self.assertEqual(event["dropped"], 0)

    def test_it_names_the_emitter_instance_and_the_catalog_it_used(self):
        event = self.publish()
        self.assertEqual(event["emitter"], {
            "id": "acesvision", "instance": "instance-1",
            "version": "9.9.9", "host": "test-host"})
        self.assertEqual(event["catalog"],
                         {"version": CATALOG.version, "sha256": CATALOG.sha256})

    def test_a_process_gets_one_instance_id_and_a_new_one_per_start(self):
        first = EmitterIdentity.for_process()
        second = EmitterIdentity.for_process()
        self.assertNotEqual(first.instance, second.instance)
        uuid.UUID(first.instance)             # a real UUID4, not a counter
        self.assertEqual(uuid.UUID(first.instance).version, 4)

    def test_sequence_numbers_start_at_one_and_never_repeat(self):
        self.assertEqual(self.bus.seq, 0)     # 0 means "nothing published yet"
        sequences = [self.publish()["seq"] for _ in range(5)]
        self.assertEqual(sequences, [1, 2, 3, 4, 5])

    def test_the_source_url_and_its_credentials_never_reach_the_wire(self):
        event = self.publish()
        self.assertEqual(event["source"], {
            "id": "desk", "kind": "network", "trusted_device": False,
            "label": "Desk cam (network: rtsp://192.168.68.40:554/Preferred)"})
        serialised = json.dumps(event)
        for secret in ("admin", "hunter2", "admin:hunter2@", "channel=1"):
            self.assertNotIn(secret, serialised, secret)

    def test_the_source_label_can_be_withheld_entirely(self):
        _, emitter, source = emitter_for(
            PublishFilter(publish_source_label=False))
        event = emitter.publish_gesture(gesture_event(), scene_for(source))
        self.assertIsNone(event["source"]["label"])
        # id and kind still route, and neither carries a hostname.
        self.assertEqual(event["source"]["id"], "desk")
        self.assertEqual(event["source"]["kind"], "network")

    def test_liveness_and_authorization_are_honestly_negative(self):
        event = self.publish()
        self.assertEqual(event["liveness_state"], "not_evaluated")
        self.assertIs(event["security_authorized"], False)


class IdentityStateTests(unittest.TestCase):
    """Four states. The old collapse made two of them indistinguishable."""

    def setUp(self):
        self.source = SourceSpec.from_mapping({"type": "webcam"})

    def wire_event(self, faces, gestures, publish_filter=None):
        """Drive the real detector so the projection is tested on real input."""
        bus = EventBus()
        emitter = GestureEmitter(bus, publish_filter=publish_filter or PublishFilter(),
                                 identity=EmitterIdentity.for_process(host="test"))
        local = []
        output = GestureEventOutput(local.append, hold_frames=1, enabled=True,
                                    clock=Mock(return_value=1.0), emitter=emitter)
        output.publish(SceneFrame(self.source, 5, 0.0, np.zeros((2, 2, 3)),
                                  faces=faces, gestures=gestures))
        published = bus.replay_since(0)
        return local[0], (published[0] if published else None)

    def test_one_enrolled_face_is_identified_and_named(self):
        local, wire = self.wire_event([Face(0, 0, 5, 5, "Toby", 0.2, True)],
                                      [Gesture("Victory", 0.9, 0, 0, 5, 5)])
        self.assertEqual(local["actor_attribution"], "unique")
        self.assertEqual(wire["identity_state"], "identified")
        self.assertEqual(wire["actor"], "Toby")

    def test_no_enrolled_face_is_unknown(self):
        _, wire = self.wire_event([Face(0, 0, 5, 5, None, 0.9, False)],
                                  [Gesture("Victory", 0.9, 0, 0, 5, 5)])
        self.assertEqual(wire["identity_state"], "unknown")
        self.assertIsNone(wire["actor"])

    def test_two_enrolled_faces_are_ambiguous_and_neither_is_named(self):
        """The case with no test before this one.

        Locally the nearest face still wins, marked as a guess, so actor-scoped
        rules keep working. On the wire that guess is published as `ambiguous`
        with no actor: a proximity guess is not an identification, and naming
        both candidates to say it might be either publishes both of them.
        """
        faces = [Face(0, 0, 10, 10, "A", 0.2, True),
                 Face(200, 0, 10, 10, "B", 0.2, True)]
        local, wire = self.wire_event(faces,
                                      [Gesture("Open_Palm", 0.9, 195, 0, 10, 10)])
        self.assertEqual((local["actor"], local["actor_attribution"]),
                         ("B", "nearest"))
        self.assertEqual(local["actor_candidates"], ["A", "B"])

        self.assertEqual(wire["identity_state"], "ambiguous")
        self.assertIsNone(wire["actor"])
        serialised = json.dumps(wire)
        self.assertNotIn("A", json.dumps(wire.get("actor")))
        for name in ("\"A\"", "\"B\"", "candidates"):
            self.assertNotIn(name, serialised, name)

    def test_two_enrolled_faces_with_no_geometry_are_ambiguous_too(self):
        Blind = namedtuple("Blind", "name known")
        _, wire = self.wire_event([Blind("A", True), Blind("B", True)],
                                  [Gesture("Victory", 0.9, 0, 0, 5, 5)])
        self.assertEqual(wire["identity_state"], "ambiguous")
        self.assertIsNone(wire["actor"])

    def test_the_operator_can_switch_identity_off_entirely(self):
        _, wire = self.wire_event([Face(0, 0, 5, 5, "Toby", 0.2, True)],
                                  [Gesture("Victory", 0.9, 0, 0, 5, 5)],
                                  publish_filter=PublishFilter(publish_identity=False))
        self.assertEqual(wire["identity_state"], "disabled")
        self.assertIsNone(wire["actor"])
        self.assertNotIn("Toby", json.dumps(wire))

    def test_disabled_says_nothing_about_the_scene(self):
        # All four scene cases collapse to `disabled`, which is the point: it
        # reports the operator's choice, not what the camera saw.
        for attribution in ("unique", "nearest", "ambiguous", "none"):
            self.assertEqual(identity_state(attribution, False), "disabled")
        self.assertEqual(
            [identity_state(a, True)
             for a in ("unique", "nearest", "ambiguous", "none", "anything")],
            ["identified", "ambiguous", "ambiguous", "unknown", "unknown"])


# --- the bus ---------------------------------------------------------------


class EventBusTests(unittest.TestCase):
    def setUp(self):
        self.bus = EventBus(ring_size=8, queue_size=4, max_subscribers=3)

    def payload(self, name="x"):
        return {"type": "gesture", "gesture": name}

    def test_publish_stamps_a_monotonic_sequence(self):
        self.assertEqual([self.bus.publish(self.payload())["seq"]
                          for _ in range(3)], [1, 2, 3])
        self.assertEqual(self.bus.seq, 3)

    def test_the_ring_replays_only_what_comes_after_since(self):
        for index in range(5):
            self.bus.publish(self.payload(f"g{index}"))
        self.assertEqual([event["seq"] for event in self.bus.replay_since(2)],
                         [3, 4, 5])
        self.assertEqual(self.bus.replay_since(5), [])
        self.assertEqual([event["seq"] for event in self.bus.replay_since(0)],
                         [1, 2, 3, 4, 5])

    def test_the_ring_is_bounded_and_says_what_it_still_holds(self):
        for index in range(20):                    # ring_size is 8
            self.bus.publish(self.payload(f"g{index}"))
        self.assertEqual(self.bus.oldest_seq(), 13)
        self.assertEqual([event["seq"] for event in self.bus.replay_since(0)],
                         [13, 14, 15, 16, 17, 18, 19, 20])

    def test_an_empty_ring_reports_no_oldest(self):
        self.assertEqual(self.bus.oldest_seq(), 0)

    def test_a_slow_subscriber_loses_its_oldest_events_and_is_told(self):
        subscription, _, _ = self.bus.subscribe()
        for index in range(7):                     # queue_size is 4
            self.bus.publish(self.payload(f"g{index}"))
        self.assertEqual(subscription.dropped, 3)

        received = [subscription.get(timeout=0.1) for _ in range(4)]
        # Newest-wins: the four that survived are the last four, not the first.
        self.assertEqual([event["seq"] for event in received], [4, 5, 6, 7])
        # And every one of them carries the gap count, so it is in-band.
        self.assertEqual([event["dropped"] for event in received], [3, 3, 3, 3])
        self.assertIsNone(subscription.get(timeout=0.05))

    def test_dropping_never_blocks_and_never_loses_the_newest(self):
        subscription, _, _ = self.bus.subscribe()
        for index in range(1000):
            self.bus.publish(self.payload(f"g{index}"))
        self.assertEqual(subscription.dropped, 996)
        newest = None
        while True:
            event = subscription.get(timeout=0.05)
            if event is None:
                break
            newest = event
        self.assertEqual(newest["seq"], 1000)

    def test_dropped_is_per_subscriber_not_per_bus(self):
        slow, _, _ = self.bus.subscribe()
        quick, _, _ = self.bus.subscribe()
        for index in range(6):
            self.bus.publish(self.payload(f"g{index}"))
            quick.get(timeout=0.1)                 # keeps up
        self.assertEqual(quick.dropped, 0)
        self.assertEqual(slow.dropped, 2)
        self.assertEqual(slow.get(timeout=0.1)["dropped"], 2)

    def test_the_ring_is_unaffected_by_a_subscriber_falling_behind(self):
        self.bus.subscribe()
        for index in range(7):
            self.bus.publish(self.payload(f"g{index}"))
        self.assertEqual([event["dropped"] for event in self.bus.replay_since(0)],
                         [0] * 7)

    def test_subscribing_is_capped(self):
        for _ in range(3):
            self.bus.subscribe()
        with self.assertRaises(TooManySubscribers):
            self.bus.subscribe()

    def test_unsubscribing_frees_a_slot_and_wakes_the_reader(self):
        subscription, _, _ = self.bus.subscribe()
        self.bus.subscribe()
        self.bus.subscribe()
        self.bus.unsubscribe(subscription)
        self.assertTrue(subscription.closed)
        self.assertIsNone(subscription.get(timeout=1.0))
        self.bus.subscribe()                       # the slot came back
        self.assertEqual(self.bus.subscriber_count, 3)

    def test_an_unsubscribed_reader_receives_nothing_further(self):
        subscription, _, _ = self.bus.subscribe()
        self.bus.unsubscribe(subscription)
        self.bus.publish(self.payload())
        self.assertIsNone(subscription.get(timeout=0.05))

    def test_replay_and_live_traffic_do_not_overlap_or_gap(self):
        """The seam a replay protocol usually leaks.

        Subscribing and taking the replay slice happen under one lock. Register
        first and anything published in between is delivered twice; read first
        and it is lost. Subscribing in the middle of a hard publish loop is what
        would expose either, so this runs the race repeatedly and demands that
        replay + live is exactly 1..TOTAL, once each, every time.

        The ring and the queue are both larger than TOTAL, so a drop here would
        be a real loss rather than the documented back-pressure.
        """
        TOTAL = 400
        for attempt in range(25):
            bus = EventBus(ring_size=1024, queue_size=1024, max_subscribers=2)
            running = threading.Event()

            def publisher():
                running.set()
                for _ in range(TOTAL):
                    bus.publish(self.payload())
                    # Paced only so that `subscribe` below reliably lands while
                    # the loop is still running. Nothing about the seam itself
                    # is synchronised — the lock is what has to hold.
                    time.sleep(0.001)

            thread = threading.Thread(target=publisher, daemon=True)
            thread.start()
            running.wait(5)
            time.sleep(0.05)
            subscription, replay, _ = bus.subscribe(since=0)
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())

            live = []
            while True:
                event = subscription.get(timeout=0.05)
                if event is None:
                    break
                live.append(event)

            self.assertEqual(subscription.dropped, 0, f"attempt {attempt}")
            sequences = [event["seq"] for event in replay] + \
                        [event["seq"] for event in live]
            self.assertEqual(sequences, list(range(1, TOTAL + 1)),
                             f"attempt {attempt}: duplicated or lost at the seam")
            # The race has to have actually straddled the seam at least once.
            if replay and live:
                break
        else:
            self.fail("subscribe never landed mid-publish; the seam went untested")

    def test_closing_the_bus_releases_every_subscriber(self):
        first, _, _ = self.bus.subscribe()
        second, _, _ = self.bus.subscribe()
        self.bus.close()
        self.assertIsNone(first.get(timeout=1.0))
        self.assertIsNone(second.get(timeout=1.0))
        self.assertEqual(self.bus.subscriber_count, 0)

    def test_a_full_queue_still_accepts_the_close_sentinel(self):
        subscription, _, _ = self.bus.subscribe()
        for _ in range(10):
            self.bus.publish(self.payload())
        subscription.close()
        for _ in range(5):                          # drains, then hits the sentinel
            if subscription.get(timeout=0.5) is None:
                return
        self.fail("close() never woke the reader")

    def test_the_bus_performs_no_io(self):
        """A publish that could touch a socket could block the capture path."""
        source = inspect.getsource(EventBus) + inspect.getsource(Subscription)
        for forbidden in ("socket", "open(", "requests", "urllib", "subprocess"):
            self.assertNotIn(forbidden, source, forbidden)


class GestureEventOutputEmitterTests(unittest.TestCase):
    """The detector feeds two sinks and neither may break the other."""

    def setUp(self):
        self.source = SourceSpec.from_mapping({"type": "webcam"})
        self.scene = SceneFrame(self.source, 3, 0.0, np.zeros((2, 2, 3)),
                                faces=[Face(0, 0, 5, 5, "Toby", 0.2, True)],
                                gestures=[Gesture("Victory", 0.9, 0, 0, 5, 5)])

    def test_the_local_callback_and_the_emitter_both_see_the_gesture(self):
        bus = EventBus()
        emitter = GestureEmitter(bus, identity=EmitterIdentity.for_process(host="t"))
        local = []
        output = GestureEventOutput(local.append, hold_frames=1, enabled=True,
                                    clock=Mock(return_value=1.0), emitter=emitter)
        output.publish(self.scene)
        self.assertEqual(len(local), 1)
        self.assertEqual(bus.seq, 1)

    def test_the_local_event_dict_is_unchanged_by_the_emitter(self):
        """The rule engine's contract must not answer to the wire's."""
        without = []
        GestureEventOutput(without.append, hold_frames=1, enabled=True,
                           clock=Mock(return_value=1.0)).publish(self.scene)
        with_emitter = []
        GestureEventOutput(
            with_emitter.append, hold_frames=1, enabled=True,
            clock=Mock(return_value=1.0),
            emitter=GestureEmitter(EventBus())).publish(self.scene)
        self.assertEqual(without, with_emitter)

    def test_a_broken_emitter_cannot_stop_the_local_rules(self):
        class Exploding:
            def publish_gesture(self, event, scene):
                raise RuntimeError("subscriber transport is on fire")

        local = []
        output = GestureEventOutput(local.append, hold_frames=1, enabled=True,
                                    clock=Mock(return_value=1.0),
                                    emitter=Exploding())
        with self.assertLogs("acesvision.events", level="WARNING") as logged:
            output.publish(self.scene)
        self.assertEqual(len(local), 1)
        self.assertIn("on fire", "\n".join(logged.output))

    def test_no_emitter_is_the_pre_existing_behaviour(self):
        local = []
        output = GestureEventOutput(local.append, hold_frames=1, enabled=True,
                                    clock=Mock(return_value=1.0))
        output.publish(self.scene)
        self.assertEqual(len(local), 1)

    def test_no_emit_serves_the_endpoint_but_publishes_nothing(self):
        bus = EventBus()
        emitter = GestureEmitter(bus, publishing=False)
        self.assertIsNone(emitter.publish_gesture(gesture_event(),
                                                  scene_for(self.source)))
        self.assertEqual(bus.seq, 0)
        # And it says so, so "switched off" reads differently from "dead".
        self.assertIs(emitter.hello(0)["publishing"], False)


# --- the publish filter ----------------------------------------------------


class PublishFilterTests(unittest.TestCase):
    def test_the_defaults_are_the_behaviour_that_already_existed(self):
        allowed = PublishFilter()
        self.assertTrue(allowed.enabled)
        self.assertIsNone(allowed.gestures)
        self.assertTrue(allowed.publish_identity)
        self.assertTrue(allowed.publish_source_label)
        self.assertEqual(allowed.allows("Open_Palm", 0.0), (True, ""))

    def test_disabled_publishes_nothing(self):
        allowed, reason = PublishFilter(enabled=False).allows("Open_Palm", 1.0)
        self.assertFalse(allowed)
        self.assertIn("disabled", reason)

    def test_an_allowlist_is_exactly_that_list(self):
        filtered = PublishFilter(gestures=("Open_Palm",))
        self.assertTrue(filtered.allows("Open_Palm", 1.0)[0])
        self.assertFalse(filtered.allows("Victory", 1.0)[0])
        self.assertIn("allowlist", filtered.allows("Victory", 1.0)[1])

    def test_an_empty_allowlist_means_none_not_all(self):
        """The one place a permissive reading would silently invert the config."""
        empty = PublishFilter.from_mapping({"gestures": []})
        self.assertEqual(empty.gestures, ())
        for gesture in CATALOG.ids:
            self.assertFalse(empty.allows(gesture, 1.0)[0], gesture)

        absent = PublishFilter.from_mapping({})
        self.assertIsNone(absent.gestures)
        for gesture in CATALOG.ids:
            self.assertTrue(absent.allows(gesture, 1.0)[0], gesture)

    def test_min_confidence_is_inclusive_at_the_threshold(self):
        filtered = PublishFilter(min_confidence=0.6)
        self.assertTrue(filtered.allows("Open_Palm", 0.6)[0])
        self.assertFalse(filtered.allows("Open_Palm", 0.5999)[0])
        self.assertIn("0.600", filtered.allows("Open_Palm", 0.1)[1])

    def test_allowlist_names_are_normalised_against_the_catalog(self):
        filtered = PublishFilter.from_mapping({"gestures": ["open_palm", "VICTORY"]})
        self.assertEqual(filtered.gestures, ("Open_Palm", "Victory"))
        self.assertTrue(filtered.allows("Open_Palm", 1.0)[0])

    def test_an_unknown_allowlist_name_is_quarantined_not_fatal(self):
        """It can only ever fail closed, and one typo must not take the emitter
        down — but it is named rather than silently swallowed."""
        filtered = PublishFilter.from_mapping({"gestures": ["Open_Palm", "wave"]})
        self.assertEqual(filtered.gestures, ("Open_Palm",))
        self.assertEqual(filtered.rejected, (("wave", "not in the gesture catalog"),))
        self.assertFalse(filtered.allows("wave", 1.0)[0])

    def test_duplicate_allowlist_entries_collapse(self):
        filtered = PublishFilter.from_mapping(
            {"gestures": ["Open_Palm", "open palm", "OPEN_PALM"]})
        self.assertEqual(filtered.gestures, ("Open_Palm",))

    def test_a_malformed_filter_is_refused(self):
        for bad in ([], "nope", 7):
            with self.assertRaises(ValueError):
                PublishFilter.from_mapping(bad)
        with self.assertRaises(ValueError):
            PublishFilter.from_mapping({"gestures": "Open_Palm"})

    def test_a_missing_file_is_the_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            filtered = PublishFilter.load(Path(directory) / "absent.json")
            self.assertEqual(filtered, PublishFilter())

    def test_a_file_is_read_from_disk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "publish.json"
            path.write_text(json.dumps({"gestures": ["Shush"],
                                        "min_confidence": 0.8,
                                        "publish_identity": False}))
            filtered = PublishFilter.load(path)
            self.assertEqual(filtered.gestures, ("Shush",))
            self.assertEqual(filtered.min_confidence, 0.8)
            self.assertFalse(filtered.publish_identity)

    def test_without_identity_narrows_and_changes_nothing_else(self):
        filtered = PublishFilter(gestures=("Shush",), min_confidence=0.4)
        narrowed = filtered.without_identity()
        self.assertFalse(narrowed.publish_identity)
        self.assertEqual(narrowed.gestures, ("Shush",))
        self.assertEqual(narrowed.min_confidence, 0.4)
        self.assertTrue(filtered.publish_identity)          # the original is frozen

    def test_the_advertised_filter_never_lists_a_person(self):
        advertised = PublishFilter(gestures=("Shush",)).as_dict()
        self.assertEqual(sorted(advertised),
                         ["enabled", "gestures", "min_confidence",
                          "publish_identity", "publish_source_label"])

    def test_a_blocked_gesture_never_reaches_the_bus(self):
        bus, emitter, source = emitter_for(
            PublishFilter(gestures=("Shush",), min_confidence=0.5))
        self.assertIsNone(emitter.publish_gesture(gesture_event(gesture="Victory"),
                                                  scene_for(source)))
        self.assertIsNone(emitter.publish_gesture(
            gesture_event(gesture="Shush", confidence=0.4), scene_for(source)))
        self.assertIsNotNone(emitter.publish_gesture(
            gesture_event(gesture="Shush", confidence=0.9), scene_for(source)))
        self.assertEqual(bus.seq, 1)
        self.assertEqual(emitter.suppressed, 2)


# --- the security decisions, each on its own -------------------------------


class BindPolicyTests(unittest.TestCase):
    def test_loopback_is_recognised_in_every_spelling(self):
        for host in ("127.0.0.1", "127.1.2.3", "::1", "localhost", "LOCALHOST",
                     "[::1]"):
            self.assertTrue(is_loopback(host), host)
        for host in ("0.0.0.0", "192.168.68.10", "example.com", "", None,
                     "127.0.0.1.evil.com"):
            self.assertFalse(is_loopback(host), host)

    def test_a_loopback_bind_needs_no_flag(self):
        self.assertEqual(resolve_bind("127.0.0.1", False), "127.0.0.1")
        self.assertEqual(resolve_bind("", False), "127.0.0.1")

    def test_a_non_loopback_bind_is_refused_without_allow_remote(self):
        for host in ("192.168.68.10", "10.0.0.5", "fd00::1"):
            with self.assertRaises(BindRefused) as caught:
                resolve_bind(host, False)
            self.assertIn("--allow-remote", str(caught.exception))
            self.assertEqual(resolve_bind(host, True), host)

    def test_a_wildcard_bind_is_refused_even_with_the_flag(self):
        """Answering only to `Host: 0.0.0.0` means every request 403s and the
        flag silently does nothing; dropping the check trades a real defence for
        a working flag. Naming the interface keeps both."""
        for host in ("0.0.0.0", "::", "*"):
            for allow_remote in (False, True):
                with self.assertRaises(BindRefused) as caught:
                    resolve_bind(host, allow_remote)
                self.assertIn("name the interface", str(caught.exception))

    def test_allowing_remote_forces_identity_publishing_off(self):
        from acesvision.__main__ import build_parser, build_emitter

        args = build_parser().parse_args(["--bind", "192.168.68.10",
                                          "--allow-remote"])
        emitter, host = build_emitter(args, publish_filter=PublishFilter())
        self.assertEqual(host, "192.168.68.10")
        self.assertFalse(emitter.filter.publish_identity)

        args = build_parser().parse_args([])
        emitter, host = build_emitter(args, publish_filter=PublishFilter())
        self.assertEqual(host, "127.0.0.1")
        self.assertTrue(emitter.filter.publish_identity)

    def test_the_runner_refuses_a_bare_non_loopback_bind(self):
        from acesvision.__main__ import build_parser, build_emitter

        args = build_parser().parse_args(["--bind", "192.168.68.10"])
        with self.assertRaises(BindRefused):
            build_emitter(args, publish_filter=PublishFilter())


class HostHeaderTests(unittest.TestCase):
    """DNS rebinding: the socket cannot tell, the Host header can."""

    def test_loopback_hosts_are_accepted_with_and_without_a_port(self):
        for header in ("127.0.0.1:8765", "127.0.0.1", "localhost:8765",
                       "[::1]:8765", "[::1]"):
            self.assertTrue(host_header_ok(header, ()), header)

    def test_a_rebound_name_is_refused_however_it_resolves(self):
        for header in ("evil.example", "evil.example:8765",
                       "acesvision.local", "127.0.0.1.evil.example"):
            self.assertFalse(host_header_ok(header, ()), header)

    def test_a_missing_host_header_fails_closed(self):
        self.assertFalse(host_header_ok("", ()))
        self.assertFalse(host_header_ok(None, ()))

    def test_an_explicitly_bound_address_is_also_answerable(self):
        self.assertTrue(host_header_ok("192.168.68.10:8765", {"192.168.68.10"}))
        self.assertFalse(host_header_ok("192.168.68.11:8765", {"192.168.68.10"}))


class TokenTests(unittest.TestCase):
    def test_the_comparison_is_constant_time(self):
        """== returns as soon as two bytes differ, so its timing tells the
        caller how much of the prefix was right."""
        with patch.object(server_module.hmac, "compare_digest",
                          wraps=hmac.compare_digest) as spy:
            self.assertTrue(token_matches("abcdef", "abcdef"))
            self.assertFalse(token_matches("abcdeg", "abcdef"))
        self.assertEqual(spy.call_count, 2)
        self.assertIn("hmac.compare_digest",
                      inspect.getsource(token_matches))

    def test_a_near_miss_is_still_a_miss(self):
        for wrong in ("abcde", "abcdef ", "abcdeF", "", "abcdefg"):
            self.assertFalse(token_matches(wrong, "abcdef"), repr(wrong))

    def test_an_absent_token_on_either_side_never_matches(self):
        self.assertFalse(token_matches("", "abcdef"))
        self.assertFalse(token_matches("abcdef", ""))
        self.assertFalse(token_matches(None, "abcdef"))

    def test_the_bearer_header_is_read_and_the_query_is_the_fallback(self):
        self.assertEqual(presented_token({"Authorization": "Bearer abc"}, {}), "abc")
        self.assertEqual(presented_token({"Authorization": "bearer abc"}, {}), "abc")
        self.assertEqual(presented_token({}, {"token": ["abc"]}), "abc")
        self.assertEqual(presented_token({}, {}), "")
        # A malformed Authorization header is not silently downgraded to the
        # query string — the caller said which mechanism it was using.
        self.assertEqual(presented_token({"Authorization": "Basic abc"},
                                         {"token": ["abc"]}), "")

    def test_the_token_file_is_created_once_and_kept_private(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config" / "emitter.token"
            token, created = load_or_create_token(path)
            self.assertTrue(created)
            self.assertGreaterEqual(len(token), 40)     # 32 bytes, url-safe
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(path.parent.stat().st_mode), 0o700)

            again, created_again = load_or_create_token(path)
            self.assertEqual((again, created_again), (token, False))

    def test_a_world_readable_token_is_refused_not_repaired(self):
        """Repairing it would hide that something already had the chance to
        read it."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emitter.token"
            path.write_text("leaked-token")
            path.chmod(0o644)
            with self.assertRaises(PermissionError) as caught:
                load_or_create_token(path)
            self.assertIn("group or others", str(caught.exception))
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o644)

    def test_an_existing_loose_directory_is_tightened(self):
        """mkdir(mode=0o700, exist_ok=True) applies the mode only when it
        creates the directory. ~/.config/acesvision usually already exists —
        the rule store makes it — so the mode argument alone was a comment
        dressed as a control."""
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "acesvision"
            config.mkdir(mode=0o775)
            self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o775)

            load_or_create_token(config / "emitter.token")
            self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o700)

    def test_an_already_written_token_still_gets_its_directory_tightened(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "acesvision"
            config.mkdir(mode=0o700)
            token, _ = load_or_create_token(config / "emitter.token")
            config.chmod(0o755)                     # as if written before this check
            again, created = load_or_create_token(config / "emitter.token")
            self.assertEqual((again, created), (token, False))
            self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o700)

    def test_hardening_never_widens_a_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "narrow"
            config.mkdir(mode=0o500)
            harden_directory(config)
            self.assertEqual(stat.S_IMODE(config.stat().st_mode), 0o500)

    def test_writing_a_secret_refuses_to_follow_a_planted_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "emitter.token"
            write_secret_file(path, "first")
            with self.assertRaises(FileExistsError):
                write_secret_file(path, "second")
            self.assertEqual(path.read_text(), "first")

    def test_generated_tokens_are_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            tokens = set()
            for index in range(20):
                tokens.add(load_or_create_token(
                    Path(directory) / f"t{index}.token")[0])
            self.assertEqual(len(tokens), 20)


class SseFramingTests(unittest.TestCase):
    def test_a_frame_carries_id_event_and_data(self):
        self.assertEqual(sse_frame("gesture", {"a": 1}, event_id=7),
                         b'id: 7\nevent: gesture\ndata: {"a":1}\n\n')

    def test_an_id_is_optional(self):
        self.assertEqual(sse_frame("hello", "x"), b"event: hello\ndata: x\n\n")

    def test_a_newline_in_the_data_becomes_a_second_data_line(self):
        """A bare newline inside a value would otherwise end the frame early."""
        frame = sse_frame("gesture", "one\ntwo", event_id=1)
        self.assertEqual(frame, b"id: 1\nevent: gesture\ndata: one\ndata: two\n\n")
        self.assertEqual(frame.count(b"\n\n"), 1)

    def test_the_payload_is_canonical_json(self):
        frame = sse_frame("gesture", {"b": 2, "a": 1})
        self.assertIn(b'data: {"a":1,"b":2}', frame)

    def test_a_keepalive_is_a_comment_and_carries_no_data(self):
        self.assertEqual(keepalive_frame(), b": keepalive\n\n")


class ReplayPointTests(unittest.TestCase):
    def test_the_last_event_id_header_wins_over_the_query(self):
        """An EventSource reconnect re-requests the original URL and adds the
        header. Honouring the query would replay the same range every time."""
        self.assertEqual(parse_since({"since": ["5"]},
                                     {"Last-Event-ID": "42"}), (42, None))

    def test_the_query_is_used_when_there_is_no_header(self):
        self.assertEqual(parse_since({"since": ["5"]}, {}), (5, None))

    def test_no_replay_requested_is_not_a_replay_from_zero(self):
        self.assertEqual(parse_since({}, {}), (None, None))

    def test_a_malformed_query_is_an_error_and_a_malformed_header_is_not(self):
        since, error = parse_since({"since": ["abc"]}, {})
        self.assertIsNone(since)
        self.assertIn("must be an integer", error)
        # The caller did not type the header; their HTTP client did.
        self.assertEqual(parse_since({}, {"Last-Event-ID": "abc"}), (None, None))

    def test_negative_replay_points_clamp_to_zero(self):
        self.assertEqual(parse_since({"since": ["-9"]}, {}), (0, None))


# --- the HTTP surface, over a real socket ----------------------------------


class EmitterEndpointTests(EmitterHarnessCase):
    def test_health_is_open_and_says_nothing_else(self):
        status, payload = self.harness.json("/api/health", token=None)
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"ok": True, "schema": SCHEMA_GESTURE})

    def test_health_is_the_only_open_endpoint(self):
        for path in ("/", "/index.html", "/latest.jpg", "/api/state",
                     "/api/catalog", "/api/events", "/api/events/recent"):
            status, _ = self.harness.request(path, token=None)
            self.assertEqual(status, 401, path)

    def test_the_live_camera_frame_is_no_longer_readable_without_a_token(self):
        """It used to be served to any process that could guess the port."""
        self.assertEqual(self.harness.request("/latest.jpg", token=None)[0], 401)
        status, body = self.harness.request("/latest.jpg")
        self.assertEqual(status, 200)
        self.assertEqual(body, b"\xff\xd8jpeg")

    def test_a_wrong_token_is_refused_and_a_right_one_is_not(self):
        self.assertEqual(self.harness.request("/api/catalog",
                                              token="wrong")[0], 401)
        self.assertEqual(self.harness.request("/api/catalog",
                                              token=TEST_TOKEN[:-1])[0], 401)
        self.assertEqual(self.harness.request("/api/catalog")[0], 200)

    def test_a_token_may_travel_in_the_query_for_browsers(self):
        status, _ = self.harness.request(f"/api/catalog?token={TEST_TOKEN}",
                                         token=None)
        self.assertEqual(status, 200)
        status, _ = self.harness.request("/api/catalog?token=wrong", token=None)
        self.assertEqual(status, 401)

    def test_an_origin_header_is_refused_outright(self):
        for origin in ("https://evil.example", "null", "http://127.0.0.1:8765"):
            status, body = self.harness.request(
                "/api/events", headers={"Origin": origin})
            self.assertEqual(status, 403, origin)
            self.assertIn(b"cross-origin", body)

    def test_a_rebound_host_header_is_refused(self):
        status, body = self.harness.request(
            "/api/catalog", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)
        self.assertIn(b"Host", body)

    def test_the_rebinding_check_runs_before_the_token_check(self):
        """So a hostile page cannot use the status code to probe a guess."""
        status, _ = self.harness.request(
            "/api/catalog", token="wrong", headers={"Host": "evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.harness.request(
            "/api/catalog", token="wrong", headers={"Origin": "https://evil.example"})
        self.assertEqual(status, 403)

    def test_health_is_not_exempt_from_the_rebinding_check(self):
        status, _ = self.harness.request("/api/health", token=None,
                                         headers={"Host": "evil.example"})
        self.assertEqual(status, 403)

    def test_the_catalog_is_served_with_a_hash_a_subscriber_can_recompute(self):
        status, payload = self.harness.json("/api/catalog")
        self.assertEqual(status, 200)
        served_hash = payload.pop("sha256")
        self.assertEqual(served_hash, CATALOG.sha256)
        self.assertEqual(
            hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
            served_hash)
        self.assertEqual([entry["id"] for entry in payload["gestures"]],
                         list(CATALOG.ids))

    def test_no_endpoint_enumerates_identities(self):
        self.harness.publish(actor="Toby", attribution="unique")
        for path in ("/api/catalog", "/api/state", "/api/health", "/",
                     "/api/events/recent"):
            _, body = self.harness.request(path)
            self.assertNotIn(b"Toby", body, path)

    def test_an_unknown_path_is_a_json_404(self):
        status, payload = self.harness.json("/api/nope")
        self.assertEqual(status, 404)
        self.assertEqual(payload, {"error": "not found"})


class RecentEndpointTests(EmitterHarnessCase):
    def test_a_first_poll_gets_a_starting_point_not_a_ring_dump(self):
        for _ in range(3):
            self.harness.publish()
        status, payload = self.harness.json("/api/events/recent")
        self.assertEqual(status, 200)
        self.assertEqual(payload["events"], [])
        self.assertEqual(payload["seq"], 3)
        self.assertEqual(payload["oldest_available"], 1)

    def test_since_returns_everything_after_it(self):
        for name in ("Open_Palm", "Victory", "Shush"):
            self.harness.publish(gesture=name)
        _, payload = self.harness.json("/api/events/recent?since=1")
        self.assertEqual([event["gesture"] for event in payload["events"]],
                         ["Victory", "Shush"])
        self.assertEqual([event["seq"] for event in payload["events"]], [2, 3])

    def test_the_polled_events_are_whole_schema_objects(self):
        self.harness.publish()
        _, payload = self.harness.json("/api/events/recent?since=0")
        event = payload["events"][0]
        self.assertEqual(event["schema"], SCHEMA_GESTURE)
        self.assertEqual(event["dropped"], 0)   # gaps show as gaps in seq here

    def test_a_malformed_since_is_a_400(self):
        status, payload = self.harness.json("/api/events/recent?since=abc")
        self.assertEqual(status, 400)
        self.assertIn("integer", payload["error"])


class SseStreamTests(EmitterHarnessCase):
    def test_hello_bootstraps_a_subscriber_in_one_round_trip(self):
        self.harness.publish()
        connection, _, stream = self.harness.stream()
        self.addCleanup(connection.close)
        event, event_id, hello = stream.frame()
        self.assertEqual(event, "hello")
        self.assertEqual(event_id, 1)
        self.assertEqual(hello["schema"], SCHEMA_HELLO)
        self.assertEqual(hello["seq"], 1)
        self.assertEqual(hello["supported_schemas"], [SCHEMA_GESTURE])
        self.assertEqual(hello["emitter"]["instance"], "instance-1")
        self.assertEqual(hello["catalog"]["sha256"], CATALOG.sha256)
        self.assertEqual([entry["id"] for entry in hello["catalog"]["gestures"]],
                         list(CATALOG.ids))
        self.assertTrue(hello["publishing"])
        self.assertNotIn("replay", hello)       # none was requested

    def test_a_live_gesture_arrives_on_the_stream(self):
        connection, _, stream = self.harness.stream()
        self.addCleanup(connection.close)
        stream.frame()                          # hello
        self.harness.publish(gesture="Shush", actor=None, attribution="none")
        event, event_id, payload = stream.frame()
        self.assertEqual(event, "gesture")
        self.assertEqual(event_id, 1)
        self.assertEqual(payload["gesture"], "Shush")
        self.assertEqual(payload["seq"], 1)
        self.assertEqual(payload["identity_state"], "unknown")

    def test_the_content_type_is_the_sse_one(self):
        connection, response, _ = self.harness.stream()
        self.addCleanup(connection.close)
        self.assertEqual(response.status, 200)
        self.assertIn("text/event-stream", response.getheader("Content-Type"))
        self.assertEqual(response.getheader("Cache-Control"), "no-store")

    def test_frames_survive_being_read_one_byte_at_a_time(self):
        """A frame boundary and a read boundary have nothing to do with each
        other on a real socket."""
        connection, _, stream = self.harness.stream(chunk=1)
        self.addCleanup(connection.close)
        self.assertEqual(stream.frame()[0], "hello")
        for name in ("Open_Palm", "Victory", "Shush"):
            self.harness.publish(gesture=name)
        received = [stream.frame() for _ in range(3)]
        self.assertEqual([payload["gesture"] for _, _, payload in received],
                         ["Open_Palm", "Victory", "Shush"])
        self.assertEqual([event_id for _, event_id, _ in received], [1, 2, 3])

    def test_replay_by_since_then_live_traffic(self):
        for name in ("Open_Palm", "Victory", "Shush"):
            self.harness.publish(gesture=name)
        connection, _, stream = self.harness.stream("/api/events?since=1")
        self.addCleanup(connection.close)
        _, hello_id, hello = stream.frame()
        self.assertEqual(hello_id, 1)
        self.assertEqual(hello["replay"],
                         {"requested_since": 1, "oldest_available": 1,
                          "gap": False})
        replayed = [stream.frame() for _ in range(2)]
        self.assertEqual([payload["gesture"] for _, _, payload in replayed],
                         ["Victory", "Shush"])
        self.harness.publish(gesture="Thumb_Up")
        self.assertEqual(stream.frame()[2]["gesture"], "Thumb_Up")

    def test_replay_by_last_event_id_is_the_reconnect_path(self):
        for name in ("Open_Palm", "Victory"):
            self.harness.publish(gesture=name)
        connection, _, stream = self.harness.stream(
            headers={"Last-Event-ID": "1"})
        self.addCleanup(connection.close)
        stream.frame()
        self.assertEqual(stream.frame()[2]["gesture"], "Victory")

    def test_the_header_beats_the_query_on_reconnect(self):
        for name in ("Open_Palm", "Victory", "Shush"):
            self.harness.publish(gesture=name)
        connection, _, stream = self.harness.stream(
            "/api/events?since=0", headers={"Last-Event-ID": "2"})
        self.addCleanup(connection.close)
        stream.frame()
        # since=0 would have replayed all three; the header says two are known.
        self.assertEqual(stream.frame()[2]["gesture"], "Shush")

    def test_a_gap_the_ring_cannot_cover_is_declared(self):
        harness = EmitterHarness(ring_size=2, poll_s=0.02)
        self.addCleanup(harness.stop)
        for _ in range(6):
            harness.publish()
        connection, _, stream = harness.stream("/api/events?since=1")
        self.addCleanup(connection.close)
        _, _, hello = stream.frame()
        self.assertEqual(hello["replay"],
                         {"requested_since": 1, "oldest_available": 5,
                          "gap": True})

    def test_a_ring_that_still_covers_the_request_reports_no_gap(self):
        harness = EmitterHarness(ring_size=8, poll_s=0.02)
        self.addCleanup(harness.stop)
        for _ in range(4):
            harness.publish()
        connection, _, stream = harness.stream("/api/events?since=3")
        self.addCleanup(connection.close)
        self.assertFalse(stream.frame()[2]["replay"]["gap"])

    def test_keepalives_separate_an_idle_emitter_from_a_dead_one(self):
        harness = EmitterHarness(keepalive_s=0.15, poll_s=0.02)
        self.addCleanup(harness.stop)
        connection, _, stream = harness.stream()
        self.addCleanup(connection.close)
        self.assertEqual(stream.frame()[0], "hello")
        comments = [stream.raw_frame() for _ in range(3)]
        self.assertEqual(comments, [": keepalive"] * 3)

    def test_a_keepalive_never_displaces_a_real_event(self):
        harness = EmitterHarness(keepalive_s=0.15, poll_s=0.02)
        self.addCleanup(harness.stop)
        connection, _, stream = harness.stream()
        self.addCleanup(connection.close)
        stream.frame()
        time.sleep(0.4)                          # let some keepalives go out
        harness.publish(gesture="Victory")
        event, _, payload = stream.frame()       # skips the comments
        self.assertEqual((event, payload["gesture"]), ("gesture", "Victory"))

    def test_a_ninth_subscriber_is_refused_with_503(self):
        harness = EmitterHarness(max_subscribers=2, poll_s=0.02)
        self.addCleanup(harness.stop)
        held = []
        for _ in range(2):
            connection, response, stream = harness.stream()
            held.append(connection)
            self.addCleanup(connection.close)
            stream.frame()                       # connected, not merely opened
        status, body = harness.request("/api/events")
        self.assertEqual(status, 503)
        self.assertIn(b"limit 2", body)

    def test_a_departed_subscriber_frees_its_slot(self):
        harness = EmitterHarness(max_subscribers=1, poll_s=0.02)
        self.addCleanup(harness.stop)
        connection, _, stream = harness.stream()
        stream.frame()
        self.assertEqual(harness.request("/api/events")[0], 503)
        connection.close()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            harness.publish()                    # a write is what notices the hangup
            if harness.bus.subscriber_count == 0:
                break
            time.sleep(0.05)
        second, _, stream = harness.stream()
        self.addCleanup(second.close)
        self.assertEqual(stream.frame()[0], "hello")

    def test_two_subscribers_both_receive_the_same_event(self):
        connection_a, _, stream_a = self.harness.stream()
        connection_b, _, stream_b = self.harness.stream()
        self.addCleanup(connection_a.close)
        self.addCleanup(connection_b.close)
        stream_a.frame()
        stream_b.frame()
        self.harness.publish(gesture="ILoveYou")
        for stream in (stream_a, stream_b):
            self.assertEqual(stream.frame()[2]["gesture"], "ILoveYou")

    def test_a_malformed_since_is_a_400_before_any_stream_starts(self):
        status, payload = self.harness.json("/api/events?since=abc")
        self.assertEqual(status, 400)
        self.assertEqual(self.harness.bus.subscriber_count, 0)

    def test_stopping_the_server_releases_a_parked_subscriber(self):
        harness = EmitterHarness(keepalive_s=30.0, poll_s=0.02)
        connection, _, stream = harness.stream()
        self.addCleanup(connection.close)
        stream.frame()
        started = time.monotonic()
        harness.stop()                           # must not wait on the keepalive
        self.assertLess(time.monotonic() - started, 10.0)
        with self.assertRaises(EOFError):
            stream.raw_frame()


class NoEmitStreamTests(unittest.TestCase):
    def test_the_stream_stays_open_and_says_it_is_not_publishing(self):
        harness = EmitterHarness(publishing=False, keepalive_s=0.15, poll_s=0.02)
        self.addCleanup(harness.stop)
        connection, _, stream = harness.stream()
        self.addCleanup(connection.close)
        _, _, hello = stream.frame()
        self.assertFalse(hello["publishing"])
        harness.publish()
        self.assertEqual(stream.raw_frame(), ": keepalive")


class HeadlessEmitterRunnerTests(unittest.TestCase):
    def test_the_new_flags_exist_with_safe_defaults(self):
        from acesvision.__main__ import build_parser

        args = build_parser().parse_args([])
        self.assertEqual(args.bind, "127.0.0.1")
        self.assertEqual(args.port, 8765)
        self.assertFalse(args.no_emit)
        self.assertFalse(args.print_token)
        self.assertFalse(args.allow_remote)

    def test_the_old_preview_port_spelling_still_works(self):
        from acesvision.__main__ import build_parser

        self.assertEqual(build_parser().parse_args(["--preview-port", "9001"]).port,
                         9001)
        self.assertEqual(build_parser().parse_args(["--port", "9002"]).port, 9002)

    def test_no_emit_reaches_the_emitter(self):
        from acesvision.__main__ import build_parser, build_emitter

        args = build_parser().parse_args(["--no-emit"])
        emitter, _ = build_emitter(args, publish_filter=PublishFilter())
        self.assertFalse(emitter.publishing)

    def test_the_gesture_output_is_wired_to_the_emitter(self):
        from acesvision.__main__ import build_parser, build_gesture_output

        bus = EventBus()
        emitter = GestureEmitter(bus)
        output = build_gesture_output(build_parser().parse_args([]),
                                      callback=lambda event: None,
                                      emitter=emitter)
        self.assertIs(output.emitter, emitter)
        output.publish(SceneFrame(
            SourceSpec.from_mapping({"type": "webcam"}), 1, 0.0,
            np.zeros((2, 2, 3)),
            gestures=[Gesture("Victory", 0.9, 0, 0, 5, 5)]))
        self.assertEqual(bus.seq, 0)      # hold_frames is 6 by default
        for _ in range(5):
            output.publish(SceneFrame(
                SourceSpec.from_mapping({"type": "webcam"}), 1, 0.0,
                np.zeros((2, 2, 3)),
                gestures=[Gesture("Victory", 0.9, 0, 0, 5, 5)]))
        self.assertEqual(bus.seq, 1)


# ---------------------------------------------------------------------------
# ArcFace: alignment, metric direction, thresholds, providers, concurrency.
#
# These cover the recogniser swap (YuNet + dlib -> YuNet + ArcFace). The
# expensive risk in that change is not the model, it is the METRIC FLIP: the
# calibrated 0.50 was a Euclidean distance where lower is better, and ArcFace
# scores cosine similarity where higher is better. A wrong-direction
# comparison does not crash or look odd, it silently accepts everybody.
# ---------------------------------------------------------------------------

import arcface
import engine as face_engine
import matching


def executable_source(source):
    """``source`` with comments and string literals blanked out.

    A source-scanning test that reads prose is a test that fails the moment
    someone documents the very rule it enforces — and these rules are exactly
    the kind that deserve a paragraph of explanation next to them. Comment and
    string spans are overwritten with spaces rather than deleted, so every
    surviving line keeps its original layout and an assertion can still look
    for real code like ``with _LOCK:``.
    """
    import io
    import textwrap
    import tokenize

    lines = textwrap.dedent(source).splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO("".join(lines)).readline):
        if token.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (start_row, start_col), (end_row, end_col) = token.start, token.end
        for row in range(start_row, end_row + 1):
            line = lines[row - 1]
            first = start_col if row == start_row else 0
            last = end_col if row == end_row else len(line.rstrip("\n"))
            lines[row - 1] = (line[:first] + " " * (last - first) + line[last:])
    return "".join(lines)


def rotate_scale_translate(points, degrees, scale, shift):
    """Apply a known similarity transform to a point set."""
    theta = np.deg2rad(degrees)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    return (scale * (np.asarray(points, dtype=np.float64) @ rotation.T)
            + np.asarray(shift, dtype=np.float64))


def apply_affine(matrix, points):
    pts = np.asarray(points, dtype=np.float64)
    return pts @ matrix[:, :2].T + matrix[:, 2]


def yunet_row(landmarks, box=(10, 20, 100, 100), score=0.9):
    """A YuNet detection row: [x, y, w, h, *10 landmark coords, score]."""
    return np.array(list(box) + list(np.asarray(landmarks).ravel()) + [score],
                    dtype=np.float64)


class AlignmentTests(unittest.TestCase):
    """The 112x112 ArcFace crop, built from YuNet's own five keypoints."""

    def test_reference_template_is_the_arcface_one(self):
        # Five points, and the subject's right eye (index 0) sits on the LEFT
        # of the image, which is what makes YuNet's ordering line up with the
        # template without a re-order.
        self.assertEqual(arcface.ARCFACE_REFERENCE_5PT.shape, (5, 2))
        self.assertLess(arcface.ARCFACE_REFERENCE_5PT[0][0],
                        arcface.ARCFACE_REFERENCE_5PT[1][0])
        # Eyes above nose above mouth.
        eyes_y = arcface.ARCFACE_REFERENCE_5PT[:2, 1].mean()
        nose_y = arcface.ARCFACE_REFERENCE_5PT[2, 1]
        mouth_y = arcface.ARCFACE_REFERENCE_5PT[3:, 1].mean()
        self.assertLess(eyes_y, nose_y)
        self.assertLess(nose_y, mouth_y)

    def test_umeyama_recovers_a_known_similarity_transform(self):
        reference = arcface.ARCFACE_REFERENCE_5PT
        moved = rotate_scale_translate(reference, degrees=17.0, scale=2.5,
                                       shift=(120.0, -35.0))
        matrix = arcface.umeyama(moved, reference)[:2, :]
        recovered = apply_affine(matrix, moved)
        np.testing.assert_allclose(recovered, reference, atol=1e-9)

    def test_alignment_matrix_maps_landmarks_onto_the_template(self):
        landmarks = rotate_scale_translate(arcface.ARCFACE_REFERENCE_5PT,
                                           degrees=-30.0, scale=1.8,
                                           shift=(400.0, 250.0))
        matrix = arcface.alignment_matrix(landmarks)
        np.testing.assert_allclose(apply_affine(matrix, landmarks),
                                   arcface.ARCFACE_REFERENCE_5PT, atol=1e-9)

    def test_alignment_of_the_template_itself_is_the_identity(self):
        matrix = arcface.alignment_matrix(arcface.ARCFACE_REFERENCE_5PT)
        np.testing.assert_allclose(matrix, np.array([[1.0, 0.0, 0.0],
                                                     [0.0, 1.0, 0.0]]),
                                   atol=1e-9)

    def test_alignment_never_mirrors_the_face(self):
        # A mirrored landmark set must still be fitted with a rotation, not a
        # reflection: det > 0. A mirrored crop is not a valid alignment, it is
        # a different face as far as the embedder is concerned.
        mirrored = arcface.ARCFACE_REFERENCE_5PT.copy()
        mirrored[:, 0] = 112.0 - mirrored[:, 0]
        matrix = arcface.alignment_matrix(mirrored)
        self.assertGreater(np.linalg.det(matrix[:, :2]), 0.0)

    def test_align_places_a_marked_eye_at_the_template_eye(self):
        # A synthetic frame with one bright pixel at the subject's right eye.
        landmarks = rotate_scale_translate(arcface.ARCFACE_REFERENCE_5PT,
                                           degrees=12.0, scale=2.0,
                                           shift=(150.0, 90.0))
        frame = np.zeros((400, 500, 3), dtype=np.uint8)
        eye = np.round(landmarks[0]).astype(int)
        frame[eye[1] - 1:eye[1] + 2, eye[0] - 1:eye[0] + 2] = 255
        crop = arcface.align(frame, landmarks)
        self.assertEqual(crop.shape, (112, 112, 3))
        brightest = np.unravel_index(np.argmax(crop[:, :, 0]), crop.shape[:2])
        expected = arcface.ARCFACE_REFERENCE_5PT[0]
        self.assertLess(abs(brightest[1] - expected[0]), 2.5)
        self.assertLess(abs(brightest[0] - expected[1]), 2.5)

    def test_yunet_row_slice_is_the_five_landmarks(self):
        landmarks = np.arange(10, dtype=np.float64).reshape(5, 2)
        np.testing.assert_allclose(
            arcface.yunet_landmarks(yunet_row(landmarks)), landmarks)

    def test_short_yunet_row_is_refused(self):
        with self.assertRaises(ValueError):
            arcface.yunet_landmarks(np.zeros(6))

    def test_wrong_landmark_count_is_refused(self):
        with self.assertRaises(ValueError):
            arcface.alignment_matrix(np.zeros((3, 2)))

    def test_degenerate_landmarks_are_refused_not_silently_fitted(self):
        with self.assertRaises(ValueError):
            arcface.alignment_matrix(np.zeros((5, 2)))

    def test_l2_normalise_puts_rows_on_the_unit_sphere(self):
        rows = arcface.l2_normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
        self.assertAlmostEqual(float(np.linalg.norm(rows[0])), 1.0, places=6)
        # A zero row stays zero rather than becoming NaN; it can only ever
        # score 0 similarity, which is the correct "no opinion".
        self.assertTrue(np.all(np.isfinite(rows[1])))
        self.assertAlmostEqual(float(np.linalg.norm(rows[1])), 0.0, places=6)


class MetricDirectionTests(unittest.TestCase):
    """The flip: lower-is-better became higher-is-better."""

    def test_the_two_engines_do_not_share_a_metric(self):
        self.assertEqual(matching.THRESHOLDS["dlib"].metric,
                         matching.EUCLIDEAN_L2)
        self.assertEqual(matching.THRESHOLDS["arcface"].metric,
                         matching.COSINE_SIMILARITY)
        self.assertFalse(matching.METRICS[matching.EUCLIDEAN_L2].higher_is_better)
        self.assertTrue(matching.METRICS[matching.COSINE_SIMILARITY].higher_is_better)

    def test_a_cosine_threshold_rejects_a_euclidean_shaped_comparison(self):
        cosine = matching.THRESHOLDS["arcface"]
        with self.assertRaises(matching.MetricMismatch):
            cosine.accepts(0.42, matching.EUCLIDEAN_L2)

    def test_a_euclidean_threshold_rejects_a_cosine_shaped_comparison(self):
        with self.assertRaises(matching.MetricMismatch):
            matching.THRESHOLDS["dlib"].accepts(0.83, matching.COSINE_SIMILARITY)

    def test_lbph_is_its_own_metric_and_compares_to_neither(self):
        lbph = matching.THRESHOLDS["lbph"]
        self.assertNotIn(lbph.metric,
                         (matching.EUCLIDEAN_L2, matching.COSINE_SIMILARITY))
        with self.assertRaises(matching.MetricMismatch):
            lbph.accepts(0.4, matching.EUCLIDEAN_L2)

    def test_acceptance_runs_the_right_way_for_each_metric(self):
        euclidean = matching.Threshold("t", matching.EUCLIDEAN_L2, 0.50)
        self.assertTrue(euclidean.accepts(0.40, matching.EUCLIDEAN_L2))
        self.assertFalse(euclidean.accepts(0.60, matching.EUCLIDEAN_L2))

        cosine = matching.Threshold("t", matching.COSINE_SIMILARITY, 0.30)
        self.assertTrue(cosine.accepts(0.62, matching.COSINE_SIMILARITY))
        self.assertFalse(cosine.accepts(0.05, matching.COSINE_SIMILARITY))

    def test_best_index_follows_the_metric_not_the_habit(self):
        scores = np.array([0.9, 0.2, 0.5])
        self.assertEqual(matching.best_index(scores, matching.COSINE_SIMILARITY), 0)
        self.assertEqual(matching.best_index(scores, matching.EUCLIDEAN_L2), 1)

    def test_match_refuses_a_bare_float(self):
        # This is the exact shape of the bug: 0.50 is strict for dlib and wide
        # open for ArcFace, and a float cannot tell you which it meant.
        gallery = [np.array([1.0, 0.0])]
        with self.assertRaises(TypeError):
            matching.match(gallery, ["Toby"], np.array([1.0, 0.0]), 0.50)

    def test_the_dlib_number_read_as_cosine_would_admit_a_stranger(self):
        # Documents the size of the trap numerically. 0.50 as a cosine FLOOR
        # accepts a 0.55-similarity face; 0.50 as a distance CEILING rejects
        # the same pair. Same digits, opposite verdict.
        as_cosine = matching.Threshold("wrong", matching.COSINE_SIMILARITY, 0.50)
        as_distance = matching.Threshold("right", matching.EUCLIDEAN_L2, 0.50)
        self.assertTrue(as_cosine.accepts(0.55, matching.COSINE_SIMILARITY))
        self.assertFalse(as_distance.accepts(0.55, matching.EUCLIDEAN_L2))

    def test_score_gallery_computes_each_metric_correctly(self):
        gallery = np.array([[1.0, 0.0], [0.0, 1.0]])
        query = np.array([1.0, 0.0])
        np.testing.assert_allclose(
            matching.score_gallery(gallery, query, matching.COSINE_SIMILARITY),
            [1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(
            matching.score_gallery(gallery, query, matching.EUCLIDEAN_L2),
            [0.0, np.sqrt(2.0)], atol=1e-9)

    def test_an_empty_gallery_returns_the_worst_score_in_this_metric(self):
        cosine = matching.Threshold("t", matching.COSINE_SIMILARITY, 0.30)
        name, score, known = matching.match([], [], np.array([1.0, 0.0]), cosine)
        self.assertIsNone(name)
        self.assertFalse(known)
        # 0.0, not 1.0: an empty gallery must not look like a perfect match.
        self.assertEqual(score, 0.0)

        euclidean = matching.Threshold("t", matching.EUCLIDEAN_L2, 0.50)
        _, distance, _ = matching.match([], [], np.array([1.0, 0.0]), euclidean)
        self.assertEqual(distance, float("inf"))

    def test_format_score_names_the_metric_on_screen(self):
        self.assertIn("cos", matching.format_score(0.62, matching.COSINE_SIMILARITY))
        self.assertIn("d", matching.format_score(0.42, matching.EUCLIDEAN_L2))


class PerEngineThresholdTests(unittest.TestCase):
    """One threshold per engine, never one shared constant."""

    def test_every_engine_has_its_own_entry(self):
        self.assertEqual(sorted(matching.THRESHOLDS),
                         ["arcface", "dlib", "lbph", "yunet"])
        self.assertEqual(sorted(matching.THRESHOLD_ENV), sorted(matching.THRESHOLDS))

    def test_threshold_for_returns_the_matching_metric(self):
        self.assertEqual(matching.threshold_for("arcface", env={}).metric,
                         matching.COSINE_SIMILARITY)
        for name in ("dlib", "yunet"):
            self.assertEqual(matching.threshold_for(name, env={}).metric,
                             matching.EUCLIDEAN_L2)

    def test_the_dlib_tolerance_variable_does_not_reach_arcface(self):
        env = {"FACE_ID_TOLERANCE": "0.50"}
        self.assertEqual(matching.threshold_for("dlib", env=env).value, 0.50)
        arc = matching.threshold_for("arcface", env=env)
        self.assertEqual(arc.value, matching.THRESHOLDS["arcface"].value)
        self.assertIsNot(arc.provenance, None)

    def test_the_arcface_variable_does_not_reach_dlib(self):
        env = {"FACE_ID_ARCFACE_THRESHOLD": "0.9"}
        self.assertEqual(matching.threshold_for("arcface", env=env).value, 0.9)
        self.assertEqual(matching.threshold_for("dlib", env=env).value,
                         matching.DLIB_TOLERANCE.value)

    def test_an_override_keeps_the_metric_but_drops_the_evidence(self):
        overridden = matching.threshold_for(
            "arcface", env={"FACE_ID_ARCFACE_THRESHOLD": "0.42"})
        self.assertEqual(overridden.metric, matching.COSINE_SIMILARITY)
        self.assertEqual(overridden.value, 0.42)
        self.assertIsNone(overridden.far_percent)
        with self.assertRaises(matching.NotCalibrated):
            overridden.require_evidence()

    def test_an_unknown_engine_is_refused(self):
        with self.assertRaises(KeyError):
            matching.threshold_for("insightface", env={})

    def test_an_unknown_metric_is_refused_at_construction(self):
        with self.assertRaises(KeyError):
            matching.Threshold("t", "made_up_metric", 0.5)

    def test_a_threshold_without_a_measured_far_is_refused(self):
        guess = matching.Threshold("t", matching.COSINE_SIMILARITY, 0.3)
        with self.assertRaises(matching.NotCalibrated):
            guess.require_evidence()

    def test_the_shipped_arcface_threshold_carries_its_evidence(self):
        shipped = matching.THRESHOLDS["arcface"]
        shipped.require_evidence()
        self.assertIsNotNone(shipped.n_impostors)
        self.assertGreaterEqual(shipped.n_impostors, 1000)
        self.assertNotIn("UNCALIBRATED", shipped.provenance)
        # And it is not the dlib number wearing a new hat.
        self.assertNotEqual(shipped.value, matching.DLIB_TOLERANCE.value)

    def test_lbph_is_honest_about_never_having_been_calibrated(self):
        with self.assertRaises(matching.NotCalibrated):
            matching.THRESHOLDS["lbph"].require_evidence()


class ProviderAssertionTests(unittest.TestCase):
    """ROCm reports healthy on this host and runs on the CPU anyway."""

    def test_default_is_cpu_only(self):
        self.assertEqual(arcface.default_providers(env={}),
                         ["CPUExecutionProvider"])

    def test_the_operator_can_opt_in_explicitly(self):
        self.assertEqual(
            arcface.default_providers(env={"FACE_ID_ARCFACE_PROVIDERS":
                                           "ROCMExecutionProvider, CPUExecutionProvider"}),
            ["ROCMExecutionProvider", "CPUExecutionProvider"])

    def test_thread_count_is_chosen_not_left_to_onnxruntime(self):
        # Left to itself onnxruntime sizes the intra-op pool to every core,
        # which on this 12-core host measured 48.5 ms per r50 embedding
        # against 27.6 ms at six threads. The session is also shared by every
        # camera thread, so oversubscribing turns extra cameras into
        # contention rather than throughput.
        self.assertEqual(arcface.default_intra_op_threads(env={}, cpu_count=12), 6)
        self.assertEqual(arcface.default_intra_op_threads(env={}, cpu_count=4), 2)
        self.assertEqual(arcface.default_intra_op_threads(env={}, cpu_count=1), 1)
        self.assertEqual(arcface.default_intra_op_threads(env={}, cpu_count=64), 6)

    def test_thread_count_can_be_handed_back_to_onnxruntime(self):
        env = {"FACE_ID_ARCFACE_THREADS": "0"}
        self.assertEqual(arcface.default_intra_op_threads(env=env, cpu_count=12), 0)
        env = {"FACE_ID_ARCFACE_THREADS": "3"}
        self.assertEqual(arcface.default_intra_op_threads(env=env, cpu_count=12), 3)

    def test_a_bound_provider_passes(self):
        self.assertEqual(
            arcface.check_providers(["CPUExecutionProvider"],
                                    ["CPUExecutionProvider"]),
            ["CPUExecutionProvider"])

    def test_an_unbound_provider_raises_instead_of_running_slowly(self):
        with self.assertRaises(arcface.ProviderUnavailable) as caught:
            arcface.check_providers(["CPUExecutionProvider"],
                                    ["ROCMExecutionProvider"])
        self.assertIn("ROCMExecutionProvider", str(caught.exception))

    def test_the_loader_asks_the_session_never_the_build(self):
        # onnxruntime.get_available_providers() lists ROCMExecutionProvider on
        # this host for a build whose ROCm libraries are missing. It is a claim
        # about the wheel, not about the session, and it must never be the
        # thing that is checked. Comments and docstrings are stripped first:
        # the rule is about what runs, and the prose is allowed to name the
        # function it is warning against.
        source = executable_source(inspect.getsource(arcface.ArcFaceEmbedder.load))
        self.assertIn("session.get_providers()", source)
        self.assertNotIn("get_available_providers", source)

    def test_no_executable_line_anywhere_consults_the_build(self):
        self.assertNotIn("get_available_providers",
                         executable_source(inspect.getsource(arcface)))


# --- fakes for the engine-level tests --------------------------------------

class FakeVariant:
    def __init__(self, name="w600k_r50", dim=512):
        self.name = name
        self.dim = dim
        self.input_size = 112


class FakeArcFacePipeline:
    """Stands in for arcface.ArcFacePipeline without onnxruntime or a model file."""

    def __init__(self, space="arcface:fake/yunet-5pt-align112", dim=4,
                 vector=None, detections=(), on_detect=None):
        self.variant = FakeVariant(name=space.split(":")[-1].split("/")[0],
                                   dim=dim)
        self._space = space
        self._vector = np.ones(dim) / np.sqrt(dim) if vector is None else vector
        self._detections = list(detections)
        self._on_detect = on_detect
        self.encode_calls = []

    def embedding_space(self):
        return self._space

    def encode_file(self, path):
        self.encode_calls.append(Path(path))
        return self._vector

    def detect(self, frame):
        if self._on_detect is not None:
            self._on_detect()
        return self._detections


def fake_detection(embedding, box=(1, 2, 3, 4)):
    return arcface.DetectedFace(box[0], box[1], box[2], box[3],
                                np.zeros((5, 2)), 0.99,
                                np.asarray(embedding, dtype=np.float64))


class EmbeddingSpaceCacheTests(unittest.TestCase):
    """A stale gallery must not survive an engine or variant switch."""

    def setUp(self):
        face_engine.clear_known_cache()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(face_engine.clear_known_cache)
        root = Path(self.tmp.name)
        for person, count in (("Toby", 2), ("Ada", 1)):
            (root / person).mkdir()
            for i in range(count):
                (root / person / f"{i}.jpg").write_bytes(b"")
        patcher = patch.object(face_engine, "KNOWN_DIR", root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_enrolment_reads_the_photos_on_disk(self):
        names = [name for name, _ in face_engine.enrolled_photos()]
        self.assertEqual(names, ["Ada", "Toby", "Toby"])

    def test_a_gallery_is_built_once_per_space(self):
        calls = []

        def encode(path):
            calls.append(path)
            return np.array([1.0, 0.0])

        first = face_engine.known_embeddings("space-a", encode)
        second = face_engine.known_embeddings("space-a", encode)
        self.assertIs(first, second)
        self.assertEqual(len(calls), 3)

    def test_a_different_space_gets_its_own_gallery(self):
        a = face_engine.known_embeddings("space-a", lambda p: np.zeros(128))
        b = face_engine.known_embeddings("space-b", lambda p: np.zeros(512))
        self.assertEqual(a[0][0].shape, (128,))
        self.assertEqual(b[0][0].shape, (512,))

    def test_photos_without_a_face_are_dropped_from_both_lists(self):
        encodings, names = face_engine.known_embeddings(
            "space-skip",
            lambda p: None if p.parent.name == "Ada" else np.zeros(8))
        self.assertEqual(len(encodings), len(names))
        self.assertEqual(names, ["Toby", "Toby"])

    def test_switching_arcface_variant_rebuilds_rather_than_reuses(self):
        r50 = FakeArcFacePipeline(space=arcface.embedding_space_id("w600k_r50"), dim=512)
        mbf = FakeArcFacePipeline(space=arcface.embedding_space_id("w600k_mbf"), dim=512,
                           vector=np.eye(512)[7])
        encs_r50, _, _ = face_engine.arcface_gallery(pipeline=r50)
        encs_mbf, _, _ = face_engine.arcface_gallery(pipeline=mbf)
        # Both pipelines were actually asked to encode: the second did not
        # silently inherit the first one's vectors.
        self.assertEqual(len(r50.encode_calls), 3)
        self.assertEqual(len(mbf.encode_calls), 3)
        self.assertFalse(np.allclose(encs_r50[0], encs_mbf[0]))

    def test_the_two_arcface_variants_do_not_share_a_space_id(self):
        self.assertNotEqual(arcface.embedding_space_id("w600k_r50"),
                            arcface.embedding_space_id("w600k_mbf"))
        self.assertNotEqual(arcface.embedding_space_id("w600k_r50"),
                            face_engine.DLIB_EMBEDDING_SPACE)

    def test_mismatched_dimensions_raise_instead_of_scoring(self):
        # The failure mode this closes: a 128-d dlib gallery meeting a 512-d
        # ArcFace query would otherwise need to be caught by someone noticing
        # the numbers looked odd.
        with self.assertRaises(ValueError):
            matching.score_gallery(np.zeros((3, 128)), np.zeros(512),
                                   matching.COSINE_SIMILARITY)

    def test_clear_known_cache_forces_a_rebuild(self):
        calls = []
        face_engine.known_embeddings("space-c", lambda p: calls.append(p) or np.zeros(4))
        face_engine.clear_known_cache()
        face_engine.known_embeddings("space-c", lambda p: calls.append(p) or np.zeros(4))
        self.assertEqual(len(calls), 6)


class NoCachedEncodingArtifactTests(unittest.TestCase):
    """Re-enrolment is cheap because nothing is persisted."""

    def test_no_encoding_artifact_is_written_to_disk(self):
        # If a .npy/.pkl/.npz gallery ever appears, an engine switch could
        # load embeddings produced by a model that is no longer in use, and
        # the space-keyed in-memory cache would not save us.
        repo = Path(__file__).parent
        skip = {".venv", ".git", "__pycache__", "node_modules"}
        found = [
            path for pattern in ("*.npy", "*.pkl", "*.npz", "*.pickle")
            for path in repo.rglob(pattern)
            if not skip & set(path.relative_to(repo).parts)
        ]
        self.assertEqual(found, [], f"unexpected persisted encodings: {found}")


class ArcFaceEngineTests(unittest.TestCase):
    """engine.build_detector's ArcFace path."""

    def setUp(self):
        face_engine.clear_known_cache()
        self.addCleanup(face_engine.clear_known_cache)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "Toby").mkdir()
        (root / "Toby" / "0.jpg").write_bytes(b"")
        patcher = patch.object(face_engine, "KNOWN_DIR", root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.enrolled = np.eye(8)[0]

    def build(self, detections=(), on_detect=None):
        pipeline = FakeArcFacePipeline(dim=8, vector=self.enrolled,
                               detections=list(detections), on_detect=on_detect)
        return face_engine._build_arcface(pipeline=pipeline), pipeline

    def test_faces_carry_the_metric_they_were_scored_in(self):
        detect, _ = self.build([fake_detection(self.enrolled)])
        face = detect(np.zeros((10, 10, 3), dtype=np.uint8))[0]
        self.assertEqual(face.metric, matching.COSINE_SIMILARITY)
        self.assertTrue(face.known)
        self.assertEqual(face.name, "Toby")
        self.assertAlmostEqual(face.conf, 1.0, places=6)

    def test_a_stranger_is_rejected_and_scores_low_not_high(self):
        detect, _ = self.build([fake_detection(np.eye(8)[3])])
        face = detect(np.zeros((10, 10, 3), dtype=np.uint8))[0]
        self.assertFalse(face.known)
        self.assertIsNone(face.name)
        # Higher is better here: an orthogonal stranger scores ~0.
        self.assertLess(face.conf, matching.THRESHOLDS["arcface"].value)

    def test_the_detector_advertises_its_metric_and_threshold(self):
        detect, _ = self.build()
        self.assertEqual(detect.engine, "arcface")
        self.assertEqual(detect.metric, matching.COSINE_SIMILARITY)
        self.assertEqual(detect.threshold, matching.THRESHOLDS["arcface"])
        self.assertEqual(detect.people, ["Toby"])

    def test_face_records_its_metric_field(self):
        self.assertEqual(face_engine.Face._fields,
                         ("x", "y", "w", "h", "name", "conf", "known", "metric"))

    def test_an_unknown_engine_name_is_refused(self):
        # It used to fall through to the yunet branch, so a typo in
        # FACE_ID_ENGINE silently ran a different recogniser.
        with self.assertRaises(ValueError):
            face_engine.build_detector(engine="arcfase")

    def test_arcface_is_the_default_engine(self):
        self.assertEqual(face_engine.DEFAULT_ENGINE, "arcface")

    def test_match_helper_refuses_a_bare_tolerance(self):
        with self.assertRaises(TypeError):
            face_engine._match([np.zeros(8)], ["Toby"], np.zeros(8), 0.50)


class ArcFaceConcurrencyTests(unittest.TestCase):
    """The process-wide dlib lock must not gate the ArcFace path."""

    def setUp(self):
        face_engine.clear_known_cache()
        self.addCleanup(face_engine.clear_known_cache)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        (root / "Toby").mkdir()
        (root / "Toby" / "0.jpg").write_bytes(b"")
        patcher = patch.object(face_engine, "KNOWN_DIR", root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_three_camera_threads_are_inside_detect_at_the_same_time(self):
        # A barrier can only be cleared if all three threads are genuinely
        # concurrent. Re-introduce the lock around the ArcFace path and this
        # test times out rather than merely running slower.
        barrier = threading.Barrier(3, timeout=5.0)
        detect = face_engine._build_arcface(
            pipeline=FakeArcFacePipeline(dim=8, detections=[], on_detect=barrier.wait))

        errors = []

        def run():
            try:
                detect(np.zeros((8, 8, 3), dtype=np.uint8))
            except Exception as exc:      # BrokenBarrierError on serialisation
                errors.append(exc)

        threads = [threading.Thread(target=run) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)
            self.assertFalse(t.is_alive())
        self.assertEqual(errors, [])

    def test_detect_completes_while_another_thread_holds_the_dlib_lock(self):
        detect = face_engine._build_arcface(pipeline=FakeArcFacePipeline(dim=8))
        done = threading.Event()

        def run():
            detect(np.zeros((8, 8, 3), dtype=np.uint8))
            done.set()

        with face_engine._LOCK:
            worker = threading.Thread(target=run)
            worker.start()
            self.assertTrue(done.wait(timeout=5.0),
                            "ArcFace detect blocked on the dlib lock")
        worker.join(timeout=5.0)

    def test_the_dlib_engines_still_take_the_lock(self):
        # The lock is a dlib workaround, not dead code: dlib's detector and
        # encoder are process-global C++ objects and concurrent calls segfault.
        for builder in (face_engine._build_yunet, face_engine._build_dlib,
                        face_engine._build_lbph):
            self.assertIn("with _LOCK:",
                          executable_source(inspect.getsource(builder)))
        self.assertNotIn("_LOCK",
                         executable_source(inspect.getsource(face_engine._build_arcface)))


_ONNX_MODEL = arcface.model_path(arcface.DEFAULT_VARIANT)


def _onnxruntime_available():
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return _ONNX_MODEL.is_file()


@unittest.skipUnless(_onnxruntime_available(),
                     "needs onnxruntime and the ArcFace weights installed")
class ArcFaceSessionTests(unittest.TestCase):
    """Checks that need a real ONNX session. Skipped when it is absent."""

    @classmethod
    def setUpClass(cls):
        cls.embedder = arcface.ArcFaceEmbedder.load(arcface.DEFAULT_VARIANT)

    def test_the_session_bound_the_provider_it_was_asked_for(self):
        self.assertIn("CPUExecutionProvider", self.embedder.providers)

    def test_batched_matches_serial_exactly(self):
        # The InsightFace graphs declare a static (1, 512) output, so any
        # batch > 1 makes onnxruntime log a shape warning. The annotation is
        # stale, not the arithmetic — this is the check that says so, and the
        # reason arcface.py raises the ORT log level.
        rng = np.random.default_rng(7)
        crops = [rng.integers(0, 256, (112, 112, 3), dtype=np.uint8)
                 for _ in range(4)]
        batched = self.embedder.embed_aligned_batch(crops)
        serial = np.stack([self.embedder.embed_aligned(c) for c in crops])
        np.testing.assert_array_equal(batched, serial)

    def test_embeddings_come_out_l2_normalised(self):
        crop = np.zeros((112, 112, 3), dtype=np.uint8)
        vector = self.embedder.embed_aligned(crop)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_a_missing_model_is_reported_not_downloaded(self):
        with self.assertRaises(arcface.ModelMissing) as caught:
            arcface.ArcFaceEmbedder.load(arcface.DEFAULT_VARIANT,
                                         models_dir="/nonexistent")
        self.assertIn("never downloads models automatically",
                      str(caught.exception))


# ---------------------------------------------------------------------------
# Backpressure — the fan-out mode a recorder needs and a preview must not get.
# ---------------------------------------------------------------------------


class BlockingOutput:
    """An encoder that has fallen behind: publish + close, and nothing else.

    That is the entire ``FrameOutput`` protocol, deliberately — an output that
    does not care about loss must not have to grow a method to say so, so the
    worker looks ``note_dropped`` up rather than requiring it.

    ``release`` lets exactly ``count`` frames through, so a test can put the
    queue in a known state instead of racing it.
    """

    def __init__(self):
        self.published = []
        self.closed = 0
        self._gate = threading.Semaphore(0)

    def release(self, count=1):
        for _ in range(count):
            self._gate.release()

    def publish(self, scene):
        self._gate.acquire()
        self.published.append(scene.sequence)

    def close(self):
        self.closed += 1
        self._gate.release()          # never leave the worker parked on stop


class LossAwareOutput(BlockingOutput):
    """The same, plus the optional hook a recorder implements."""

    def __init__(self):
        super().__init__()
        self.dropped_notices = 0

    def note_dropped(self, count=1):
        self.dropped_notices += count


class OutputWorkerModeTests(unittest.TestCase):
    """``_OutputWorker`` has two mailboxes and one contract."""

    def worker(self, output, **kwargs):
        worker = _OutputWorker(output, **kwargs)
        self.addCleanup(worker.join, 5.0)
        self.addCleanup(worker.stop)
        return worker

    def scene(self, sequence):
        source = SourceSpec.from_mapping({"type": "webcam"})
        return SceneFrame(source, sequence, float(sequence), None)

    def test_the_default_mailbox_still_drops_old_frames(self):
        """The preview's contract, unchanged: newest wins, the rest counted."""
        output = BlockingOutput()
        worker = self.worker(output)
        worker.start()
        for sequence in range(10):
            worker.submit(self.scene(sequence))
        self.assertFalse(worker.lossless)
        self.assertGreaterEqual(worker.dropped, 8)
        # And the producer never waited: ten submits against a blocked
        # consumer returned without a single frame being taken.
        self.assertEqual(output.published, [])

    def test_a_lossless_worker_keeps_every_frame_it_is_given(self):
        output = BlockingOutput()
        worker = self.worker(output, lossless=True)
        worker.start()
        for sequence in range(10):
            worker.submit(self.scene(sequence))
        output.release(10)
        deadline = time.monotonic() + 5.0
        while len(output.published) < 10 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(output.published, list(range(10)))
        self.assertEqual(worker.dropped, 0)

    def test_a_full_queue_makes_the_producer_wait(self):
        """Backpressure, measured: the submit that fills the queue blocks.

        This is the whole point of the mode. Capture fps dips here, visibly,
        instead of the recording losing time invisibly.
        """
        output = BlockingOutput()
        worker = self.worker(output, lossless=True, maxsize=2,
                             put_timeout_s=0.2)
        worker.start()
        worker.submit(self.scene(0))      # taken by the consumer, or queued
        worker.submit(self.scene(1))
        worker.submit(self.scene(2))
        started = time.monotonic()
        worker.submit(self.scene(3))      # queue full: this one has to wait
        waited = time.monotonic() - started
        self.assertGreaterEqual(waited, 0.15)
        self.assertEqual(worker.dropped, 1)

    def test_an_overflow_is_counted_and_the_output_is_told(self):
        output = LossAwareOutput()
        worker = self.worker(output, lossless=True, maxsize=1,
                             put_timeout_s=0.01)
        worker.start()
        for sequence in range(6):
            worker.submit(self.scene(sequence))
        self.assertGreater(worker.dropped, 0)
        # Counted in the worker *and* handed to the output, because only the
        # output writes the file that has the gap in it.
        self.assertEqual(output.dropped_notices, worker.dropped)

    def test_note_dropped_is_optional(self):
        """``FrameOutput`` is publish + close. The loss hook is extra."""
        output = BlockingOutput()
        self.assertFalse(hasattr(output, "note_dropped"))
        worker = self.worker(output, lossless=True, maxsize=1,
                             put_timeout_s=0.01)
        worker.start()
        for sequence in range(6):
            worker.submit(self.scene(sequence))
        self.assertGreater(worker.dropped, 0)
        self.assertEqual(worker.last_error, "")

    def test_a_lossless_worker_drains_what_is_queued_before_it_closes(self):
        """Stop must not put the gap at the end of the file instead."""
        output = BlockingOutput()
        worker = self.worker(output, lossless=True)
        worker.start()
        for sequence in range(5):
            worker.submit(self.scene(sequence))
        worker.stop()
        output.release(5)
        worker.join(5.0)
        self.assertEqual(output.published, list(range(5)))
        self.assertEqual(output.closed, 1)

    def test_close_runs_on_the_lossless_path_too(self):
        """The finalisation seam. Without it the MP4 has no moov atom."""
        for lossless in (False, True):
            with self.subTest(lossless=lossless):
                output = BlockingOutput()
                worker = _OutputWorker(output, lossless=lossless)
                worker.start()
                worker.stop()
                worker.join(5.0)
                self.assertFalse(worker.is_alive())
                self.assertEqual(output.closed, 1)

    def test_a_worker_out_of_drain_time_counts_what_it_abandons(self):
        """Even the give-up path is accounted for. The tail of the recording
        is then short by a number somebody can read."""
        output = LossAwareOutput()
        worker = self.worker(output, lossless=True, drain_timeout_s=0.0)
        worker.start()
        for sequence in range(5):
            worker.submit(self.scene(sequence))
        worker.stop()
        worker.join(5.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(worker.dropped + len(output.published), 5)
        self.assertEqual(output.dropped_notices, worker.dropped)
        self.assertEqual(output.closed, 1)

    def test_a_frame_submitted_after_the_loop_exits_is_not_lost_silently(self):
        """``remove_output`` stops a worker while the capture loop is still
        submitting, so a frame can land after the consumer has given up."""
        output = LossAwareOutput()
        # Never started: the point is the state of the queue *after* the
        # consumer loop has stopped reading it.
        worker = _OutputWorker(output, lossless=True)
        worker._queue.put(self.scene(0))
        worker._queue.put(self.scene(1))
        worker._drain_abandoned()
        self.assertEqual(worker.dropped, 2)
        self.assertEqual(output.dropped_notices, 2)
        self.assertTrue(worker._queue.empty())

    def test_a_lossless_worker_is_given_longer_to_shut_down(self):
        drop_old = _OutputWorker(BlockingOutput())
        lossless = _OutputWorker(BlockingOutput(), lossless=True)
        self.assertGreater(lossless.join_timeout_s, drop_old.join_timeout_s)


class PipelineDropMetricTests(unittest.TestCase):
    """The drop count reaches pipeline metrics, where the GUI can read it."""

    def workers(self, *specs):
        made = []
        for lossless, dropped in specs:
            worker = _OutputWorker(BlockingOutput(), lossless=lossless)
            worker.dropped = dropped
            made.append(worker)
        return tuple(made)

    def test_only_lossless_drops_are_reported(self):
        """A preview behind a 60 fps camera drops most frames and is fine.

        Summing the two would report a healthy system as lossy and bury the
        number that means something.
        """
        metrics = VisionPipeline._output_drop_metrics(
            self.workers((False, 5000), (True, 3)))
        self.assertEqual(metrics["dropped_output_frames"], 3)
        self.assertEqual(list(metrics["output_drops"]), ["BlockingOutput"])

    def test_a_healthy_recorder_publishes_a_zero_not_an_absence(self):
        metrics = VisionPipeline._output_drop_metrics(self.workers((True, 0)))
        self.assertEqual(metrics["dropped_output_frames"], 0)
        self.assertEqual(metrics["output_drops"], {"BlockingOutput": 0})

    def test_two_recorders_do_not_collide_on_one_key(self):
        """A dict merge here would silently delete one of the two counts."""
        metrics = VisionPipeline._output_drop_metrics(
            self.workers((True, 1), (True, 2)))
        self.assertEqual(metrics["dropped_output_frames"], 3)
        self.assertEqual(sorted(metrics["output_drops"].values()), [1, 2])
        self.assertEqual(len(metrics["output_drops"]), 2)

    def test_the_metric_is_json_safe_for_the_event_api(self):
        json.dumps(VisionPipeline._output_drop_metrics(self.workers((True, 4))))

    def test_the_live_pipeline_publishes_the_count(self):
        """End to end through the real capture loop, not the helper."""
        source = SourceSpec.from_mapping({"type": "webcam", "index": 0})
        # Real arrays: the capture loop runs a quality probe over the frame
        # every 30 frames, and it is cv2 all the way down.
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        capture = ScriptedCapture([frame] * 400)
        seen = threading.Event()

        def processor(frame, src, sequence, captured_at):
            return SceneFrame(src, sequence, captured_at, frame)

        pipeline = VisionPipeline(source, processor, [],
                                  opener=lambda _: capture,
                                  retry_min_s=0.001, retry_max_s=0.002)
        wedged = LossAwareOutput()
        pipeline.add_output(wedged, lossless=True)
        pipeline.add_output(CallbackOutput(lambda scene: seen.set()))
        pipeline.start()
        self.addCleanup(pipeline.join, 20.0)
        self.addCleanup(pipeline.stop)
        self.assertTrue(seen.wait(5.0))

        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            metrics = pipeline.state().metrics
            if metrics.get("dropped_output_frames", 0) > 0:
                break
            time.sleep(0.02)
        self.assertGreater(metrics.get("dropped_output_frames", 0), 0)
        self.assertEqual(sum(metrics["output_drops"].values()),
                         metrics["dropped_output_frames"])
        self.assertEqual(wedged.dropped_notices,
                         metrics["dropped_output_frames"])
        wedged.release(200)


# ---------------------------------------------------------------------------
# RecordingOutput — CFR, sidecar, finalisation, and where files may be written.
# ---------------------------------------------------------------------------


class FakeEncoder:
    """A stand-in ffmpeg process. Counts what was piped into it."""

    def __init__(self, command, frame_bytes=None, returncode=0):
        self.command = list(command)
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"")
        self.returncode = returncode
        self.waited = False
        self.killed = False
        self._frame_bytes = frame_bytes

    def wait(self, timeout=None):
        self.waited = True
        return self.returncode

    def kill(self):
        self.killed = True

    @property
    def frames(self):
        if not self._frame_bytes:
            return 0
        return len(self.stdin.getvalue()) // self._frame_bytes


class RecordingTestCase(unittest.TestCase):
    WIDTH, HEIGHT = 32, 24

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="acesvision-record-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.source = SourceSpec.from_mapping({"id": "webcam",
                                               "type": "webcam", "index": 0})
        self.raw = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
        self.spawned = []

    def spawn(self, command):
        encoder = FakeEncoder(command, frame_bytes=self.WIDTH * self.HEIGHT * 3)
        self.spawned.append(encoder)
        return encoder

    def recorder(self, **kwargs):
        kwargs.setdefault("fps", 30)
        kwargs.setdefault("spawn", self.spawn)
        output = RecordingOutput(self.tmp / "clip.mp4", **kwargs)
        self.addCleanup(output.close)
        return output

    def scene(self, sequence, captured_at, **kwargs):
        return SceneFrame(self.source, sequence, captured_at, self.raw,
                          metadata=kwargs.pop("metadata", {}), **kwargs)

    def sidecar(self, output):
        return json.loads(output.sidecar_path.read_text())


class RecordingPathTests(RecordingTestCase):
    def test_the_name_carries_the_time_and_the_source(self):
        name = recording_name("droidcam", datetime(2026, 8, 20, 15, 4, 5))
        self.assertEqual(name, "acesvision-20260820-150405-droidcam.mp4")

    def test_a_hostile_source_id_cannot_escape_the_filename(self):
        name = recording_name("../../etc/passwd",
                              datetime(2026, 8, 20, 15, 4, 5))
        self.assertNotIn("/", name)
        self.assertTrue(name.endswith(".mp4"))

    def test_the_environment_overrides_the_default_directory(self):
        self.assertEqual(recordings_dir({RECORDINGS_ENV: str(self.tmp)}),
                         self.tmp.resolve())
        self.assertEqual(recordings_dir({}),
                         Path("~/Videos/AcesVision").expanduser().resolve())

    def test_writing_inside_the_repository_is_refused(self):
        """AGPL, published, and a recording is a video of somebody's room."""
        for candidate in (REPO_ROOT, REPO_ROOT / "clip.mp4",
                          REPO_ROOT / "acesvision" / "deep" / "clip.mp4"):
            with self.subTest(candidate=candidate):
                with self.assertRaises(ValueError) as caught:
                    refuse_repo_path(candidate)
                self.assertIn("inside the repository", str(caught.exception))

    def test_the_refusal_covers_the_environment_variable_too(self):
        with self.assertRaises(ValueError):
            recordings_dir({RECORDINGS_ENV: str(REPO_ROOT / "recordings")})

    def test_gitignore_is_the_second_lock_on_the_same_door(self):
        ignore = (REPO_ROOT / ".gitignore").read_text().splitlines()
        self.assertIn("*.mp4", ignore)
        self.assertIn("*.mp4.json", ignore)

    def test_a_directory_argument_gets_the_generated_name(self):
        path = resolve_path(str(self.tmp), "webcam",
                            now=datetime(2026, 8, 20, 15, 4, 5))
        self.assertEqual(path,
                         self.tmp.resolve() / "acesvision-20260820-150405-webcam.mp4")

    def test_an_explicit_file_argument_is_taken_as_given(self):
        self.assertEqual(resolve_path(str(self.tmp / "a.mp4"), "webcam"),
                         (self.tmp / "a.mp4").resolve())

    def test_no_argument_falls_back_to_the_recordings_directory(self):
        path = resolve_path("", "webcam", now=datetime(2026, 8, 20, 15, 4, 5),
                            environ={RECORDINGS_ENV: str(self.tmp)})
        self.assertEqual(path.parent, self.tmp.resolve())

    def test_the_sidecar_sits_beside_the_video_and_cannot_shadow_a_json(self):
        output = self.recorder()
        self.assertEqual(output.sidecar_path.name, "clip.mp4.json")
        self.assertEqual(output.sidecar_path.parent, output.path.parent)


class RecordingCommandTests(RecordingTestCase):
    def test_the_software_command_is_browser_and_social_playable(self):
        command = software_command("/tmp/a.mp4", 1280, 720, 30)
        self.assertEqual(command[-1], "/tmp/a.mp4")
        # yuv420p and +faststart are not tuning knobs. Without the first,
        # browsers refuse the file; without the second, playback cannot start
        # until the whole file has been fetched.
        self.assertIn("yuv420p", command)
        self.assertIn("+faststart", command)
        self.assertIn("libx264", command)
        self.assertEqual(command[command.index("-crf") + 1], "20")
        self.assertEqual(command[command.index("-s") + 1], "1280x720")
        self.assertEqual(command[command.index("-i") + 1], "-")
        self.assertIn("bgr24", command)

    def test_the_hardware_command_uploads_to_vaapi(self):
        command = hardware_command("/tmp/a.mp4", 1280, 720, 30)
        self.assertIn("h264_vaapi", command)
        self.assertIn("format=nv12,hwupload", command)
        self.assertEqual(command[command.index("-vaapi_device") + 1],
                         "/dev/dri/renderD128")
        self.assertIn("+faststart", command)
        self.assertNotIn("libx264", command)

    def test_the_recorder_picks_the_command_from_the_flag(self):
        for hardware, codec in ((False, "libx264"), (True, "h264_vaapi")):
            with self.subTest(hardware=hardware):
                output = self.recorder(hardware=hardware)
                output.publish(self.scene(0, 100.0))
                self.assertIn(codec, output.command)
                self.assertEqual(output.command[output.command.index("-s") + 1],
                                 f"{self.WIDTH}x{self.HEIGHT}")


class RecordingClockTests(RecordingTestCase):
    """CFR against a fake clock — captured_at is the clock, so it is exact."""

    def publish_at(self, output, times):
        for sequence, captured_at in enumerate(times):
            output.publish(self.scene(sequence, captured_at))

    def test_capture_at_the_target_rate_writes_one_frame_each(self):
        output = self.recorder(fps=30)
        self.publish_at(output, [100.0 + n / 30.0 for n in range(30)])
        self.assertEqual(output.frames_written, 30)
        self.assertEqual(output.frames_duplicated, 0)
        self.assertEqual(output.scenes_dropped_early, 0)
        self.assertEqual(self.spawned[0].frames, 30)

    def test_capture_slower_than_the_target_duplicates_to_fill_the_gap(self):
        """15 fps in, 30 fps out: every frame is written twice."""
        output = self.recorder(fps=30)
        self.publish_at(output, [100.0 + n / 15.0 for n in range(10)])
        self.assertEqual(output.frames_written, 19)   # slots 0..18
        self.assertEqual(output.frames_duplicated, 9)
        self.assertEqual(output.scenes_dropped_early, 0)
        self.assertEqual(self.spawned[0].frames, 19)

    def test_capture_faster_than_the_target_drops_from_the_video(self):
        """60 fps in, 30 fps out: half the scenes land inside a filled slot."""
        output = self.recorder(fps=30)
        self.publish_at(output, [100.0 + n / 60.0 for n in range(20)])
        self.assertEqual(output.frames_written, 10)
        self.assertEqual(output.scenes_dropped_early, 10)
        self.assertEqual(output.scenes_published, 20)

    def test_a_long_stall_is_filled_rather_than_left_as_a_gap(self):
        """The artefact this whole design exists to prevent.

        A VFR writer would put a 1 s hole in the timeline here. CFR fills it,
        and the sidecar still says the truth about when frames arrived.
        """
        output = self.recorder(fps=30)
        self.publish_at(output, [100.0, 101.0])
        self.assertEqual(output.frames_written, 31)
        self.assertEqual(output.frames_duplicated, 29)
        self.assertAlmostEqual(output.summary()["duration_s"], 31 / 30,
                               places=5)

    def test_the_slot_boundary_does_not_lose_a_frame_to_float_error(self):
        """elapsed * fps landing on 0.9999999 would cost a frame every time."""
        output = self.recorder(fps=30)
        self.publish_at(output, [0.0 + n * (1.0 / 30.0) for n in range(600)])
        self.assertEqual(output.frames_written, 600)
        self.assertEqual(output.scenes_dropped_early, 0)

    def test_a_frame_arriving_before_the_start_does_not_rewind_the_clock(self):
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0))
        output.publish(self.scene(1, 99.0))       # a clock that went backwards
        self.assertEqual(output.frames_written, 1)
        self.assertEqual(output.scenes_dropped_early, 1)

    def test_the_video_duration_is_the_frame_count_over_the_rate(self):
        output = self.recorder(fps=25)
        self.publish_at(output, [100.0 + n / 25.0 for n in range(50)])
        self.assertEqual(output.summary()["duration_s"], 2.0)


class RecordingSidecarTests(RecordingTestCase):
    def test_the_sidecar_carries_every_frame_s_true_capture_time(self):
        output = self.recorder(fps=30)
        times = [100.0, 100.02, 100.031, 100.9]
        for sequence, captured_at in enumerate(times):
            output.publish(self.scene(sequence, captured_at,
                                      metadata={"inference_fps": 40.0}))
        output.close()

        document = self.sidecar(output)
        self.assertEqual(document["schema"], SIDECAR_SCHEMA)
        self.assertEqual([f["captured_at"] for f in document["frames"]], times)
        self.assertEqual([f["sequence"] for f in document["frames"]],
                         [0, 1, 2, 3])
        self.assertEqual(document["frames"][0]["metadata"],
                         {"inference_fps": 40.0})

    def test_a_scene_dropped_from_the_video_is_still_in_the_record(self):
        """Nothing is lost by the CFR drop — that is why it is allowed."""
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0))
        output.publish(self.scene(1, 100.001))    # inside slot 0
        output.close()

        frames = self.sidecar(output)["frames"]
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[1]["video_frames"], 0)
        self.assertEqual(frames[1]["video_frame"], 0)   # shares slot 0
        self.assertEqual(frames[1]["captured_at"], 100.001)

    def test_the_sidecar_carries_the_inference_results(self):
        output = self.recorder(fps=30)
        output.publish(self.scene(
            0, 100.0,
            objects=[Detection(1, 2, 3, 4, "person", 0.9, 7)],
            faces=[Face(1, 2, 3, 4, "Toby", 0.31, True)],
            gestures=[Gesture("Victory", 0.8, 1, 2, 3, 4)]))
        output.close()

        frame = self.sidecar(output)["frames"][0]
        self.assertEqual(frame["objects"][0]["label"], "person")
        self.assertEqual(frame["objects"][0]["track_id"], 7)
        self.assertEqual(frame["faces"][0]["name"], "Toby")
        self.assertEqual(frame["gestures"][0]["name"], "Victory")

    def test_numpy_scalars_from_the_detectors_survive_serialisation(self):
        """YOLO hands back float32. json.dump refuses it by default."""
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0, objects=[
            Detection(np.int64(1), np.int64(2), np.int64(3), np.int64(4),
                      "person", np.float32(0.5), np.int64(7))]))
        output.close()
        self.assertAlmostEqual(
            self.sidecar(output)["frames"][0]["objects"][0]["score"], 0.5,
            places=5)

    def test_the_dropped_frame_count_lands_in_the_sidecar(self):
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0))
        output.note_dropped(4)
        output.note_dropped()
        output.close()
        document = self.sidecar(output)
        self.assertEqual(document["frames_dropped_by_backpressure"], 5)
        self.assertEqual(document["scenes_published"], 1)

    def test_a_healthy_recording_reports_zero_rather_than_omitting_it(self):
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0))
        output.close()
        self.assertEqual(self.sidecar(output)["frames_dropped_by_backpressure"],
                         0)

    def test_the_sidecar_names_the_source_and_the_encoder(self):
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0))
        output.close()
        document = self.sidecar(output)
        self.assertEqual(document["source"]["id"], "webcam")
        self.assertEqual(document["video"], "clip.mp4")
        self.assertFalse(document["encoder"]["hardware"])
        self.assertIn("libx264", document["encoder"]["command"])
        self.assertEqual((document["width"], document["height"]),
                         (self.WIDTH, self.HEIGHT))

    def test_no_frames_means_no_sidecar_and_no_process(self):
        output = self.recorder(fps=30)
        output.close()
        self.assertEqual(self.spawned, [])
        self.assertFalse(output.sidecar_path.exists())


class RecordingFinalisationTests(RecordingTestCase):
    def test_close_shuts_the_pipe_and_waits_for_the_moov_atom(self):
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0))
        output.close()
        encoder = self.spawned[0]
        self.assertTrue(encoder.stdin.closed)
        self.assertTrue(encoder.waited)

    def test_close_is_idempotent(self):
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0))
        output.close()
        output.close()
        self.assertEqual(len(self.spawned), 1)

    def test_publishing_after_close_does_not_resurrect_the_encoder(self):
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0))
        output.close()
        output.publish(self.scene(1, 101.0))
        self.assertEqual(len(self.spawned), 1)
        self.assertEqual(output.frames_written, 1)

    def test_a_non_zero_exit_is_recorded_rather_than_swallowed(self):
        def spawn(command):
            encoder = FakeEncoder(command, returncode=1)
            encoder.stderr = io.BytesIO(b"height not divisible by 2")
            self.spawned.append(encoder)
            return encoder

        output = RecordingOutput(self.tmp / "clip.mp4", fps=30, spawn=spawn)
        output.publish(self.scene(0, 100.0))
        output.close()
        self.assertIn("ffmpeg exited 1", output.error)
        self.assertIn("divisible by 2", output.error)
        self.assertIn("divisible by 2", self.sidecar(output)["error"])

    def test_a_missing_ffmpeg_fails_loudly_at_the_first_frame(self):
        def spawn(command):
            raise FileNotFoundError("no ffmpeg")

        output = RecordingOutput(self.tmp / "clip.mp4", fps=30, spawn=spawn)
        with self.assertRaises(RecordingError):
            output.publish(self.scene(0, 100.0))

    def test_a_broken_pipe_does_not_claim_frames_the_file_does_not_have(self):
        """An inflated frames_written would inflate duration_s in the sidecar,
        and a record that claims seconds the video does not contain is worse
        than one that reports an error."""
        class Exploding(FakeEncoder):
            def __init__(self, command):
                super().__init__(command, returncode=1)
                self.stdin = self

            closed = False

            def write(self, payload):
                raise BrokenPipeError("gone")

            def close(self):
                Exploding.closed = True

        def spawn(command):
            encoder = Exploding(command)
            self.spawned.append(encoder)
            return encoder

        output = RecordingOutput(self.tmp / "clip.mp4", fps=30, spawn=spawn)
        with self.assertRaises(RecordingError):
            output.publish(self.scene(0, 100.0))
        self.assertEqual(output.frames_written, 0)
        self.assertEqual(self.sidecar(output)["duration_s"], 0.0)
        self.assertIn("stopped accepting frames", output.error)

    def test_ffmpeg_is_detached_from_the_terminal_s_signal_group(self):
        """Ctrl-C signals the process *group*, and ffmpeg is in it.

        Measured on a real run: the terminal's SIGINT reached ffmpeg, which
        wrote the trailer but exited 255, so a perfectly good recording was
        reported as a failure. Detached, close() is the only thing that ends
        the encode.
        """
        import acesvision.recording as recording

        seen = {}

        def fake_popen(command, **kwargs):
            seen.update(kwargs)
            return FakeEncoder(command)

        with patch.object(recording.subprocess, "Popen", fake_popen):
            recording._spawn(["ffmpeg"])
        self.assertTrue(seen.get("start_new_session"))
        self.assertIs(seen.get("stdin"), subprocess.PIPE)

    def test_the_worker_finally_finalises_a_lossless_recorder(self):
        """The seam Ctrl-C depends on: worker.run's finally -> close()."""
        output = self.recorder(fps=30)
        worker = _OutputWorker(output, lossless=True)
        worker.start()
        worker.submit(self.scene(0, 100.0))
        deadline = time.monotonic() + 5.0
        while not self.spawned and time.monotonic() < deadline:
            time.sleep(0.01)
        worker.stop()
        worker.join(10.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(self.spawned[0].stdin.closed)
        self.assertTrue(output.sidecar_path.exists())


class RecordingOwnershipTests(RecordingTestCase):
    def test_the_recorder_owns_its_own_smoother(self):
        """Each output sees a different frame subset through its mailbox, so a
        shared smoother would advance its clock against frames this recorder
        never received."""
        first, second = self.recorder(), self.recorder()
        self.assertIsNot(first._smoother, second._smoother)
        self.assertIsNot(first._smoother, LatestFrameOutput()._smoother)
        self.assertIsNot(first._smoother, ObsVirtualCameraOutput()._smoother)
        self.assertIsInstance(first._smoother, SceneSmoother)

    def test_the_sidecar_records_the_unsmoothed_detections(self):
        """The eased boxes are the picture; the raw detections are the record.

        Same split as ``LatestFrameOutput.snapshot``.
        """
        output = self.recorder(fps=30)
        start = Detection(10, 10, 20, 20, "person", 0.9, 1)
        moved = start._replace(x=110)
        metadata = {"inference_fps": 40.0}
        output.publish(self.scene(0, 100.0, objects=[start],
                                  metadata=dict(metadata)))
        output.publish(self.scene(1, 100.01, objects=[moved],
                                  metadata=dict(metadata)))
        output.close()

        recorded = [f["objects"][0]["x"] for f in self.sidecar(output)["frames"]]
        self.assertEqual(recorded, [10, 110])

    def test_a_source_switch_is_resized_rather_than_corrupting_the_stream(self):
        output = self.recorder(fps=30)
        output.publish(self.scene(0, 100.0))
        other = SceneFrame(self.source, 1, 100.1,
                           np.zeros((self.HEIGHT * 2, self.WIDTH * 2, 3),
                                    dtype=np.uint8))
        output.publish(other)
        encoder = self.spawned[0]
        # Every frame in the pipe is still the size the stream was opened at.
        self.assertEqual(len(encoder.stdin.getvalue()) %
                         (self.WIDTH * self.HEIGHT * 3), 0)
        self.assertEqual(encoder.frames, output.frames_written)


class RecordingCliTests(RecordingTestCase):
    def parse(self, argv):
        from acesvision.__main__ import build_parser

        return build_parser().parse_args(argv)

    def test_recording_is_off_unless_asked_for(self):
        from acesvision.__main__ import build_recorder

        args = self.parse([])
        self.assertIsNone(args.record)
        self.assertIsNone(build_recorder(args, self.source))

    def test_the_bare_flag_uses_the_recordings_directory(self):
        from acesvision.__main__ import build_recorder

        recorder = build_recorder(self.parse(["--record"]), self.source,
                                  now=datetime(2026, 8, 20, 15, 4, 5),
                                  environ={RECORDINGS_ENV: str(self.tmp)})
        self.assertEqual(recorder.path,
                         self.tmp.resolve() /
                         "acesvision-20260820-150405-webcam.mp4")
        self.assertEqual(recorder.fps, 30)
        self.assertFalse(recorder.hardware)

    def test_the_flags_reach_the_recorder(self):
        from acesvision.__main__ import build_recorder

        args = self.parse(["--record", str(self.tmp / "a.mp4"),
                           "--record-fps", "60", "--record-hw"])
        recorder = build_recorder(args, self.source)
        self.assertEqual(recorder.path, (self.tmp / "a.mp4").resolve())
        self.assertEqual(recorder.fps, 60)
        self.assertTrue(recorder.hardware)

    def test_a_path_inside_the_repository_is_refused_at_startup(self):
        from acesvision.__main__ import build_recorder

        args = self.parse(["--record", str(REPO_ROOT / "a.mp4")])
        with self.assertRaises(ValueError):
            build_recorder(args, self.source)

    def test_the_runner_adds_the_recorder_lossless(self):
        """Read off the source, because getting this wrong is silent: the file
        would simply have gaps in it and nothing would say so."""
        import acesvision.__main__ as runner

        source = inspect.getsource(runner.main)
        self.assertRegex(source, r"add_output\(\s*recorder,\s*lossless=True")


if __name__ == "__main__":
    unittest.main()
