"""Non-invasive local webcam inventory and bounded DroidCam discovery."""
from __future__ import annotations

import ipaddress
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class WebcamDevice:
    index: int
    name: str
    path: str
    kind: str
    label: str
    stable_path: str | None = None

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DroidCamDevice:
    host: str
    port: int
    url: str
    label: str

    def as_dict(self):
        return asdict(self)


def discover_webcams(sysfs=Path("/sys/class/video4linux")):
    """List primary V4L2 capture nodes without opening or locking a camera."""
    devices = []
    if not sysfs.exists():
        return devices
    for node in sorted(sysfs.glob("video*"), key=_video_number):
        try:
            video_index = _video_number(node)
            interface_index = int((node / "index").read_text().strip())
            name = (node / "name").read_text().strip()
        except (OSError, ValueError):
            continue
        # UVC devices normally expose capture as index 0 and metadata as index 1.
        # Metadata nodes cannot produce frames and should not be user choices.
        if interface_index != 0:
            continue
        kind = classify_camera(name, node)
        display_kind = {
            "colour": "Colour",
            "infrared": "IR",
            "virtual": "Virtual",
            "camera": "Camera",
        }[kind]
        path = f"/dev/video{video_index}"
        stable_path = _stable_v4l_path(path)
        devices.append(WebcamDevice(
            video_index, name, path, kind,
            f"Camera {video_index}  |  {name}  |  {display_kind}  |  {path}",
            stable_path,
        ))
    return devices


def preferred_webcam(devices):
    """Choose a real colour camera first, then another real capture device."""
    devices = list(devices)
    return next((device for device in devices if device.kind == "colour"),
                next((device for device in devices if device.kind != "virtual"), None))


def _video_number(path):
    return int(Path(path).name.removeprefix("video"))


def _stable_v4l_path(device_path, roots=None):
    """Return a reconnect-stable V4L symlink for a transient /dev/video node."""
    roots = roots or (Path("/dev/v4l/by-id"), Path("/dev/v4l/by-path"))
    target = Path(device_path)
    try:
        target = target.resolve()
    except OSError:
        return None
    for root in roots:
        if not root.exists():
            continue
        for link in sorted(root.iterdir()):
            if not link.name.endswith("video-index0"):
                continue
            try:
                if link.resolve() == target:
                    return str(link)
            except OSError:
                continue
    return None


def classify_camera(name, node=None):
    lowered = name.lower()
    if ("infrared" in lowered or "ir camera" in lowered or
            "ir camer" in lowered or lowered.endswith(" ir")):
        return "infrared"
    if any(word in lowered for word in ("loopback", "virtual", "obs")):
        return "virtual"
    if node is not None:
        try:
            resolved = str((Path(node) / "device").resolve()).lower()
            if "virtual" in resolved:
                return "virtual"
        except OSError:
            pass
    if any(word in lowered for word in ("webcam", "camera", "uvc")):
        return "colour"
    return "camera"


def local_scan_networks(addresses=None):
    """Return private /24s containing this host. Never returns public networks."""
    if addresses is None:
        addresses = set()
        try:
            for result in socket.getaddrinfo(socket.gethostname(), None,
                                             family=socket.AF_INET):
                addresses.add(result[4][0])
        except socket.gaierror:
            pass
    networks = set()
    for address in addresses:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.version == 4 and ip.is_private and not ip.is_loopback:
            networks.add(ipaddress.ip_network(f"{ip}/24", strict=False))
    return sorted(networks, key=str)


def scan_droidcam(networks=None, port=4747, timeout_s=0.12,
                  connector=socket.create_connection, max_workers=32):
    """Find hosts accepting DroidCam's default TCP port on private local /24s."""
    networks = list(networks if networks is not None else local_scan_networks())
    candidates = []
    for network in networks:
        network = ipaddress.ip_network(network, strict=False)
        if not network.is_private or network.version != 4:
            continue
        # Bound every requested network to a /24 to avoid broad network scans.
        if network.prefixlen < 24:
            continue
        candidates.extend(str(host) for host in network.hosts())

    def probe(host):
        connection = None
        try:
            connection = connector((host, port), timeout=timeout_s)
            return DroidCamDevice(
                host, port, f"http://{host}:{port}/video",
                f"Possible DroidCam  |  {host}:{port}",
            )
        except (OSError, TimeoutError):
            return None
        finally:
            if connection is not None:
                connection.close()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        found = [device for device in pool.map(probe, candidates) if device]
    return sorted(found, key=lambda device: ipaddress.ip_address(device.host))
