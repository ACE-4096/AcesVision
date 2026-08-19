"""Non-invasive local webcam inventory and bounded DroidCam discovery."""
from __future__ import annotations

import ipaddress
import os
import socket
import struct
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path

try:
    import fcntl
except ImportError:                     # pragma: no cover - runtime target is Linux
    # Only the interface-address lookup needs it, and it degrades to "this host
    # has no enumerable interfaces", which is a loud failure rather than a
    # silent empty scan. The rest of this module still imports.
    fcntl = None


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


# ---------------------------------------------------------------------------
# Local network discovery
#
# Two halves, kept separable on purpose: ACQUISITION asks the kernel which
# interfaces exist and what kind they are, SELECTION decides which of those
# this program is allowed to scan.
#
# Only selection used to exist. Acquisition was
# ``socket.getaddrinfo(socket.gethostname())``, which on Debian and Ubuntu
# resolves the hostname through /etc/hosts to 127.0.1.1 and therefore returns
# nothing but loopback. Selection then correctly discarded loopback, discovery
# returned [], and the Sources page found no DroidCam on any standard Linux
# host, ever. Every test passed, because every test injected ``addresses=`` and
# so exercised selection only. The injection seam is still here — it is how
# selection stays unit-testable — but acquisition is now real code with its own
# fake-interface-table tests, and it fails loudly instead of returning [].
# ---------------------------------------------------------------------------

# Interface flags, from <linux/if.h>. /sys/class/net/<name>/flags is that same
# word, in hex. IFF_RUNNING is deliberately not consulted: this host reports
# 0x1003 (UP|BROADCAST|MULTICAST) for a working, cabled, carrying NIC.
IFF_UP = 0x1
IFF_LOOPBACK = 0x8
IFF_POINTOPOINT = 0x10

# ARP hardware types, from <linux/if_arp.h>. A DroidCam phone can only be
# reached over a link that carries ethernet frames on a shared segment.
ARPHRD_ETHER = 1
ARPHRD_LOOPBACK = 772
ARPHRD_IEEE80211 = 801
ARPHRD_IEEE80211_PRISM = 802
ARPHRD_IEEE80211_RADIOTAP = 803
ETHERNET_ARP_TYPES = frozenset({ARPHRD_ETHER})
WIRELESS_ARP_TYPES = frozenset({
    ARPHRD_IEEE80211, ARPHRD_IEEE80211_PRISM, ARPHRD_IEEE80211_RADIOTAP,
})

# ioctls, from <linux/sockios.h>. /sys exposes an interface's flags and its
# hardware type but not its addresses, and the standard library has no netlink
# client, so this is the stdlib-only way to read an interface address on Linux.
SIOCGIFADDR = 0x8915
SIOCGIFNETMASK = 0x891B

SYSFS_NET = Path("/sys/class/net")

# Interfaces this program will not scan even when they carry a private IPv4.
# Scanning them would mean port-scanning VPN peers, hypervisor guests and
# container networks — other people's machines, on a machine whose owner asked
# only to find their own phone. Name matching is one of two gates, never the
# only one: names can be changed, so the flag, hardware-type and
# physical-adapter checks in interface_exclusion_reason() must pass as well.
EXCLUDED_INTERFACE_PATTERNS = {
    "loopback": ("lo", "lo[0-9]*"),
    "tunnel": ("tun*", "tap*", "ppp*", "sit*", "gre*", "ip6tnl*"),
    "VPN overlay": ("wg*", "tailscale*", "utun*", "zt*", "nordlynx*", "proton*"),
    "hypervisor guest network": ("virbr*", "vnet*", "vboxnet*", "vmnet*",
                                 "xenbr*"),
    "container network": ("docker*", "br-*", "veth*", "cni*", "flannel*",
                          "cali*", "kube*"),
    "synthetic device": ("dummy*", "ifb*", "teql*"),
}

# Never widen a scan past a /24 (254 hosts). A /16 sweep is 65,534 hosts: too
# slow to be usable and far too broad to be defensible. An interface on a wider
# subnet is scanned only across the /24 that contains this host.
MAX_SCAN_HOSTS_PREFIX = 24

# Operator overrides. ACESVISION_SCAN_NETWORKS is the more specific of the two
# and wins. Both are honoured over auto-detection; neither can lift the
# private-address and /24 bounds, which are safety properties, not defaults.
SCAN_NETWORKS_ENV = "ACESVISION_SCAN_NETWORKS"
SCAN_INTERFACES_ENV = "ACESVISION_SCAN_INTERFACES"


class NoScannableNetwork(RuntimeError):
    """Raised when no interface on this host may be scanned.

    Discovery used to answer this case with an empty list, which the GUI showed
    as "No DroidCam devices found" — the same words it uses for a network that
    was scanned and held no phone. The two are not the same answer and must not
    read the same.
    """


@dataclass(frozen=True)
class NetworkInterface:
    """One interface as the kernel describes it, before any policy is applied."""

    name: str
    address: str | None = None
    netmask: str | None = None
    kind: str = "unknown"          # ethernet, wireless, loopback, tunnel, virtual
    is_up: bool = False
    is_physical: bool = False
    is_point_to_point: bool = False

    def as_dict(self):
        return asdict(self)

    def scan_network(self):
        """The network to scan for this interface, capped at a /24.

        A narrower subnet than /24 is honoured as-is — it is cheaper and no
        broader. A wider one (a /16 docker bridge, a /8) is cut down to the /24
        around this host's own address.
        """
        if not self.address:
            return None
        prefix = MAX_SCAN_HOSTS_PREFIX
        if self.netmask:
            try:
                configured = ipaddress.ip_network(f"0.0.0.0/{self.netmask}").prefixlen
            except ValueError:
                configured = prefix
            prefix = max(prefix, configured)
        return ipaddress.ip_network(f"{self.address}/{prefix}", strict=False)

    def describe(self):
        address = f"{self.address}/{self.netmask}" if self.address else "no IPv4"
        return f"{self.name}  |  {self.kind}  |  {address}"


@dataclass(frozen=True)
class ScanPlan:
    """What discovery is about to scan, what it skipped, and why.

    Exists so the answer is inspectable before any packet is sent. Silent
    network scanning is not acceptable behaviour for this program, and scanning
    the wrong network quietly is worse than finding nothing.
    """

    networks: tuple = ()
    selected: tuple = ()
    excluded: tuple = ()
    origin: str = "interfaces"

    def describe(self):
        lines = []
        if self.networks:
            lines.append("Networks to scan: "
                         + ", ".join(str(network) for network in self.networks))
        else:
            lines.append("Networks to scan: none")
        for interface in self.selected:
            lines.append(f"  scan    {interface.describe()} "
                         f"-> {interface.scan_network()}")
        for name, reason in self.excluded:
            lines.append(f"  skip    {name}  |  {reason}")
        lines.append(f"Source: {self.origin}")
        lines.append(f"Override with {SCAN_INTERFACES_ENV}=<names> or "
                     f"{SCAN_NETWORKS_ENV}=<cidrs>.")
        return "\n".join(lines)

    def summary(self):
        """One line, for the GUI."""
        if not self.networks:
            return "No scannable network found"
        names = ", ".join(interface.name for interface in self.selected)
        networks = ", ".join(str(network) for network in self.networks)
        if self.origin != "interfaces":
            return f"{networks} (from {self.origin})"
        return f"{networks} ({names})" if names else networks


def _sysfs_text(node, name):
    try:
        return (node / name).read_text().strip()
    except OSError:
        return ""


def _sysfs_int(node, name):
    text = _sysfs_text(node, name)
    try:
        return int(text, 0)
    except ValueError:
        return None


def _ipv4_address(name):
    """(address, netmask) for one interface, asked of the kernel directly."""
    if fcntl is None:                   # pragma: no cover - runtime target is Linux
        return None, None
    request = struct.pack("256s", name[:15].encode("utf-8"))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        address = socket.inet_ntoa(
            fcntl.ioctl(sock.fileno(), SIOCGIFADDR, request)[20:24])
        netmask = socket.inet_ntoa(
            fcntl.ioctl(sock.fileno(), SIOCGIFNETMASK, request)[20:24])
        return address, netmask
    except OSError:
        # No IPv4 configured on this interface. Not an error: a tap device for
        # a stopped VM looks exactly like this.
        return None, None
    finally:
        sock.close()


def _interface_kind(node, flags, arp_type, is_physical):
    if flags & IFF_LOOPBACK or arp_type == ARPHRD_LOOPBACK:
        return "loopback"
    if flags & IFF_POINTOPOINT or (arp_type is not None
                                   and arp_type not in ETHERNET_ARP_TYPES
                                   and arp_type not in WIRELESS_ARP_TYPES):
        # tun0 and tailscale0 report ARPHRD_NONE (65534) and IFF_POINTOPOINT.
        return "tunnel"
    if arp_type in WIRELESS_ARP_TYPES or (node / "wireless").exists() or \
            (node / "phy80211").exists():
        return "wireless"
    if is_physical:
        return "ethernet"
    # Ethernet-flagged but backed by no hardware: a bridge (virbr0, docker0,
    # br-<id>) or a veth. Real ethernet frames, nobody's LAN.
    return "virtual"


def read_interface_table(sysfs=SYSFS_NET, address_lookup=_ipv4_address):
    """Every network interface on this host, as the kernel reports it.

    This is the acquisition half. It reads /sys/class/net for identity and
    flags, and asks the kernel for addresses through SIOCGIFADDR, because
    hostname resolution answers a different question than "what is this machine
    plugged into".
    """
    sysfs = Path(sysfs)
    interfaces = []
    if not sysfs.exists():
        return interfaces
    for node in sorted(sysfs.iterdir(), key=lambda path: path.name):
        name = node.name
        flags = _sysfs_int(node, "flags") or 0
        arp_type = _sysfs_int(node, "type")
        operstate = _sysfs_text(node, "operstate")
        # A device symlink means a real adapter behind the interface — PCI, USB,
        # SDIO. Bridges, tunnels and veth pairs have none. It is the single most
        # reliable "is this physical" signal sysfs offers.
        is_physical = (node / "device").exists()
        try:
            address, netmask = address_lookup(name)
        except OSError:
            address, netmask = None, None
        interfaces.append(NetworkInterface(
            name=name,
            address=address,
            netmask=netmask,
            kind=_interface_kind(node, flags, arp_type, is_physical),
            is_up=bool(flags & IFF_UP) and operstate != "down",
            is_physical=is_physical,
            is_point_to_point=bool(flags & IFF_POINTOPOINT),
        ))
    return interfaces


def interface_name_exclusion(name):
    """(pattern, category) this interface name is excluded by, or None."""
    for category, patterns in EXCLUDED_INTERFACE_PATTERNS.items():
        for pattern in patterns:
            if fnmatch(name, pattern):
                return pattern, category
    return None


def interface_exclusion_reason(interface, trusted=False):
    """Why this interface must not be scanned, or None if it may be.

    ``trusted`` is the operator override: they named this interface explicitly,
    so the type and naming policy steps aside. The address checks do not — a
    public network is never scanned, whoever asks.
    """
    if not trusted:
        excluded_by_name = interface_name_exclusion(interface.name)
        if excluded_by_name is not None:
            pattern, category = excluded_by_name
            return f"{category} (name matches {pattern})"
        if interface.kind not in ("ethernet", "wireless"):
            return f"{interface.kind} interface, not a physical LAN adapter"
        if not interface.is_physical:
            return "no hardware behind it (bridge or virtual device)"
        if interface.is_point_to_point:
            return "point-to-point link, not a shared LAN segment"
        if not interface.is_up:
            return "interface is down"
    if not interface.address:
        return "no IPv4 address"
    try:
        ip = ipaddress.ip_address(interface.address)
    except ValueError:
        return f"unparsable address {interface.address!r}"
    if ip.version != 4:
        return "not IPv4"
    if ip.is_loopback:
        return "loopback address"
    if not ip.is_private:
        return "public address"
    return None


def parse_scan_networks(text):
    """Parse an operator's ACESVISION_SCAN_NETWORKS value.

    Refuses anything this program would not scan on its own: a public range, or
    a prefix wider than /24. Refusing beats silently narrowing, because the
    operator asked for something specific and should be told it was not done.
    """
    networks = []
    for item in str(text).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            network = ipaddress.ip_network(item, strict=False)
        except ValueError as exc:
            raise ValueError(f"{SCAN_NETWORKS_ENV}: {item!r} is not a network "
                             f"({exc})") from exc
        if network.version != 4:
            raise ValueError(f"{SCAN_NETWORKS_ENV}: {item!r} is not IPv4")
        if not network.is_private or network.is_loopback:
            raise ValueError(f"{SCAN_NETWORKS_ENV}: {item!r} is not a private "
                             f"local network — refusing to scan it")
        if network.prefixlen < MAX_SCAN_HOSTS_PREFIX:
            raise ValueError(
                f"{SCAN_NETWORKS_ENV}: {item!r} is wider than a "
                f"/{MAX_SCAN_HOSTS_PREFIX} — refusing to sweep "
                f"{network.num_addresses} addresses")
        networks.append(network)
    return tuple(sorted(set(networks), key=str))


def scan_plan(interfaces=None, env=None, sysfs=SYSFS_NET,
              address_lookup=_ipv4_address):
    """Decide what to scan, and record what was skipped and why."""
    env = os.environ if env is None else env

    override = (env.get(SCAN_NETWORKS_ENV) or "").strip()
    if override:
        return ScanPlan(networks=parse_scan_networks(override),
                        origin=SCAN_NETWORKS_ENV)

    if interfaces is None:
        interfaces = read_interface_table(sysfs=sysfs,
                                          address_lookup=address_lookup)
    interfaces = list(interfaces)

    wanted = [name.strip() for name
              in (env.get(SCAN_INTERFACES_ENV) or "").split(",") if name.strip()]
    if wanted:
        known = {interface.name for interface in interfaces}
        unknown = [name for name in wanted if name not in known]
        if unknown:
            raise ValueError(
                f"{SCAN_INTERFACES_ENV} names no such interface: "
                f"{', '.join(unknown)} (this host has: "
                f"{', '.join(sorted(known)) or 'none'})")

    selected, excluded, networks = [], [], []
    for interface in interfaces:
        if wanted and interface.name not in wanted:
            reason = f"not named in {SCAN_INTERFACES_ENV}"
        else:
            reason = interface_exclusion_reason(interface, trusted=bool(wanted))
        if reason is not None:
            excluded.append((interface.name, reason))
            continue
        selected.append(interface)
        networks.append(interface.scan_network())

    return ScanPlan(
        networks=tuple(sorted(set(networks), key=str)),
        selected=tuple(selected),
        excluded=tuple(excluded),
        origin=SCAN_INTERFACES_ENV if wanted else "interfaces",
    )


def local_scan_networks(addresses=None, **planning):
    """Return the private /24s to scan. Never returns a public network.

    ``addresses`` is the injection seam: given a list, this filters it and
    nothing else, which is how the selection rules stay testable without a
    network. Given nothing, it enumerates this host's real interfaces and
    raises NoScannableNetwork rather than answering [] — an empty answer here
    is a fault, not a result.
    """
    if addresses is not None:
        networks = set()
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
            except ValueError:
                continue
            if ip.version == 4 and ip.is_private and not ip.is_loopback:
                networks.add(ipaddress.ip_network(
                    f"{ip}/{MAX_SCAN_HOSTS_PREFIX}", strict=False))
        return sorted(networks, key=str)

    plan = scan_plan(**planning)
    if not plan.networks:
        raise NoScannableNetwork(
            "No scannable local network on this host.\n" + plan.describe())
    return list(plan.networks)


# DroidCam's default listening port. Configurable in the phone app, so it is a
# default here rather than a constant in a format string.
DROIDCAM_PORT = 4747

# How long one probe waits for a phone to say anything at all — accept or
# refuse. This was 0.12 s, and 0.12 s does not find a phone on Wi-Fi.
#
# Measured against the development phone on this LAN, timing how long a
# *definitive* TCP answer took to come back:
#
#     ten probes, one at a time      median 211 ms   max 335 ms   min 2.3 ms
#     six full /24 sweeps            99, 113, 125, 145, 161, 252 ms
#
# One of those sixteen measurements came back inside 120 ms. The phone was
# awake and on the same subnet the whole time; the delay is its Wi-Fi radio
# power-saving, where the access point buffers the frame until the next beacon
# and a first contact costs a beacon interval or several. The 2.3 ms readings
# are the radio still being awake from the probe immediately before.
#
# That is the whole "finds the device, then loses it" symptom: at 0.12 s
# whether discovery sees the phone is close to a coin toss, and which call site
# ran is coincidence. One second is roughly three times the worst measurement,
# and sits just under Linux's 1 s initial SYN retransmit timer, so a probe
# still costs exactly one SYN. Going past 1 s buys a retransmit and doubles the
# floor for dead hosts, and on a /24 nearly every host is a dead host.
DEFAULT_PROBE_TIMEOUT_S = 1.0

# Raised with the timeout, and for its sake. A sweep costs roughly
# ceil(non-answering hosts / workers) x timeout, because a host that is not
# there burns the full timeout and a /24 is almost entirely hosts that are not
# there. At 32 workers a 1 s timeout would take a /24 from ~1 s to ~8 s. These
# threads are blocked in connect(), not working, so widening the batch is cheap
# and keeps the sweep near its single-probe cost.
DEFAULT_SCAN_WORKERS = 128


def scan_droidcam(networks=None, port=DROIDCAM_PORT,
                  timeout_s=DEFAULT_PROBE_TIMEOUT_S,
                  connector=socket.create_connection,
                  max_workers=DEFAULT_SCAN_WORKERS, deadline_s=30.0):
    """Find hosts accepting DroidCam's default TCP port on private local /24s.

    Bounded three ways: only private IPv4 networks, never wider than a /24, and
    never longer than ``deadline_s`` in total. The deadline matters because this
    runs on a GUI worker thread — a connector that blocks past its own timeout
    must not leave the Sources page saying "Scanning..." for the rest of the
    session.
    """
    networks = list(networks if networks is not None else local_scan_networks())
    candidates = []
    for network in networks:
        network = ipaddress.ip_network(network, strict=False)
        if not network.is_private or network.version != 4:
            continue
        # Bound every requested network to a /24 to avoid broad network scans.
        if network.prefixlen < MAX_SCAN_HOSTS_PREFIX:
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

    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        probes = [pool.submit(probe, host) for host in candidates]
        finished, _ = wait(probes, timeout=deadline_s)
        found = [probe_result for probe_result in
                 (future.result() for future in probes if future in finished)
                 if probe_result]
    finally:
        # Never join: an overrunning probe is exactly the case this guards, and
        # its socket already carries its own timeout.
        pool.shutdown(wait=False, cancel_futures=True)
    return sorted(found, key=lambda device: ipaddress.ip_address(device.host))
