"""Webcam helper — the one place that opens a V4L2 capture device.

Device *discovery* is not done here. `acesvision.discovery.discover_webcams()`
is the single source of truth: it reads /sys/class/video4linux, keeps only the
primary capture node of each device (interface index 0, so metadata nodes that
cannot produce frames never appear) and classifies colour / IR / virtual. This
module orders those devices and opens one.

The previous hard-coded CANDIDATES list probed five nodes that do not exist on
this machine, probed a metadata node that cannot capture, and omitted the IR
node entirely — which made the IR camera unreachable and silently blocked the
colour+IR liveness design in docs/VISION_CONTROL_SPEC.md:241-248.

Two gotchas this handles:
  1. The colour node only yields frames with the V4L2 backend + MJPG format
     (its default YUYV mode hands OpenCV nothing), so we force both.
  2. A naive probe can grab the IR node first and you get a greyscale picture.
     open_camera() checks each candidate's colour content and prefers a real
     colour camera, only falling back to greyscale (the IR sensor) if nothing
     else works.

Failure is typed, because "busy" and "missing" need different fixes:
    CameraBusyError      — the device exists but another process holds it.
                           Diagnose with `fuser -v /dev/videoN`.
    CameraNotFoundError  — no usable capture device is present at all.
Both subclass CameraError -> RuntimeError, so existing callers that catch
RuntimeError keep working.

Overrides (env vars):
    FACE_ID_CAM=1          force a specific index or /dev path
    FACE_ID_W / FACE_ID_H  requested capture resolution (default 1280x720)
"""
import errno
import os
from pathlib import Path

import cv2
import numpy as np

PREF_W = int(os.environ.get("FACE_ID_W", "1280"))
PREF_H = int(os.environ.get("FACE_ID_H", "720"))
PREF_FPS = int(os.environ.get("FACE_ID_FPS", "30"))
MANUAL_EXPOSURE = int(os.environ.get("ACESVISION_EXPOSURE", "166"))
CHROMA_MIN = 6.0   # mean channel spread above this == a colour image

DEVICE_AVAILABLE = "available"
DEVICE_BUSY = "busy"
DEVICE_MISSING = "missing"
DEVICE_DENIED = "denied"
DEVICE_UNUSABLE = "unusable"

# Probe order by discovery kind: real colour first, unclassified capture next,
# IR last (it is monochrome, so it is a fallback rather than a default).
_KIND_ORDER = {"colour": 0, "camera": 1, "infrared": 2}


class CameraError(RuntimeError):
    """Base class for camera-open failures."""


class CameraBusyError(CameraError):
    """The device node exists but is held open by another process."""


class CameraNotFoundError(CameraError):
    """No usable capture device is present."""


def _chroma(frame):
    b, g, r = cv2.split(frame.astype("int16"))
    return float((np.abs(b - g) + np.abs(g - r) + np.abs(b - r)).mean())


def device_path(device):
    """Filesystem path for an int index or an absolute path; else None.

    Absolute paths rather than /dev/ only, because discovery hands out stable
    /dev/v4l/by-id symlinks and callers may pass either.
    """
    if isinstance(device, bool):
        return None
    if isinstance(device, int):
        return f"/dev/video{device}"
    if isinstance(device, str) and device.startswith("/"):
        return device
    return None


def device_status(device, opener=os.open, closer=os.close):
    """Cheap pre-open classification: missing, denied, EBUSY, or available.

    Measured on this hardware: a UVC node accepts a *second* open() and only
    refuses at stream setup, so DEVICE_AVAILABLE here does NOT mean free. It
    means "worth trying". Contention is diagnosed after a failed open by
    device_holders(). Some drivers do return EBUSY at open(), so that branch
    still earns its keep.
    """
    path = device_path(device)
    if path is None:
        return DEVICE_AVAILABLE      # unrecognised form; let OpenCV decide
    try:
        handle = opener(path, os.O_RDWR | os.O_NONBLOCK)
    except FileNotFoundError:
        return DEVICE_MISSING
    except PermissionError:
        return DEVICE_DENIED
    except OSError as exc:
        if exc.errno in (errno.EBUSY, errno.EAGAIN):
            return DEVICE_BUSY
        return DEVICE_MISSING
    closer(handle)
    return DEVICE_AVAILABLE


def device_holders(device, proc_root="/proc"):
    """Processes holding this device open, as [(pid, name)], from /proc/*/fd.

    This is what turns "No working camera found" into "OBS is holding
    /dev/video1" — the same answer `fuser -v /dev/video1` gives, without
    shelling out. Processes owned by other users are not readable and are
    reported as absent, so an empty list is weak evidence, never proof of free.
    """
    path = device_path(device)
    if path is None:
        return []
    try:
        target = os.path.realpath(path)
    except OSError:
        return []
    root = Path(proc_root)
    self_pid = str(os.getpid())
    holders = []
    for entry in sorted(root.glob("[0-9]*")):
        if entry.name == self_pid:
            continue
        try:
            descriptors = list((entry / "fd").iterdir())
        except OSError:
            continue                 # gone, or owned by another user
        for descriptor in descriptors:
            try:
                if os.path.realpath(descriptor) != target:
                    continue
                name = (entry / "comm").read_text().strip()
            except OSError:
                continue
            holders.append((int(entry.name), name))
            break
    return holders


def explain_failure(device):
    """Why did this device not give frames? -> (status, human detail)."""
    path = device_path(device) or str(device)
    status = device_status(device)
    if status == DEVICE_MISSING:
        return DEVICE_MISSING, f"{path} does not exist"
    if status == DEVICE_DENIED:
        return DEVICE_DENIED, f"{path} is not readable by this user"
    if status == DEVICE_BUSY:
        return DEVICE_BUSY, f"{path} is in use (EBUSY)"
    holders = device_holders(device)
    if holders:
        who = ", ".join(f"{name} (pid {pid})" for pid, name in holders)
        return DEVICE_BUSY, f"{path} is held by {who}"
    return DEVICE_UNUSABLE, f"{path} opened but produced no frames"


def candidate_devices(devices=None):
    """Ordered capture candidates from discovery — colour, then other, then IR.

    Virtual/loopback nodes are excluded: they are outputs, not cameras.
    """
    if devices is None:
        from acesvision.discovery import discover_webcams
        devices = discover_webcams()
    real = [device for device in devices if device.kind != "virtual"]
    return sorted(real, key=lambda device: (_KIND_ORDER.get(device.kind, 3),
                                            device.index))


def _try_open(device, manual_exposure=False):
    """Open an index/path with V4L2+MJPG, returning (cap, frame) or none."""
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        return None, None
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREF_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREF_H)
    cap.set(cv2.CAP_PROP_FPS, PREF_FPS)
    if manual_exposure:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE)
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    frame = None
    valid_frames = 0
    for _ in range(8):              # let format and exposure controls settle
        ok, frame = cap.read()
        if ok and frame is not None:
            valid_frames += 1
            if valid_frames >= 3:
                return cap, frame
    cap.release()
    return None, None


def _busy_error(details):
    return CameraBusyError(
        "Camera in use by another process — " + "; ".join(details)
        + ". Free it, or choose another device with FACE_ID_CAM=<n>."
    )


def _attempt(device, manual_exposure, opener, status, explain):
    """Try one device. -> (cap, frame) on success, or (None, (status, detail))."""
    if status(device) == DEVICE_MISSING:
        return None, (DEVICE_MISSING, f"{device_path(device) or device} does not exist")
    cap, frame = opener(device, manual_exposure=manual_exposure)
    if cap is not None:
        return cap, frame
    return None, explain(device)


def open_camera(preferred=None, manual_exposure=False, devices=None,
                opener=_try_open, status=device_status, explain=explain_failure):
    """Return an opened cv2.VideoCapture (colour preferred), or raise.

    Raises CameraBusyError when a device exists but another process holds it,
    and CameraNotFoundError when nothing usable is present. The two need
    different fixes, so they are never collapsed into one message.
    """
    env = os.environ.get("FACE_ID_CAM")
    forced = [device for device in (preferred, _parse_env_device(env))
              if device is not None]

    if forced:
        busy, missing = [], []
        for device in forced:
            cap, outcome = _attempt(device, manual_exposure, opener, status, explain)
            if cap is not None:
                frame = outcome
                h, w = frame.shape[:2]
                tag = "colour" if _chroma(frame) > CHROMA_MIN else "greyscale"
                print(f"[camera] forced device {device} ({w}x{h}, {tag})")
                return cap
            outcome_status, detail = outcome
            (busy if outcome_status == DEVICE_BUSY else missing).append(detail)
        if busy:
            # An explicit choice that is merely contended must not be silently
            # replaced by a different camera — the caller asked for that one.
            raise _busy_error(busy)
        # Stale or removed device path: warn loudly, then fall through to
        # discovery rather than failing outright.
        print(f"[camera] WARNING: forced device(s) unusable — {'; '.join(missing)}; "
              f"falling back to discovery")

    candidates = candidate_devices(devices)
    if not candidates:
        raise CameraNotFoundError(
            "No V4L2 capture device found. Check `ls /dev/video*` and that the "
            "camera is connected."
        )

    fallback = None
    busy, unusable = [], []
    for device in candidates:
        cap, outcome = _attempt(device.path, manual_exposure, opener, status,
                                explain)
        if cap is None:
            outcome_status, detail = outcome
            (busy if outcome_status == DEVICE_BUSY else unusable).append(detail)
            continue
        frame = outcome
        h, w = frame.shape[:2]
        if _chroma(frame) > CHROMA_MIN:
            print(f"[camera] using colour {device.path} — {device.name} ({w}x{h})")
            return cap
        if fallback is None:
            fallback = (cap, device, w, h)   # keep one greyscale handle open
        else:
            cap.release()

    if fallback:
        cap, device, w, h = fallback
        why = f" ({'; '.join(busy)})" if busy else ""
        print(f"[camera] WARNING: no colour camera available{why} — falling back "
              f"to {device.kind} {device.path} ({w}x{h}). "
              f"Force a device with FACE_ID_CAM=<n>.")
        return cap

    if busy:
        raise _busy_error(busy)

    detail = f" Tried: {'; '.join(unusable)}." if unusable else ""
    raise CameraNotFoundError(
        f"No working camera found.{detail} Try FACE_ID_CAM=<n> "
        f"(check `ls /dev/video*`)."
    )


def _parse_env_device(value):
    """FACE_ID_CAM accepts an index or a /dev path; anything else is ignored."""
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.startswith("/dev/"):
        return value
    try:
        return int(value)
    except ValueError:
        print(f"[camera] WARNING: ignoring unparsable FACE_ID_CAM={value!r}")
        return None
