"""reolink.py — build stream / snapshot URLs for Reolink IP cameras.

Reolink exposes RTSP (and an HTTP snapshot CGI). OpenCV opens the RTSP URL via
FFMPEG, so a Reolink camera plugs straight into roomview.open_source once the
URL is built. A camera config entry looks like:

    {"name": "Front Door", "type": "reolink",
     "ip": "192.168.1.50", "user": "admin", "password": "secret",
     "stream": "sub", "channel": 1}

    stream:  "sub" (lower-res, lighter on CPU/bandwidth — good for detection)
             or "main" (full-res).
    channel: 1-based (1 for a standalone camera; 1..N behind an NVR).
    port:    RTSP port, default 554.
    path:    optional explicit RTSP path, overriding the built one (some older
             firmware uses 'Preview_01_main' without the 'h264' prefix).

Reolink's RTSP path convention:  h264Preview_<ch:02d>_<main|sub>
"""
from urllib.parse import quote


def rtsp_url(spec):
    """Build the RTSP URL for a Reolink camera spec (dict)."""
    ip = spec["ip"]
    user = quote(str(spec.get("user", "admin")), safe="")
    pwd = quote(str(spec.get("password", "")), safe="")
    port = spec.get("port", 554)
    stream = "main" if str(spec.get("stream", "sub")).lower() == "main" else "sub"
    ch = int(spec.get("channel", 1))
    path = spec.get("path") or f"h264Preview_{ch:02d}_{stream}"
    cred = f"{user}:{pwd}@" if pwd else f"{user}@"
    return f"rtsp://{cred}{ip}:{port}/{path}"


def snapshot_url(spec):
    """Build the HTTP snapshot CGI URL (single JPEG) for a Reolink camera."""
    ip = spec["ip"]
    user = quote(str(spec.get("user", "admin")), safe="")
    pwd = quote(str(spec.get("password", "")), safe="")
    ch = int(spec.get("channel", 1)) - 1  # snapshot CGI is 0-based
    return (f"http://{ip}/cgi-bin/api.cgi?cmd=Snap&channel={ch}"
            f"&rs=watch&user={user}&password={pwd}")
