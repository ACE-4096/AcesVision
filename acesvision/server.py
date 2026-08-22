"""The local HTTP surface: preview frames, and the event API subscribers use.

This started as a browser preview — a page, a JPEG, and a status blob, served on
loopback with no authentication at all. It is now also the emitter's public
contract, so the security posture had to become deliberate rather than
incidental:

* **Loopback by default.** Binding anywhere else takes ``--allow-remote`` and
  forces identity publishing off (``PublishFilter.without_identity``).
* **A token on everything but ``/api/health``.** Including ``/`` and
  ``/latest.jpg``, which served live frames of whoever is in front of the camera
  to any process on the machine that could guess a port. That was already a
  hole; adding an event stream beside it would have made it a bigger one.
* **No ``Origin``, ever.** A browser attaches ``Origin`` to cross-site requests.
  A legitimate subscriber has no reason to send one, so its presence is
  sufficient to refuse — no allowlist to get wrong.
* **``Host`` must be loopback.** DNS rebinding gets a page on an attacker's
  origin to resolve a name to 127.0.0.1 and talk to us. The socket cannot tell;
  the ``Host`` header can, because it still carries the attacker's name.
* **Bounded subscribers.** ``EventBus`` caps concurrency; beyond it, 503.

The security decisions are module-level functions, not handler methods, so each
one is testable on its own without standing a socket up.

Endpoints
---------
``GET /api/health``          ``{"ok":true,"schema":...}``. The only open one.
``GET /api/catalog``         The gesture vocabulary and its sha256.
``GET /api/events``          Server-Sent Events. Replay via ``Last-Event-ID``/``?since``.
``GET /api/events/recent``   Polling fallback: ``{"events":[...],"seq":N}``.
``GET /api/state``           Pipeline status (pre-existing).
``GET /``, ``GET /latest.jpg``  The browser preview (pre-existing).
"""
from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from .emitter import (
    SCHEMA_GESTURE,
    TooManySubscribers,
    harden_directory,
    read_secret_file,
    write_secret_file,
)

log = logging.getLogger(__name__)

#: Seconds of silence after which the stream emits a ``: keepalive`` comment.
#: Without it an idle emitter and a dead one look identical to a subscriber, and
#: the difference matters: one means "nobody has gestured", the other means
#: "you are not going to hear about it when they do".
KEEPALIVE_S = 15.0

#: How often the stream wakes to check for shutdown. Only relevant when there is
#: nothing to send; a queued event wakes the reader immediately.
POLL_S = 0.5

DEFAULT_TOKEN_PATH = Path.home() / ".config" / "acesvision" / "emitter.token"

#: 32 bytes of entropy, URL-safe so it survives being pasted into a query string.
TOKEN_BYTES = 32

LOOPBACK_NAMES = frozenset({"localhost", "localhost.localdomain"})

PAGE = b"""<!doctype html><html><head><meta charset=utf-8>
<title>AcesVision Preview</title>
<style>body{margin:0;background:#101216;color:#eee;font:14px sans-serif}
main{max-width:1100px;margin:auto;padding:16px}img{width:100%;background:#000}
#state{padding:8px 0;color:#aaa}</style></head><body><main>
<h1>AcesVision</h1><div id=state>connecting</div><img id=feed>
<script>const image=document.getElementById('feed'),state=document.getElementById('state');
// The token stays in the URL the operator opened and is appended to each
// request. The page is never handed a token it was not already given.
const token=new URLSearchParams(location.search).get('token')||'';
const auth=path=>path+(path.includes('?')?'&':'?')+'token='+encodeURIComponent(token);
setInterval(()=>{image.src=auth('/latest.jpg?t='+Date.now())},100);
setInterval(async()=>{try{const r=await fetch(auth('/api/state'));const s=await r.json();
state.textContent=s.status+' | '+s.source+' | frame '+s.sequence}catch(e){}},500);
</script></main></body></html>"""


class BindRefused(ValueError):
    """A non-loopback bind was requested without ``--allow-remote``."""


def is_loopback(host: str) -> bool:
    """Is this host string a loopback address or the loopback name?"""
    host = (host or "").strip().strip("[]")
    if not host:
        return False
    if host.lower() in LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


#: Wildcard binds. Refused even with ``--allow-remote`` — see ``resolve_bind``.
WILDCARD_BINDS = frozenset({"0.0.0.0", "::", "*"})


def resolve_bind(bind: str, allow_remote: bool) -> str:
    """The address to bind, or ``BindRefused``.

    Reaching the network is an explicit act. The flag is not a convenience for
    silencing an error — it is the record that somebody decided a camera event
    stream should leave this machine.

    A **wildcard** bind is refused even with the flag. The ``Host`` allowlist is
    built from the address we bound, and a wildcard names no address, so the
    rebinding defence would have nothing to check against. The choice would be
    between answering only to ``Host: 0.0.0.0`` — which no client ever sends, so
    every request 403s and the flag silently does nothing — or dropping the
    check, which trades a working flag for a real defence. Naming the interface
    is the option that leaves both intact.
    """
    bind = (bind or "127.0.0.1").strip()
    if is_loopback(bind):
        return bind
    if bind in WILDCARD_BINDS:
        raise BindRefused(
            f"refusing to bind the wildcard address {bind}: name the interface "
            "you mean (for example --bind 192.168.1.20). The Host header "
            "allowlist that defeats DNS rebinding is built from the bound "
            "address, and a wildcard names none.")
    if allow_remote:
        return bind
    raise BindRefused(
        f"refusing to bind {bind}: it is not a loopback address. Pass "
        "--allow-remote to publish beyond this machine (identity publishing "
        "is forced off when you do).")


def host_header_ok(host_header: str, allowed_hosts) -> bool:
    """Does the ``Host`` header name something we agreed to answer to?

    The defence against DNS rebinding. The kernel only knows the connection
    arrived on 127.0.0.1; it cannot know the browser thinks it is talking to
    ``evil.example``. The ``Host`` header does know, so a name we never bound is
    refused. A missing header is refused too — fail closed; every HTTP client
    written this century sends one.
    """
    if not host_header:
        return False
    hostname = host_header.rsplit(":", 1)[0] if not host_header.startswith("[") \
        else host_header.split("]")[0] + "]"
    hostname = hostname.strip().strip("[]").lower()
    if not hostname:
        return False
    if is_loopback(hostname):
        return True
    return hostname in {str(allowed).lower() for allowed in allowed_hosts}


def presented_token(headers, query) -> str:
    """The token the caller offered: ``Authorization: Bearer`` or ``?token=``.

    The header is preferred. A query token is supported because a browser cannot
    set headers on ``<img src>`` or ``EventSource``, and the preview page needs
    both — but a URL leaks into shell history and referrers, so the header is
    the documented way for a real subscriber.
    """
    authorization = (headers.get("Authorization") or "").strip()
    if authorization:
        scheme, _, value = authorization.partition(" ")
        if scheme.lower() == "bearer" and value.strip():
            return value.strip()
        return ""
    values = query.get("token") or []
    return values[0] if values else ""


def token_matches(presented: str, expected: str) -> bool:
    """Constant-time comparison.

    ``==`` on a secret returns as soon as two bytes differ, so how long it took
    tells the caller how much of the prefix was right. ``compare_digest`` does
    not leak that. The length check first is safe: the token length is fixed and
    public.
    """
    if not presented or not expected:
        return False
    return hmac.compare_digest(presented, expected)


def load_or_create_token(path=None) -> tuple[str, bool]:
    """The emitter token, generating it on first run. Returns ``(token, created)``."""
    path = Path(path or DEFAULT_TOKEN_PATH)
    if path.exists():
        # Tighten the directory on the read path too. A token written before
        # this check existed sits in whatever directory the rule store made.
        harden_directory(path.parent)
        token = read_secret_file(path)
        if token:
            return token, False
        path.unlink()
    token = secrets.token_urlsafe(TOKEN_BYTES)
    write_secret_file(path, token + "\n")
    return token, True


def sse_frame(event: str, data, event_id=None) -> bytes:
    """One Server-Sent Events frame.

    ``data`` is split across ``data:`` lines because a bare newline inside a
    value would otherwise terminate the frame early. JSON with the compact
    separators has no newlines, but a serialiser that gains an ``indent`` should
    not silently corrupt the stream.
    """
    if not isinstance(data, str):
        data = json.dumps(data, separators=(",", ":"), sort_keys=True)
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    lines.extend(f"data: {line}" for line in data.split("\n"))
    return ("\n".join(lines) + "\n\n").encode("utf-8")


def keepalive_frame() -> bytes:
    """An SSE comment. Ignored by every client, and proof the socket is alive."""
    return b": keepalive\n\n"


def parse_since(query, headers):
    """The replay point: ``(since, error)``.

    ``Last-Event-ID`` wins over ``?since``. On an ``EventSource`` reconnect the
    browser re-requests the original URL *and* sends the header, so honouring
    the query string would replay from the connection's original point every
    time and duplicate everything since.

    A malformed ``?since`` is a 400 — the caller typed it. A malformed
    ``Last-Event-ID`` is ignored — the caller did not.
    """
    header = (headers.get("Last-Event-ID") or "").strip()
    if header:
        try:
            return max(0, int(header)), None
        except ValueError:
            log.debug("ignoring unparsable Last-Event-ID %r", header)
    values = query.get("since") or []
    if not values:
        return None, None
    try:
        return max(0, int(values[0])), None
    except ValueError:
        return None, f"since must be an integer, got {values[0]!r}"


class VisionServer:
    """The threaded HTTP server, its token, and the routing.

    ``emitter`` and ``bus`` are optional so the preview can still be stood up
    without an event surface; when they are absent the ``/api/events*`` and
    ``/api/catalog`` routes answer 404 rather than pretending.
    """

    def __init__(self, latest_output, pipeline, host="127.0.0.1", port=8765, *,
                 bus=None, emitter=None, token=None, allowed_hosts=(),
                 keepalive_s: float = KEEPALIVE_S, poll_s: float = POLL_S):
        self.latest_output = latest_output
        self.pipeline = pipeline
        self.host = host
        self.port = port
        self.bus = bus
        self.emitter = emitter
        self.token = token
        self.allowed_hosts = frozenset(allowed_hosts) | {host}
        self.keepalive_s = keepalive_s
        self.poll_s = poll_s
        self._server = None
        self._thread = None
        self._stopping = threading.Event()

    # -- routing ----------------------------------------------------------

    def url(self, path="/") -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        base = f"http://{host}:{self.bound_port}{path}"
        if not self.token:
            return base
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}token={self.token}"

    @property
    def bound_port(self) -> int:
        """The port actually in use — resolves ``port=0`` after ``start()``."""
        if self._server is not None:
            return self._server.server_address[1]
        return self.port

    def start(self):
        self._stopping.clear()
        self._server = ThreadingHTTPServer((self.host, self.port),
                                           self._handler_class())
        self.port = self._server.server_address[1]
        self.allowed_hosts = frozenset(self.allowed_hosts) | {self.host}
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        daemon=True, name="vision-server")
        self._thread.start()

    def stop(self):
        self._stopping.set()
        if self.bus is not None:
            # Wake every streaming handler instead of leaving it parked on its
            # keepalive timeout while the process tries to exit.
            self.bus.close()
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _handler_class(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "AcesVision"

            def log_message(self, *_):
                pass

            def do_GET(self):
                split = urlsplit(self.path)
                path, query = split.path, parse_qs(split.query)

                # Order matters. Rebinding and cross-origin checks come before
                # authentication so that a hostile page cannot use response
                # timing or status to probe whether its guessed token was right.
                if self.headers.get("Origin") is not None:
                    self._reply_json(403, {"error": "cross-origin requests are refused"})
                    return
                if not host_header_ok(self.headers.get("Host", ""),
                                      outer.allowed_hosts):
                    self._reply_json(403, {"error": "unrecognised Host header"})
                    return

                if path == "/api/health":
                    # Deliberately says nothing else. A probe that can reach the
                    # port learns the emitter is up and which schema it speaks,
                    # and nothing about the camera, the machine, or the people.
                    self._reply_json(200, {"ok": True, "schema": SCHEMA_GESTURE})
                    return

                if not outer._authorised(self.headers, query):
                    self._reply_json(401, {"error": "missing or invalid token"},
                                     {"WWW-Authenticate": "Bearer"})
                    return

                if path == "/" or path.startswith("/index.html"):
                    self._reply(200, "text/html; charset=utf-8", PAGE)
                    return
                if path == "/latest.jpg":
                    _, jpeg = outer.latest_output.snapshot()
                    if not jpeg:
                        self._reply(503, "text/plain", b"camera unavailable")
                    else:
                        self._reply(200, "image/jpeg", jpeg,
                                    {"Cache-Control": "no-store"})
                    return
                if path == "/api/state":
                    state = outer.pipeline.state()
                    scene, _ = outer.latest_output.snapshot()
                    self._reply_json(200, {
                        "status": state.status,
                        "source": state.source.safe_label(),
                        "sequence": state.sequence,
                        "last_error": state.last_error,
                        "metrics": state.metrics,
                        "scene_counts": {
                            "objects": len(scene.objects) if scene else 0,
                            "faces": len(scene.faces) if scene else 0,
                            "gestures": len(scene.gestures) if scene else 0,
                        },
                    })
                    return
                if path == "/api/catalog":
                    if outer.emitter is None:
                        self._reply_json(404, {"error": "no emitter"})
                    else:
                        self._reply_json(200, outer.emitter.catalog.as_payload())
                    return
                if path == "/api/events/recent":
                    outer._recent(self, query)
                    return
                if path == "/api/events":
                    outer._stream(self, query)
                    return
                self._reply_json(404, {"error": "not found"})

            def _reply(self, status, content_type, body, headers=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(body)

            def _reply_json(self, status, payload, headers=None):
                body = json.dumps(payload).encode()
                merged = {"Cache-Control": "no-store"}
                merged.update(headers or {})
                self._reply(status, "application/json", body, merged)

        return Handler

    # -- behaviour --------------------------------------------------------

    def _authorised(self, headers, query) -> bool:
        if not self.token:
            return True
        return token_matches(presented_token(headers, query), self.token)

    def _recent(self, handler, query):
        """The polling fallback, for a subscriber that cannot hold a stream open."""
        if self.bus is None:
            handler._reply_json(404, {"error": "no emitter"})
            return
        since, error = parse_since(query, handler.headers)
        if error:
            handler._reply_json(400, {"error": error})
            return
        # `since` absent means "just tell me where you are" — a first poll gets
        # the sequence to resume from, not a ring dump it has no context for.
        events = self.bus.replay_since(since) if since is not None else []
        handler._reply_json(200, {
            "events": events,
            "seq": self.bus.seq,
            "oldest_available": self.bus.oldest_seq(),
        })

    def _stream(self, handler, query):
        """Server-Sent Events: hello, then replay, then live, then keepalives."""
        if self.bus is None or self.emitter is None:
            handler._reply_json(404, {"error": "no emitter"})
            return
        since, error = parse_since(query, handler.headers)
        if error:
            handler._reply_json(400, {"error": error})
            return

        try:
            subscription, replay, seq = self.bus.subscribe(since=since)
        except TooManySubscribers as exc:
            handler._reply_json(503, {"error": str(exc)},
                                {"Retry-After": "5"})
            return

        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Cache-Control", "no-store")
        handler.send_header("Connection", "close")
        # Belt and braces for anyone who puts a reverse proxy in front of this.
        handler.send_header("X-Accel-Buffering", "no")
        handler.end_headers()

        try:
            # hello carries `id:` = the sequence the subscriber is caught up to,
            # so a browser EventSource that drops before the first real event
            # still reconnects from the right place instead of from zero.
            hello_id = since if since is not None else seq
            handler.wfile.write(sse_frame(
                "hello",
                self.emitter.hello(seq, requested_since=since,
                                   oldest_seq=self.bus.oldest_seq()),
                event_id=hello_id))
            for event in replay:
                handler.wfile.write(sse_frame(event["type"], event,
                                              event_id=event["seq"]))
            self._pump(handler, subscription)
        except (BrokenPipeError, ConnectionResetError):
            pass                      # the subscriber hung up; entirely normal
        finally:
            self.bus.unsubscribe(subscription)

    def _pump(self, handler, subscription):
        last_write = time.monotonic()
        while not self._stopping.is_set() and not subscription.closed:
            event = subscription.get(timeout=self.poll_s)
            if event is not None:
                handler.wfile.write(sse_frame(event["type"], event,
                                              event_id=event["seq"]))
                last_write = time.monotonic()
                continue
            if subscription.closed:
                break
            if time.monotonic() - last_write >= self.keepalive_s:
                handler.wfile.write(keepalive_frame())
                last_write = time.monotonic()
