"""Typed outbound connectors — the action-execution path AcesVision never had.

Until this module existed, ``policy.Decision(outcome="approved")`` was terminal:
``grep -rnE "subprocess|Popen|dbus" acesvision/`` found only the YOLO worker
spawn, so an approved rule reached the end of the pipeline and stopped there.
``CONNECTORS["acergb"]`` (policy.py) was a name in a dict with a risk level.

Two rules shape everything here.

**Typed calls, not shell strings.** The predecessor design shelled out
``ledctl theme next``. On 2026-08-15 ``~/bin/ledctl`` was found to be a stale
build whose usage did not list the ``theme`` subcommands at all: it printed a
usage dump and **exited 0**. Every caller read that as success. A D-Bus method
call cannot fail that way — an unexported member comes back as
``org.freedesktop.DBus.Error.UnknownMethod``, which this module raises as
``MethodFailedError``.

**Errors are loud.** Every failure mode is a distinct exception carrying a
``kind``, and ``ConnectorRegistry.dispatch`` converts it to an
``ExecutionResult(ok=False, error_kind=...)`` that the rule engine folds into
the ``Decision`` and the GUI shows. Nothing is swallowed. In particular
``NextTheme`` returning ``""`` is a *failure* (``no_effect``), not a success:
the daemon returns the empty string when it applied nothing, when the engine is
gone, and when it is not ready yet (acergbd ``control.rs:386,395,500``).

The registry stays open. ``gesture_catalog.CONNECTOR_BINDINGS`` already maps
catalog actions onto ``(connector, action)`` pairs for mpris, pipewire and the
rest; ``policy.CONNECTORS`` declares them all. Only ``acergb`` has an executor
today, and dispatching an unexecuted connector reports ``not_implemented``
loudly rather than doing nothing quietly. Adding mpris/pipewire/kde/
notification/home_assistant means writing a class with ``name``, ``actions()``
and ``execute()`` and registering it — no change to dispatch.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

ACERGB_BUS_NAME = "org.acergb.Daemon"
ACERGB_OBJECT_PATH = "/org/acergb/Daemon"
ACERGB_INTERFACE = "org.acergb.Daemon"


class ConnectorError(Exception):
    """Base for every connector failure. ``kind`` is the machine-readable tag."""

    kind = "connector_error"

    def __init__(self, message, *, connector="", action=""):
        super().__init__(message)
        self.connector = connector
        self.action = action


class TransportUnavailableError(ConnectorError):
    """The D-Bus client library is not installed."""

    kind = "transport_missing"


class BusUnavailableError(ConnectorError):
    """There is no session bus to talk to (no DBUS_SESSION_BUS_ADDRESS)."""

    kind = "bus_absent"


class DaemonUnavailableError(ConnectorError):
    """The bus name is not on the bus — the daemon is not running.

    Distinct from ``DaemonNotReadyError``: absent means nothing owns the name.
    """

    kind = "daemon_absent"


class DaemonNotReadyError(ConnectorError):
    """The daemon owns the name but ``IsReady()`` stayed false.

    acergbd's control plane registers before its engine is usable, so this
    window is real. It is never inferred from an empty ``NextTheme`` string —
    ``""`` legitimately means "no saved themes".
    """

    kind = "not_ready"


class MethodFailedError(ConnectorError):
    """The daemon returned a D-Bus error reply (including UnknownMethod)."""

    kind = "method_error"


class ConnectorTimeoutError(ConnectorError):
    """No reply arrived inside the call deadline."""

    kind = "timeout"


class ActionHadNoEffectError(ConnectorError):
    """The call succeeded at the protocol level but changed nothing."""

    kind = "no_effect"


class UnsupportedActionError(ConnectorError):
    """This connector does not implement the requested action."""

    kind = "unsupported"


class InvalidParameterError(ConnectorError):
    """The action was called with a parameter it cannot accept."""

    kind = "bad_parameter"


#: Every failure tag this module can produce. Kept as data so the GUI and the
#: tests can enumerate them instead of hard-coding strings in three places.
ERROR_KINDS = (
    TransportUnavailableError.kind,
    BusUnavailableError.kind,
    DaemonUnavailableError.kind,
    DaemonNotReadyError.kind,
    MethodFailedError.kind,
    ConnectorTimeoutError.kind,
    ActionHadNoEffectError.kind,
    UnsupportedActionError.kind,
    InvalidParameterError.kind,
    "not_implemented",
    "unexpected",
)


@dataclass(frozen=True)
class ExecutionResult:
    """What actually happened when a connector was asked to act."""

    connector: str
    action: str
    ok: bool
    detail: str
    error_kind: Optional[str] = None

    def summary(self):
        verdict = "executed" if self.ok else f"failed ({self.error_kind})"
        return f"{self.connector}.{self.action} {verdict}: {self.detail}"


# --------------------------------------------------------------------------
# Transport
# --------------------------------------------------------------------------
#
# jeepney is the D-Bus client. It is pure Python with zero dependencies, needs
# no compiled extension, no libdbus, no glib main loop and — the deciding
# point — no Qt, so the headless runner stays importable on a machine with no
# GUI stack. PySide6.QtDBus would have worked for the GUI process and only the
# GUI process; dbus-python needs a C build against system libdbus; dbus-next is
# asyncio-first and this call site is synchronous. See the report for the full
# comparison.


class JeepneySession:
    """One open connection to the session bus, mapping D-Bus errors to ours."""

    _ABSENT = {
        "org.freedesktop.DBus.Error.ServiceUnknown",
        "org.freedesktop.DBus.Error.NameHasNoOwner",
    }
    _TIMEOUT = {
        "org.freedesktop.DBus.Error.NoReply",
        "org.freedesktop.DBus.Error.Timeout",
        "org.freedesktop.DBus.Error.TimedOut",
    }

    def __init__(self, connection, address, timeout_s, connector=""):
        self._connection = connection
        self._address = address
        self._timeout_s = float(timeout_s)
        self._connector = connector

    def call(self, member, signature="", body=(), action=""):
        from jeepney import MessageType, new_method_call

        message = new_method_call(self._address, member, signature, tuple(body))
        try:
            reply = self._connection.send_and_get_reply(
                message, timeout=self._timeout_s)
        except TimeoutError as exc:
            raise ConnectorTimeoutError(
                f"{member} did not answer within {self._timeout_s:g}s",
                connector=self._connector, action=action) from exc
        if reply.header.message_type is not MessageType.error:
            return tuple(reply.body)

        name = reply.header.fields.get(4, "org.freedesktop.DBus.Error.Failed")
        detail = reply.body[0] if reply.body else ""
        if name in self._ABSENT:
            raise DaemonUnavailableError(
                f"{self._address.bus_name} is not on the session bus "
                f"({name}): {detail}",
                connector=self._connector, action=action)
        if name in self._TIMEOUT:
            raise ConnectorTimeoutError(
                f"{member} timed out ({name}): {detail}",
                connector=self._connector, action=action)
        raise MethodFailedError(
            f"{member} failed ({name}): {detail}",
            connector=self._connector, action=action)

    def close(self):
        try:
            self._connection.close()
        except Exception:       # noqa: BLE001 - closing must never mask a result
            pass


class JeepneyTransport:
    """Opens one short-lived session-bus connection per action.

    Per-action rather than cached on purpose: acergbd is restarted often enough
    (systemd user unit, and it drops its name entirely when it dies) that a
    cached socket would hand back a stale-connection error whose text says
    nothing useful about the daemon. Reconnecting costs a local auth handshake.
    """

    def __init__(self, bus_name, object_path, interface, bus="SESSION",
                 timeout_s=5.0, connector=""):
        self.bus_name = bus_name
        self.object_path = object_path
        self.interface = interface
        self.bus = bus
        self.timeout_s = float(timeout_s)
        self.connector = connector

    def open(self, action=""):
        try:
            from jeepney import DBusAddress
            from jeepney.io.blocking import open_dbus_connection
        except ImportError as exc:
            raise TransportUnavailableError(
                "jeepney is not installed — no D-Bus client available "
                "(pip install jeepney)",
                connector=self.connector, action=action) from exc

        address = DBusAddress(self.object_path, bus_name=self.bus_name,
                              interface=self.interface)
        try:
            connection = open_dbus_connection(bus=self.bus)
        except KeyError as exc:
            raise BusUnavailableError(
                f"no {self.bus.lower()} bus address in the environment ({exc})",
                connector=self.connector, action=action) from exc
        except (OSError, ValueError) as exc:
            raise BusUnavailableError(
                f"cannot reach the {self.bus.lower()} bus: {exc}",
                connector=self.connector, action=action) from exc
        return JeepneySession(connection, address, self.timeout_s,
                              connector=self.connector)


# --------------------------------------------------------------------------
# Connectors
# --------------------------------------------------------------------------


class OverlayConnector:
    """GUI-local presentation actions.

    The callback is intentionally supplied by the GUI rather than importing
    Qt here. That keeps the headless runner free of GUI imports; a saved overlay
    rule run headlessly is reported as unwired rather than silently affecting
    an unrelated process.
    """

    name = "overlay"

    def __init__(self, toggle_clean):
        self._toggle_clean = toggle_clean

    @staticmethod
    def actions():
        return ("toggle_clean",)

    def execute(self, action, params=None):
        if action not in self.actions():
            raise UnsupportedActionError(
                f"overlay cannot do {action!r}. Actions: "
                + ", ".join(self.actions()),
                connector=self.name, action=action)
        self._toggle_clean()
        return "toggled clean camera overlay"


class AceRgbConnector:
    """Dispatches ``CONNECTORS["acergb"]`` actions as typed D-Bus calls.

    Targets ``org.acergb.Daemon`` at ``/org/acergb/Daemon`` on the session bus.
    Readiness is *polled* via ``IsReady() -> b``, never inferred from the
    ``NextTheme`` return string.
    """

    name = "acergb"

    def __init__(self, transport=None, ready_timeout_s=2.0, call_timeout_s=5.0,
                 poll_interval_s=0.05, clock=time.monotonic, sleep=time.sleep):
        self.transport = transport or JeepneyTransport(
            ACERGB_BUS_NAME, ACERGB_OBJECT_PATH, ACERGB_INTERFACE,
            timeout_s=call_timeout_s, connector=self.name)
        self.ready_timeout_s = float(ready_timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self._clock = clock
        self._sleep = sleep

    @staticmethod
    def actions():
        return ("next_theme", "off", "brightness")

    def execute(self, action, params=None):
        """Run one action. Returns a human detail string, or raises ConnectorError."""
        params = dict(params or {})
        if action not in self.actions():
            raise UnsupportedActionError(
                f"acergb cannot do {action!r}. Actions: "
                + ", ".join(self.actions()),
                connector=self.name, action=action)
        session = self.transport.open(action=action)
        try:
            self._await_ready(session, action)
            if action == "next_theme":
                return self._next_theme(session, action)
            if action == "off":
                return self._off(session, action)
            return self._brightness(session, action, params)
        finally:
            session.close()

    def _await_ready(self, session, action):
        """Poll ``IsReady()`` until true or the deadline passes.

        A missing daemon surfaces as DaemonUnavailableError from the very first
        call — the bus name is simply absent — so "absent" and "not ready" are
        never conflated.
        """
        deadline = self._clock() + self.ready_timeout_s
        while True:
            body = session.call("IsReady", action=action)
            if body and bool(body[0]):
                return
            if self._clock() >= deadline:
                raise DaemonNotReadyError(
                    f"{ACERGB_BUS_NAME} is on the bus but IsReady() stayed "
                    f"false for {self.ready_timeout_s:g}s",
                    connector=self.name, action=action)
            self._sleep(self.poll_interval_s)

    def _next_theme(self, session, action):
        body = session.call("NextTheme", action=action)
        applied = str(body[0]) if body else ""
        if not applied:
            raise ActionHadNoEffectError(
                "NextTheme applied nothing (empty reply) — no saved themes, or "
                "the engine did not answer. The daemon is ready, so this is "
                "not a readiness problem.",
                connector=self.name, action=action)
        return f"applied theme {applied!r}"

    def _off(self, session, action):
        session.call("Off", action=action)
        # Off() -> void: acceptance by the daemon is all the protocol confirms.
        # Read the theme back so the detail carries observed state rather than
        # an assumption. A failure here is still a real failure and propagates.
        body = session.call("CurrentTheme", action=action)
        current = str(body[0]) if body else ""
        return f"Off() accepted (CurrentTheme now {current!r})"

    def _brightness(self, session, action, params):
        raw = params.get("level", params.get("brightness"))
        try:
            level = float(raw)
        except (TypeError, ValueError) as exc:
            raise InvalidParameterError(
                f"brightness needs a numeric 'level' in 0.0..1.0, got {raw!r}",
                connector=self.name, action=action) from exc
        if not 0.0 <= level <= 1.0:
            raise InvalidParameterError(
                f"brightness level must be within 0.0..1.0, got {level!r}",
                connector=self.name, action=action)
        session.call("SetBrightness", "d", (level,), action=action)
        return f"SetBrightness({level:g}) accepted"

    def current_theme(self):
        """Read ``CurrentTheme()`` — used for before/after proof, not by rules."""
        session = self.transport.open(action="current_theme")
        try:
            body = session.call("CurrentTheme", action="current_theme")
            return str(body[0]) if body else ""
        finally:
            session.close()


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


class ConnectorRegistry:
    """Name -> connector. Dispatch never raises and never returns silence."""

    def __init__(self, connectors=()):
        self._connectors = {}
        for connector in connectors:
            self.register(connector)

    def register(self, connector):
        self._connectors[connector.name] = connector
        return self

    def names(self):
        return sorted(self._connectors)

    def get(self, name):
        return self._connectors.get(name)

    def supports(self, connector, action):
        target = self._connectors.get(connector)
        return target is not None and action in target.actions()

    def dispatch(self, connector, action, params=None):
        target = self._connectors.get(connector)
        if target is None:
            wired = ", ".join(self.names()) or "(none)"
            return ExecutionResult(
                connector, action, False,
                f"no executor is registered for {connector}.{action}. "
                f"Wired connectors: {wired}",
                error_kind="not_implemented")
        try:
            detail = target.execute(action, params)
        except ConnectorError as exc:
            return ExecutionResult(connector, action, False, str(exc),
                                   error_kind=exc.kind)
        except Exception as exc:            # noqa: BLE001 - never swallow
            return ExecutionResult(
                connector, action, False,
                f"{type(exc).__name__}: {exc}", error_kind="unexpected")
        return ExecutionResult(connector, action, True, str(detail))


def default_registry():
    """The connectors that have a real executor today. Only acergb."""
    return ConnectorRegistry([AceRgbConnector()])
