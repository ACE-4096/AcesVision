"""Typed gesture rules and deny-by-default action policy.

Rule identifiers are validated against a shared vocabulary at entry. They used
to be free text matched exactly (gui.py addRule -> RuleEngine._matches), which
made every mistyped rule a permanent silent no-op: "open_palm" never equals the
emitted "Open_Palm", "sampleperson" never equals the enrolled "SamplePerson".

``dry_run`` is a **per-rule** field defaulting to True. It used to be a global:
hardcoded ``True`` at the RuleEngine construction site and written
unconditionally into every saved file by ``RuleStore.save``, so there was no way
to arm one convenience rule without arming all of them — including any
``sensitive`` rule someone added later. A rule loaded from a file written before
this change has no ``dry_run`` key and therefore defaults to True: existing
rules stay dry-run.

An approved rule now reaches a connector (see ``connectors.py``). Whether the
action actually happened is in the ``Decision`` — outcome ``executed`` or
``failed``, with ``error_kind`` naming the failure mode.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from gesture_catalog import ANY_ACTOR, GESTURE_IDS, require_gesture

RISK_CONVENIENCE = "convenience"
RISK_PERSONAL = "personal"
RISK_SENSITIVE = "sensitive"

KNOWN_FACES_DIR = Path(__file__).resolve().parents[1] / "known_faces"

CONNECTORS = {
    # Local presentation only: clean keeps the camera stream running but hides
    # every rendered box, label and gesture annotation. It is deliberately
    # convenience class because a second gesture restores the previous view.
    "overlay": {"toggle_clean": RISK_CONVENIENCE},
    "acergb": {
        "next_theme": RISK_CONVENIENCE,
        "off": RISK_CONVENIENCE,
        "brightness": RISK_CONVENIENCE,
    },
    "mpris": {
        "play_pause": RISK_CONVENIENCE,
        "next": RISK_CONVENIENCE,
        "previous": RISK_CONVENIENCE,
    },
    "pipewire": {
        "volume_up": RISK_CONVENIENCE,
        "volume_down": RISK_CONVENIENCE,
        "mute": RISK_CONVENIENCE,
    },
    "kde": {"next_desktop": RISK_PERSONAL},
    "notification": {"show": RISK_PERSONAL},
    "home_assistant": {"webhook": RISK_PERSONAL},
}


@dataclass(frozen=True)
class Rule:
    id: str
    gesture: str
    connector: str
    action: str
    actor: str = "*"
    source: str = "*"
    enabled: bool = True
    require_liveness: bool = False
    require_confirmation: bool = False
    # Per-rule, and True unless the operator says otherwise. Arming is a
    # deliberate per-rule act, never a side effect of arming something else.
    dry_run: bool = True

    @classmethod
    def create(cls, gesture, connector, action, actor="*", source="*",
               dry_run=True):
        validate_action(connector, action)
        risk = CONNECTORS[connector][action]
        return cls(
            id=str(uuid.uuid4()),
            gesture=validate_gesture(gesture),
            connector=connector,
            action=action,
            actor=validate_actor(actor),
            source=source or "*",
            require_liveness=risk == RISK_SENSITIVE,
            require_confirmation=risk == RISK_SENSITIVE,
            dry_run=bool(dry_run),
        )

    @property
    def risk(self):
        return CONNECTORS[self.connector][self.action]


#: Terminal outcomes a Decision can carry.
#: ``blocked``  - policy refused it.
#: ``dry_run``  - the rule is not armed; nothing was attempted.
#: ``approved`` - policy allowed it and no executor was wired (the old
#:                terminal state; now only reachable with executor=None).
#: ``executed`` - a connector ran it and the daemon confirmed.
#: ``failed``   - a connector tried and the attempt failed. Loud.
OUTCOME_BLOCKED = "blocked"
OUTCOME_DRY_RUN = "dry_run"
OUTCOME_APPROVED = "approved"
OUTCOME_EXECUTED = "executed"
OUTCOME_FAILED = "failed"


@dataclass(frozen=True)
class Decision:
    rule_id: str
    outcome: str
    reason: str
    connector: str
    action: str
    risk: str
    #: Set only when outcome == "failed" — the connectors.ERROR_KINDS tag.
    error_kind: str = ""

    @property
    def ok(self):
        """Did the world actually change? Only ``executed`` says yes."""
        return self.outcome == OUTCOME_EXECUTED

    def describe(self):
        head = f"{self.outcome}: {self.connector}.{self.action}"
        if self.error_kind:
            head += f" [{self.error_kind}]"
        return f"{head}, {self.reason}"


def validate_action(connector, action):
    if connector not in CONNECTORS:
        raise ValueError(f"unsupported connector: {connector}")
    if action not in CONNECTORS[connector]:
        raise ValueError(f"unsupported action for {connector}: {action}")


def validate_gesture(gesture):
    """Canonical gesture id from the shared catalog, or ValueError.

    Normalises case and separators, so "open_palm" resolves to the emitted
    "Open_Palm" instead of becoming an unfireable rule.
    """
    return require_gesture(str(gesture).strip())


def known_actors(root=None):
    """Enrolled identities — the directory names under known_faces/."""
    root = Path(root or KNOWN_FACES_DIR)
    if not root.is_dir():
        return []
    return sorted(entry.name for entry in root.iterdir() if entry.is_dir())


def validate_actor(actor, actors=None):
    """Canonical enrolled name, or '*' for anyone. Raises on an unknown actor.

    Matching is case-insensitive so "sampleperson" resolves to the enrolled "SamplePerson"
    rather than matching nothing forever.
    """
    text = str(actor or "").strip()
    if not text or text == ANY_ACTOR:
        return ANY_ACTOR
    enrolled = list(actors if actors is not None else known_actors())
    for name in enrolled:
        if name.lower() == text.lower():
            return name
    raise ValueError(
        f"unknown actor: {actor!r}. Enrolled: "
        + (", ".join(enrolled) if enrolled else "(nobody)")
        + f". Use '{ANY_ACTOR}' for anyone."
    )


def normalise_rule(rule, actors=None):
    """Return the rule with canonical gesture/actor, or raise ValueError.

    Used when loading persisted rules so pre-validation files are repaired
    rather than silently kept in an unfireable state.
    """
    validate_action(rule.connector, rule.action)
    return replace(rule,
                   gesture=validate_gesture(rule.gesture),
                   actor=validate_actor(rule.actor, actors=actors))


class RuleEngine:
    """Evaluates rules against gesture events and dispatches the survivors.

    There is deliberately no engine-wide ``dry_run``. It used to take one and
    every construction site passed True, which meant the only way to arm a rule
    was to arm every rule. Arming is now ``Rule.dry_run=False`` on the one rule
    you mean.

    ``executor`` is anything exposing ``dispatch(connector, action, params) ->
    ExecutionResult`` — normally ``connectors.default_registry()``. With no
    executor an allowed rule stops at ``approved``, which is exactly the
    behaviour that existed before connectors did.
    """

    def __init__(self, rules=None, executor=None):
        self.rules = list(rules or [])
        self.executor = executor

    def evaluate(self, event):
        return [self._evaluate(rule, event) for rule in self.rules
                if self._matches(rule, event)]

    @staticmethod
    def _matches(rule, event):
        if not rule.enabled or rule.gesture != event.get("gesture"):
            return False
        if rule.actor != "*" and rule.actor != event.get("actor"):
            return False
        if rule.source != "*" and rule.source != event.get("source"):
            return False
        return True

    def _evaluate(self, rule, event):
        if rule.risk in {RISK_PERSONAL, RISK_SENSITIVE} and not event.get("actor"):
            return self._decision(rule, OUTCOME_BLOCKED, "actor is not associated")
        if rule.require_liveness and event.get("liveness_state") != "live":
            return self._decision(rule, OUTCOME_BLOCKED,
                                  "live presentation not verified")
        if rule.risk == RISK_SENSITIVE and not event.get("security_authorized", False):
            return self._decision(rule, OUTCOME_BLOCKED,
                                  "security authorization unavailable")
        if rule.require_confirmation and not event.get("confirmed", False):
            return self._decision(rule, OUTCOME_BLOCKED,
                                  "confirmation or passkey required")
        if rule.dry_run:
            return self._decision(rule, OUTCOME_DRY_RUN, "would execute")
        if self.executor is None:
            return self._decision(rule, OUTCOME_APPROVED,
                                  "ready for typed connector dispatch")
        result = self.executor.dispatch(rule.connector, rule.action,
                                        self._params(rule, event))
        if result.ok:
            return self._decision(rule, OUTCOME_EXECUTED, result.detail)
        return self._decision(rule, OUTCOME_FAILED, result.detail,
                              error_kind=result.error_kind or "unexpected")

    @staticmethod
    def _params(rule, event):
        """Action parameters. Rules carry none yet; the event may supply some."""
        params = event.get("params")
        return dict(params) if isinstance(params, dict) else {}

    @staticmethod
    def _decision(rule, outcome, reason, error_kind=""):
        return Decision(rule.id, outcome, reason, rule.connector,
                        rule.action, rule.risk, error_kind)


class RuleStore:
    def __init__(self, path=None):
        self.path = Path(path or Path.home() / ".config" / "acesvision" / "rules.json")
        self.rejected = []

    def load(self, strict=True, actors=None):
        """Load rules, validating every gesture/actor against the catalog.

        strict=True (default) raises on the first invalid rule — the historical
        behaviour. strict=False repairs what it can (case/separator differences)
        and quarantines the rest into self.rejected, so one bad rule cannot
        blank out a whole rule set.
        """
        self.rejected = []
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        rules = []
        for raw in data.get("rules", []):
            try:
                validate_action(raw["connector"], raw["action"])
                rules.append(normalise_rule(Rule(**raw), actors=actors))
            except (ValueError, TypeError, KeyError) as exc:
                if strict:
                    raise
                self.rejected.append((raw, str(exc)))
        return rules

    def save(self, rules):
        """Persist rules. Each carries its own ``dry_run``.

        The old top-level ``"dry_run": true`` is gone. It was written
        unconditionally on every save, so it could never describe anything and
        silently contradicted any attempt to arm a rule. Files that still have
        it load fine — the key is ignored, and each rule's own default is True.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "version": 1,
            "rules": [asdict(rule) for rule in rules],
        }, indent=2) + "\n"
        fd, tmp_name = tempfile.mkstemp(prefix=".rules-", suffix=".json",
                                        dir=self.path.parent)
        try:
            with os.fdopen(fd, "w") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self.path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
