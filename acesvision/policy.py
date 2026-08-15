"""Typed gesture rules and deny-by-default action policy."""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

RISK_CONVENIENCE = "convenience"
RISK_PERSONAL = "personal"
RISK_SENSITIVE = "sensitive"

CONNECTORS = {
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

    @classmethod
    def create(cls, gesture, connector, action, actor="*", source="*"):
        validate_action(connector, action)
        risk = CONNECTORS[connector][action]
        return cls(
            id=str(uuid.uuid4()),
            gesture=gesture,
            connector=connector,
            action=action,
            actor=actor or "*",
            source=source or "*",
            require_liveness=risk == RISK_SENSITIVE,
            require_confirmation=risk == RISK_SENSITIVE,
        )

    @property
    def risk(self):
        return CONNECTORS[self.connector][self.action]


@dataclass(frozen=True)
class Decision:
    rule_id: str
    outcome: str
    reason: str
    connector: str
    action: str
    risk: str


def validate_action(connector, action):
    if connector not in CONNECTORS:
        raise ValueError(f"unsupported connector: {connector}")
    if action not in CONNECTORS[connector]:
        raise ValueError(f"unsupported action for {connector}: {action}")


class RuleEngine:
    def __init__(self, rules=None, dry_run=True):
        self.rules = list(rules or [])
        self.dry_run = bool(dry_run)

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
            return self._decision(rule, "blocked", "actor is not associated")
        if rule.require_liveness and event.get("liveness_state") != "live":
            return self._decision(rule, "blocked", "live presentation not verified")
        if rule.risk == RISK_SENSITIVE and not event.get("security_authorized", False):
            return self._decision(rule, "blocked", "security authorization unavailable")
        if rule.require_confirmation and not event.get("confirmed", False):
            return self._decision(rule, "blocked", "confirmation or passkey required")
        if self.dry_run:
            return self._decision(rule, "dry_run", "would execute")
        return self._decision(rule, "approved", "ready for typed connector dispatch")

    @staticmethod
    def _decision(rule, outcome, reason):
        return Decision(rule.id, outcome, reason, rule.connector,
                        rule.action, rule.risk)


class RuleStore:
    def __init__(self, path=None):
        self.path = Path(path or Path.home() / ".config" / "acesvision" / "rules.json")

    def load(self):
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text())
        rules = []
        for raw in data.get("rules", []):
            validate_action(raw["connector"], raw["action"])
            rules.append(Rule(**raw))
        return rules

    def save(self, rules):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "version": 1,
            "dry_run": True,
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
