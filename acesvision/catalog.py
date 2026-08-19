"""The gesture vocabulary, as data.

The list of recognisable gestures used to be a tuple literal inside
``gesture_catalog.py``. That made it code: a subscriber in another process — or
another language — could not read it, and there was no way to tell whether the
vocabulary it had compiled against was still the vocabulary the emitter was
using. Two projects sharing a vocabulary by both hardcoding it is how the live
rules ended up permanently unfireable in the first place.

So the vocabulary now lives in ``gestures.json`` at the repository root and this
module is the only thing that reads it. Two properties follow, and both are the
point:

``version``
    An integer the operator bumps whenever the file changes. A subscriber can
    refuse to run against a version it does not understand.

``sha256``
    A hash over the *canonical* serialisation — ``sort_keys=True`` with the
    tight separators — so it is a fingerprint of the vocabulary's **content**,
    not of the file's whitespace or key order. Reformat the JSON and the hash is
    unchanged; add a gesture and it is not. A subscriber that pins the hash
    detects a silently edited catalog, which the version alone cannot do because
    an editor can always forget to bump it.

The pair travels on every emitted event (see ``acesvision/emitter.py``) and is
served whole at ``GET /api/catalog``.

Pure data and pure functions. No cv2, no mediapipe, no network. The one read of
``gestures.json`` happens once at import.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

#: Repository-root ``gestures.json``. This module lives in ``acesvision/``, so
#: one level up is the root that also holds ``gesture_catalog.py``.
CATALOG_PATH = Path(__file__).resolve().parents[1] / "gestures.json"


class CatalogError(ValueError):
    """``gestures.json`` is missing, malformed, or self-contradictory.

    A ValueError subclass so existing callers that catch ValueError around
    vocabulary loading keep working.
    """


@dataclass(frozen=True)
class GestureSpec:
    """One entry in the recognisable-gesture vocabulary."""

    id: str
    label: str
    builtin: bool = True   # True == the MediaPipe model emits this label itself

    def as_dict(self):
        return asdict(self)


def canonical_json(document) -> str:
    """The one serialisation the sha256 is taken over.

    Sorted keys and no incidental whitespace, so the hash describes the parsed
    document rather than the bytes on disk. Anything that reproduces this string
    reproduces the hash — including a subscriber written in another language.
    """
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def catalog_sha256(document) -> str:
    return hashlib.sha256(canonical_json(document).encode("utf-8")).hexdigest()


def _fold(name) -> str:
    """Fold a hand-typed name onto a comparison key: case and separators ignored."""
    return "".join(character for character in str(name).lower()
                   if character.isalnum())


class GestureCatalog:
    """An immutable, validated gesture vocabulary and its fingerprint."""

    def __init__(self, document, origin: str = "<memory>"):
        self.origin = origin
        self._document = _validate(document, origin)
        self.version = int(self._document["catalog_version"])
        self.sha256 = catalog_sha256(self._document)
        self.gestures = tuple(
            GestureSpec(entry["id"], entry["label"], bool(entry["builtin"]))
            for entry in self._document["gestures"]
        )
        self.ids = tuple(spec.id for spec in self.gestures)
        self._by_key = {_fold(spec.id): spec for spec in self.gestures}

    def by_id(self, gesture_id):
        """The spec for a loosely typed name, or None."""
        return self._by_key.get(_fold(gesture_id))

    def normalise(self, name):
        """The canonical gesture id for a loosely typed name, or None."""
        spec = self._by_key.get(_fold(name))
        return spec.id if spec else None

    def is_known(self, name) -> bool:
        return self.normalise(name) is not None

    def require(self, name) -> str:
        """Canonical gesture id, or ValueError naming the whole vocabulary.

        Strict on purpose: a name that is not in the catalog is a typo or a
        version skew, and both are better as a loud failure at entry than as a
        rule that silently never fires.
        """
        canonical = self.normalise(name)
        if canonical is None:
            raise CatalogError(
                f"unknown gesture: {name!r}. Known gestures: "
                + ", ".join(self.ids)
            )
        return canonical

    def as_document(self) -> dict:
        """A deep copy of the parsed catalog. ``canonical_json`` of this
        reproduces ``sha256`` exactly."""
        return json.loads(canonical_json(self._document))

    def as_payload(self) -> dict:
        """The catalog as served at ``GET /api/catalog``: the document plus the
        fingerprint a subscriber pins against."""
        payload = self.as_document()
        payload["sha256"] = self.sha256
        return payload

    def stamp(self) -> dict:
        """The two-field catalog reference carried on every emitted event."""
        return {"version": self.version, "sha256": self.sha256}


def _validate(document, origin) -> dict:
    """Reject a catalog that cannot mean one thing, naming what is wrong.

    A vocabulary that two processes disagree about is the failure this whole
    module exists to prevent, so ambiguity is refused rather than repaired.
    """
    if not isinstance(document, dict):
        raise CatalogError(f"{origin}: top level must be an object")

    version = document.get("catalog_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise CatalogError(
            f"{origin}: 'catalog_version' must be an integer >= 1, got "
            f"{version!r}"
        )

    entries = document.get("gestures")
    if not isinstance(entries, list) or not entries:
        raise CatalogError(f"{origin}: 'gestures' must be a non-empty array")

    seen = {}
    for position, entry in enumerate(entries):
        where = f"{origin}: gestures[{position}]"
        if not isinstance(entry, dict):
            raise CatalogError(f"{where} must be an object")
        missing = [key for key in ("id", "label", "builtin") if key not in entry]
        if missing:
            raise CatalogError(f"{where} is missing {', '.join(missing)}")
        gesture_id = entry["id"]
        if not isinstance(gesture_id, str) or not gesture_id.strip():
            raise CatalogError(f"{where} has a non-string or empty 'id'")
        if not isinstance(entry["label"], str) or not entry["label"].strip():
            raise CatalogError(f"{where} has a non-string or empty 'label'")
        if not isinstance(entry["builtin"], bool):
            raise CatalogError(f"{where} has a non-boolean 'builtin'")
        # Uniqueness is checked on the *folded* key, not the raw id. Names are
        # matched case- and separator-insensitively, so "open_palm" and
        # "Open_Palm" are one gesture wearing two hats and normalisation would
        # pick between them arbitrarily.
        key = _fold(gesture_id)
        if key in seen:
            raise CatalogError(
                f"{where}: id {gesture_id!r} collides with {seen[key]!r} — "
                "names are matched ignoring case and separators"
            )
        seen[key] = gesture_id
    return document


def load_catalog(path=None) -> GestureCatalog:
    """Read and validate a catalog file."""
    path = Path(path or CATALOG_PATH)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"cannot read gesture catalog {path}: {exc}") from exc
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"{path} is not valid JSON: {exc}") from exc
    return GestureCatalog(document, origin=str(path))


#: The process-wide catalog. Loaded once; everything else in the repo — the
#: vocabulary re-exported by ``gesture_catalog``, rule validation, the emitter's
#: per-event stamp, ``GET /api/catalog`` — reads through this one object.
CATALOG = load_catalog()

CATALOG_VERSION = CATALOG.version
CATALOG_SHA256 = CATALOG.sha256
GESTURES = CATALOG.gestures
GESTURE_IDS = CATALOG.ids

gesture_by_id = CATALOG.by_id
normalise_gesture = CATALOG.normalise
is_known_gesture = CATALOG.is_known
require_gesture = CATALOG.require
catalog_document = CATALOG.as_document
catalog_payload = CATALOG.as_payload
catalog_stamp = CATALOG.stamp
