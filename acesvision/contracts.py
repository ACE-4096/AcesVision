"""Versioned source and scene contracts shared by AcesVision adapters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

CONTRACT_VERSION = 1


@dataclass(frozen=True)
class SourceSpec:
    id: str
    name: str
    kind: str
    index: int | None = None
    url: str | None = None
    trusted_device: bool = False
    device_path: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "SourceSpec":
        name = str(raw.get("name") or "Camera")
        source_id = str(raw.get("id") or name.lower().replace(" ", "-"))
        kind = str(raw.get("type") or raw.get("kind") or "").lower()
        if not kind:
            kind = "network" if raw.get("url") else "webcam"

        trusted = bool(raw.get("trusted_device", False))
        if kind in {"webcam", "camera", "local"}:
            index = raw.get("index")
            return cls(source_id, name, "webcam",
                       index=None if index is None else int(index),
                       trusted_device=trusted,
                       device_path=(str(raw.get("device_path"))
                                    if raw.get("device_path") else None))

        if kind == "droidcam":
            url = raw.get("url")
            if not url:
                host = str(raw.get("host") or "").strip()
                if not host:
                    raise ValueError("DroidCam source needs 'url' or 'host'")
                port = int(raw.get("port", 4747))
                path = str(raw.get("path") or "/video")
                if not path.startswith("/"):
                    path = "/" + path
                url = f"http://{host}:{port}{path}"
            return cls(source_id, name, "droidcam", url=str(url),
                       trusted_device=False)

        if kind in {"network", "url", "rtsp", "mjpeg"}:
            url = str(raw.get("url") or "").strip()
            if not url:
                raise ValueError(f"{kind} source needs 'url'")
            return cls(source_id, name, "network", url=url,
                       trusted_device=False)

        raise ValueError(f"unsupported camera source type: {kind}")

    def safe_label(self) -> str:
        if not self.url:
            index = "auto" if self.index is None else str(self.index)
            return f"{self.name} (webcam {index})"
        parts = urlsplit(self.url)
        host = parts.hostname or ""
        if parts.port:
            host += f":{parts.port}"
        redacted = urlunsplit((parts.scheme, host, parts.path, "", ""))
        return f"{self.name} ({self.kind}: {redacted})"


@dataclass
class SceneFrame:
    """One immutable-by-convention inference result shared by every output."""

    source: SourceSpec
    sequence: int
    captured_at: float
    raw: Any
    objects: list[Any] = field(default_factory=list)
    faces: list[Any] = field(default_factory=list)
    gestures: list[Any] = field(default_factory=list)
    poses: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    contract_version: int = CONTRACT_VERSION
