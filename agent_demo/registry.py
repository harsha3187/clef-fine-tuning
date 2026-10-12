"""Versioned model registry: the catalog the router chooses from, editable without code changes."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SKILLS = ("coding", "reasoning", "tools", "general")
TIER_RANK = {"small": 0, "mid": 1, "frontier": 2}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_MUTABLE = {"healthy", "allowed", "live"}


@dataclass
class ModelProfile:
    id: str
    display: str
    provider: str
    deployment: str
    tier: str
    price_in: float
    price_out: float
    price_source: str
    lat_base_ms: float
    lat_per_token_ms: float
    skills: dict[str, float]
    context_window: int = 128_000
    regions: list[str] = field(default_factory=lambda: ["eastus2"])
    live: bool = False
    healthy: bool = True
    allowed: bool = True
    supports_tools: bool = True
    notes: str = ""

    def validate(self) -> ModelProfile:
        if not _ID.fullmatch(self.id) or not _ID.fullmatch(self.deployment):
            raise ValueError("Model and deployment ids use letters, digits, dot, dash and underscore")
        if self.tier not in TIER_RANK:
            raise ValueError(f"tier must be one of {sorted(TIER_RANK)}")
        if set(self.skills) != set(SKILLS) or any(not 0 <= v <= 1 for v in self.skills.values()):
            raise ValueError(f"skills must give each of {list(SKILLS)} a score between 0 and 1")
        numbers = (self.price_in, self.price_out, self.lat_base_ms, self.lat_per_token_ms)
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0 for v in numbers):
            raise ValueError("Prices and latency priors must be non-negative numbers")
        if not isinstance(self.context_window, int) or self.context_window <= 0:
            raise ValueError("context_window must be a positive integer")
        if not self.regions or not all(isinstance(r, str) and r for r in self.regions):
            raise ValueError("regions must list at least one region")
        for flag in ("live", "healthy", "allowed", "supports_tools"):
            if not isinstance(getattr(self, flag), bool):
                raise ValueError(f"{flag} must be true or false")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _canonical(models: list[ModelProfile]) -> str:
    return json.dumps([m.to_dict() for m in models], sort_keys=True, separators=(",", ":"))


class Registry:
    """Thread-safe catalog. Every change bumps the revision, so decisions cite an exact version."""

    def __init__(self, models: list[ModelProfile], path: Path | None = None, revision: int = 1) -> None:
        ids = [m.id for m in models]
        if len(ids) != len(set(ids)):
            raise ValueError("Model ids must be unique")
        self._models = [m.validate() for m in models]
        self._path = path
        self._revision = revision
        self._lock = threading.RLock()

    @classmethod
    def load(cls, seed: Path, runtime: Path | None = None) -> Registry:
        """Prefer the runtime copy (it holds health and onboarding changes) over the shipped seed."""
        source = runtime if runtime is not None and runtime.is_file() else seed
        data = json.loads(source.read_text(encoding="utf-8-sig"))
        models = [ModelProfile(**item) for item in data["models"]]
        return cls(models, runtime, int(data.get("revision", 1)))

    @property
    def version(self) -> str:
        with self._lock:
            return f"r{self._revision}.{hashlib.sha256(_canonical(self._models).encode()).hexdigest()[:8]}"

    def snapshot(self) -> list[ModelProfile]:
        with self._lock:
            return [ModelProfile(**m.to_dict()) for m in self._models]

    def get(self, model_id: str) -> ModelProfile:
        with self._lock:
            for model in self._models:
                if model.id == model_id:
                    return ModelProfile(**model.to_dict())
        raise KeyError(f"Unknown model: {model_id}")

    def update(self, model_id: str, **changes: Any) -> ModelProfile:
        """Change health, allowlist membership or liveness only; profiles are otherwise immutable."""
        if not changes or set(changes) - _MUTABLE or any(not isinstance(v, bool) for v in changes.values()):
            raise ValueError(f"Only {sorted(_MUTABLE)} can be changed, and each must be true or false")
        with self._lock:
            for model in self._models:
                if model.id == model_id:
                    for key, value in changes.items():
                        setattr(model, key, value)
                    self._bump()
                    return ModelProfile(**model.to_dict())
        raise KeyError(f"Unknown model: {model_id}")

    def add(self, data: dict[str, Any]) -> ModelProfile:
        """Onboard a model by profile alone (BR-013); no application code changes."""
        try:
            model = ModelProfile(**data).validate()
        except TypeError as error:
            raise ValueError(f"Invalid model profile: {error}") from error
        with self._lock:
            if any(m.id == model.id for m in self._models):
                raise ValueError(f"Model {model.id} already exists")
            self._models.append(model)
            self._bump()
            return ModelProfile(**model.to_dict())

    def _bump(self) -> None:
        self._revision += 1
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"schema": 1, "revision": self._revision, "models": [m.to_dict() for m in self._models]}
            self._path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
