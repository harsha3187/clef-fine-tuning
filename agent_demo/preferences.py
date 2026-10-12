"""The three preferences a user sets, and how they become routing parameters.

Quality, latency and cost are chosen as plain levels. The policy table in policy.json turns each level into the
numbers the router uses, so the mapping is data, visible in the UI, and versioned with the policy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

LEVELS: dict[str, tuple[str, ...]] = {
    "quality": ("average", "good", "very_good"),
    "latency": ("low", "medium", "high"),
    "cost": ("low", "medium", "high"),
}
PRESETS: dict[str, dict[str, str]] = {
    "lean": {"quality": "average", "latency": "low", "cost": "low"},
    "balanced": {"quality": "good", "latency": "medium", "cost": "medium"},
    "premium": {"quality": "very_good", "latency": "high", "cost": "high"},
}


@dataclass(frozen=True)
class Preferences:
    quality: str = "good"
    latency: str = "medium"
    cost: str = "medium"

    def __post_init__(self) -> None:
        for field, levels in LEVELS.items():
            if getattr(self, field) not in levels:
                raise ValueError(f"{field} must be one of {list(levels)}")

    @classmethod
    def from_dict(cls, data: Any) -> Preferences:
        if not isinstance(data, dict):
            raise ValueError("Preferences must be an object")
        unknown = set(data) - set(LEVELS)
        if unknown:
            raise ValueError(f"Unknown preference fields: {sorted(unknown)}")
        return cls(**{k: data[k] for k in LEVELS if k in data})

    def to_dict(self) -> dict[str, str]:
        return asdict(self)

    def preset(self) -> str | None:
        """The named preset these levels match, if any."""
        return next((name for name, levels in PRESETS.items() if levels == self.to_dict()), None)
