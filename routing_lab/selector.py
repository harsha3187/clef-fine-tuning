"""The selector: Clef's tier and skill probabilities plus measured endpoint profiles in, one endpoint out.

It is deterministic and never reads the request text. The only signals are the two probability distributions from the
capability router and numbers measured on the benchmark. The same function runs live (shadow mode in the agents) and in
the offline evaluation, so what is reported is what runs.

    expected success(endpoint) = sum over tiers P(tier) x sum over skills P(skill) x measured pass rate(endpoint, skill, tier)

Endpoints below the user's quality floor are dropped, and the rest are ranked by
`expected success - cost weight x relative cost - latency weight x relative latency`.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SELECTOR_VERSION = "selector-v1"
TIERS = ("low", "medium", "high")
MIN_CELL = 4  # fewer measured tasks than this and a (skill, tier) cell is not trusted on its own
MIN_MASS = 0.8  # share of the probability mass that must be covered by measured cells
VERBOSITY_RANGE = (0.5, 3.0)

RateFn = Callable[[str, str | None, str], float | None]


@dataclass(frozen=True)
class Endpoint:
    id: str
    deployment: str
    price_in: float  # USD per 1M tokens
    price_out: float
    supports_tools: bool
    verified_price: bool
    mean_out_tokens: float
    lat_base_ms: float
    lat_per_token_ms: float
    context_window: int = 128_000
    api: str = "responses"  # which client can call it; "chat" endpoints have no Responses API
    allowed: bool = True
    healthy: bool = True


@dataclass(frozen=True)
class Settings:
    floor: float
    cost_weight: float
    latency_weight: float
    confidence_threshold: float = 0.55  # below this the top tier is not trusted
    high_tier_probability: float = 0.25  # this much mass on "high" counts as a warning
    uncertainty_margin: float = 0.05  # added to the floor when Clef is unsure


# The three presets of the agent screen, as selector settings. A test keeps them equal to agent_demo/data/policy.json.
PRESET_SETTINGS = {
    "lean": Settings(floor=0.55, cost_weight=1.2, latency_weight=1.0),
    "balanced": Settings(floor=0.70, cost_weight=0.6, latency_weight=0.4),
    "premium": Settings(floor=0.85, cost_weight=0.1, latency_weight=0.05),
}


@dataclass(frozen=True)
class Constraints:
    allow: frozenset[str] | None = None
    needs_tools: bool = False
    input_tokens: int = 600
    output_tokens: int = 400


@dataclass
class Scored:
    endpoint: str
    status: str  # "selected", "eligible", "below_floor" or "eliminated"
    reasons: list[str] = field(default_factory=list)
    expected_success: float | None = None
    est_cost_usd: float | None = None
    est_latency_ms: float | None = None
    utility: float | None = None
    price_verified: bool = True


@dataclass
class Selection:
    choice: str | None
    expected_success: float | None
    floor: float
    effective_floor: float
    below_floor: bool
    uncertain: bool
    uncertainty_reasons: list[str]
    confidence: float
    candidates: list[Scored]
    reason: str
    version: str = SELECTOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize(probabilities: dict[str, float] | None, keys: tuple[str, ...] | None = None) -> dict[str, float]:
    if not probabilities:
        return {}
    total = sum(max(0.0, v) for v in probabilities.values())
    if total <= 0:
        return {}
    return {k: max(0.0, v) / total for k, v in probabilities.items() if keys is None or k in keys}


def expected_success(rate: RateFn, endpoint: str, tiers: dict[str, float], skills: dict[str, float]) -> float | None:
    """Probability-weighted measured pass rate; None when too little of the probability mass is measured."""
    weighted = covered = 0.0
    for tier, p_tier in tiers.items():
        if p_tier <= 0:
            continue
        for skill, p_skill in (skills or {None: 1.0}).items():
            mass = p_tier * p_skill
            if mass <= 0:
                continue
            value = rate(endpoint, skill, tier)
            if value is not None:
                weighted += mass * value
                covered += mass
    return weighted / covered if covered >= MIN_MASS else None


def _verbosity(endpoints: list[Endpoint]) -> dict[str, float]:
    """How many output tokens an endpoint tends to write relative to the typical one, measured on the benchmark."""
    typical = statistics.median(e.mean_out_tokens for e in endpoints) or 1.0
    low, high = VERBOSITY_RANGE
    return {e.id: min(high, max(low, e.mean_out_tokens / typical)) for e in endpoints}


def select(endpoints: list[Endpoint], rate: RateFn, tier_probabilities: dict[str, float],
           skill_probabilities: dict[str, float] | None, settings: Settings,
           constraints: Constraints = Constraints()) -> Selection:  # noqa: B008 - frozen dataclass default
    tiers = _normalize(tier_probabilities, TIERS)
    if not tiers:
        raise ValueError("Tier probabilities are required")
    skills = _normalize(skill_probabilities)
    confidence = max(tiers.values())
    warnings = []
    if confidence < settings.confidence_threshold:
        warnings.append(f"Clef's top tier has only {confidence:.0%} probability")
    if tiers.get("high", 0.0) >= settings.high_tier_probability:
        warnings.append(f"{tiers['high']:.0%} of the probability is on the high tier")
    effective_floor = round(settings.floor + (settings.uncertainty_margin if warnings else 0.0), 4)

    verbosity = _verbosity(endpoints)
    scored: dict[str, Scored] = {}
    live: list[tuple[Endpoint, float, float, float]] = []
    for e in endpoints:
        item = Scored(e.id, "eliminated", price_verified=e.verified_price)
        scored[e.id] = item
        if not e.allowed or (constraints.allow is not None and e.id not in constraints.allow):
            item.reasons.append("not on the allow-list")
        elif not e.healthy:
            item.reasons.append("marked unhealthy")
        elif constraints.needs_tools and not e.supports_tools:
            item.reasons.append("cannot call tools")
        elif constraints.input_tokens > e.context_window:
            item.reasons.append("request does not fit the context window")
        else:
            success = expected_success(rate, e.id, tiers, skills)
            if success is None:
                item.reasons.append("not measured for this kind of request")
            else:
                out = constraints.output_tokens * verbosity[e.id]
                cost = constraints.input_tokens * e.price_in / 1e6 + out * e.price_out / 1e6
                latency = e.lat_base_ms + e.lat_per_token_ms * out
                item.expected_success, item.est_cost_usd, item.est_latency_ms = round(success, 4), cost, latency
                live.append((e, success, cost, latency))
    if not live:
        return Selection(None, None, settings.floor, effective_floor, False, bool(warnings), warnings, confidence,
                         list(scored.values()), "No endpoint is eligible and measured for this request.")

    top_cost, top_latency = max(c for _, _, c, _ in live) or 1.0, max(lat for _, _, _, lat in live) or 1.0
    for e, success, cost, latency in live:
        item = scored[e.id]
        item.utility = round(success - settings.cost_weight * cost / top_cost - settings.latency_weight * latency / top_latency, 4)
        item.status = "eligible" if success >= effective_floor else "below_floor"
    passing = [(e, s, c) for e, s, c, _ in live if s >= effective_floor]
    if passing:
        pick = max(passing, key=lambda row: (scored[row[0].id].utility, -row[2], row[0].id))
        below = False
        reason = (f"Best fit for the cost and latency settings among endpoints expected to succeed {pick[1]:.0%} of the time, "
                  f"against a floor of {effective_floor:.0%}.")
    else:
        pick = max(((e, s, c) for e, s, c, _ in live), key=lambda row: (row[1], -row[2], row[0].id))
        below = True
        reason = (f"No endpoint reaches the {effective_floor:.0%} floor; the most reliable one "
                  f"({pick[1]:.0%} expected) was chosen.")
    scored[pick[0].id].status = "selected"
    if warnings:
        reason += " The floor was raised because " + " and ".join(w[0].lower() + w[1:] for w in warnings) + "."
    order = sorted(scored.values(), key=lambda s: (s.status != "selected", s.status == "eliminated", -(s.utility or -9), s.endpoint))
    return Selection(pick[0].id, round(pick[1], 4), settings.floor, effective_floor, below, bool(warnings), warnings,
                     round(confidence, 4), order, reason)


class MeasuredRegistry:
    """Endpoint profiles measured on the benchmark: pass rates by skill and tier, price, latency, output length.

    Nothing here is assumed. Every cell carries its sample size, and an endpoint that was never measured for a kind
    of request is not selectable for it.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self.data = data
        self.endpoints = [Endpoint(**{k: v for k, v in item.items() if k in Endpoint.__dataclass_fields__ and k != "cells"})
                          for item in data["endpoints"]]
        self._cells: dict[str, dict[str, dict[str, list[float]]]] = {item["id"]: item["cells"] for item in data["endpoints"]}
        self._tier_pool: dict[tuple[str, str], list[float]] = {}
        for endpoint, by_skill in self._cells.items():
            for tiers in by_skill.values():
                for tier, (passes, n) in tiers.items():
                    cell = self._tier_pool.setdefault((endpoint, tier), [0.0, 0.0])
                    cell[0] += passes
                    cell[1] += n

    @classmethod
    def load(cls, path: Path) -> MeasuredRegistry:
        return cls(json.loads(path.read_text(encoding="utf-8-sig")))

    @property
    def version(self) -> str:
        return str(self.data["version"])

    def rate(self, endpoint: str, skill: str | None, tier: str) -> float | None:
        """The (skill, tier) pass rate when it has enough samples, else the endpoint's pooled rate for the tier."""
        if skill is not None:
            passes, n = self._cells.get(endpoint, {}).get(skill, {}).get(tier, (0.0, 0.0))
            if n >= MIN_CELL:
                return passes / n
        passes, n = self._tier_pool.get((endpoint, tier), (0.0, 0.0))
        return passes / n if n >= MIN_CELL else None

    def evidence(self, endpoint: str, tier_probabilities: dict[str, float], skill_probabilities: dict[str, float] | None) -> dict[str, float]:
        """How much of an endpoint's expected success rests on tasks of exactly this skill and tier, and how much on pooled results."""
        tiers, skills = _normalize(tier_probabilities, TIERS), _normalize(skill_probabilities) or {None: 1.0}
        shares = {"exact": 0.0, "pooled": 0.0, "missing": 0.0}
        for tier, p_tier in tiers.items():
            for skill, p_skill in skills.items():
                cell = self._cells.get(endpoint, {}).get(skill, {}).get(tier, (0.0, 0.0)) if skill is not None else (0.0, 0.0)
                pool = self._tier_pool.get((endpoint, tier), (0.0, 0.0))
                shares["exact" if cell[1] >= MIN_CELL else "pooled" if pool[1] >= MIN_CELL else "missing"] += p_tier * p_skill
        total = sum(shares.values()) or 1.0
        return {f"{k}_share": round(v / total, 4) for k, v in shares.items()}

    def with_changes(self, **flags: dict[str, bool]) -> list[Endpoint]:
        """Endpoints with allow-list or health overrides, such as allowed={"phi-4": False}."""
        out = []
        for e in self.endpoints:
            changes = {k: v[e.id] for k, v in flags.items() if e.id in v}
            out.append(Endpoint(**{**asdict(e), **changes}) if changes else e)
        return out
