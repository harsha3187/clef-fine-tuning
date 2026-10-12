"""Routing policy: how a step's type, difficulty and the user's mode become a model choice.

Utility = expected quality - lambda * normalized cost - mu * normalized latency (BRD section 11),
over models that survive the hard constraints. Every number used is returned for display.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_demo.registry import ModelProfile

_SIGNALS = {
    "constraints": re.compile(
        r"\b(must|should|ensure|without|only|exactly|at least|at most|unless|except|never|always)\b", re.I),
    "reasoning": re.compile(
        r"\b(why|root cause|trade-?off|optimi[sz]|architecture|design|prove|analy[sz]e|security|race condition|"
        r"concurren\w*|deadlock|performance|refactor|thread[- ]?safe)\b", re.I),
    "multipart": re.compile(r"(?m)^\s*(?:\d+[.)]|[-*])\s+|;\s|\bthen\b|\band then\b|\bafterwards\b", re.I),
    "risk": re.compile(
        r"\b(furious|angry|unacceptable|outrageous|chargeback|lawyer|legal|complain\w*|manager|escalate|"
        r"compensat\w*|never again|disgusted|exception|VIP|platinum)\b|\$\s?\d{3,}", re.I),
}


@dataclass(frozen=True)
class Mode:
    key: str
    label: str
    lambda_cost: float
    mu_latency: float
    quality_floor: float
    requirement_shift: float


@dataclass(frozen=True)
class RouteProfile:
    skill: str
    base: float
    out_tokens: int
    needs_tools: bool = False
    executes: bool = True


@dataclass(frozen=True)
class Derived:
    """A user's preferences turned into routing parameters, with a plain-language reading of each."""

    mode: Mode
    gate_min_score: int
    gate_tiers: tuple[str, ...]
    notes: dict[str, str]


@dataclass
class Complexity:
    score: float
    signals: dict[str, float]


@dataclass
class Requirement:
    total: float
    parts: list[tuple[str, float]]


@dataclass
class Candidate:
    model_id: str
    display: str
    tier: str
    status: str  # "selected", "eligible", "below_floor" or "eliminated"
    reasons: list[str] = field(default_factory=list)
    quality: float | None = None
    est_cost: float | None = None
    est_latency_ms: float | None = None
    utility: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Selection:
    candidates: list[Candidate]
    selected: str | None
    below_floor: bool
    budget_overridden: bool
    note: str


class Policy:
    def __init__(self, data: dict[str, Any]) -> None:
        try:
            self.data = data
            self.modes = {key: Mode(key=key, **value) for key, value in data["modes"].items()}
            self.profiles = {
                taxonomy: {route: RouteProfile(**spec) for route, spec in routes.items()}
                for taxonomy, routes in data["profiles"].items()
            }
            self.sigmoid_k = float(data["sigmoid_k"])
            self.complexity_max_bump = float(data["complexity_max_bump"])
            self.low_confidence = data["low_confidence"]
            self.escalation = data["escalation"]
            self.limits = data["limits"]
            self.fallback_route = data["fallback_route"]
            self.preference_table = data["preferences"]
            self.customer_facing_bump = float(data["customer_facing_bump"])
            self.stakes_bump = float(data["stakes_bump"])
            self.stakes = re.compile(data["stakes_pattern"], re.I)
            self.router_cost_usd = float(data["router_cost_per_decision_usd"])
            self.baseline_model = data["baseline_model"]
            self.planner_model = data["planner_model"]
            self.judge_model = data["judge_model"]
            self.safe_fallback_model = data["safe_fallback_model"]
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid routing policy: {error}") from error
        if not {"fast", "balanced", "deep"} <= set(self.modes):
            raise ValueError("Policy needs fast, balanced and deep modes")
        self.version = hashlib.sha256(
            json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:8]

    @classmethod
    def load(cls, path: Path) -> Policy:
        return cls(json.loads(path.read_text(encoding="utf-8-sig")))

    def mode(self, key: str) -> Mode:
        if key not in self.modes:
            raise ValueError(f"Unknown mode {key!r}; choose one of {sorted(self.modes)}")
        return self.modes[key]

    def derive(self, prefs: Any) -> Derived:
        """Turn quality, latency and cost levels into one routing mode and a result-check threshold."""
        table = self.preference_table
        quality, latency, cost = table["quality"][prefs.quality], table["latency"][prefs.latency], table["cost"][prefs.cost]
        mode = Mode(key="custom", label="Your preferences", lambda_cost=cost["weight"], mu_latency=latency["weight"],
                    quality_floor=quality["floor"], requirement_shift=quality["shift"])
        notes = {
            "quality": f"A model is used only if it is expected to get a step right at least {quality['floor']:.0%} "
                       f"of the time. When a {' or '.join(quality['gate_tiers'])}-tier model is used, a result scoring under "
                       f"{quality['gate_min_score']}/5 on the quality check is retried on a stronger model.",
            "latency": latency["note"],
            "cost": cost["note"],
        }
        return Derived(mode, int(quality["gate_min_score"]), tuple(quality["gate_tiers"]), notes)

    def profile(self, taxonomy: str, route: str) -> RouteProfile:
        try:
            return self.profiles[taxonomy][route]
        except KeyError as error:
            raise ValueError(f"No routing profile for {taxonomy}/{route}") from error


def assess_complexity(text: str, policy: Policy | None = None) -> Complexity:
    """Cheap, deterministic difficulty signals computed by the agent, not by the model."""
    tokens = max(1, len(text) // 4)
    code_lines = sum(1 for line in text.splitlines()
                     if re.match(r"\s*(def |class |import |from \S+ import |return |for |while |if |\w+\(.*\)\s*$)", line))
    signals = {
        "length": min(1.0, tokens / 600),
        "code": min(1.0, code_lines / 20) if "def " in text or "class " in text or code_lines > 3 else 0.0,
        "constraints": min(1.0, len(_SIGNALS["constraints"].findall(text)) / 4),
        "multipart": min(1.0, len(_SIGNALS["multipart"].findall(text)) / 4),
        "reasoning": min(1.0, len(_SIGNALS["reasoning"].findall(text)) / 3),
    }
    signals["risk"] = min(1.0, len(_SIGNALS["risk"].findall(text)) / 3)
    weights = {"length": 0.25, "code": 0.20, "constraints": 0.20, "multipart": 0.15, "reasoning": 0.20, "risk": 0.20}
    return Complexity(round(sum(signals[k] * weights[k] for k in weights), 4),
                      {k: round(v, 3) for k, v in signals.items()})


def compute_requirement(
    policy: Policy, profile: RouteProfile, route: str, complexity: Complexity, confidence: float | None,
    mode: Mode, prior_failures: int, customer_facing: bool = False, stakes: bool = False,
) -> Requirement:
    """How capable a model must be for this step, with every contribution listed."""
    parts = [(f"step type: {route}", profile.base)]
    bump = policy.complexity_max_bump * complexity.score
    if bump >= 0.005:
        parts.append(("difficulty signals", round(bump, 4)))
    low = policy.low_confidence
    if confidence is not None and confidence < low["threshold"]:
        parts.append((f"router unsure ({confidence:.0%})", low["requirement_bump"]))
    if customer_facing:
        parts.append(("customer-facing message", policy.customer_facing_bump))
    if stakes:
        parts.append(("changes money or an order", policy.stakes_bump))
    if mode.requirement_shift:
        parts.append((f"{mode.label} setting", mode.requirement_shift))
    if prior_failures:
        parts.append((f"escalation x{prior_failures}", policy.escalation["requirement_step"] * prior_failures))
    total = min(0.98, max(0.10, sum(delta for _, delta in parts)))
    return Requirement(round(total, 4), [(name, round(delta, 4)) for name, delta in parts])


def expected_quality(capability: float, required: float, k: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (capability - required)))


def _estimate(model: ModelProfile, tokens_in: int, tokens_out: int) -> tuple[float, float]:
    cost = tokens_in * model.price_in / 1e6 + tokens_out * model.price_out / 1e6
    latency = model.lat_base_ms + model.lat_per_token_ms * tokens_out
    return cost, latency


def select_model(
    policy: Policy, models: list[ModelProfile], profile: RouteProfile, requirement: Requirement, mode: Mode,
    tokens_in: int, tokens_out: int, *, remaining_budget: float | None = None,
    allowlist: set[str] | None = None, regions: set[str] | None = None, exclude: frozenset[str] = frozenset(),
) -> Selection:
    """Hard constraints first (BR-004), then rank the survivors by utility under the quality floor."""
    total_tokens = tokens_in + tokens_out
    candidates: dict[str, Candidate] = {}
    survivors: list[ModelProfile] = []
    for model in models:
        reasons = []
        if not model.live:
            reasons.append("not deployed (catalog only)")
        if not model.healthy:
            reasons.append("marked unhealthy")
        if not model.allowed or (allowlist is not None and model.id not in allowlist):
            reasons.append("not on the organization allowlist")
        if regions is not None and not set(model.regions) & regions:
            reasons.append("region not permitted")
        if model.context_window < total_tokens:
            reasons.append("context window too small")
        if profile.needs_tools and not model.supports_tools:
            reasons.append("no tool-calling support")
        if model.id in exclude:
            reasons.append("already tried in this step")
        cost, latency = _estimate(model, tokens_in, tokens_out)
        candidate = Candidate(model.id, model.display, model.tier, "eliminated", reasons,
                              est_cost=round(cost, 6), est_latency_ms=round(latency))
        candidates[model.id] = candidate
        if not reasons:
            survivors.append(model)

    def rank(pool: list[ModelProfile]) -> tuple[list[Candidate], float]:
        top_cost = max(candidates[m.id].est_cost for m in pool) or 1.0
        top_latency = max(candidates[m.id].est_latency_ms for m in pool) or 1.0
        ranked = []
        for model in pool:
            c = candidates[model.id]
            c.quality = round(expected_quality(model.skills[profile.skill], requirement.total, policy.sigmoid_k), 4)
            c.utility = round(c.quality - mode.lambda_cost * c.est_cost / top_cost
                              - mode.mu_latency * c.est_latency_ms / top_latency, 4)
            c.status = "eligible"
            ranked.append(c)
        return ranked, top_cost

    note = ""
    overridden = False
    pool = survivors
    if remaining_budget is not None:
        affordable = [m for m in survivors if candidates[m.id].est_cost <= remaining_budget]
        for model in survivors:
            if model not in affordable:
                candidates[model.id].reasons.append(
                    f"estimated ${candidates[model.id].est_cost:.4f} exceeds remaining budget ${remaining_budget:.4f}")
                candidates[model.id].status = "eliminated"
        if survivors and not affordable:
            overridden = True
            note = "No model fits the remaining budget, so the cheapest eligible model was used."
        else:
            pool = affordable
    if not pool:
        return Selection(list(candidates.values()), None, False, False, "No eligible model remains.")

    ranked, _ = rank(pool)
    if overridden:
        chosen = min(ranked, key=lambda c: c.est_cost)
        below = chosen.quality < mode.quality_floor
    else:
        passing = [c for c in ranked if c.quality >= mode.quality_floor]
        below = not passing
        chosen = max(passing or ranked, key=lambda c: (c.utility if passing else c.quality, -c.est_cost))
    for c in ranked:
        if c.quality < mode.quality_floor and c is not chosen:
            c.status = "below_floor"
            c.reasons.append(f"expected quality {c.quality:.0%} is under the {mode.label} floor "
                             f"of {mode.quality_floor:.0%}")
    chosen.status = "selected"
    if below and not overridden:
        note = ("No model reaches the quality floor; the most capable eligible model was used.")
    return Selection(sorted(candidates.values(), key=lambda c: (c.status != "selected", -(c.utility or -9))),
                     chosen.model_id, below, overridden, note)
