"""The Adaptive Intelligence Engine's router: one traceable model decision per agent step."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from agent_demo import ROUTER_VERSION
from agent_demo.clef_decider import ClefDecider, ClefDecision
from agent_demo.policy import Mode, Policy, assess_complexity, compute_requirement, select_model
from agent_demo.registry import Registry
from agent_demo.telemetry import Telemetry

DOMAINS = ("code", "tools")


class RoutingError(RuntimeError):
    """No model could be selected, not even by the safe fallback."""


@dataclass
class RoutingContext:
    step_text: str
    domain: str
    mode: str | Mode = "balanced"
    case_text: str = ""  # the whole customer message, used for risk signals the single step may not carry
    customer_facing: bool = False
    stakes: bool | None = None  # None: let the policy's own pattern decide
    preceding: tuple[str, ...] = ()  # earlier steps this one may depend on, shown to Clef as context
    task_id: str = ""
    step_id: str = ""
    tools: list[dict[str, Any]] | None = None
    remaining_budget_usd: float | None = None
    allowlist: set[str] | None = None
    regions: set[str] | None = None
    prior_failures: int = 0
    exclude: frozenset[str] = frozenset()
    prior_clef: ClefDecision | None = None
    shadow: bool = False
    attempt: int = 0


@dataclass
class RoutingDecision:
    trace_id: str
    task_id: str
    step_id: str
    mode: str
    shadow: bool
    attempt: int
    clef: dict[str, Any]
    complexity: dict[str, Any]
    requirement: dict[str, Any]
    profile: dict[str, Any]
    selected_model: str
    selected_tier: str
    selected_deployment: str
    below_floor: bool
    fallback_used: bool
    note: str
    estimates: dict[str, int]
    candidates: list[dict[str, Any]]
    router_ms: float
    clef_ms: float
    router_version: str
    registry_version: str
    policy_version: str
    reason: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdaptiveRouter:
    def __init__(self, registry: Registry, policy: Policy, decider: ClefDecider, telemetry: Telemetry) -> None:
        self.registry, self.policy, self.decider, self.telemetry = registry, policy, decider, telemetry

    def route(self, ctx: RoutingContext) -> RoutingDecision:
        if ctx.domain not in DOMAINS:
            raise ValueError(f"domain must be one of {DOMAINS}")
        if not ctx.step_text.strip():
            raise ValueError("The step text must not be empty")
        started = time.perf_counter()
        mode = ctx.mode if isinstance(ctx.mode, Mode) else self.policy.mode(ctx.mode)
        clef = ctx.prior_clef or self.decider.decide(ctx.domain, ctx.step_text, tools=ctx.tools, preceding=ctx.preceding)
        profile = self.policy.profile(clef.taxonomy, clef.route)
        complexity = assess_complexity(f"{ctx.step_text}\n{ctx.case_text}", self.policy)
        # An unreachable router is treated as maximally unsure, so the safe path errs toward capability.
        confidence = 0.0 if clef.source == "fallback" else clef.confidence
        requirement = compute_requirement(
            self.policy, profile, clef.route, complexity, confidence, mode, ctx.prior_failures, ctx.customer_facing,
            stakes=(ctx.stakes if ctx.stakes is not None
                    else not ctx.customer_facing and bool(self.policy.stakes.search(ctx.step_text))))
        tools_chars = len(json.dumps(ctx.tools)) if ctx.tools else 0
        tokens_in = (len(ctx.step_text) + tools_chars) // 4 + 200
        tokens_out = int(profile.out_tokens * (1 + complexity.score))
        models = self.registry.snapshot()
        selection = select_model(
            self.policy, models, profile, requirement, mode, tokens_in, tokens_out,
            remaining_budget=ctx.remaining_budget_usd, allowlist=ctx.allowlist, regions=ctx.regions,
            exclude=ctx.exclude)

        warnings: list[str] = []
        fallback_used = False
        selected_id = selection.selected
        if selected_id is None:
            selected_id = self._safe_fallback(models, ctx)
            fallback_used = True
            warnings.append("No candidate was eligible; the configured safe fallback model was used.")
        if clef.source == "fallback":
            warnings.append(f"The decision model was unavailable ({clef.error}); a deterministic default step type was used.")
        if selection.note:
            warnings.append(selection.note)
        chosen = next(m for m in models if m.id == selected_id)
        router_ms = round((time.perf_counter() - started) * 1000, 1)
        reason = self._reason(clef, profile.skill, requirement.total, chosen.display, mode.label, selection.below_floor)
        decision = RoutingDecision(
            trace_id="tr_" + uuid.uuid4().hex[:12], task_id=ctx.task_id, step_id=ctx.step_id, mode=mode.key,
            shadow=ctx.shadow, attempt=ctx.attempt, clef=clef.to_dict(), complexity=asdict(complexity),
            requirement={"total": requirement.total, "parts": requirement.parts},
            profile=asdict(profile), selected_model=chosen.id, selected_tier=chosen.tier,
            selected_deployment=chosen.deployment, below_floor=selection.below_floor, fallback_used=fallback_used,
            note=selection.note, estimates={"input_tokens": tokens_in, "output_tokens": tokens_out},
            candidates=[c.to_dict() for c in selection.candidates], router_ms=router_ms,
            clef_ms=0.0 if ctx.prior_clef else clef.latency_ms, router_version=ROUTER_VERSION,
            registry_version=self.registry.version, policy_version=self.policy.version, reason=reason,
            warnings=warnings)
        self._log(decision, ctx, clef)
        return decision

    def _safe_fallback(self, models: list, ctx: RoutingContext) -> str:
        usable = [m for m in models if m.live and m.healthy and m.allowed and m.id not in ctx.exclude]
        preferred = [m for m in usable if m.id == self.policy.safe_fallback_model]
        if preferred or usable:
            return (preferred or usable)[0].id
        raise RoutingError("No live, healthy and allowed model is available.")

    @staticmethod
    def _reason(clef: ClefDecision, skill: str, required: float, model: str, mode: str, below: bool) -> str:
        source = (f"The decision model classified the step as {clef.route} ({clef.confidence:.0%})" if clef.source == "clef"
                  else f"The decision model was unavailable, so the safe default step type {clef.route} was used")
        tail = ", but no model fully reached the quality floor" if below else ""
        return (f"{source}. That needs {skill} capability of about {required:.0%} in {mode} mode, so "
                f"{model} gives the best utility among eligible models{tail}.")

    def _log(self, d: RoutingDecision, ctx: RoutingContext, clef: ClefDecision) -> None:
        self.telemetry.log_decision({
            "trace_id": d.trace_id, "task_id": d.task_id, "step_id": d.step_id,
            "created_utc": datetime.now(timezone.utc).isoformat(), "mode": d.mode, "shadow": int(d.shadow),
            "router_version": d.router_version, "registry_version": d.registry_version,
            "policy_version": d.policy_version, "domain": ctx.domain, "taxonomy": clef.taxonomy,
            "route": clef.route, "confidence": clef.confidence, "clef_source": clef.source,
            "clef_ms": d.clef_ms, "router_ms": d.router_ms, "requirement": d.requirement["total"],
            "text_sha256": hashlib.sha256(ctx.step_text.encode()).hexdigest(), "text_chars": len(ctx.step_text),
            "complexity": d.complexity["score"], "selected_model": d.selected_model,
            "selected_tier": d.selected_tier, "below_floor": int(d.below_floor), "attempt": d.attempt,
            "candidates": json.dumps(d.candidates),
        })

    def record_outcome(self, trace_id: str, **fields: Any) -> None:
        self.telemetry.log_outcome(trace_id, **fields)
