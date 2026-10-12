"""Shadow mode: run the capability router and the selector next to the agent, change nothing, and record what they would do.

At every model-call boundary (the plan and each step) the request goes to Clef, the selector picks an endpoint from the
measured registry, and the decision is logged. The two lanes keep running exactly as before, so nothing here can
alter an outcome or delay a run: decisions are made on a background thread, failures become events, and the final
comparison is made after both lanes have finished.

The cost comparison is a counterfactual and says so: it prices the endpoint the selector chose on the tokens the
always-strongest lane actually used for the same step. Quality is not re-run; it is the selector's expected success from
measured pass rates, shown beside the judge's score of what really ran.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from routing_lab.backends import CAPABILITY_PROMPT_VERSION
from routing_lab.selector import SELECTOR_VERSION, Constraints, Endpoint, MeasuredRegistry, Selection, Settings, select

Emit = Callable[[dict[str, Any]], None]
JOIN_SECONDS = 45.0
REPLAY_JOIN_SECONDS = 240.0
SHOWN_ALTERNATIVES = 4
AUX_SKILLS = {"planner": "reasoning", "judge": "reasoning"}  # the agent's own calls that Clef also routes
DISPLAY = {"gpt-5.6-sol": "GPT-5.6 Sol", "gpt-5.6-terra": "GPT-5.6 Terra", "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.4-nano": "GPT-5.4 Nano",
           "gpt-4.1": "GPT-4.1", "gpt-oss-120b": "gpt-oss-120B", "deepseek-v4-pro": "DeepSeek V4 Pro", "deepseek-v4-flash": "DeepSeek V4 Flash",
           "llama-4-maverick": "Llama 4 Maverick", "phi-4": "Phi-4"}


def display_name(endpoint_id: str) -> str:
    return DISPLAY.get(endpoint_id, endpoint_id)


def endpoint_tier(endpoint: Endpoint) -> str:
    """small, mid or frontier by output price, the same words the quality setting uses to decide which models get a result check."""
    return "small" if endpoint.price_out <= 4 else "mid" if endpoint.price_out <= 15 else "frontier"


@dataclass(frozen=True)
class ShadowItem:
    key: str  # "plan" or the step number
    kind: str  # "plan", "step", "reply", or "planner" / "judge" for the agent's own routed calls
    text: str
    needs_tools: bool
    input_tokens: int
    output_tokens: int


class ShadowSelector:
    """Capability router plus selector. `decide` never raises: an outage becomes an error entry."""

    def __init__(self, router: Any, registry: MeasuredRegistry, *, baseline: str, router_cost_usd: float,
                 log_path: Path | None = None, adapter_sha256: str | None = None, use_skill: bool = True) -> None:
        """`use_skill=False` ignores the skill head, for a router fine-tuned without skill labels (its skill answer is untrained)."""
        self.router, self.registry, self.baseline, self.use_skill = router, registry, baseline, use_skill
        self.router_cost_usd = router_cost_usd
        self.log_path = log_path
        self.versions = {"router": getattr(router, "deployment", None), "adapter_sha256": adapter_sha256,
                         "prompt": CAPABILITY_PROMPT_VERSION, "skill_head_used": use_skill,
                         "registry": registry.version, "selector": SELECTOR_VERSION}
        self.last: dict[str, Any] | None = None  # outcome of the latest router call, for the status badge
        self._lock = threading.Lock()  # the Clef endpoint serves one request at a time
        self._log_lock = threading.Lock()

    def skills_for(self, item: ShadowItem, result: Any) -> dict[str, float] | None:
        """The skill distribution the selector uses.

        Without a trained skill head the agent supplies the skill hint the contract allows, from what the step is for: a step
        that calls tools is tool work and a final answer is writing. This is the agent's own knowledge of the step, not a
        reading of the text.
        """
        if self.use_skill:
            return result.skill_probabilities
        if item.kind in AUX_SKILLS:  # planning and reviewing are reasoning over the request, whatever the request is about
            return {AUX_SKILLS[item.kind]: 1.0}
        if item.needs_tools:
            return {"tools": 1.0}
        return {"writing": 1.0} if item.kind == "reply" else None

    def reselect(self, event: dict[str, Any], item: ShadowItem, settings: Settings, allow: frozenset[str] | None = None) -> Selection:
        """Choose again from the same Clef answer, with a higher floor or fewer endpoints, without calling Clef a second time."""
        skills = event["skill_probabilities"] if self.use_skill else event.get("skill_hint")
        return select(self.registry.endpoints, self.registry.rate, event["tier_probabilities"], skills, settings,
                      Constraints(allow, item.needs_tools, item.input_tokens, item.output_tokens))

    def warm_up(self) -> None:
        """Pay the first call's connection and token setup before a user is waiting (it took about 10 s cold)."""
        started = time.perf_counter()
        try:
            with self._lock:
                result = self.router.classify("Warm up the router.")
            self.last = {"ok": True, "latency_ms": result.latency_ms}
        except Exception:  # noqa: BLE001 - warming is best effort, but the badge should say the router is not answering
            self.last = {"ok": False, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}

    def decide(self, item: ShadowItem, settings: Settings, allow: frozenset[str] | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            with self._lock:
                result = self.router.classify(item.text)
            skills = self.skills_for(item, result)
            selection = select(self.registry.endpoints, self.registry.rate, result.tier_probabilities,
                               skills, settings,
                               Constraints(allow, item.needs_tools, item.input_tokens, item.output_tokens))
        except Exception as error:  # noqa: BLE001 - shadow mode must never break or slow the agent
            self.last = {"ok": False, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
            return {"type": "shadow_error", "key": item.key, "kind": item.kind,
                    "error": f"{type(error).__name__}: {str(error)[:160]}", "versions": self.versions}
        self.last = {"ok": True, "latency_ms": result.latency_ms}
        return self._event(item, result, selection, (time.perf_counter() - started) * 1000, skills)

    def _event(self, item: ShadowItem, result: Any, selection: Selection, total_ms: float,
               skills: dict[str, float] | None = None) -> dict[str, Any]:
        live = [c for c in selection.candidates if c.status != "eliminated"]
        base = next((c for c in selection.candidates if c.endpoint == self.baseline), None)
        return {
            "type": "shadow_decision", "key": item.key, "kind": item.kind, "needs_tools": item.needs_tools,
            "tier": result.tier, "tier_probabilities": result.tier_probabilities, "skill": result.skill,
            "skill_probabilities": result.skill_probabilities, "skill_used": self.use_skill, "skill_hint": None if self.use_skill else skills, "confidence": round(result.confidence, 4),
            "router_ms": result.latency_ms, "decision_ms": round(total_ms, 1),
            "choice": selection.choice, "expected_success": selection.expected_success,
            "floor": selection.floor, "effective_floor": selection.effective_floor, "below_floor": selection.below_floor,
            "uncertain": selection.uncertain, "uncertainty_reasons": selection.uncertainty_reasons,
            "reason": selection.reason,
            "evidence": (self.registry.evidence(selection.choice, result.tier_probabilities, skills)
                         if selection.choice else None),
            "baseline": {"endpoint": self.baseline, "expected_success": base.expected_success if base else None,
                         "eligible": bool(base and base.status != "eliminated")},
            "alternatives": [{"endpoint": c.endpoint, "status": c.status, "expected_success": c.expected_success,
                              "est_cost_usd": c.est_cost_usd, "est_latency_ms": c.est_latency_ms, "utility": c.utility,
                              "price_verified": c.price_verified} for c in live[:SHOWN_ALTERNATIVES]],
            "eliminated": [{"endpoint": c.endpoint, "reasons": c.reasons} for c in selection.candidates if c.status == "eliminated"],
            "versions": self.versions,
        }

    def log(self, row: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        try:
            with self._log_lock:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass  # a full disk must not affect the run

    def endpoint(self, endpoint: str) -> Endpoint | None:
        return next((e for e in self.registry.endpoints if e.id == endpoint), None)

    def price(self, endpoint: str, tokens_in: int, tokens_out: int) -> float | None:
        for e in self.registry.endpoints:
            if e.id == endpoint:
                return tokens_in * e.price_in / 1e6 + tokens_out * e.price_out / 1e6
        return None


class ShadowRun:
    """Decisions for one agent run, made one after another on a background thread."""

    def __init__(self, selector: ShadowSelector, settings: Settings, emit: Emit, task_id: str,
                 allow: frozenset[str] | None = None) -> None:
        self.selector, self.settings, self.emit, self.task_id, self.allow = selector, settings, emit, task_id, allow
        self.items: dict[str, ShadowItem] = {}
        self.decisions: dict[str, dict[str, Any]] = {}
        self.replay: dict[str, Any] = {"steps": {}, "checks": None, "error": None}
        self._thread: threading.Thread | None = None
        self._replay_thread: threading.Thread | None = None
        self._cond = threading.Condition()

    def start(self, items: list[ShadowItem]) -> None:
        self.items = {item.key: item for item in items}
        self._thread = threading.Thread(target=self._work, args=(items,), daemon=True)
        self._thread.start()

    def _work(self, items: list[ShadowItem]) -> None:
        for item in items:
            event = self.selector.decide(item, self.settings, self.allow)
            with self._cond:
                self.decisions[item.key] = event
                self._cond.notify_all()
            self.emit(event)
            self.selector.log({"task_id": self.task_id, "at": round(time.time(), 3), **event})

    def wait(self, key: str, timeout: float = JOIN_SECONDS) -> dict[str, Any] | None:
        """The decision for a call, once it has been made."""
        with self._cond:
            self._cond.wait_for(lambda: key in self.decisions, timeout)
            return self.decisions.get(key)

    def start_replay(self, target: Callable[[], None]) -> None:
        """Run a replay of the chosen endpoints on its own thread; any failure is recorded, never raised."""
        def guarded() -> None:
            try:
                target()
            except Exception as error:  # noqa: BLE001 - the replay is a measurement and must not touch the run
                self.replay["error"] = f"{type(error).__name__}: {str(error)[:160]}"
        self._replay_thread = threading.Thread(target=guarded, daemon=True)
        self._replay_thread.start()

    def finish(self, timeout: float = JOIN_SECONDS) -> dict[str, dict[str, Any]]:
        if self._thread is not None:
            self._thread.join(timeout)
        return dict(self.decisions)

    def finish_replay(self, timeout: float = REPLAY_JOIN_SECONDS) -> dict[str, Any] | None:
        if self._replay_thread is None:
            return None
        self._replay_thread.join(timeout)
        return dict(self.replay, finished=not self._replay_thread.is_alive())


def replay_summary(rows: list[dict[str, Any]], replay: dict[str, Any], checks: dict[str, list[list[int]]], gate_min: int) -> dict[str, Any]:
    """What happened when the selector's choices were actually run: cost, judged quality and outcome checks.

    Judged scores compare only steps that both the replay and the always-strongest lane scored.
    """
    steps = replay["steps"]
    ran = [s for s in steps.values() if s["status"] == "ok"]
    base_cost_of = {r["key"]: r["actual"]["cost_usd"] for r in rows if "actual" in r}
    base_score_of = {r["key"]: r["actual"]["score"] for r in rows if "actual" in r}
    paired = [(s["score"], base_score_of.get(s["key"])) for s in ran if s["score"] is not None and base_score_of.get(s["key"]) is not None]
    return {
        "steps_run": len(ran), "failed": sum(s["status"] == "failed" for s in steps.values()),
        "skipped": sum(s["status"] == "skipped" for s in steps.values()),
        "cost_usd": round(sum(s["cost_usd"] for s in ran), 6),
        "baseline_cost_usd": round(sum(base_cost_of.get(s["key"], 0.0) for s in ran), 6),
        "judge_cost_usd": round(sum(s["judge_cost_usd"] for s in ran), 6),
        "mean_score": round(sum(a for a, _ in paired) / len(paired), 2) if paired else None,
        "baseline_mean_score": round(sum(b for _, b in paired) / len(paired), 2) if paired else None,
        "scored_steps": len(paired), "below_baseline": sum(a < b for a, b in paired),
        "below_gate": sum(a < gate_min for a, _ in paired), "gate_min_score": gate_min,
        "checks": checks, "error": replay.get("error"), "finished": replay.get("finished", True),
    }


def _saved(base: float, mine: float) -> dict[str, float | None]:
    return {"value": round(base - mine, 6), "pct": round(100 * (base - mine) / base, 1) if base else None}


def summarize(selector: ShadowSelector, decisions: dict[str, dict[str, Any]], actuals: dict[str, dict[str, Any]],
              clef_lane_cost: float, baseline_lane_cost: float, plan_actual: dict[str, Any],
              replay: dict[str, Any] | None = None, checks: dict[str, list[list[int]]] | None = None,
              gate_min: int = 4) -> dict[str, Any]:
    """Line up each shadow decision with what actually ran, and price the selector's choices on the same tokens.

    `actuals` maps a step key to the always-strongest lane's observed tokens and cost and the Clef lane's model.
    """
    rows: list[dict[str, Any]] = []
    est_total = n_decisions = agree = unpriced = 0
    expected, observed_scores = [], []
    for key, decision in decisions.items():
        actual = plan_actual if key == "plan" else actuals.get(key)
        row: dict[str, Any] = {"key": key, "kind": decision["kind"], "decision": decision}
        if decision["type"] == "shadow_decision" and actual is not None:
            est = selector.price(decision["choice"], actual["tokens_in"], actual["tokens_out"]) if decision["choice"] else None
            row.update(actual={k: actual.get(k) for k in ("model", "tokens_in", "tokens_out", "cost_usd", "score", "clef_lane_model")},
                       est_cost_usd=est, router_cost_usd=selector.router_cost_usd)
            if est is None:
                unpriced += 1
            else:
                est_total += est + selector.router_cost_usd
                n_decisions += 1
                row["saved_vs_baseline"] = _saved(actual["cost_usd"], est + selector.router_cost_usd)
            if decision["expected_success"] is not None:
                expected.append(decision["expected_success"])
            if actual.get("clef_lane_model") and actual["clef_lane_model"] == decision["choice"]:
                agree += 1
            if actual.get("score") is not None:
                observed_scores.append(actual["score"])
        rows.append(row)
    for row in rows:
        if replay is not None:
            row["replay"] = replay["steps"].get(row["key"])
    errors = sum(d["type"] == "shadow_error" for d in decisions.values())
    base_total = sum(r["actual"]["cost_usd"] for r in rows if "actual" in r and r.get("est_cost_usd") is not None)
    return {
        "rows": rows, "decisions": n_decisions, "errors": errors, "unpriced": unpriced,
        "selector_est_cost_usd": round(est_total, 6), "priced_baseline_cost_usd": round(base_total, 6),
        "saved_vs_baseline": _saved(base_total, est_total) if base_total else None,
        "clef_lane_cost_usd": round(clef_lane_cost, 6), "baseline_lane_cost_usd": round(baseline_lane_cost, 6),
        "mean_expected_success": round(sum(expected) / len(expected), 4) if expected else None,
        "same_choice_as_clef_lane": agree, "versions": selector.versions,
        "replay": replay_summary(rows, replay, checks or {}, gate_min) if replay is not None else None,
        "note": "Counterfactual: the endpoint the selector chose, priced on the tokens the always-strongest lane used. "
                + ("Replayed steps were run on the chosen endpoint, judged by the same reviewer, and checked against the same expected outcomes."
                   if replay is not None else "Quality is the selector's expected success from measured pass rates; nothing was re-run."),
    }


def aggregate(path: Path) -> dict[str, Any]:
    """Totals over every logged decision: how the selector behaves across runs."""
    counts: dict[str, int] = {}
    tiers: dict[str, int] = {}
    n = errors = uncertain = below = thin = 0
    expected: list[float] = []
    runs: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("type") == "shadow_error":
            errors += 1
        elif row.get("type") == "shadow_decision":
            n += 1
            runs.add(row.get("task_id", ""))
            counts[row["choice"]] = counts.get(row["choice"], 0) + 1
            tiers[row["tier"]] = tiers.get(row["tier"], 0) + 1
            uncertain += bool(row["uncertain"])
            below += bool(row["below_floor"])
            thin += bool(row.get("evidence") and row["evidence"]["pooled_share"] + row["evidence"]["missing_share"] >= 0.5)
            if row["expected_success"] is not None:
                expected.append(row["expected_success"])
    return {"decisions": n, "errors": errors, "runs": len(runs), "by_endpoint": counts, "by_tier": tiers,
            "uncertain": uncertain, "below_floor": below, "thin_evidence": thin,
            "mean_expected_success": round(sum(expected) / len(expected), 4) if expected else None}
