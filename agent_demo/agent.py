"""An agent that proves the routing idea by running every case twice, in any domain.

One shared plan is executed by two lanes in parallel, each on its own copy of the world:

* **With Clef**: each step is classified by the fine-tuned Clef model, then the policy (shaped by the user's quality,
  latency and cost preferences) picks the model, and a quality check can escalate a weak result.
* **Without Clef**: the common default of sending every step to the strongest model with every tool attached.

Both lanes are scored by the same blind judge and checked against the same expected outcomes, so the savings
shown are measured, not estimated. A domain (support, travel) supplies the tools, prompts and plan shape.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Any

from agent_demo import ROUTER_VERSION
from agent_demo.clef_decider import ClefDecision
from agent_demo.domains import DOMAINS, Domain
from agent_demo.llm import AzureResponsesClient, LLMError, LLMResult
from agent_demo.policy import Policy
from agent_demo.preferences import Preferences
from agent_demo.registry import ModelProfile, Registry
from agent_demo.router import AdaptiveRouter, RoutingContext, RoutingError
from agent_demo.shadow import ShadowItem, ShadowRun, ShadowSelector, display_name, endpoint_tier, summarize
from agent_demo.telemetry import Telemetry
from routing_lab.selector import Settings

Emit = Callable[[dict[str, Any]], None]
LANES = ("clef", "baseline")
MAX_TOOL_TURNS = 6
REPLAY_MIN_TOKENS = 3000
CLARIFY_CONFIDENCE = 0.8
JUDGE_FLOOR_BONUS = 0.10  # a weak reviewer would let weak work through, so the reviewer is held to a higher bar than the work
PLANNER_TOKENS, JUDGE_TOKENS = 4000, 3000  # reasoning models spend output on thinking before the answer
STRUCTURED_OUTPUT = Path(__file__).resolve().parent / "data" / "structured_output.json"


def load_json_exclusions(path: Path = STRUCTURED_OUTPUT) -> dict[str, frozenset[str]]:
    """Endpoints that did not return usable JSON on every probe call, for the planner and for the reviewer.

    The measured registry says how often an endpoint solves a task, not whether it can answer in JSON, so this comes from
    `python -m agent_demo.probe_structured`. No file means no exclusions; the fixed model is still the fallback.
    """
    try:
        endpoints = json.loads(path.read_text(encoding="utf-8-sig"))["endpoints"]
    except (OSError, ValueError, KeyError):
        return {"planner": frozenset(), "judge": frozenset()}
    return {role: frozenset(name for name, row in endpoints.items() if row[role][0] < row[role][1]) for role in ("planner", "judge")}


def cost_of(model: ModelProfile, tokens_in: int, tokens_out: int) -> float:
    return tokens_in * model.price_in / 1e6 + tokens_out * model.price_out / 1e6


SPECIALISTS = (("flight", "flights"), ("hotel", "hotels"), ("weather", "weather"), ("book", "booking"),
               ("hold", "booking"), ("restaurant", "dining"), ("dining", "dining"), ("food", "dining"),
               ("destination", "destination"), ("sight", "sights"), ("attraction", "sights"))


def normalize_specialist(label: Any, instruction: str) -> str:
    """Planners sometimes name a tool instead of a role; map either to one of a few readable specialists."""
    text = f"{label or ''}".lower()
    for needle, name in SPECIALISTS:
        if needle in text:
            return name
    for needle, name in SPECIALISTS:
        if needle in instruction.lower()[:80]:
            return name
    return "research"


def parse_plan(text: str, max_steps: int, *, parallel: bool = False) -> list[dict[str, Any]]:
    """Validate the planner's JSON; anything malformed raises ValueError so the caller can fall back."""
    try:
        raw = json.loads(text)["steps"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError("Planner did not return a steps list") from error
    steps: list[dict[str, Any]] = []
    for item in raw[:max_steps] if isinstance(raw, list) else []:
        instruction = str(item.get("instruction", "")).strip() if isinstance(item, dict) else ""
        if instruction:
            specialist = normalize_specialist(item.get("specialist"), instruction) if parallel else ""
            steps.append({"title": str(item.get("title") or instruction[:40]).strip()[:80],
                          "instruction": instruction[:3000], "reply": item.get("reply") is True,
                          "specialist": specialist,
                          "parallel": parallel and item.get("parallel") is True and specialist != "booking"})
    if not steps:
        raise ValueError("Planner returned no usable steps")
    for step in steps[:-1]:
        step["reply"] = False
    if not steps[-1]["reply"]:
        if len(steps) == max_steps:
            steps[-1]["reply"] = True
        else:
            steps.append({"title": "Reply to the customer" if not parallel else "Write the itinerary", "reply": True,
                          "specialist": "", "parallel": False,
                          "instruction": "Write the final answer using the results of the earlier steps."})
    steps[-1]["parallel"], steps[-1]["specialist"] = False, ""
    return steps


def fallback_plan(message: str) -> list[dict[str, Any]]:
    return [{"title": "Look into the request", "instruction": message, "reply": False, "specialist": "", "parallel": False},
            {"title": "Reply to the customer", "reply": True, "specialist": "", "parallel": False,
             "instruction": "Write the final answer using the results of the earlier step."}]


def merge_lane(into: dict[str, Any], part: dict[str, Any]) -> None:
    """Fold one step's own totals into its lane (steps of a parallel group run on separate threads)."""
    for key in ("exec_cost", "check_cost", "router_cost", "eval_cost", "tokens_in", "tokens_out", "check_tokens",
                "clef_tokens", "latency_ms", "escalations", "failed"):
        into[key] += part[key]
    into["scores"].extend(part["scores"])
    if part["reply_score"] is not None:
        into["reply_score"] = part["reply_score"]
    for model, entry in part["models"].items():
        slot = into["models"].setdefault(model, {"steps": 0, "cost_usd": 0.0})
        slot["steps"] += entry["steps"]
        slot["cost_usd"] = round(slot["cost_usd"] + entry["cost_usd"], 6)


def _fresh_lane() -> dict[str, Any]:
    return {"exec_cost": 0.0, "check_cost": 0.0, "router_cost": 0.0, "eval_cost": 0.0, "tokens_in": 0,
            "tokens_out": 0, "check_tokens": 0, "clef_tokens": 0, "latency_ms": 0.0, "scores": [],
            "reply_score": None, "escalations": 0, "steps_done": 0, "failed": 0, "models": {}}


def lane_view(t: dict[str, Any]) -> dict[str, Any]:
    scores = t["scores"]
    return {"cost_usd": round(t["exec_cost"] + t["check_cost"] + t["router_cost"], 6),
            "exec_cost_usd": round(t["exec_cost"], 6), "check_cost_usd": round(t["check_cost"], 6),
            "router_cost_usd": round(t["router_cost"], 6), "eval_cost_usd": round(t["eval_cost"], 6),
            "tokens_in": t["tokens_in"], "tokens_out": t["tokens_out"], "check_tokens": t["check_tokens"],
            "exec_tokens": t["tokens_in"] + t["tokens_out"],
            "tokens_total": t["tokens_in"] + t["tokens_out"] + t["check_tokens"], "clef_tokens": t["clef_tokens"],
            "latency_ms": round(t["latency_ms"], 1),
            "avg_score": round(sum(scores) / len(scores), 2) if scores else None,
            "reply_score": t["reply_score"], "escalations": t["escalations"], "steps_done": t["steps_done"],
            "failed_steps": t["failed"], "models": t["models"]}


def _saved(base: float, mine: float) -> dict[str, float | None]:
    return {"value": round(base - mine, 6), "pct": round(100 * (base - mine) / base, 1) if base else None}


def _blank_view() -> dict[str, Any]:
    return {"model": None, "tier": None, "cost_usd": 0.0, "tokens": 0, "exec_tokens": 0, "check_tokens": 0,
            "router_cost_usd": 0.0, "tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0, "score": None, "failed": False, "attempts": 0, "route": None, "confidence": None, "output": "", "specialist": ""}


class Agent:
    def __init__(self, router: AdaptiveRouter, llm: AzureResponsesClient, registry: Registry, policy: Policy,
                 telemetry: Telemetry, domains: dict[str, Domain] | None = None,
                 shadow: ShadowSelector | None = None, shadow_replay: bool = False, apply_selector: bool = False,
                 route_planner: bool = False, route_judge: bool = False) -> None:
        self.router, self.llm, self.registry, self.policy = router, llm, registry, policy
        self.telemetry, self.domains, self.shadow = telemetry, domains or DOMAINS, shadow
        # With apply_selector the Clef lane runs on what the capability router and selector choose, so a replay of the same
        # choices on a second desk would only repeat that work.
        self.apply_selector = apply_selector and shadow is not None
        self.shadow_replay = shadow_replay and shadow is not None and not self.apply_selector
        # The planner and the reviewer are model calls too. When routed, Clef reads the planning or review task and the selector
        # picks the endpoint; the fixed models in the policy stay as the fallback if that endpoint or Clef fails.
        self.route_planner = route_planner and self.apply_selector
        self.route_judge = route_judge and self.apply_selector
        self.json_excluded = load_json_exclusions()

    # ------------------------------------------------------------------ public

    def run(self, message: str, *, prefs: Preferences, expectations: list[dict[str, Any]] | None = None,
            agent: str = "support", emit: Emit = lambda event: None) -> dict[str, Any]:
        limits = self.policy.limits
        if agent not in self.domains:
            raise ValueError(f"Unknown agent {agent!r}")
        domain = self.domains[agent]
        if not message.strip() or len(message) > limits["max_task_chars"]:
            raise ValueError(f"The request must be 1 to {limits['max_task_chars']} characters")
        derived = self.policy.derive(prefs)
        task_id = "tk_" + uuid.uuid4().hex[:10]
        started = time.perf_counter()
        baseline = self._price(self.policy.baseline_model)
        emit({"type": "run_started", "task_id": task_id, "agent": domain.id, "request": message[:900],
              "preferences": prefs.to_dict(), "mode": {"quality_floor": derived.mode.quality_floor, "cost_weight": derived.mode.lambda_cost,
                       "latency_weight": derived.mode.mu_latency, "gate_min_score": derived.gate_min_score},
              "baseline_model": baseline.id, "router_version": ROUTER_VERSION,
              "registry_version": self.registry.version, "policy_version": self.policy.version})
        steps, plan_cost, plan_tokens = self._plan(message, emit, domain, derived, task_id)
        world = domain.new_shared()
        shared = {"task_id": task_id, "message": message, "derived": derived, "steps": steps, "domain": domain,
                  "pending": {}, "lock": threading.Lock(), "world": world,
                  "lane_locks": {lane: threading.Lock() for lane in LANES}}
        desks = {lane: domain.make_desk(self.llm, world) for lane in LANES}
        shadow_run = self._start_shadow(task_id, message, steps, derived, plan_tokens, domain, emit, shared, expectations or [])
        shared["shadow"] = shadow_run
        totals = {lane: _fresh_lane() for lane in LANES}
        workers = [threading.Thread(target=self._guarded, args=(lane, desks[lane], totals[lane], shared, emit),
                                    daemon=True) for lane in LANES]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        return self._finish(task_id, started, plan_cost, totals, desks, expectations or [], emit, prefs, domain, world,
                            shadow=(shadow_run, shared["pending"], plan_tokens))

    # ----------------------------------------------------------------- internals

    def _price(self, model_id: str) -> ModelProfile:
        return self.registry.get(model_id)

    def _route_aux(self, role: str, text: str, derived: Any, task_id: str, tokens: tuple[int, int], bonus: float = 0.0) -> dict[str, Any]:
        """Ask Clef which endpoint should do one of the agent's own calls (the planner or the reviewer).

        Returns the decision for the screen. `endpoint` is None when Clef did not answer or nothing qualified; the caller then
        uses the fixed model from the policy.
        """
        assert self.shadow is not None
        base = Settings(derived.mode.quality_floor, derived.mode.lambda_cost, derived.mode.mu_latency)
        settings = replace(base, floor=min(0.97, base.floor + bonus))
        left_out = self.json_excluded[role]
        allow = frozenset(e.id for e in self.shadow.registry.endpoints if e.api == "responses") - left_out
        event = self.shadow.decide(ShadowItem(role, role, text, False, *tokens), settings, allow)
        self.shadow.log({"task_id": task_id, **event})
        if event["type"] != "shadow_decision" or not event["choice"]:
            reason = event.get("error") or "no endpoint was eligible"
            return {"role": role, "endpoint": None, "source": "fallback", "error": reason, "router_ms": 0.0, "router_cost_usd": 0.0}
        top = event["tier_probabilities"]
        return {"role": role, "endpoint": self.shadow.endpoint(event["choice"]), "source": "clef", "model": event["choice"],
                "display": display_name(event["choice"]), "route": event["tier"], "probabilities": top, "confidence": event["confidence"],
                "reason": event["reason"], "floor": round(settings.floor, 4), "left_out_for_json": sorted(left_out), "expected_success": event["expected_success"],
                "router_ms": event["router_ms"], "router_cost_usd": self.policy.router_cost_usd}

    def _judge_choice(self, event: dict[str, Any], derived: Any) -> dict[str, Any]:
        """Pick the reviewer for a step from the answer Clef already gave for that step, so checking costs no extra Clef call.

        The work being checked and the check are about equally hard, so the reviewer is chosen for the same capability need. It is
        chosen as reasoning work (judging, not doing the step), and held to a higher bar than the work: a reviewer that is too weak
        lets weak work through.
        """
        assert self.shadow is not None
        base = Settings(derived.mode.quality_floor, derived.mode.lambda_cost, derived.mode.mu_latency)
        settings = replace(base, floor=min(0.97, base.floor + JUDGE_FLOOR_BONUS))
        left_out = self.json_excluded["judge"]
        allow = frozenset(e.id for e in self.shadow.registry.endpoints if e.api == "responses") - left_out
        reasoning = {"reasoning": 1.0}
        selection = self.shadow.reselect({**event, "skill_probabilities": reasoning, "skill_hint": reasoning},
                                         ShadowItem("judge", "judge", "", False, 1500, 150), settings, allow)
        if selection.choice is None:
            return {"role": "judge", "endpoint": None, "source": "fallback", "error": "no endpoint was eligible", "router_ms": 0.0,
                    "router_cost_usd": 0.0}
        return {"role": "judge", "endpoint": self.shadow.endpoint(selection.choice), "source": "clef", "model": selection.choice,
                "display": display_name(selection.choice), "route": event["tier"], "probabilities": event["tier_probabilities"],
                "confidence": event["confidence"], "reason": selection.reason, "floor": round(settings.floor, 4),
                "expected_success": selection.expected_success, "left_out_for_json": sorted(left_out), "router_ms": 0.0, "router_cost_usd": 0.0}

    def _plan(self, message: str, emit: Emit, domain: Domain, derived: Any = None,
              task_id: str = "") -> tuple[list[dict[str, Any]], float, tuple[int, int]]:
        fixed = self._price(self.policy.planner_model)
        system = domain.planner_system.format(
            max_steps=domain.max_steps, tools=", ".join(f"{t['name']} ({t['description']})" for t in domain.tools))
        routing = None
        if self.route_planner and derived is not None:
            framed = f"Task: split this request into ordered steps for an assistant with tools.\n\nRequest:\n{message[:3000]}"
            routing = self._route_aux("planner", framed, derived, task_id, ((len(system) + len(message)) // 4, 700))
        attempts = [routing["endpoint"], fixed] if routing and routing["endpoint"] is not None else [fixed]
        note, cost, latency, tokens, used = None, 0.0, 0.0, (0, 0), fixed
        steps: list[dict[str, Any]] | None = None
        for position, model in enumerate(attempts):
            try:
                result = self.llm.complete(model.deployment, system=system, user=message, json_object=True,
                                           max_output_tokens=2000 if model is fixed else PLANNER_TOKENS)
            except LLMError as error:
                note = f"The planner was unavailable or invalid ({str(error)[:80]}), so the case was handled in two steps."
                continue
            cost += cost_of(model, result.input_tokens, result.output_tokens)
            latency += result.latency_ms
            tokens, used = (result.input_tokens, result.output_tokens), model
            try:
                parsed = parse_plan(result.text, domain.max_steps, parallel=domain.parallel)
                if model is not fixed and len(parsed) < 2:  # a cheaper planner once folded the whole case into the reply step, which fails both lanes
                    raise ValueError("the plan had no working step before the reply")
                steps, note = parsed, None
                break
            except ValueError as error:
                note = f"The planner was unavailable or invalid ({str(error)[:80]}), so the case was handled in two steps."
        if steps is None:
            steps = fallback_plan(message)
        elif position > 0 and routing:
            note = f"The planner Clef chose ({routing['display']}) did not give a usable plan, so {fixed.id} planned it."
        event: dict[str, Any] = {"type": "plan", "steps": steps, "planner_model": used.id, "note": note}
        if routing:
            router_cost = routing["router_cost_usd"] if routing["source"] == "clef" else 0.0
            cost += router_cost
            latency += routing["router_ms"]
            event["routing"] = {k: v for k, v in routing.items() if k != "endpoint"}
            event["fixed_model"] = fixed.id
            event["fixed_cost_usd"] = round(cost_of(fixed, *tokens), 6)  # what the fixed planner would have cost for the same tokens
        event.update(cost_usd=round(cost, 6), latency_ms=latency)
        emit(event)
        return steps, cost, tokens
    def _start_shadow(self, task_id: str, message: str, steps: list[dict[str, Any]], derived: Any,
                      plan_tokens: tuple[int, int], domain: Domain, emit: Emit, shared: dict[str, Any],
                      expectations: list[dict[str, Any]]) -> ShadowRun | None:
        """Start the shadow selector on the plan and every step. It only observes; nothing it does reaches a lane."""
        if self.shadow is None:
            return None
        settings = Settings(derived.mode.quality_floor, derived.mode.lambda_cost, derived.mode.mu_latency)
        items = [] if self.route_planner else [ShadowItem("plan", "plan", message, False, plan_tokens[0] or 600, plan_tokens[1] or 400)]
        for number, step in enumerate(steps, start=1):
            text = step["instruction"]
            if step["reply"]:  # a reply step only says "write the answer"; the customer's request carries its difficulty
                text = f"{text}\n\nRequest being answered:\n{message[:1500]}"
            items.append(ShadowItem(str(number), "reply" if step["reply"] else "step", text, not step["reply"],
                                    min(len(text) // 4 + 600, 4000), domain.reply_tokens if step["reply"] else 400))
        allow = frozenset(e.id for e in self.shadow.registry.endpoints if e.api == "responses") if self.apply_selector else None
        run = ShadowRun(self.shadow, settings, emit, task_id, allow)
        emit({"type": "shadow_started", "versions": self.shadow.versions, "decisions": len(items), "applied": self.apply_selector,
              "settings": {"floor": settings.floor, "cost_weight": settings.cost_weight, "latency_weight": settings.latency_weight}})
        run.start(items)
        if self.shadow_replay:
            run.start_replay(lambda: self._replay(run, shared, expectations, emit))
        return run

    def _guarded(self, lane: str, desk: Any, totals: dict[str, Any], shared: dict[str, Any],
                 emit: Emit) -> None:
        try:
            emit({"type": "lane_started", "lane": lane})
            self._run_lane(lane, desk, totals, shared, emit)
        except Exception as error:  # noqa: BLE001 - one lane failing must not hide the other
            emit({"type": "lane_error", "lane": lane, "error": f"{type(error).__name__}: {str(error)[:200]}"})
        emit({"type": "lane_done", "lane": lane, "totals": lane_view(totals)})

    def _run_lane(self, lane: str, desk: Any, totals: dict[str, Any], shared: dict[str, Any], emit: Emit) -> None:
        domain: Domain = shared["domain"]
        steps = shared["steps"]
        summaries: list[str] = []
        i = 0
        while i < len(steps):
            j = i + 1
            if steps[i]["parallel"]:
                while j < len(steps) and steps[j]["parallel"]:
                    j += 1
            group = list(range(i, j))
            snapshot = list(summaries)
            if len(group) == 1:
                views = [self._do_step(lane, k + 1, steps[k], shared, desk, totals, snapshot, emit) for k in group]
            else:
                with ThreadPoolExecutor(max_workers=len(group)) as pool:
                    views = list(pool.map(lambda k: self._do_step(lane, k + 1, steps[k], shared, desk, totals, snapshot, emit), group))
            if len(group) > 1:
                with shared["lane_locks"][lane]:
                    totals["latency_ms"] += max(v["lane_ms"] for v in views)
                    snapshot_totals = lane_view(totals)
                emit({"type": "lane_totals", "lane": lane, "totals": snapshot_totals})
            for k, view in zip(group, views, strict=True):
                summaries.append(f"Step {k + 1} ({steps[k]['title']}): {view['output'][:domain.summary_chars]}")
            i = j

    def _do_step(self, lane: str, index: int, step: dict[str, Any], shared: dict[str, Any], desk: Any,
                 totals: dict[str, Any], summaries: list[str], emit: Emit) -> dict[str, Any]:
        emit({"type": "step_started", "lane": lane, "index": index, "title": step["title"],
              "instruction": step["instruction"], "reply": step["reply"], "specialist": step["specialist"],
              "parallel": step["parallel"]})
        local = _fresh_lane()
        step_fn = self._baseline_step if lane == "baseline" else self._selector_step if self.apply_selector else self._clef_step
        view = step_fn(index, step, shared, desk, local, summaries, emit)
        view["specialist"] = step["specialist"]
        view["lane_ms"] = local["latency_ms"]
        if step["parallel"]:
            local["latency_ms"] = 0.0  # added once for the whole group by _run_lane
        with shared["lane_locks"][lane]:
            merge_lane(totals, local)
            totals["steps_done"] += 1
            snapshot = lane_view(totals)
        emit({"type": "step_done", "lane": lane, "index": index,
              "step": {k: v for k, v in view.items() if k != "output"}, "totals": snapshot})
        self._compare(index, lane, view, shared, emit)
        return view

    @staticmethod
    def _compare(index: int, lane: str, view: dict[str, Any], shared: dict[str, Any], emit: Emit) -> None:
        """Once both lanes have finished a step, emit what the Clef lane saved on it."""
        with shared["lock"]:
            pair = shared["pending"].setdefault(index, {})
            pair[lane] = view
            if len(pair) < len(LANES):
                return
        clef, base = pair["clef"], pair["baseline"]

        def pick(v: dict[str, Any]) -> dict[str, Any]:
            return {k: v[k] for k in ("model", "tier", "cost_usd", "tokens", "exec_tokens", "check_tokens",
                                      "router_cost_usd", "latency_ms", "score", "failed")}

        emit({"type": "step_compare", "index": index, "clef": pick(clef), "baseline": pick(base),
              "saved": {"cost": _saved(base["cost_usd"], clef["cost_usd"]),
                        "tokens": _saved(base["tokens"], clef["tokens"]),
                        "latency": _saved(base["latency_ms"], clef["latency_ms"])}})

    # ---- lane: with Clef

    def _clef_step(self, index: int, step: dict[str, Any], shared: dict[str, Any], desk: Any,
                   totals: dict[str, Any], summaries: list[str], emit: Emit) -> dict[str, Any]:
        derived, task_id, domain = shared["derived"], shared["task_id"], shared["domain"]
        step_id = f"{task_id}.s{index}"
        excluded: set[str] = set()
        clef: ClefDecision | None = None
        view = _blank_view()
        max_escalations = self.policy.escalation["max_escalations"]
        for attempt in range(max_escalations + 1):
            ctx = RoutingContext(
                step_text=step["instruction"], domain="tools", mode=derived.mode, case_text=shared["message"],
                customer_facing=step["reply"], task_id=task_id, step_id=step_id, tools=domain.router_tools,
                prior_failures=attempt, exclude=frozenset(excluded), prior_clef=clef, attempt=attempt,
                stakes=None if domain.stakes is None else (not step["reply"] and bool(domain.stakes.search(step["instruction"]))),
                preceding=tuple(s["instruction"] for s in shared["steps"][:index - 1] if not s["parallel"]))
            try:
                decision = self.router.route(ctx)
            except RoutingError as error:
                emit({"type": "step_failed", "lane": "clef", "index": index, "error": str(error)})
                totals["failed"] += 1
                view.update(failed=True, output=f"Failed: {error}")
                return view
            clef = ClefDecision(**decision.clef)
            first = attempt == 0
            router_cost = self.policy.router_cost_usd if clef.source == "clef" and first else 0.0
            totals["router_cost"] += router_cost
            totals["clef_tokens"] += (clef.input_tokens or 0) if first else 0
            view["cost_usd"] += router_cost
            view["router_cost_usd"] += router_cost
            emit({"type": "decision", "lane": "clef", "index": index, "attempt": attempt,
                  "decision": decision.to_dict(), "router_cost_usd": round(router_cost, 6)})
            model = self._price(decision.selected_model)
            # Planner-marked work steps keep their tools unless Clef is confident that details are missing. Later steps
            # may get their details from earlier results, so only the first step can turn into a question.
            asks = not decision.profile["executes"] and (clef.confidence or 0) >= CLARIFY_CONFIDENCE and not summaries
            style = "reply" if step["reply"] else "clarify" if asks else "tools"
            changes_before = len(desk.changes())
            try:
                run = self._execute(model, style, step, shared["message"], summaries, desk, domain,
                                    domain.reply_tokens if step["reply"] else max(1500, int(decision.profile["out_tokens"]) * 4))
            except LLMError as error:
                self.router.record_outcome(decision.trace_id, outcome="error_rerouted", exec_ms=0.0)
                emit({"type": "exec_error", "lane": "clef", "index": index, "model": model.id,
                      "error": str(error)[:200]})
                excluded.add(model.id)
                if attempt >= max_escalations:
                    totals["failed"] += 1
                    view.update(failed=True, output=f"Failed: {str(error)[:120]}")
                    return view
                totals["escalations"] += 1
                continue
            cost = cost_of(model, run["input_tokens"], run["output_tokens"])
            step_ms = decision.router_ms + run["latency_ms"]
            totals["exec_cost"] += cost
            totals["tokens_in"] += run["input_tokens"]
            totals["tokens_out"] += run["output_tokens"]
            totals["latency_ms"] += step_ms
            counted = totals["models"].setdefault(model.id, {"steps": 0, "cost_usd": 0.0})
            counted["steps"] += 1
            counted["cost_usd"] = round(counted["cost_usd"] + cost, 6)
            view.update(model=model.id, tier=model.tier, route=clef.route, confidence=clef.confidence,
                        attempts=attempt + 1, output=run["text"])
            view["cost_usd"] += cost
            view["tokens"] += run["input_tokens"] + run["output_tokens"]
            view["exec_tokens"] += run["input_tokens"] + run["output_tokens"]
            view["tokens_in"] += run["input_tokens"]
            view["tokens_out"] += run["output_tokens"]
            view["latency_ms"] += step_ms
            emit({"type": "exec", "lane": "clef", "index": index, "attempt": attempt, "model": model.id,
                  "tier": model.tier, "style": style, "latency_ms": run["latency_ms"], "router_ms": decision.router_ms,
                  "input_tokens": run["input_tokens"], "output_tokens": run["output_tokens"],
                  "cost_usd": round(cost, 6), "output": run["text"][:self._cap(domain, step)], "tool_calls": run["tool_calls"]})
            fields: dict[str, Any] = {
                "exec_ms": run["latency_ms"], "input_tokens": run["input_tokens"],
                "output_tokens": run["output_tokens"], "cost_usd": round(cost, 8),
                "baseline_cost_usd": round(cost_of(self._price(self.policy.baseline_model), run["input_tokens"],
                                                   run["output_tokens"]), 8)}
            # Only the tiers named by the quality setting are gated; other checks are measurement and cost nothing here.
            gate_active = model.tier in derived.gate_tiers
            judged = self._judge(shared["message"], step, run["text"], desk, domain, emit, "clef", index,
                                 role="gate" if gate_active else "evaluation")
            if judged["score"] is not None:
                fields.update(judge_score=judged["score"], judge_cost_usd=round(judged["cost"], 8))
            if gate_active:
                totals["check_cost"] += judged["cost"]
                totals["check_tokens"] += judged["tokens"]
                totals["latency_ms"] += judged["latency_ms"]
                view["cost_usd"] += judged["cost"]
                view["tokens"] += judged["tokens"]
                view["check_tokens"] += judged["tokens"]
                view["latency_ms"] += judged["latency_ms"]
            else:
                totals["eval_cost"] += judged["cost"]
            view["score"] = judged["score"]
            weak = judged["score"] is not None and judged["score"] < derived.gate_min_score
            acted = len(desk.changes()) > changes_before
            own_fault = judged.get("fault", "this_output") == "this_output"
            if weak and not own_fault:
                emit({"type": "note", "lane": "clef", "index": index,
                      "text": "The check traced the problem to earlier work, so retrying this step would not help."})
            if weak and own_fault and gate_active and attempt < max_escalations and not acted:
                self.router.record_outcome(decision.trace_id, outcome="escalated", **fields)
                excluded.add(model.id)
                totals["escalations"] += 1
                emit({"type": "escalation", "lane": "clef", "index": index, "from_model": model.id,
                      "attempt": attempt + 1,
                      "reason": f"The check scored {judged['score']}/5, under the {derived.gate_min_score}/5 your "
                                f"quality setting needs, so the step moves to a stronger model."})
                continue
            if weak and acted:
                emit({"type": "note", "lane": "clef", "index": index,
                      "text": "The weak result was kept because the step already changed the store and must not repeat."})
            self.router.record_outcome(decision.trace_id, outcome="weak" if weak else "ok", **fields)
            break
        self._finish_scores(step, view, totals)
        return view

    # ---- lane: with Clef, applying the capability router and the selector

    def _selector_decision(self, event: dict[str, Any] | None, selection: Any, settings: Settings, base_floor: float, attempt: int,
                           error: str | None) -> dict[str, Any]:
        """The decision in the shape the screen already renders: what Clef saw, what was needed, every endpoint considered."""
        fallback = selection is None
        if fallback:
            probabilities, route, confidence, router_ms = {}, "high", None, 0.0
        else:
            probabilities, route = event["tier_probabilities"], event["tier"]
            confidence, router_ms = event["confidence"], event["router_ms"] if attempt == 0 else 0.0
        parts = [["quality floor", base_floor]]
        if not fallback and selection.uncertain and selection.effective_floor > selection.floor:
            parts.append(["router unsure", round(selection.effective_floor - selection.floor, 4)])
        if attempt:
            parts.append(["moved up after a weak result", round(settings.floor - base_floor, 4)])
        chosen = self.policy.baseline_model if fallback else selection.choice
        endpoint = self.shadow.endpoint(chosen)
        candidates = [] if fallback else [
            {"model_id": c.endpoint, "display": display_name(c.endpoint), "tier": endpoint_tier(self.shadow.endpoint(c.endpoint)),
             "status": c.status, "reasons": c.reasons, "quality": c.expected_success, "est_cost": c.est_cost_usd,
             "est_latency_ms": c.est_latency_ms, "utility": c.utility} for c in selection.candidates if self.shadow.endpoint(c.endpoint)]
        reason = ("Clef did not answer" + (f" ({error})" if error else "") + ", so the strongest model was used.") if fallback else selection.reason
        return {"selected_model": chosen, "selected_tier": endpoint_tier(endpoint), "reason": reason, "candidates": candidates,
                "requirement": {"total": round(sum(v for _, v in parts), 4), "parts": parts}, "router_ms": router_ms, "clef_ms": router_ms,
                "clef": {"taxonomy_id": "capability-v1", "route": route, "probabilities": probabilities, "confidence": confidence,
                         "source": "fallback" if fallback else "clef", "error": error if fallback else None},
                "below_floor": False if fallback else selection.below_floor}

    def _selector_step(self, index: int, step: dict[str, Any], shared: dict[str, Any], desk: Any, totals: dict[str, Any],
                       summaries: list[str], emit: Emit) -> dict[str, Any]:
        derived, domain, message = shared["derived"], shared["domain"], shared["message"]
        run: ShadowRun = shared["shadow"]
        item = run.items[str(index)]
        event = run.wait(str(index))
        answered = event is not None and event.get("type") == "shadow_decision"
        error = None if answered else (event or {}).get("error", "no answer in time")
        responses = frozenset(e.id for e in self.shadow.registry.endpoints if e.api == "responses")
        base = run.settings
        excluded: set[str] = set()
        view = _blank_view()
        max_escalations = self.policy.escalation["max_escalations"]
        for attempt in range(max_escalations + 1):
            settings = Settings(min(0.97, base.floor + self.policy.escalation["requirement_step"] * attempt), base.cost_weight,
                                base.latency_weight, base.confidence_threshold, base.high_tier_probability, base.uncertainty_margin)
            selection = self.shadow.reselect(event, item, settings, responses - excluded) if answered else None
            if answered and selection.choice is None:
                if attempt:  # nothing stronger is left to try: keep the last result
                    break
                selection, error = None, "no endpoint is eligible"
            decision = self._selector_decision(event if answered else None, selection, settings, base.floor, attempt, error)
            model = self.shadow.endpoint(decision["selected_model"])
            first = attempt == 0
            router_cost = self.policy.router_cost_usd if answered and first else 0.0
            totals["router_cost"] += router_cost
            view["cost_usd"] += router_cost
            view["router_cost_usd"] += router_cost
            emit({"type": "decision", "lane": "clef", "index": index, "attempt": attempt, "decision": decision,
                  "router_cost_usd": round(router_cost, 6)})
            changes_before = len(desk.changes())
            style = "reply" if step["reply"] else "tools"
            limit = domain.reply_tokens if step["reply"] else max(domain.baseline_tokens, REPLAY_MIN_TOKENS)  # reasoning models spend output on thinking
            try:
                result = self._execute(model, style, step, message, summaries, desk, domain, limit)
            except LLMError as error_:
                emit({"type": "exec_error", "lane": "clef", "index": index, "model": model.id, "error": str(error_)[:200]})
                excluded.add(model.id)
                if attempt >= max_escalations or not answered:
                    totals["failed"] += 1
                    view.update(failed=True, output=f"Failed: {str(error_)[:120]}")
                    return view
                totals["escalations"] += 1
                continue
            cost = cost_of(model, result["input_tokens"], result["output_tokens"])
            step_ms = decision["router_ms"] + result["latency_ms"]
            tier = decision["selected_tier"]
            totals["exec_cost"] += cost
            totals["tokens_in"] += result["input_tokens"]
            totals["tokens_out"] += result["output_tokens"]
            totals["latency_ms"] += step_ms
            counted = totals["models"].setdefault(model.id, {"steps": 0, "cost_usd": 0.0})
            counted["steps"] += 1
            counted["cost_usd"] = round(counted["cost_usd"] + cost, 6)
            view.update(model=model.id, tier=tier, route=decision["clef"]["route"], confidence=decision["clef"]["confidence"],
                        attempts=attempt + 1, output=result["text"])
            view["cost_usd"] += cost
            view["tokens"] += result["input_tokens"] + result["output_tokens"]
            view["exec_tokens"] += result["input_tokens"] + result["output_tokens"]
            view["tokens_in"] += result["input_tokens"]
            view["tokens_out"] += result["output_tokens"]
            view["latency_ms"] += step_ms
            emit({"type": "exec", "lane": "clef", "index": index, "attempt": attempt, "model": model.id, "tier": tier, "style": style,
                  "latency_ms": result["latency_ms"], "router_ms": decision["router_ms"], "input_tokens": result["input_tokens"],
                  "output_tokens": result["output_tokens"], "cost_usd": round(cost, 6),
                  "output": result["text"][:self._cap(domain, step)], "tool_calls": result["tool_calls"]})
            gate_active = tier in derived.gate_tiers
            judge_routing = self._judge_choice(event, derived) if self.route_judge and answered else None
            judged = self._judge(message, step, result["text"], desk, domain, emit, "clef", index, role="gate" if gate_active else "evaluation",
                                 routing=judge_routing)
            if gate_active:
                totals["check_cost"] += judged["cost"]
                totals["check_tokens"] += judged["tokens"]
                totals["latency_ms"] += judged["latency_ms"]
                view["cost_usd"] += judged["cost"]
                view["tokens"] += judged["tokens"]
                view["check_tokens"] += judged["tokens"]
                view["latency_ms"] += judged["latency_ms"]
            else:
                totals["eval_cost"] += judged["cost"]
            view["score"] = judged["score"]
            weak = judged["score"] is not None and judged["score"] < derived.gate_min_score
            acted = len(desk.changes()) > changes_before
            own_fault = judged.get("fault", "this_output") == "this_output"
            if weak and not own_fault:
                emit({"type": "note", "lane": "clef", "index": index,
                      "text": "The check traced the problem to earlier work, so retrying this step would not help."})
            if weak and own_fault and gate_active and answered and attempt < max_escalations and not acted:
                excluded.add(model.id)
                totals["escalations"] += 1
                emit({"type": "escalation", "lane": "clef", "index": index, "from_model": model.id, "attempt": attempt + 1,
                      "reason": f"The check scored {judged['score']}/5, under the {derived.gate_min_score}/5 your quality setting needs, "
                                "so the step moves to a stronger choice."})
                continue
            if weak and acted:
                emit({"type": "note", "lane": "clef", "index": index,
                      "text": "The weak result was kept because the step already changed the store and must not repeat."})
            break
        self._finish_scores(step, view, totals)
        return view

    # ---- lane: without Clef

    def _baseline_step(self, index: int, step: dict[str, Any], shared: dict[str, Any], desk: Any,
                       totals: dict[str, Any], summaries: list[str], emit: Emit) -> dict[str, Any]:
        model = self._price(self.policy.baseline_model)
        style = "reply" if step["reply"] else "tools"
        view = _blank_view()
        view.update(model=model.id, tier=model.tier, attempts=1)
        try:
            run = self._execute(model, style, step, shared["message"], summaries, desk, shared["domain"],
                                shared["domain"].baseline_tokens)
        except LLMError as error:
            emit({"type": "exec_error", "lane": "baseline", "index": index, "model": model.id,
                  "error": str(error)[:200]})
            totals["failed"] += 1
            view.update(failed=True, output=f"Failed: {str(error)[:120]}")
            return view
        cost = cost_of(model, run["input_tokens"], run["output_tokens"])
        totals["exec_cost"] += cost
        totals["tokens_in"] += run["input_tokens"]
        totals["tokens_out"] += run["output_tokens"]
        totals["latency_ms"] += run["latency_ms"]
        counted = totals["models"].setdefault(model.id, {"steps": 0, "cost_usd": 0.0})
        counted["steps"] += 1
        counted["cost_usd"] = round(counted["cost_usd"] + cost, 6)
        view.update(cost_usd=cost, tokens=run["input_tokens"] + run["output_tokens"], tokens_in=run["input_tokens"],
                    tokens_out=run["output_tokens"],
                    exec_tokens=run["input_tokens"] + run["output_tokens"], latency_ms=run["latency_ms"],
                    output=run["text"])
        emit({"type": "exec", "lane": "baseline", "index": index, "attempt": 0, "model": model.id,
              "tier": model.tier, "style": style, "latency_ms": run["latency_ms"], "router_ms": 0.0,
              "input_tokens": run["input_tokens"], "output_tokens": run["output_tokens"],
              "cost_usd": round(cost, 6), "output": run["text"][:self._cap(shared["domain"], step)], "tool_calls": run["tool_calls"]})
        judged = self._judge(shared["message"], step, run["text"], desk, shared["domain"], emit, "baseline", index,
                             role="evaluation")
        totals["eval_cost"] += judged["cost"]
        view["score"] = judged["score"]
        self._finish_scores(step, view, totals)
        return view

    @staticmethod
    def _finish_scores(step: dict[str, Any], view: dict[str, Any], totals: dict[str, Any]) -> None:
        if view["score"] is not None:
            totals["scores"].append(view["score"])
            if step["reply"]:
                totals["reply_score"] = view["score"]

    # ---- shared pieces

    def _execute(self, model: ModelProfile, style: str, step: dict[str, Any], message: str, summaries: list[str],
                 desk: Any, domain: Domain, limit: int) -> dict[str, Any]:
        context = ("Earlier steps:\n" + "\n".join(summaries) + "\n\n") if summaries else ""
        user = (f"Request (for reference; do only your step):\n{message}\n\n{context}"
                f"Your step:\n{step['instruction']}")
        system = {"reply": domain.reply_system, "clarify": domain.clarify_system, "general": domain.general_system,
                  "tools": domain.tools_system}[style]
        if style != "tools":
            result = self.llm.complete(model.deployment, system=system, user=user, max_output_tokens=limit,
                                       json_object=style == "reply" and domain.reply_json)
            return self._pack(result, result.latency_ms, result.input_tokens, result.output_tokens, [])
        calls: list[dict[str, Any]] = []
        total_in = total_out = 0
        latency = 0.0
        result = self.llm.complete(model.deployment, system=system, user=user, tools=domain.responses_tools,
                                   max_output_tokens=limit)
        for turn in range(MAX_TOOL_TURNS + 1):
            total_in += result.input_tokens
            total_out += result.output_tokens
            latency += result.latency_ms
            if not result.function_calls or turn == MAX_TOOL_TURNS:
                break
            outputs = []
            for call in result.function_calls:
                outcome = desk.execute(call["name"], call["arguments"])
                calls.append({"name": call["name"], "arguments": call["arguments"], "result": outcome})
                outputs.append({"type": "function_call_output", "call_id": call["call_id"],
                                "output": json.dumps(outcome)})
            result = self.llm.complete(model.deployment, input_items=outputs,
                                       previous_response_id=result.response_id, tools=domain.responses_tools,
                                       max_output_tokens=limit)
        return self._pack(result, latency, total_in, total_out, calls)

    @staticmethod
    def _cap(domain: Domain, step: dict[str, Any]) -> int:
        """How much of a step's output is sent to the screen: a full itinerary is much longer than a tool summary."""
        return 24000 if step["reply"] and domain.reply_json else 5000

    @staticmethod
    def _pack(result: LLMResult, latency: float, tokens_in: int, tokens_out: int,
              calls: list[dict[str, Any]]) -> dict[str, Any]:
        return {"text": result.text, "input_tokens": tokens_in, "output_tokens": tokens_out,
                "latency_ms": round(latency, 1), "tool_calls": calls}

    def _judge(self, message: str, step: dict[str, Any], output: str, desk: Any, domain: Domain, emit: Emit,
               lane: str, index: int, *, role: str, routing: dict[str, Any] | None = None) -> dict[str, Any]:
        judge = self._price(self.policy.judge_model)
        if step["reply"] and domain.validate_reply is not None:
            problem = domain.validate_reply(output)
            if problem:  # a structural failure needs no reviewer
                emit({"type": "judge", "lane": lane, "index": index, "role": role, "score": 1, "reason": problem,
                      "fault": "this_output", "model": "structure check", "cost_usd": 0.0, "latency_ms": 0.0})
                return {"score": 1, "cost": 0.0, "tokens": 0, "latency_ms": 0.0, "fault": "this_output"}
        cap = domain.evidence_chars
        shown = 12000 if step["reply"] and domain.reply_json else 3000
        evidence = json.dumps([{"tool": c["name"], "args": c["arguments"], "result": c["result"]}
                               for c in desk.calls], default=str)[-cap:]
        user = (f"REQUEST:\n{message[:1500]}\n\nSTEP:\n{step['instruction'][:1200]}"
                f"{' (this step is the final answer to the user)' if step['reply'] else ''}\n\n"
                f"TOOL CALLS SO FAR IN THIS CASE:\n{evidence}\n\nOUTPUT:\n{output[:shown]}")
        empty = {"score": None, "cost": 0.0, "tokens": 0, "latency_ms": 0.0}
        models = [routing["endpoint"], judge] if routing and routing["endpoint"] is not None else [judge]
        cost = latency = 0.0
        tokens = 0
        used, score, reason, fault, last_error = judge, None, "The check did not return a usable score.", "this_output", None
        for position, model in enumerate(models):
            try:
                result = self.llm.complete(model.deployment, system=domain.judge_system, user=user, json_object=True,
                                           max_output_tokens=300 if model is judge else JUDGE_TOKENS)
            except LLMError as error:
                last_error = error
                continue
            last_error = None
            used = model
            cost += cost_of(model, result.input_tokens, result.output_tokens)
            latency += result.latency_ms
            tokens += result.input_tokens + result.output_tokens
            try:
                data = json.loads(result.text)
                parsed = int(data["score"])
                score, reason = (parsed if 1 <= parsed <= 5 else None), str(data.get("reason", ""))[:240]
                fault = "earlier_work" if data.get("fault") == "earlier_work" else "this_output"
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                score = None
            if score is not None:
                break
        if last_error is not None and tokens == 0:
            emit({"type": "judge", "lane": lane, "index": index, "role": role, "score": None,
                  "reason": f"The check was unavailable ({str(last_error)[:60]})."})
            return empty
        extra: dict[str, Any] = {}
        if routing:
            router_cost = routing["router_cost_usd"] if routing["source"] == "clef" else 0.0
            cost += router_cost
            latency += routing["router_ms"]
            extra = {"routing": {k: v for k, v in routing.items() if k != "endpoint"}, "router_cost_usd": round(router_cost, 6),
                     "fell_back": used is judge and routing["endpoint"] is not None}
        emit({"type": "judge", "lane": lane, "index": index, "role": role, "score": score, "reason": reason,
              "fault": fault, "model": used.id, "cost_usd": round(cost, 6), "latency_ms": round(latency, 1), **extra})
        return {"score": score, "cost": cost, "tokens": tokens, "latency_ms": latency, "fault": fault}
    def _tool_usage(self, world: Any) -> dict[str, Any]:
        """Shared tool spend (web search), reported apart from the comparison because both lanes use the same results."""
        stats = dict(getattr(world, "stats", None) or {})
        if stats.get("web_searches"):
            model = self._price(self.policy.planner_model)
            stats["search_cost_usd"] = round(cost_of(model, stats["search_input_tokens"], stats["search_output_tokens"]), 6)
        return stats

    def _replay(self, run: ShadowRun, shared: dict[str, Any], expectations: list[dict[str, Any]], emit: Emit) -> None:
        """Run the plan on the endpoints the selector chose, on a fresh desk, and judge it like the lanes.

        This is what turns the shadow decisions into evidence: the same steps, the same judge, the same outcome
        checks, but with the selector's endpoint and no retry, so a weak choice shows up instead of being hidden.
        """
        domain, steps, message = shared["domain"], shared["steps"], shared["message"]
        desk = domain.make_desk(self.llm, shared["world"])
        summaries: list[str] = []
        i = 0
        while i < len(steps):
            j = i + 1
            if steps[i]["parallel"]:
                while j < len(steps) and steps[j]["parallel"]:
                    j += 1
            snapshot, produced = list(summaries), []
            for k in range(i, j):  # group members see the same earlier results, as they do in the lanes
                row = self._replay_step(run, k + 1, steps[k], message, desk, domain, snapshot)
                run.replay["steps"][row["key"]] = {key: v for key, v in row.items() if key != "output"}
                emit({"type": "replay_step", **run.replay["steps"][row["key"]]})
                produced.append(f"Step {k + 1} ({steps[k]['title']}): {row.get('output', '')[:domain.summary_chars]}")
            summaries.extend(produced)
            i = j
        if expectations:
            run.replay["checks"] = desk.check(expectations)
        emit({"type": "replay_done", "steps": len(run.replay["steps"])})

    def _replay_step(self, run: ShadowRun, index: int, step: dict[str, Any], message: str, desk: Any, domain: Domain,
                     summaries: list[str]) -> dict[str, Any]:
        key = str(index)
        row: dict[str, Any] = {"key": key, "status": "skipped", "endpoint": None, "reason": "", "tokens_in": 0,
                               "tokens_out": 0, "cost_usd": 0.0, "judge_cost_usd": 0.0, "score": None, "latency_ms": 0.0}
        decision = run.wait(key)
        if decision is None or decision["type"] != "shadow_decision" or not decision["choice"]:
            return {**row, "reason": "The selector made no choice for this step."}
        endpoint = self.shadow.endpoint(decision["choice"]) if self.shadow else None
        row["endpoint"] = decision["choice"]
        if endpoint is None or endpoint.api != "responses":
            return {**row, "reason": "This endpoint needs the chat API, which the agent client does not call."}
        style = "reply" if step["reply"] else "tools"
        limit = max(domain.reply_tokens if step["reply"] else domain.baseline_tokens, REPLAY_MIN_TOKENS)  # reasoning models spend output on thinking
        try:
            result = self._execute(SimpleNamespace(deployment=endpoint.deployment), style, step, message, summaries, desk, domain, limit)
        except LLMError as error:
            return {**row, "status": "failed", "reason": f"The endpoint failed: {str(error)[:100]}"}
        judged = self._judge(message, step, result["text"], desk, domain, lambda event: None, "replay", index, role="evaluation")
        cost = self.shadow.price(decision["choice"], result["input_tokens"], result["output_tokens"]) if self.shadow else None
        return {**row, "status": "ok", "tokens_in": result["input_tokens"], "tokens_out": result["output_tokens"],
                "cost_usd": cost or 0.0, "judge_cost_usd": judged["cost"], "score": judged["score"],
                "latency_ms": result["latency_ms"], "output": result["text"]}

    def _shadow_summary(self, shadow: tuple[ShadowRun | None, dict[int, dict[str, Any]], tuple[int, int]],
                        views: dict[str, dict[str, Any]], plan_cost: float, emit: Emit, task_id: str,
                        checks: dict[str, list[dict[str, Any]]], gate_min: int) -> None:
        """Wait briefly for the shadow decisions, line them up with what really ran, and report the counterfactual."""
        run, pending, plan_tokens = shadow
        if run is None or self.shadow is None:
            return
        decisions = run.finish()
        actuals = {}
        for index, pair in pending.items():
            if "baseline" in pair and "clef" in pair:
                base = pair["baseline"]
                actuals[str(index)] = {"model": base["model"], "tokens_in": base["tokens_in"], "tokens_out": base["tokens_out"],
                                       "cost_usd": base["cost_usd"], "score": base["score"],
                                       "clef_lane_model": pair["clef"]["model"]}
        planner = self._price(self.policy.planner_model)
        plan_actual = {"model": planner.id, "tokens_in": plan_tokens[0], "tokens_out": plan_tokens[1], "cost_usd": plan_cost,
                       "score": None, "clef_lane_model": planner.id}
        replay = run.finish_replay()
        tally = {lane: [sum(c["passed"] for c in rows), len(rows)] for lane, rows in checks.items()}
        if replay is not None and replay.get("checks") is not None:
            tally["replay"] = [sum(c["passed"] for c in replay["checks"]), len(replay["checks"])]
        summary = summarize(self.shadow, decisions, actuals, views["clef"]["cost_usd"] + plan_cost,
                            views["baseline"]["cost_usd"] + plan_cost, plan_actual, replay, tally, gate_min)
        emit({"type": "shadow_summary", **summary})
        self.shadow.log({"type": "shadow_run", "task_id": task_id,
                         **{k: v for k, v in summary.items() if k != "rows"}})

    def _finish(self, task_id: str, started: float, plan_cost: float, totals: dict[str, dict[str, Any]],
                desks: dict[str, Any], expectations: list[dict[str, Any]], emit: Emit,
                prefs: Preferences, domain: Domain, world: Any,
                shadow: tuple[ShadowRun | None, dict[int, dict[str, Any]], tuple[int, int]] = (None, {}, (0, 0))) -> dict[str, Any]:
        views = {lane: lane_view(totals[lane]) for lane in LANES}
        checks = {lane: desks[lane].check(expectations) for lane in LANES} if expectations else {}
        self._shadow_summary(shadow, views, plan_cost, emit, task_id, checks, self.policy.derive(prefs).gate_min_score)
        for lane in LANES:
            emit({"type": "outcome", "lane": lane, "checks": checks.get(lane, []), "changes": desks[lane].changes()})
        clef, base = views["clef"], views["baseline"]
        reply = (None if clef["reply_score"] is None or base["reply_score"] is None
                 else round(clef["reply_score"] - base["reply_score"], 2))
        summary = {
            "task_id": task_id, "agent": domain.id, "preferences": prefs.to_dict(),
            "tool_usage": self._tool_usage(world), "elapsed_s": round(time.perf_counter() - started, 2),
            "lanes": views, "planner_cost_usd": round(plan_cost, 6),
            "saved": {"cost": _saved(base["cost_usd"], clef["cost_usd"]),
                      "tokens": _saved(base["tokens_total"], clef["tokens_total"]),
                      "latency": _saved(base["latency_ms"], clef["latency_ms"])},
            "reply_score_delta": reply,
            "checks_passed": {lane: [sum(c["passed"] for c in rows), len(rows)] for lane, rows in checks.items()},
            "router_cost_note": self.policy.data.get("router_cost_note", ""),
        }
        self.telemetry.log_run(summary)
        emit({"type": "run_done", **summary})
        return summary
