"""The router and the agent loop, end to end with scripted models and a scripted Clef."""

from __future__ import annotations

from pathlib import Path

import pytest
from agent_demo_fakes import FakeDecider, ScriptedLLM, collect, plan_json

from agent_demo.agent import Agent, cost_of, fallback_plan, lane_view, _fresh_lane, parse_plan
from agent_demo.policy import Policy
from agent_demo.registry import Registry
from agent_demo.router import AdaptiveRouter, RoutingContext, RoutingError
from agent_demo.telemetry import Telemetry
from agent_demo.preferences import Preferences

DATA = Path(__file__).resolve().parents[1] / "agent_demo" / "data"


@pytest.fixture
def parts(tmp_path):
    registry = Registry.load(DATA / "registry.json", tmp_path / "r.json")
    policy = Policy.load(DATA / "policy.json")
    telemetry = Telemetry(":memory:")
    return registry, policy, telemetry


def make_router(parts, decider=None):
    registry, policy, telemetry = parts
    return AdaptiveRouter(registry, policy, decider or FakeDecider(), telemetry)


def make_agent(parts, decider=None, llm=None):
    registry, policy, telemetry = parts
    router = AdaptiveRouter(registry, policy, decider or FakeDecider(), telemetry)
    return Agent(router, llm or ScriptedLLM(), registry, policy, telemetry), router


# -------------------------------------------------------------------------------- router
def test_a_decision_carries_versions_candidates_reason_and_a_trace(parts):
    router = make_router(parts)
    d = router.route(RoutingContext(step_text="What is the weather in Dubai?", domain="tools", task_id="t", step_id="t.s1"))
    assert d.selected_model == "gpt-5.6-luna" and d.selected_tier == "small" and d.clef["route"] == "tool_lookup"
    assert (d.router_version, d.registry_version, d.policy_version) == ("aie-poc-0.1.0", parts[0].version, parts[1].version)
    assert len(d.candidates) == 5 and sum(c["status"] == "selected" for c in d.candidates) == 1
    assert "tool_lookup" in d.reason and "GPT-5.6 Luna" in d.reason
    trace = parts[2].trace(d.trace_id)
    assert trace["selected_model"] == "gpt-5.6-luna" and trace["registry_version"] == parts[0].version
    assert [c["model_id"] for c in trace["candidates"]] and "weather" not in str(trace)


def test_the_same_clef_decision_is_reused_when_escalating(parts):
    decider = FakeDecider()
    router = make_router(parts, decider)
    first = router.route(RoutingContext(step_text="Create a ticket.", domain="tools"))
    from agent_demo.clef_decider import ClefDecision

    second = router.route(RoutingContext(step_text="Create a ticket.", domain="tools", prior_failures=1, attempt=1,
                                         prior_clef=ClefDecision(**first.clef), exclude=frozenset({first.selected_model})))
    assert len(decider.calls) == 1 and second.clef_ms == 0.0
    assert second.selected_model != first.selected_model and second.requirement["total"] > first.requirement["total"]


def test_clef_outage_degrades_to_a_safe_default_that_errs_toward_capability(parts):
    ok = make_router(parts).route(RoutingContext(step_text="Fix it", domain="code"))
    down = make_router(parts, FakeDecider(fail=True)).route(RoutingContext(step_text="Fix it", domain="code"))
    assert down.clef["source"] == "fallback" and down.clef["route"] == "debugging"
    assert any("Clef was unavailable" in w for w in down.warnings) and "safe default" in down.reason
    assert any("unsure" in name for name, _ in down.requirement["parts"])
    assert down.requirement["total"] > ok.requirement["total"]


def test_no_eligible_model_uses_the_configured_safe_fallback_then_fails_loudly(parts):
    registry, _, _ = parts
    for model in ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"):
        registry.update(model, healthy=False)
    d = make_router(parts).route(RoutingContext(step_text="Fix it", domain="code", regions={"nowhere"}))
    assert d.selected_model == "gpt-4.1" and d.fallback_used and any("safe fallback" in w for w in d.warnings)
    registry.update("gpt-4.1", healthy=False)
    with pytest.raises(RoutingError, match="No live"):
        make_router(parts).route(RoutingContext(step_text="Fix it", domain="code", regions={"nowhere"}))


def test_routing_rejects_bad_input_before_calling_clef(parts):
    decider = FakeDecider()
    router = make_router(parts, decider)
    for kwargs in ({"step_text": "x", "domain": "pdf"}, {"step_text": "  ", "domain": "code"},
                   {"step_text": "x", "domain": "code", "mode": "turbo"}):
        with pytest.raises(ValueError):
            router.route(RoutingContext(**kwargs))
    assert decider.calls == []


def test_outcomes_are_recorded_against_the_trace(parts):
    router = make_router(parts)
    d = router.route(RoutingContext(step_text="Fix it", domain="code"))
    router.record_outcome(d.trace_id, outcome="ok", cost_usd=0.01)
    assert parts[2].trace(d.trace_id)["outcome"] == "ok"


# ---------------------------------------------------------------------------------- agent
REFUND = {"name": "issue_refund", "arguments": {"order_id": "A-1042", "amount": 138.99, "reason": "damaged"}}
EXPECT_REFUND = [{"label": "refund", "tool": "issue_refund", "args": {"order_id": "A-1042", "amount": 138.99}}]


def run_case(parts, *, decider=None, llm=None, prefs=None, expectations=None, message="Order A-1042 arrived broken."):
    agent, _ = make_agent(parts, decider, llm)
    events: list[dict] = []
    summary = agent.run(message, prefs=prefs or Preferences(), expectations=expectations, emit=collect(events))
    return summary, events


def lane_events(events, lane, kind):
    return [e for e in events if e.get("lane") == lane and e["type"] == kind]


def test_a_case_runs_both_lanes_on_one_plan_and_measures_the_difference(parts):
    summary, events = run_case(parts)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "run_started" and kinds[1] == "plan" and kinds[-1] == "run_done" and kinds.count("plan") == 1
    assert kinds.count("lane_started") == 2 and kinds.count("lane_done") == 2 and kinds.count("step_compare") == 2
    base_models = {e["model"] for e in lane_events(events, "baseline", "exec")}
    clef_models = {e["model"] for e in lane_events(events, "clef", "exec")}
    assert base_models == {"gpt-5.6-sol"} and clef_models != base_models
    assert lane_events(events, "baseline", "decision") == [] and len(lane_events(events, "clef", "decision")) == 2
    assert summary["saved"]["cost"]["value"] > 0 and summary["saved"]["cost"]["pct"] > 0
    clef, base = summary["lanes"]["clef"], summary["lanes"]["baseline"]
    assert clef["steps_done"] == base["steps_done"] == 2 and clef["router_cost_usd"] > 0 and base["router_cost_usd"] == 0
    assert clef["cost_usd"] == pytest.approx(clef["exec_cost_usd"] + clef["check_cost_usd"] + clef["router_cost_usd"], abs=1e-6)
    assert base["tokens_total"] == base["exec_tokens"] and clef["tokens_total"] == clef["exec_tokens"] + clef["check_tokens"]
    assert parts[2].run_aggregates()["runs"] == 1


def test_step_comparison_events_carry_the_per_step_savings(parts):
    _, events = run_case(parts)
    compare = [e for e in events if e["type"] == "step_compare"]
    assert sorted(e["index"] for e in compare) == [1, 2]
    for e in compare:
        assert e["baseline"]["model"] == "gpt-5.6-sol" and e["saved"]["cost"]["value"] == pytest.approx(
            e["baseline"]["cost_usd"] - e["clef"]["cost_usd"], abs=1e-6)
        assert e["saved"]["tokens"]["value"] == e["baseline"]["tokens"] - e["clef"]["tokens"]
        assert {"value", "pct"} <= set(e["saved"]["latency"])


def test_each_lane_gets_its_own_store_and_outcomes_are_checked_per_lane(parts):
    llm = ScriptedLLM(tool_calls=[REFUND])
    summary, events = run_case(parts, llm=llm, expectations=EXPECT_REFUND)
    outcomes = {e["lane"]: e for e in events if e["type"] == "outcome"}
    assert all(o["checks"] == [{"label": "refund", "passed": True}] for o in outcomes.values())
    assert all(o["changes"] == ["Refunded $138.99 on A-1042"] for o in outcomes.values())
    assert summary["checks_passed"] == {"clef": [1, 1], "baseline": [1, 1]}


def test_without_expectations_no_checks_are_reported(parts):
    summary, events = run_case(parts)
    assert summary["checks_passed"] == {} and all(e["checks"] == [] for e in events if e["type"] == "outcome")


def test_a_weak_small_model_result_escalates_and_both_attempts_are_paid_for(parts):
    llm = ScriptedLLM(model_scores={"gpt-5.6-luna": 2})
    summary, events = run_case(parts, llm=llm)
    escalations = lane_events(events, "clef", "escalation")
    assert escalations and "under the 4/5" in escalations[0]["reason"]
    execs = [e for e in lane_events(events, "clef", "exec") if e["index"] == 1]
    assert [e["model"] for e in execs][0] == "gpt-5.6-luna" and execs[-1]["model"] != "gpt-5.6-luna"
    assert summary["lanes"]["clef"]["escalations"] >= 1 and summary["lanes"]["clef"]["check_cost_usd"] > 0
    paid = sum(e["cost_usd"] for e in lane_events(events, "clef", "exec"))
    assert summary["lanes"]["clef"]["exec_cost_usd"] == pytest.approx(paid, abs=1e-5)


def test_quality_setting_decides_which_tiers_are_checked(parts):
    llm = ScriptedLLM(model_scores={"gpt-4.1": 2})
    _, events = run_case(parts, llm=llm, prefs=Preferences(quality="very_good", latency="low", cost="low"))
    assert any(e["role"] == "gate" and e["lane"] == "clef" for e in events if e["type"] == "judge")
    llm = ScriptedLLM()
    summary, events = run_case(parts, llm=llm, prefs=Preferences(quality="average", latency="high", cost="high"))
    roles = {e["role"] for e in lane_events(events, "clef", "judge")}
    assert roles <= {"gate", "evaluation"} and summary["lanes"]["baseline"]["check_cost_usd"] == 0


def test_a_step_that_already_changed_the_store_is_never_retried(parts):
    llm = ScriptedLLM(tool_calls=[REFUND], model_scores={"gpt-5.6-luna": 2})
    summary, events = run_case(parts, llm=llm, expectations=EXPECT_REFUND)
    step1 = [e for e in lane_events(events, "clef", "exec") if e["index"] == 1]
    assert len(step1) == 1 and not [e for e in lane_events(events, "clef", "escalation") if e["index"] == 1]
    assert any("must not repeat" in e["text"] for e in lane_events(events, "clef", "note"))


def test_a_fault_in_earlier_work_does_not_trigger_a_pointless_retry(parts):
    llm = ScriptedLLM(model_scores={"gpt-5.6-luna": 2}, model_faults={"gpt-5.6-luna": "earlier_work"})
    _, events = run_case(parts, llm=llm)
    assert not lane_events(events, "clef", "escalation")
    assert any("earlier work" in e["text"] for e in lane_events(events, "clef", "note"))


def test_clef_outage_falls_back_safely_and_the_lane_still_finishes(parts):
    summary, events = run_case(parts, decider=FakeDecider(fail=True))
    decisions = lane_events(events, "clef", "decision")
    assert all(d["decision"]["clef"]["source"] == "fallback" and d["router_cost_usd"] == 0 for d in decisions)
    assert summary["lanes"]["clef"]["steps_done"] == 2 and summary["lanes"]["clef"]["router_cost_usd"] == 0


def test_a_failing_baseline_lane_does_not_hide_the_clef_lane(parts):
    summary, events = run_case(parts, llm=ScriptedLLM(fail_models={"gpt-5.6-sol"}))
    assert summary["lanes"]["baseline"]["failed_steps"] == 2 and summary["lanes"]["clef"]["failed_steps"] == 0
    assert any(e["type"] == "exec_error" and e["lane"] == "baseline" for e in events)
    assert events[-1]["type"] == "run_done"


def test_a_failing_cheap_model_is_routed_around(parts):
    summary, events = run_case(parts, llm=ScriptedLLM(fail_models={"gpt-5.6-luna"}))
    assert summary["lanes"]["clef"]["failed_steps"] == 0
    assert all(e["model"] != "gpt-5.6-luna" for e in lane_events(events, "clef", "exec"))
    assert summary["lanes"]["clef"]["escalations"] >= 1


def test_confident_clarification_asks_instead_of_acting_but_only_for_store_steps(parts):
    decider = FakeDecider(routes={"tools": "clarification", "code": "debugging"}, confidence=0.95)
    llm = ScriptedLLM()
    _, events = run_case(parts, decider=decider, llm=llm)
    styles = {(e["lane"], e["index"]): e["style"] for e in events if e["type"] == "exec"}
    assert styles[("clef", 1)] == "clarify" and styles[("baseline", 1)] == "tools"
    assert styles[("clef", 2)] == styles[("baseline", 2)] == "reply"
    unsure = FakeDecider(routes={"tools": "clarification", "code": "debugging"}, confidence=0.6)
    _, events = run_case(parts, decider=unsure)
    assert {(e["lane"], e["index"]): e["style"] for e in events if e["type"] == "exec"}[("clef", 1)] == "tools"


def test_tools_are_offered_only_to_store_steps_never_to_replies(parts):
    llm = ScriptedLLM()
    run_case(parts, llm=llm)
    replies = [c for c in llm.calls if c["system"] and c["system"].startswith("You write the final reply")]
    stores = [c for c in llm.calls if c["system"] and c["system"].startswith("You are a support agent")]
    assert replies and not any(c["tools"] for c in replies) and stores and all(c["tools"] for c in stores)


def test_the_preferences_in_force_are_announced_and_change_the_routing(parts):
    lean = Preferences(quality="average", latency="low", cost="low")
    premium = Preferences(quality="very_good", latency="high", cost="high")
    _, lean_events = run_case(parts, prefs=lean)
    _, premium_events = run_case(parts, prefs=premium)
    started = [e for e in lean_events if e["type"] == "run_started"][0]
    assert started["preferences"] == lean.to_dict() and started["mode"]["gate_min_score"] == 3
    tiers = lambda events: {e["tier"] for e in lane_events(events, "clef", "exec")}  # noqa: E731
    assert "small" in tiers(lean_events) and "small" not in tiers(premium_events)


def test_a_broken_planner_falls_back_to_two_steps_and_says_so(parts):
    summary, events = run_case(parts, llm=ScriptedLLM(plan_error=True))
    plan = [e for e in events if e["type"] == "plan"][0]
    assert len(plan["steps"]) == 2 and plan["steps"][1]["reply"] and "planner" in plan["note"].lower()
    assert summary["lanes"]["clef"]["steps_done"] == 2


@pytest.mark.parametrize(("message", "problem"), [("", "1 to"), ("   ", "1 to"), ("x" * 3001, "1 to")])
def test_agent_rejects_bad_requests_before_spending(parts, message, problem):
    llm = ScriptedLLM()
    agent, _ = make_agent(parts, llm=llm)
    with pytest.raises(ValueError, match=problem):
        agent.run(message, prefs=Preferences())
    assert llm.calls == []


def test_plan_parsing_is_strict_and_always_ends_with_one_reply():
    steps = parse_plan(plan_json(("A", "Look up.", True), ("B", "Refund.", False)), 5)
    assert [s["reply"] for s in steps] == [False, False, True] and steps[2]["title"] == "Reply to the customer"
    ordered = parse_plan(plan_json(("A", "Look up.", False), ("B", "Reply.", True)), 5)
    assert [s["reply"] for s in ordered] == [False, True] and ordered[1]["title"] == "B"
    appended = parse_plan(plan_json(("A", "Look up.", False)), 5)
    assert len(appended) == 2 and appended[1]["reply"] and appended[1]["title"] == "Reply to the customer"
    capped = parse_plan(plan_json(*[(f"S{i}", "Do it.", False) for i in range(9)]), 3)
    assert len(capped) == 3 and capped[-1]["reply"] and [s["reply"] for s in capped[:-1]] == [False, False]
    for bad in ("not json", "{}", '{"steps": []}', '{"steps": [{"title": "t"}]}', '{"steps": "x"}'):
        with pytest.raises(ValueError):
            parse_plan(bad, 5)
    assert [s["reply"] for s in fallback_plan("Help")] == [False, True]


def test_cost_arithmetic_and_lane_views_are_consistent(parts):
    luna = parts[0].get("gpt-5.6-luna")
    assert cost_of(luna, 1_000_000, 1_000_000) == pytest.approx(luna.price_in + luna.price_out)
    lane = _fresh_lane()
    lane.update(exec_cost=0.01, check_cost=0.002, router_cost=0.0004, tokens_in=100, tokens_out=50, check_tokens=30, scores=[4, 5])
    view = lane_view(lane)
    assert view["cost_usd"] == pytest.approx(0.0124) and view["tokens_total"] == 180 and view["avg_score"] == 4.5


def test_only_the_first_step_can_become_a_question_and_clef_sees_earlier_steps(parts):
    plan = plan_json(("Look up", "Look up order A-1042.", False), ("Check refund", "Check the order's refund status.", False),
                     ("Reply", "Write the reply.", True))
    decider = FakeDecider(routes={"tools": "clarification", "code": "debugging"}, confidence=0.95)
    _, events = run_case(parts, decider=decider, llm=ScriptedLLM(plan=plan))
    styles = {e["index"]: e["style"] for e in events if e["type"] == "exec" and e["lane"] == "clef"}
    assert styles == {1: "clarify", 2: "tools", 3: "reply"}
    assert decider.preceding == ("Look up order A-1042.", "Check the order's refund status.")
