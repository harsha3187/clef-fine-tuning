"""The With Clef lane applies the capability router and selector: each step runs on the endpoint the selector picks."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest
from agent_demo_fakes import FakeDecider, ScriptedLLM, collect

from agent_demo.agent import Agent
from agent_demo.policy import Policy
from agent_demo.preferences import Preferences
from agent_demo.registry import Registry
from agent_demo.router import AdaptiveRouter
from agent_demo.shadow import ShadowSelector
from agent_demo.telemetry import Telemetry
from routing_lab.capability import CapabilityResult
from routing_lab.selector import MeasuredRegistry

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "agent_demo" / "data"
MEASURED = ROOT / "data" / "registry" / "measured.json"
pytestmark = pytest.mark.skipif(not MEASURED.exists(), reason="run python -m capability_lab registry")
SMALL = {"gpt-oss-120b", "gpt-5.6-luna", "deepseek-v4-flash", "gpt-5.4-nano", "deepseek-v4-pro", "llama-4-maverick"}


class Router:
    deployment = "fake-cap"

    def __init__(self, tier=None, fail=False):
        self.tier = tier or {"low": 0.9, "medium": 0.08, "high": 0.02}
        self.fail, self.calls, self.lock = fail, [], threading.Lock()

    def classify(self, request, context=""):
        with self.lock:
            self.calls.append(request)
        if self.fail:
            raise RuntimeError("endpoint down")
        return CapabilityResult(self.tier, {"coding": 0.1, "reasoning": 0.2, "tools": 0.5, "writing": 0.1, "extraction": 0.1}, 40.0, 120, "fake-cap")


def build(tmp_path, router=None, llm=None, apply=True, route_planner=False, route_judge=False):
    registry = Registry.load(DATA / "registry.json", tmp_path / "r.json")
    policy = Policy.load(DATA / "policy.json")
    telemetry = Telemetry(":memory:")
    selector = ShadowSelector(router or Router(), MeasuredRegistry.load(MEASURED), baseline=policy.baseline_model,
                              router_cost_usd=policy.router_cost_usd, use_skill=False)
    llm = llm or ScriptedLLM()
    agent = Agent(AdaptiveRouter(registry, policy, FakeDecider(), telemetry), llm, registry, policy, telemetry, shadow=selector,
                  shadow_replay=True, apply_selector=apply, route_planner=route_planner, route_judge=route_judge)
    return agent, llm, selector


def run(agent, prefs=None, message="Order A-1042 arrived broken. I want my money back."):
    events: list[dict] = []
    summary = agent.run(message, prefs=prefs or Preferences(), emit=collect(events))
    return summary, events


def test_the_clef_lane_runs_each_step_on_the_endpoint_the_selector_chose(tmp_path):
    agent, llm, selector = build(tmp_path)
    summary, events = run(agent)
    clef_models = set(summary["lanes"]["clef"]["models"])
    assert clef_models and "gpt-5.6-sol" not in clef_models  # an easy request does not need the strongest model
    assert set(summary["lanes"]["baseline"]["models"]) == {"gpt-5.6-sol"}
    executed = {c["deployment"] for c in llm.calls if c["deployment"] not in ("gpt-4.1",)}
    assert clef_models <= executed  # the chosen deployments were really called
    decisions = [e for e in events if e["type"] == "decision"]
    assert [d["decision"]["selected_model"] for d in decisions if d["attempt"] == 0][:1] and all(d["lane"] == "clef" for d in decisions)
    started = next(e for e in events if e["type"] == "shadow_started")
    assert started["applied"] is True


def test_the_decision_has_everything_the_screen_shows_and_costs_use_the_endpoints_own_prices(tmp_path):
    agent, _, selector = build(tmp_path)
    summary, events = run(agent)
    first = next(e for e in events if e["type"] == "decision")
    d = first["decision"]
    assert d["clef"]["route"] == "low" and d["clef"]["source"] == "clef" and d["clef"]["probabilities"]["low"] == 0.9
    assert d["selected_tier"] in ("small", "mid", "frontier") and d["reason"] and d["candidates"]
    assert {"model_id", "display", "tier", "status", "reasons", "quality", "est_cost"} <= set(d["candidates"][0])
    assert d["requirement"]["parts"][0][0] == "quality floor" and d["requirement"]["total"] > 0.5
    execs = [e for e in events if e["type"] == "exec" and e["lane"] == "clef"]
    for e in execs:
        endpoint = selector.endpoint(e["model"])
        expected = e["input_tokens"] * endpoint.price_in / 1e6 + e["output_tokens"] * endpoint.price_out / 1e6
        assert e["cost_usd"] == pytest.approx(expected, abs=1e-6)
    assert summary["lanes"]["clef"]["cost_usd"] < summary["lanes"]["baseline"]["cost_usd"]
    assert summary["lanes"]["clef"]["router_cost_usd"] > 0  # Clef's own cost is charged to this lane


def test_a_second_replay_of_the_same_choices_is_not_run_when_they_are_applied(tmp_path):
    agent, _, _ = build(tmp_path)
    _, events = run(agent)
    assert not any(e["type"] in ("replay_step", "replay_done") for e in events)
    assert next(e for e in events if e["type"] == "shadow_summary")["replay"] is None
    assert len([e for e in events if e["type"] == "shadow_decision"]) == 3  # plan and two steps: Clef is asked once per call, not twice


def test_an_endpoint_that_needs_the_chat_api_is_never_executed(tmp_path):
    agent, llm, _ = build(tmp_path, Router({"low": 0.97, "medium": 0.02, "high": 0.01}))
    for message in ("Where is order A-1042?", "Thank the customer for waiting."):
        run(agent, message=message)
    assert all(c["deployment"] != "phi-4" for c in llm.calls)


def test_when_clef_does_not_answer_the_lane_falls_back_to_the_strongest_model_and_says_so(tmp_path):
    agent, _, _ = build(tmp_path, Router(fail=True))
    summary, events = run(agent)
    assert set(summary["lanes"]["clef"]["models"]) == {"gpt-5.6-sol"} and summary["lanes"]["clef"]["failed_steps"] == 0
    decisions = [e["decision"] for e in events if e["type"] == "decision"]
    assert decisions and all(d["clef"]["source"] == "fallback" and "endpoint down" in d["reason"] and d["selected_model"] == "gpt-5.6-sol" for d in decisions)
    assert summary["lanes"]["clef"]["router_cost_usd"] == 0


def test_a_weak_result_moves_the_step_to_a_more_reliable_endpoint(tmp_path):
    weak = {name: 2 for name in SMALL}  # the checker scores every small endpoint's output 2 out of 5
    agent, llm, _ = build(tmp_path, llm=ScriptedLLM(model_scores=weak))
    summary, events = run(agent, Preferences(quality="good", latency="medium", cost="low"))
    assert summary["lanes"]["clef"]["escalations"] >= 1
    moves = [e for e in events if e["type"] == "escalation"]
    assert moves and all(m["lane"] == "clef" for m in moves)
    later = [e["decision"] for e in events if e["type"] == "decision" and e["attempt"] >= 1]
    assert later and any(p[0] == "moved up after a weak result" for d in later for p in d["requirement"]["parts"])
    final = set(summary["lanes"]["clef"]["models"])
    assert final - SMALL, final  # at least one step ended on a stronger, non-small endpoint
    first = [e for e in events if e["type"] == "decision" and e["attempt"] == 0 and e["index"] == 1][0]["decision"]["selected_model"]
    retried = [e["decision"]["selected_model"] for e in events if e["type"] == "decision" and e["index"] == 1 and e["attempt"] >= 1]
    assert retried and first not in retried  # never the same endpoint twice


def test_a_failing_endpoint_is_skipped_and_the_step_is_retried_elsewhere(tmp_path):
    agent, llm, _ = build(tmp_path, llm=ScriptedLLM(fail_models=SMALL))
    summary, events = run(agent)
    assert summary["lanes"]["clef"]["failed_steps"] == 0 and summary["lanes"]["clef"]["escalations"] >= 1
    assert not set(summary["lanes"]["clef"]["models"]) & SMALL
    assert any(e["type"] == "exec_error" and e["lane"] == "clef" for e in events)


def test_with_the_selector_not_applied_the_old_router_still_drives_the_lane(tmp_path):
    agent, _, _ = build(tmp_path, apply=False)
    _, events = run(agent)
    decisions = [e["decision"] for e in events if e["type"] == "decision"]
    assert decisions and all(d["clef"]["route"] == "tool_lookup" for d in decisions)  # the step-type router, via the fake decider
    assert next(e for e in events if e["type"] == "shadow_started")["applied"] is False
    assert any(e["type"] == "replay_step" for e in events)


def test_applying_needs_a_selector(tmp_path):
    registry = Registry.load(DATA / "registry.json", tmp_path / "r.json")
    policy = Policy.load(DATA / "policy.json")
    agent = Agent(AdaptiveRouter(registry, policy, FakeDecider(), Telemetry(":memory:")), ScriptedLLM(), registry, policy,
                  Telemetry(":memory:"), apply_selector=True)
    assert agent.apply_selector is False  # no selector, so the lane stays on the old router

def aux_calls(llm, system_prefix):
    return [c for c in llm.calls if (c["system"] or "").startswith(system_prefix)]


def test_a_routed_planner_runs_on_the_endpoint_clef_chose_and_is_reported(tmp_path):
    router = Router()
    agent, llm, _ = build(tmp_path, router=router, route_planner=True)
    summary, events = run(agent)
    plan_event = next(e for e in events if e["type"] == "plan")
    routing = plan_event["routing"]
    assert routing["source"] == "clef" and routing["model"] != "gpt-4.1" and plan_event["planner_model"] == routing["model"]
    assert [c["deployment"] for c in aux_calls(llm, "You plan")] == [routing["model"]]
    assert plan_event["fixed_model"] == "gpt-4.1" and 0 < plan_event["cost_usd"]
    assert "Task: split this request" in router.calls[0]  # the planning task is what Clef reads
    assert plan_event["cost_usd"] >= agent.policy.router_cost_usd  # Clef's own call is part of the planner's cost
    assert summary["planner_cost_usd"] == pytest.approx(plan_event["cost_usd"], abs=1e-6)
    started = next(e for e in events if e["type"] == "shadow_started")
    assert started["decisions"] == 2  # the two steps; the plan was decided before it ran


def test_the_planner_falls_back_to_the_fixed_model_when_clef_is_down_or_the_routed_plan_is_unusable(tmp_path):
    agent, llm, _ = build(tmp_path, router=Router(fail=True), route_planner=True)
    _, events = run(agent)
    plan_event = next(e for e in events if e["type"] == "plan")
    assert plan_event["routing"]["source"] == "fallback" and plan_event["planner_model"] == "gpt-4.1"
    assert [c["deployment"] for c in aux_calls(llm, "You plan")] == ["gpt-4.1"]

    class BadPlanner(ScriptedLLM):
        def complete(self, deployment, **kwargs):
            if deployment != "gpt-4.1" and (kwargs.get("system") or "").startswith("You plan"):
                self.calls.append({"deployment": deployment, "system": kwargs["system"], "user": kwargs.get("user"), "tools": False, "previous": None})
                from agent_demo.llm import LLMResult
                return LLMResult("not json", 100, 10, 5.0, "completed", "x")
            return super().complete(deployment, **kwargs)

    agent, llm, _ = build(tmp_path, llm=BadPlanner(), route_planner=True)
    _, events = run(agent)
    plan_event = next(e for e in events if e["type"] == "plan")
    assert plan_event["planner_model"] == "gpt-4.1" and "did not give a usable plan" in plan_event["note"]
    assert len(aux_calls(llm, "You plan")) == 2


def test_a_routed_judge_checks_only_the_clef_lane_and_is_held_to_a_higher_floor(tmp_path):
    router = Router()
    agent, llm, _ = build(tmp_path, router=router, route_judge=True)
    _, events = run(agent)
    judges = [e for e in events if e["type"] == "judge"]
    clef_judges = [e for e in judges if e["lane"] == "clef"]
    assert clef_judges and all(e["routing"]["source"] == "clef" for e in clef_judges)
    assert all(e["model"] == e["routing"]["model"] for e in clef_judges if not e["fell_back"])
    base = Preferences()
    assert clef_judges[0]["routing"]["floor"] == pytest.approx(min(0.97, agent.policy.derive(base).mode.quality_floor + 0.10), abs=1e-3)
    assert all(e["model"] == "gpt-4.1" and "routing" not in e for e in judges if e["lane"] == "baseline")  # the control keeps its fixed reviewer
    assert all(e["router_cost_usd"] == 0 for e in clef_judges) and len(router.calls) == 3  # the reviewer reuses the step's Clef answer: plan plus two steps, nothing more


def test_the_judge_falls_back_to_the_fixed_reviewer_when_the_routed_one_fails(tmp_path):
    router = Router()
    probe, _, selector = build(tmp_path, router=router, route_judge=True)
    _, events = run(probe)
    chosen = next(e for e in events if e["type"] == "judge" and e["lane"] == "clef")["routing"]["model"]
    agent, llm, _ = build(tmp_path, router=Router(), llm=ScriptedLLM(fail_models={chosen}), route_judge=True)
    _, events = run(agent)
    clef_judges = [e for e in events if e["type"] == "judge" and e["lane"] == "clef" and e.get("score") is not None]
    assert clef_judges and all(e["model"] == "gpt-4.1" and e["fell_back"] for e in clef_judges)


def test_routing_the_planner_and_judge_needs_the_selector_to_be_applied(tmp_path):
    agent, llm, _ = build(tmp_path, apply=False, route_planner=True, route_judge=True)
    assert not agent.route_planner and not agent.route_judge
    _, events = run(agent)
    assert "routing" not in next(e for e in events if e["type"] == "plan")
    assert all("routing" not in e for e in events if e["type"] == "judge")

def test_endpoints_that_failed_the_json_probe_are_left_out_of_the_planner_and_reviewer_choice(tmp_path):
    from agent_demo.agent import load_json_exclusions

    probe = tmp_path / "so.json"
    probe.write_text('{"endpoints": {"a": {"planner": [12, 12], "judge": [6, 6]}, "b": {"planner": [6, 12], "judge": [0, 6]}, '
                     '"c": {"planner": [12, 12], "judge": [5, 6]}}}', encoding="utf-8")
    assert load_json_exclusions(probe) == {"planner": frozenset({"b"}), "judge": frozenset({"b", "c"})}
    assert load_json_exclusions(tmp_path / "missing.json") == {"planner": frozenset(), "judge": frozenset()}

    # make the cheapest endpoint the one that fails the probe: it must not be chosen even though it is the best fit
    agent, llm, _ = build(tmp_path, route_planner=True, route_judge=True)
    _, events = run(agent)
    planner_choice = next(e for e in events if e["type"] == "plan")["routing"]["model"]
    judge_choice = next(e for e in events if e["type"] == "judge" and e["lane"] == "clef")["routing"]["model"]
    agent, llm, _ = build(tmp_path, route_planner=True, route_judge=True)
    agent.json_excluded = {"planner": frozenset({planner_choice}), "judge": frozenset({judge_choice})}
    _, events = run(agent)
    plan_event = next(e for e in events if e["type"] == "plan")
    assert plan_event["routing"]["model"] != planner_choice and planner_choice in plan_event["routing"]["left_out_for_json"]
    assert all(e["routing"]["model"] != judge_choice for e in events if e["type"] == "judge" and e["lane"] == "clef")

def test_a_routed_planner_that_returns_only_a_reply_step_is_replaced_by_the_fixed_planner(tmp_path):
    from agent_demo.llm import LLMResult
    from agent_demo_fakes import plan_json

    class Lazy(ScriptedLLM):
        def complete(self, deployment, **kwargs):
            if deployment != "gpt-4.1" and (kwargs.get("system") or "").startswith("You plan"):
                self.calls.append({"deployment": deployment, "system": kwargs["system"], "user": kwargs.get("user"), "tools": False, "previous": None})
                return LLMResult(plan_json(("Reply", "Write the reply.", True)), 100, 10, 5.0, "completed", "x")
            return super().complete(deployment, **kwargs)

    agent, llm, _ = build(tmp_path, llm=Lazy(), route_planner=True)
    _, events = run(agent)
    plan_event = next(e for e in events if e["type"] == "plan")
    assert plan_event["planner_model"] == "gpt-4.1" and len(plan_event["steps"]) == 2
    assert "did not give a usable plan" in plan_event["note"]
