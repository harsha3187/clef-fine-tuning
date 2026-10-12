"""Shadow mode: the selector decides beside the agent and must change nothing the agent does."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest
from agent_demo_fakes import FakeDecider, ScriptedLLM, collect

from agent_demo.agent import Agent
from agent_demo.policy import Policy
from agent_demo.preferences import Preferences
from agent_demo.registry import Registry
from agent_demo.router import AdaptiveRouter
from agent_demo.shadow import ShadowItem, ShadowSelector, aggregate
from agent_demo.telemetry import Telemetry
from routing_lab.capability import CapabilityResult
from routing_lab.selector import MeasuredRegistry, Settings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "agent_demo" / "data"
MEASURED = ROOT / "data" / "registry" / "measured.json"

pytestmark = pytest.mark.skipif(not MEASURED.exists(), reason="run python -m capability_lab registry")


class FakeCapabilityRouter:
    deployment = "fake-cap"

    def __init__(self, tier=None, fail=False):
        self.tier = tier or {"low": 0.9, "medium": 0.08, "high": 0.02}
        self.fail = fail
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def classify(self, request, context=""):
        with self.lock:
            self.calls.append(request)
        if self.fail:
            raise RuntimeError("endpoint down")
        return CapabilityResult(self.tier, {"coding": 0.1, "reasoning": 0.2, "tools": 0.5, "writing": 0.1, "extraction": 0.1}, 40.0, 120, "fake-cap")


def build(tmp_path, router=None, shadow=True, replay=False, llm=None):
    registry = Registry.load(DATA / "registry.json", tmp_path / "r.json")
    policy = Policy.load(DATA / "policy.json")
    telemetry = Telemetry(":memory:")
    selector = None
    if shadow:
        selector = ShadowSelector(router or FakeCapabilityRouter(), MeasuredRegistry.load(MEASURED), baseline=policy.baseline_model,
                                  router_cost_usd=policy.router_cost_usd, log_path=tmp_path / "shadow.jsonl")
    agent = Agent(AdaptiveRouter(registry, policy, FakeDecider(), telemetry), llm or ScriptedLLM(), registry, policy, telemetry,
                  shadow=selector, shadow_replay=replay)
    return agent, selector


def run(agent, prefs=None):
    events: list[dict] = []
    summary = agent.run("Order A-1042 arrived broken. I want my money back.", prefs=prefs or Preferences(), emit=collect(events))
    return summary, events


def test_every_boundary_gets_a_shadow_decision_and_a_counterfactual_summary(tmp_path):
    agent, selector = build(tmp_path)
    _, events = run(agent)
    decisions = [e for e in events if e["type"] == "shadow_decision"]
    assert [d["key"] for d in decisions] == ["plan", "1", "2"] and [d["kind"] for d in decisions] == ["plan", "step", "reply"]
    first = decisions[0]
    assert first["choice"] and first["versions"]["registry"] == selector.registry.version and first["alternatives"]
    assert first["tier_probabilities"]["low"] == 0.9 and first["baseline"]["endpoint"] == "gpt-5.6-sol"
    assert sum(first["evidence"].values()) == pytest.approx(1.0, abs=1e-3) and "thin_evidence" in aggregate(tmp_path / "shadow.jsonl")
    summary = next(e for e in events if e["type"] == "shadow_summary")
    assert summary["decisions"] == 3 and summary["errors"] == 0 and "Counterfactual" in summary["note"]
    assert summary["selector_est_cost_usd"] < summary["priced_baseline_cost_usd"]
    assert all("est_cost_usd" in row and row["actual"]["tokens_in"] >= 0 for row in summary["rows"])
    assert next(e for e in events if e["type"] == "shadow_started")["decisions"] == 3


def test_the_shadow_selector_changes_nothing_the_agent_does(tmp_path):
    with_shadow, _ = build(tmp_path / "a")
    without, _ = build(tmp_path / "b", shadow=False)
    llm_a, llm_b = with_shadow.llm, without.llm
    a, events_a = run(with_shadow)
    b, events_b = run(without)
    assert sorted(c["deployment"] for c in llm_a.calls) == sorted(c["deployment"] for c in llm_b.calls)
    assert len(llm_a.calls) == len(llm_b.calls)
    for lane in ("clef", "baseline"):
        assert a["lanes"][lane]["models"] == b["lanes"][lane]["models"]
        assert a["lanes"][lane]["cost_usd"] == b["lanes"][lane]["cost_usd"]
    assert not any(e["type"].startswith("shadow") for e in events_b)


def test_an_endpoint_outage_becomes_events_and_the_run_still_completes(tmp_path):
    agent, selector = build(tmp_path, FakeCapabilityRouter(fail=True))
    summary, events = run(agent)
    errors = [e for e in events if e["type"] == "shadow_error"]
    assert len(errors) == 3 and "endpoint down" in errors[0]["error"]
    assert next(e for e in events if e["type"] == "shadow_summary")["errors"] == 3
    assert events[-1]["type"] == "run_done" and summary["lanes"]["clef"]["steps_done"] == 2


def test_hard_requests_are_flagged_uncertain_and_carry_a_lower_expected_success(tmp_path):
    easy, _ = build(tmp_path / "e", FakeCapabilityRouter({"low": 0.95, "medium": 0.04, "high": 0.01}))
    hard, _ = build(tmp_path / "h", FakeCapabilityRouter({"low": 0.02, "medium": 0.28, "high": 0.7}))
    prefs = Preferences(quality="very_good", latency="medium", cost="medium")
    easy_steps = [e for e in run(easy, prefs)[1] if e["type"] == "shadow_decision"]
    hard_steps = [e for e in run(hard, prefs)[1] if e["type"] == "shadow_decision"]
    assert not any(e["uncertain"] for e in easy_steps) and all(e["uncertain"] for e in hard_steps)
    assert min(e["expected_success"] for e in easy_steps) > max(e["expected_success"] for e in hard_steps)
    assert all(e["below_floor"] for e in hard_steps) and not any(e["below_floor"] for e in easy_steps)

def test_decisions_are_logged_for_later_calibration_and_aggregated(tmp_path):
    agent, selector = build(tmp_path)
    run(agent)
    run(agent)
    rows = [json.loads(line) for line in (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(r["type"] == "shadow_decision" for r in rows) == 6 and sum(r["type"] == "shadow_run" for r in rows) == 2
    assert all(r["task_id"].startswith("tk_") for r in rows) and rows[0]["versions"]["selector"] == "selector-v1"
    totals = aggregate(tmp_path / "shadow.jsonl")
    assert totals["decisions"] == 6 and totals["runs"] == 2 and totals["errors"] == 0 and sum(totals["by_endpoint"].values()) == 6
    assert aggregate(tmp_path / "missing.jsonl")["decisions"] == 0


def test_decide_prices_the_choice_and_never_raises(tmp_path):
    _, selector = build(tmp_path)
    item = ShadowItem("1", "step", "Look up order A-1042", True, 600, 400)
    event = selector.decide(item, Settings(0.7, 0.6, 0.4))
    assert event["type"] == "shadow_decision" and event["needs_tools"] and selector.price(event["choice"], 1000, 100) > 0
    assert selector.price("unknown-model", 1, 1) is None
    broken = ShadowSelector(FakeCapabilityRouter(fail=True), selector.registry, baseline="gpt-5.6-sol", router_cost_usd=0.0004)
    assert broken.decide(item, Settings(0.7, 0.6, 0.4))["type"] == "shadow_error"

def test_replay_runs_the_selected_endpoint_judges_it_and_reports_against_the_baseline(tmp_path):
    agent, selector = build(tmp_path, replay=True)
    summary, events = run(agent)
    replay_steps = [e for e in events if e["type"] == "replay_step"]
    assert [e["key"] for e in replay_steps] == ["1", "2"] and all(e["status"] == "ok" and e["endpoint"] == "gpt-oss-120b" for e in replay_steps)
    assert all(e["score"] == 5 and e["cost_usd"] > 0 for e in replay_steps)
    called = [c["deployment"] for c in agent.llm.calls]
    assert called.count("gpt-oss-120b") >= 2  # the replay really called the chosen deployment
    result = next(e for e in events if e["type"] == "shadow_summary")["replay"]
    assert result["steps_run"] == 2 and result["failed"] == 0 and result["mean_score"] == 5.0 == result["baseline_mean_score"]
    assert result["cost_usd"] < result["baseline_cost_usd"] and result["below_baseline"] == 0 and result["finished"]
    assert any(e["type"] == "replay_done" for e in events) and events[-1]["type"] == "run_done"
    assert summary["lanes"]["clef"]["steps_done"] == 2


def test_replay_never_changes_the_lanes_and_a_failing_endpoint_is_reported_not_hidden(tmp_path):
    plain, _ = build(tmp_path / "a", shadow=False)
    failing, _ = build(tmp_path / "b", replay=True, llm=ScriptedLLM(fail_models={"gpt-oss-120b"}))
    base, _ = run(plain)
    replayed, events = run(failing)
    for lane in ("clef", "baseline"):
        assert base["lanes"][lane]["models"] == replayed["lanes"][lane]["models"]
    result = next(e for e in events if e["type"] == "shadow_summary")["replay"]
    assert result["failed"] == 2 and result["steps_run"] == 0 and result["mean_score"] is None
    assert all(e["status"] == "failed" and "is down" in e["reason"] for e in events if e["type"] == "replay_step")


def test_replay_checks_expected_outcomes_on_its_own_desk(tmp_path):
    llm = ScriptedLLM(tool_calls=[{"name": "get_order", "arguments": {"order_id": "A-1042"}}])
    agent, _ = build(tmp_path, replay=True, llm=llm)
    events: list[dict] = []
    expectations = [{"label": "Order looked up", "tool": "get_order", "args": {"order_id": "A-1042"}}]
    agent.run("Order A-1042 arrived broken.", prefs=Preferences(), expectations=expectations, emit=collect(events))
    result = next(e for e in events if e["type"] == "shadow_summary")["replay"]
    assert result["checks"]["replay"] == [1, 1] and result["checks"]["baseline"] == [1, 1]

def test_a_router_without_skill_training_gets_the_skill_from_what_the_step_is_for(tmp_path):
    seen = {}

    class Spy(FakeCapabilityRouter):
        pass

    registry = MeasuredRegistry.load(MEASURED)
    import agent_demo.shadow as shadow_module

    real = shadow_module.select

    def spy(endpoints, rate, tiers, skills, settings, constraints):
        seen["skills"] = skills
        return real(endpoints, rate, tiers, skills, settings, constraints)

    shadow_module.select = spy
    try:
        item = ShadowItem("1", "step", "Look up order A-1042", True, 600, 400)
        used = ShadowSelector(Spy(), registry, baseline="gpt-5.6-sol", router_cost_usd=0.0004).decide(item, Settings(0.7, 0.6, 0.4))
        assert seen["skills"] is not None and used["skill_used"] is True and used["versions"]["skill_head_used"] is True
        ignored = ShadowSelector(Spy(), registry, baseline="gpt-5.6-sol", router_cost_usd=0.0004, use_skill=False).decide(
            item, Settings(0.7, 0.6, 0.4))
        assert seen["skills"] == {"tools": 1.0} and ignored["skill_used"] is False and ignored["skill_hint"] == {"tools": 1.0}
        assert ignored["versions"]["skill_head_used"] is False
        reply = ShadowSelector(Spy(), registry, baseline="gpt-5.6-sol", router_cost_usd=0.0004, use_skill=False)
        reply.decide(ShadowItem("2", "reply", "Write the answer", False, 600, 400), Settings(0.7, 0.6, 0.4))
        assert seen["skills"] == {"writing": 1.0}
        reply.decide(ShadowItem("plan", "plan", "Plan it", False, 600, 400), Settings(0.7, 0.6, 0.4))
        assert seen["skills"] is None
        assert ignored["choice"] and sum(ignored["evidence"].values()) == pytest.approx(1.0, abs=1e-3)
    finally:
        shadow_module.select = real

def test_the_selector_remembers_whether_the_router_last_answered(tmp_path):
    _, selector = build(tmp_path)
    assert selector.last is None
    item = ShadowItem("1", "step", "Look up order A-1042", True, 600, 400)
    selector.decide(item, Settings(0.7, 0.6, 0.4))
    assert selector.last == {"ok": True, "latency_ms": 40.0}
    selector.router.fail = True
    selector.router = FakeCapabilityRouter(fail=True)
    assert selector.decide(item, Settings(0.7, 0.6, 0.4))["type"] == "shadow_error" and selector.last["ok"] is False

def test_warming_up_records_whether_the_router_answers_and_never_raises(tmp_path):
    _, working = build(tmp_path / "a")
    working.warm_up()
    assert working.last["ok"] is True
    _, broken = build(tmp_path / "b", FakeCapabilityRouter(fail=True))
    broken.warm_up()
    assert broken.last["ok"] is False
