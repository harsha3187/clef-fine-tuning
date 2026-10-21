"""The With Clef lane applies the capability router and selector: each step runs on the endpoint the selector picks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agent_demo_fakes import FakeCapabilityRouter, FakeDecider, ScriptedLLM, collect

from agent_demo.agent import Agent
from agent_demo.policy import Policy
from agent_demo.preferences import Preferences
from agent_demo.registry import Registry
from agent_demo.router import AdaptiveRouter
from agent_demo.shadow import ShadowSelector
from agent_demo.telemetry import Telemetry
from routing_lab.selector import MeasuredRegistry
from agent_demo.shadow import ShadowItem, aggregate, replay_summary
from routing_lab.selector import Settings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "agent_demo" / "data"
MEASURED = ROOT / "data" / "registry" / "measured.json"
pytestmark = pytest.mark.skipif(not MEASURED.exists(), reason="run python -m capability_lab registry")
SMALL = {"gpt-oss-120b", "gpt-5.6-luna", "deepseek-v4-flash", "gpt-5.4-nano", "deepseek-v4-pro", "llama-4-maverick"}


def build(tmp_path, router=None, llm=None, apply=True, route_planner=False, route_judge=False):
    registry = Registry.load(DATA / "registry.json", tmp_path / "r.json")
    policy = Policy.load(DATA / "policy.json")
    telemetry = Telemetry(":memory:")
    selector = ShadowSelector(router or FakeCapabilityRouter(), MeasuredRegistry.load(MEASURED), baseline=policy.baseline_model,
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


@pytest.mark.parametrize(("provider", "name"), [("clef", "Clef"), ("decision-1", "Microsoft Decision-1")])
def test_lane_labels_are_provider_independent_in_events_and_results(tmp_path, provider, name):
    from agent_demo.labels import LANE_LABELS

    router = FakeCapabilityRouter()
    router.provider, router.display_name = provider, name
    agent, _, _ = build(tmp_path, router=router)
    summary, events = run(agent)
    assert summary["lane_labels"] == LANE_LABELS and summary["router_name"] == name
    assert set(summary["lanes"]) == {"clef", "baseline"}
    assert all(e["lane_label"] == LANE_LABELS[e["lane"]] for e in events if e.get("lane") in LANE_LABELS)
    assert next(e for e in events if e["type"] == "run_started")["lane_labels"] == LANE_LABELS
    assert next(e for e in events if e["type"] == "shadow_summary")["lane_labels"] == LANE_LABELS


@pytest.mark.parametrize(("provider", "name"), [("clef", "Clef"), ("decision-1", "Microsoft Decision-1")])
def test_fallback_reason_uses_generic_role_not_provider_name(tmp_path, provider, name):
    router = FakeCapabilityRouter(fail=True)
    router.provider, router.display_name = provider, name
    agent, _, _ = build(tmp_path, router=router)
    _, events = run(agent)
    decisions = [e["decision"] for e in events if e["type"] == "decision"]
    assert decisions and all(d["reason"].startswith("The decision model did not answer") for d in decisions)
    assert all(name not in d["reason"] for d in decisions)


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
    agent, llm, _ = build(tmp_path, FakeCapabilityRouter({"low": 0.97, "medium": 0.02, "high": 0.01}))
    for message in ("Where is order A-1042?", "Thank the customer for waiting."):
        run(agent, message=message)
    assert all(c["deployment"] != "phi-4" for c in llm.calls)


def test_when_clef_does_not_answer_the_lane_falls_back_to_the_strongest_model_and_says_so(tmp_path):
    agent, _, _ = build(tmp_path, FakeCapabilityRouter(fail=True))
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


def test_unrecoverable_itinerary_json_fails_both_lanes_and_keeps_all_paid_attempts(tmp_path):
    plan = '{"steps":[{"title":"Write itinerary","instruction":"Write the itinerary.","reply":true}]}'
    agent, _, _ = build(tmp_path, llm=ScriptedLLM(plan=plan, reply_text="Not a JSON itinerary."))
    events = []
    summary = agent.run("A family trip to Tokyo.", prefs=Preferences(), agent="travel", emit=collect(events))
    for lane in ("clef", "baseline"):
        attempts = [e for e in events if e["type"] == "exec" and e["lane"] == lane]
        assert attempts and all(e["output_error"] for e in attempts)
        assert all(len(e["json_attempts"]) == 2 for e in attempts)
        assert summary["lanes"][lane]["failed_steps"] == 1
        assert summary["lanes"][lane]["exec_cost_usd"] == pytest.approx(sum(e["cost_usd"] for e in attempts), abs=1e-6)
        assert summary["lanes"][lane]["exec_tokens"] == sum(e["input_tokens"] + e["output_tokens"] for e in attempts)
        done = next(e for e in events if e["type"] == "step_done" and e["lane"] == lane)
        assert done["step"]["failed"] is True and done["step"]["output_error"]
    assert all(e["model"] == "structure check" for e in events if e["type"] == "judge")


def test_json_requirement_survives_quality_retries_and_uses_native_chat(tmp_path):
    from types import SimpleNamespace
    from agent_demo.llm import AzureResponsesClient

    class NativeSession:
        def __init__(self):
            self.calls = []

        def post(self, url, **kwargs):
            body = kwargs["json"]
            self.calls.append((url, body))
            model = body["model"]
            messages = body.get("input", body.get("messages"))
            system = messages[0]["content"]
            if system.startswith("You plan"):
                text = '{"steps":[{"title":"Itinerary","instruction":"Write the itinerary.","reply":true}]}'
            elif system.startswith("You grade"):
                produced = json.loads(messages[-1]["content"].split("\n\nOUTPUT:\n")[1])["title"]
                score = 2 if produced in ("deepseek-v4-pro", "llama-4-maverick") else 5
                text = json.dumps({"score": score, "fault": "this_output", "reason": "test quality gate"})
            else:
                text = json.dumps({"title": model, "days": [{"day": 1, "items": []}]})
                if model == "llama-4-maverick" and url.endswith("/responses"):
                    text = "# Markdown, not JSON"
            if url.endswith("/chat/completions"):
                payload = {"choices": [{"finish_reason": "stop", "message": {"content": text}}],
                           "usage": {"prompt_tokens": 100, "completion_tokens": 30}}
            else:
                payload = {"status": "completed", "output": [{"type": "message", "content": [
                    {"type": "output_text", "text": text}]}], "usage": {"input_tokens": 100, "output_tokens": 30}}
            return SimpleNamespace(status_code=200, json=lambda: payload)

    session = NativeSession()
    credential = SimpleNamespace(get_token=lambda _: SimpleNamespace(token="test-only", expires_on=9999999999))
    llm = AzureResponsesClient("https://r.openai.azure.com/openai/v1/", credential=credential, session=session)
    agent, _, _ = build(tmp_path, router=FakeCapabilityRouter({"low": 0.15, "medium": 0.18, "high": 0.67}), llm=llm)
    events = []
    summary = agent.run("A week in Tokyo.", prefs=Preferences(), agent="travel", emit=collect(events))
    replies = [e for e in events if e["type"] == "exec" and e["lane"] == "clef"]
    assert [e["model"] for e in replies] == ["deepseek-v4-pro", "llama-4-maverick", "deepseek-v4-flash"]
    assert all(e["output_error"] is None and json.loads(e["output"]) for e in replies)
    assert summary["lanes"]["clef"]["failed_steps"] == 0 and summary["lanes"]["clef"]["reply_score"] == 5
    assert all(e["decision"]["needs_json"] for e in events if e["type"] == "decision")
    chat_models = {b["model"] for url, b in session.calls if url.endswith("/chat/completions")}
    assert chat_models == {"llama-4-maverick", "deepseek-v4-flash"}
    assert all(b["response_format"] == {"type": "json_object"} and b["max_tokens"] == 6000
               for url, b in session.calls if url.endswith("/chat/completions"))


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
    router = FakeCapabilityRouter()
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


def test_usage_based_router_cost_is_counted_once_not_as_a_clef_gpu_estimate(tmp_path):
    from dataclasses import replace

    class TokenPricedRouter(FakeCapabilityRouter):
        provider = "decision-1"
        display_name = "Microsoft Decision-1"
        cost_note = "Usage-based token estimate."

        def classify(self, request, context=""):
            return replace(super().classify(request, context), input_tokens=1000, output_tokens=2, cost_usd=0.000042084)

    agent, _, _ = build(tmp_path, router=TokenPricedRouter(), route_planner=True, route_judge=True)
    summary, events = run(agent)
    assert next(e for e in events if e["type"] == "plan")["routing"]["router_cost_usd"] == 0.000042084
    first_attempts = [e for e in events if e["type"] == "decision" and e["attempt"] == 0]
    assert summary["lanes"]["clef"]["router_cost_usd"] == pytest.approx(len(first_attempts) * 0.000042084, abs=1e-6)
    assert summary["lanes"]["clef"]["router_tokens"] == len(first_attempts) * 1002
    assert next(e for e in events if e["type"] == "run_started")["router_name"] == "Microsoft Decision-1"
    assert summary["router_cost_note"] == "Usage-based token estimate."


def test_the_planner_falls_back_to_the_fixed_model_when_clef_is_down_or_the_routed_plan_is_unusable(tmp_path):
    agent, llm, _ = build(tmp_path, router=FakeCapabilityRouter(fail=True), route_planner=True)
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
    router = FakeCapabilityRouter()
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
    router = FakeCapabilityRouter()
    probe, _, selector = build(tmp_path, router=router, route_judge=True)
    _, events = run(probe)
    chosen = next(e for e in events if e["type"] == "judge" and e["lane"] == "clef")["routing"]["model"]
    agent, llm, _ = build(tmp_path, router=FakeCapabilityRouter(), llm=ScriptedLLM(fail_models={chosen}), route_judge=True)
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


def build_shadow(tmp_path, router=None, shadow=True, replay=False, llm=None):
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




def test_every_boundary_gets_a_shadow_decision_and_a_counterfactual_summary(tmp_path):
    agent, selector = build_shadow(tmp_path)
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
    with_shadow, _ = build_shadow(tmp_path / "a")
    without, _ = build_shadow(tmp_path / "b", shadow=False)
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
    agent, selector = build_shadow(tmp_path, FakeCapabilityRouter(fail=True))
    summary, events = run(agent)
    errors = [e for e in events if e["type"] == "shadow_error"]
    assert len(errors) == 3 and "endpoint down" in errors[0]["error"]
    assert next(e for e in events if e["type"] == "shadow_summary")["errors"] == 3
    assert events[-1]["type"] == "run_done" and summary["lanes"]["clef"]["steps_done"] == 2


def test_hard_requests_are_flagged_uncertain_and_carry_a_lower_expected_success(tmp_path):
    easy, _ = build_shadow(tmp_path / "e", FakeCapabilityRouter({"low": 0.95, "medium": 0.04, "high": 0.01}))
    hard, _ = build_shadow(tmp_path / "h", FakeCapabilityRouter({"low": 0.02, "medium": 0.28, "high": 0.7}))
    prefs = Preferences(quality="very_good", latency="medium", cost="medium")
    easy_steps = [e for e in run(easy, prefs)[1] if e["type"] == "shadow_decision"]
    hard_steps = [e for e in run(hard, prefs)[1] if e["type"] == "shadow_decision"]
    assert not any(e["uncertain"] for e in easy_steps) and all(e["uncertain"] for e in hard_steps)
    assert min(e["expected_success"] for e in easy_steps) > max(e["expected_success"] for e in hard_steps)
    assert all(e["below_floor"] for e in hard_steps) and not any(e["below_floor"] for e in easy_steps)

def test_decisions_are_logged_for_later_calibration_and_aggregated(tmp_path):
    agent, selector = build_shadow(tmp_path)
    run(agent)
    run(agent)
    rows = [json.loads(line) for line in (tmp_path / "shadow.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sum(r["type"] == "shadow_decision" for r in rows) == 6 and sum(r["type"] == "shadow_run" for r in rows) == 2
    assert all(r["task_id"].startswith("tk_") for r in rows) and rows[0]["versions"]["selector"] == "selector-v1"
    totals = aggregate(tmp_path / "shadow.jsonl")
    assert totals["decisions"] == 6 and totals["runs"] == 2 and totals["errors"] == 0 and sum(totals["by_endpoint"].values()) == 6
    assert aggregate(tmp_path / "missing.jsonl")["decisions"] == 0


def test_decide_prices_the_choice_and_never_raises(tmp_path):
    _, selector = build_shadow(tmp_path)
    item = ShadowItem("1", "step", "Look up order A-1042", True, 600, 400)
    event = selector.decide(item, Settings(0.7, 0.6, 0.4))
    assert event["type"] == "shadow_decision" and event["needs_tools"] and selector.price(event["choice"], 1000, 100) > 0
    assert selector.price("unknown-model", 1, 1) is None
    broken = ShadowSelector(FakeCapabilityRouter(fail=True), selector.registry, baseline="gpt-5.6-sol", router_cost_usd=0.0004)
    assert broken.decide(item, Settings(0.7, 0.6, 0.4))["type"] == "shadow_error"


@pytest.mark.parametrize("fail", [False, True])
def test_opt_in_prompt_capture_is_persisted_even_when_inference_fails(tmp_path, fail):
    from routing_lab.capability import capability_state

    class InspectableRouter(FakeCapabilityRouter):
        def request_payload(self, request, context=""):
            return {"state": capability_state(request, context), "questions": {"route": {"type": "choice"}}}

    router = InspectableRouter(fail=fail)
    _, selector = build_shadow(tmp_path, router)
    selector.capture_prompts = True
    allow = frozenset(e.id for e in selector.registry.endpoints if e.api == "responses")
    item = ShadowItem("1", "step", "Look up Tokyo hotels.", True, 650, 400)
    event = selector.decide(item, Settings(0.7, 0.6, 0.4), allow)
    assert router.calls == [item.text]
    assert event["router_request"] == router.request_payload(item.text)
    assert event["selection_inputs"]["settings"]["floor"] == 0.7
    assert event["selection_inputs"]["constraints"]["allow"] == sorted(allow)
    assert event["selection_inputs"]["constraints"]["input_tokens"] == 650
    if fail:
        assert event["type"] == "shadow_error"
    else:
        assert event["type"] == "shadow_decision"
        assert len(event["candidates"]) == len(selector.registry.endpoints)
        selector.capture_prompts = False
        without_capture = selector.decide(item, Settings(0.7, 0.6, 0.4), allow)
        for key in ("choice", "expected_success", "alternatives", "floor", "effective_floor"):
            assert event[key] == without_capture[key]
    selector.log({"task_id": "capture-test", **event})
    logged = json.loads((tmp_path / "shadow.jsonl").read_text(encoding="utf-8"))
    assert logged["router_request"] == event["router_request"]


def test_prompt_capture_is_off_by_default(tmp_path):
    _, selector = build_shadow(tmp_path)
    event = selector.decide(ShadowItem("1", "step", "Private request.", True, 600, 400), Settings(0.7, 0.6, 0.4))
    assert not selector.capture_prompts
    assert "router_request" not in event and "selection_inputs" not in event


def test_replay_cost_includes_paid_but_invalid_json_outputs():
    rows = [{"key": "1", "actual": {"cost_usd": 0.3, "score": 5}}]
    replay = {"steps": {"1": {"key": "1", "status": "failed", "cost_usd": 0.2,
                             "judge_cost_usd": 0.0, "score": 1}}}
    summary = replay_summary(rows, replay, {}, 4)
    assert summary["failed"] == 1 and summary["steps_run"] == 0
    assert summary["cost_usd"] == 0.2 and summary["baseline_cost_usd"] == 0.3

def test_replay_runs_the_selected_endpoint_judges_it_and_reports_against_the_baseline(tmp_path):
    agent, selector = build_shadow(tmp_path, replay=True)
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
    plain, _ = build_shadow(tmp_path / "a", shadow=False)
    failing, _ = build_shadow(tmp_path / "b", replay=True, llm=ScriptedLLM(fail_models={"gpt-oss-120b"}))
    base, _ = run(plain)
    replayed, events = run(failing)
    for lane in ("clef", "baseline"):
        assert base["lanes"][lane]["models"] == replayed["lanes"][lane]["models"]
    result = next(e for e in events if e["type"] == "shadow_summary")["replay"]
    assert result["failed"] == 2 and result["steps_run"] == 0 and result["mean_score"] is None
    assert all(e["status"] == "failed" and "is down" in e["reason"] for e in events if e["type"] == "replay_step")


def test_replay_checks_expected_outcomes_on_its_own_desk(tmp_path):
    llm = ScriptedLLM(tool_calls=[{"name": "get_order", "arguments": {"order_id": "A-1042"}}])
    agent, _ = build_shadow(tmp_path, replay=True, llm=llm)
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
    _, selector = build_shadow(tmp_path)
    assert selector.last is None
    item = ShadowItem("1", "step", "Look up order A-1042", True, 600, 400)
    selector.decide(item, Settings(0.7, 0.6, 0.4))
    assert selector.last == {"ok": True, "latency_ms": 40.0}
    selector.router.fail = True
    selector.router = FakeCapabilityRouter(fail=True)
    assert selector.decide(item, Settings(0.7, 0.6, 0.4))["type"] == "shadow_error" and selector.last["ok"] is False

def test_warming_up_records_whether_the_router_answers_and_never_raises(tmp_path):
    _, working = build_shadow(tmp_path / "a")
    working.warm_up()
    assert working.last["ok"] is True
    _, broken = build_shadow(tmp_path / "b", FakeCapabilityRouter(fail=True))
    broken.warm_up()
    assert broken.last["ok"] is False
