"""Registry, policy, telemetry, tools, Clef client and model client: the building blocks of the router."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from agent_demo import clef_decider
from agent_demo.clef_decider import ClefDecider, describe
from agent_demo.llm import AzureResponsesClient, LLMError
from agent_demo.policy import (
    Policy, RouteProfile, assess_complexity, compute_requirement, expected_quality, select_model,
)
from agent_demo.registry import ModelProfile, Registry
from agent_demo.telemetry import Telemetry
from agent_demo.preferences import PRESETS, Preferences
from agent_demo.support import SupportDesk, responses_tools, router_tools

DATA = Path(__file__).resolve().parents[1] / "agent_demo" / "data"


@pytest.fixture
def registry(tmp_path):
    return Registry.load(DATA / "registry.json", tmp_path / "runtime.json")


@pytest.fixture
def policy():
    return Policy.load(DATA / "policy.json")


def profile(**changes):
    base = dict(id="m1", display="M1", provider="p", deployment="m1", tier="mid", price_in=1.0, price_out=2.0,
                price_source="test", lat_base_ms=1000, lat_per_token_ms=10,
                skills={"coding": 0.7, "reasoning": 0.7, "tools": 0.7, "general": 0.7}, live=True)
    return ModelProfile(**{**base, **changes})


# ----------------------------------------------------------------------------- registry
def test_seed_registry_has_real_tiers_and_one_catalog_only_model(registry):
    models = {m.id: m for m in registry.snapshot()}
    assert {m.tier for m in models.values() if m.live} == {"small", "mid", "frontier"}
    assert models["gpt-4o"].live is False
    assert models["gpt-5.6-luna"].price_in < models["gpt-5.6-terra"].price_in < models["gpt-5.6-sol"].price_in


@pytest.mark.parametrize("changes", [
    {"id": "bad id"}, {"tier": "huge"}, {"price_in": -1}, {"skills": {"coding": 0.5}},
    {"skills": {"coding": 1.5, "reasoning": 0.5, "tools": 0.5, "general": 0.5}}, {"context_window": 0},
    {"regions": []}, {"live": "yes"}, {"lat_base_ms": True},
])
def test_invalid_profiles_are_rejected(changes):
    with pytest.raises(ValueError):
        profile(**changes).validate()


def test_changes_bump_the_version_and_persist(registry, tmp_path):
    before = registry.version
    registry.update("gpt-4.1", healthy=False)
    assert registry.version != before and registry.get("gpt-4.1").healthy is False
    reloaded = Registry.load(DATA / "registry.json", tmp_path / "runtime.json")
    assert reloaded.get("gpt-4.1").healthy is False and reloaded.version == registry.version


def test_only_flags_can_change_and_unknown_models_are_refused(registry):
    with pytest.raises(ValueError):
        registry.update("gpt-4.1", price_in=0.0)
    with pytest.raises(ValueError):
        registry.update("gpt-4.1", healthy="no")
    with pytest.raises(KeyError):
        registry.update("nope", healthy=True)
    with pytest.raises(KeyError):
        registry.get("nope")


def test_onboarding_needs_only_a_profile_and_ids_are_unique(registry):
    added = registry.add(profile(id="new-model").to_dict())
    assert added.id == "new-model" and registry.get("new-model").tier == "mid"
    with pytest.raises(ValueError, match="already exists"):
        registry.add(profile(id="new-model").to_dict())
    with pytest.raises(ValueError, match="Invalid model profile"):
        registry.add({"id": "x"})


def test_a_thousand_profiles_are_ranked_quickly(policy):
    """BR-006: the router must cope with at least 1,000 model profiles."""
    models = [profile(id=f"m{i}", live=i % 7 != 0, healthy=i % 11 != 0, price_in=0.1 + i / 500,
                      skills={k: 0.3 + (i % 70) / 100 for k in ("coding", "reasoning", "tools", "general")})
              for i in range(1000)]
    registry = Registry(models)
    mode = policy.mode("balanced")
    route = policy.profile("tool-api-v1", "tool_workflow")
    requirement = compute_requirement(policy, route, "tool_workflow", assess_complexity("x"), 0.9, mode, 0)
    started = time.perf_counter()
    selection = select_model(policy, registry.snapshot(), route, requirement, mode, 800, 300)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert len(selection.candidates) == 1000 and selection.selected is not None
    assert elapsed_ms < 500


# ------------------------------------------------------------------------------- policy
def test_complexity_rises_with_code_constraints_and_reasoning_words():
    easy = assess_complexity("What is the weather in Dubai?")
    hard = assess_complexity(
        "Review this cache for race conditions and explain the root cause; it must be thread-safe.\n\n"
        "class Cache:\n    def get(self, key, loader):\n        if key not in self.data:\n"
        "            self.data[key] = loader(key)\n        return self.data[key]\n")
    assert easy.score < 0.05 < 0.15 < hard.score <= 1
    assert hard.signals["code"] > 0 and hard.signals["reasoning"] > 0 and easy.signals["code"] == 0


def test_requirement_lists_every_contribution_and_is_bounded(policy):
    route = policy.profile("coding-v1", "debugging")
    mode = policy.mode("deep")
    cx = assess_complexity("Fix the race condition and explain the root cause; it must be thread-safe.")
    req = compute_requirement(policy, route, "debugging", cx, 0.4, mode, 2)
    names = [name for name, _ in req.parts]
    assert names[0] == "step type: debugging"
    assert any("difficulty" in n for n in names) and any("unsure" in n for n in names)
    assert any("Deep setting" in n for n in names) and any("escalation" in n for n in names)
    assert req.total == min(0.98, round(sum(d for _, d in req.parts), 4)) <= 0.98
    low = compute_requirement(policy, RouteProfile("general", 0.05, 100), "fallback", assess_complexity("x"), 0.99,
                              policy.mode("fast"), 0)
    assert low.total >= 0.10


def test_quality_is_a_smooth_function_of_capability_gap():
    assert expected_quality(0.5, 0.5, 14) == pytest.approx(0.5)
    assert expected_quality(0.9, 0.5, 14) > 0.99 > expected_quality(0.5, 0.9, 14) + 0.9


def run_select(policy, models, route="tool_lookup", taxonomy="tool-api-v1", mode="balanced", text="x", **kw):
    prof = policy.profile(taxonomy, route)
    m = policy.mode(mode)
    req = compute_requirement(policy, prof, route, assess_complexity(text), 0.95, m, 0)
    return select_model(policy, models, prof, req, m, 600, 200, **kw)


def test_hard_constraints_eliminate_with_a_reason(registry, policy):
    registry.update("gpt-5.6-terra", healthy=False)
    registry.update("gpt-5.6-sol", allowed=False)
    selection = run_select(policy, registry.snapshot(), regions={"westeurope"})
    reasons = {c.model_id: c.reasons for c in selection.candidates}
    assert selection.selected is None and "No eligible model" in selection.note
    assert "not deployed (catalog only)" in reasons["gpt-4o"] and "marked unhealthy" in reasons["gpt-5.6-terra"]
    assert "not on the organization allowlist" in reasons["gpt-5.6-sol"]
    assert "region not permitted" in reasons["gpt-5.6-luna"]
    small = run_select(policy, [profile(context_window=100, supports_tools=False)], route="tool_workflow")
    assert set(small.candidates[0].reasons) == {"context window too small", "no tool-calling support"}


def test_cheap_model_for_easy_steps_and_stronger_ones_for_hard_ones(registry, policy):
    models = registry.snapshot()
    assert run_select(policy, models, "tool_lookup").selected == "gpt-5.6-luna"
    hard = run_select(policy, models, "debugging", "coding-v1",
                      text="Fix the race condition in this thread-safe cache and explain the root cause.\n\nclass C:\n    def get(self, k):\n        return self.d[k]")
    assert hard.selected == "gpt-5.6-sol"


@pytest.mark.parametrize(("route", "taxonomy"), [("debugging", "coding-v1"), ("tool_workflow", "tool-api-v1"),
                                                   ("code_review", "coding-v1"), ("tool_action", "tool-api-v1")])
def test_modes_order_the_tier_and_cost_of_the_choice(registry, policy, route, taxonomy):
    models = {m.id: m for m in registry.snapshot()}
    rank = {"small": 0, "mid": 1, "frontier": 2}
    price = lambda sel: models[sel.selected].price_out  # noqa: E731
    fast, balanced, deep = (run_select(policy, list(models.values()), route, taxonomy, mode=m) for m in ("fast", "balanced", "deep"))
    assert rank[models[fast.selected].tier] <= rank[models[balanced.selected].tier] <= rank[models[deep.selected].tier]
    assert price(fast) <= price(deep)


def test_below_floor_is_flagged_when_nothing_is_capable_enough(policy):
    weak = [profile(id="weak", skills={k: 0.2 for k in ("coding", "reasoning", "tools", "general")})]
    selection = run_select(policy, weak, "debugging", "coding-v1")
    assert selection.selected == "weak" and selection.below_floor and "quality floor" in selection.note


def test_budget_removes_unaffordable_models_or_falls_back_to_the_cheapest(registry, policy):
    models = registry.snapshot()
    capped = run_select(policy, models, "tool_workflow", remaining_budget=0.0035)
    expensive = [c for c in capped.candidates if c.model_id == "gpt-5.6-sol"][0]
    assert any("exceeds remaining budget" in r for r in expensive.reasons) and capped.selected != "gpt-5.6-sol"
    broke = run_select(policy, models, "tool_workflow", remaining_budget=0.0)
    assert broke.budget_overridden and broke.selected == "gpt-5.6-luna" and "cheapest" in broke.note


def test_excluded_models_are_not_picked_again(registry, policy):
    first = run_select(policy, registry.snapshot(), "tool_action")
    again = run_select(policy, registry.snapshot(), "tool_action", exclude=frozenset({first.selected}))
    assert again.selected not in (None, first.selected)


def test_policy_validation_and_versioning():
    data = json.loads((DATA / "policy.json").read_text(encoding="utf-8-sig"))
    assert Policy(data).version == Policy(json.loads(json.dumps(data))).version
    data["modes"].pop("deep")
    with pytest.raises(ValueError, match="fast, balanced and deep"):
        Policy(data)
    with pytest.raises(ValueError, match="Invalid routing policy"):
        Policy({})
    with pytest.raises(ValueError, match="Unknown mode"):
        Policy.load(DATA / "policy.json").mode("turbo")
    with pytest.raises(ValueError, match="No routing profile"):
        Policy.load(DATA / "policy.json").profile("coding-v1", "nope")


def test_every_clef_route_has_a_policy_profile(policy):
    from routing_lab.taxonomy import taxonomy_routes

    for taxonomy in ("coding-v1", "tool-api-v1"):
        assert set(taxonomy_routes(taxonomy)) == set(policy.profiles[taxonomy])


# --------------------------------------------------------------------------- telemetry
def decision_row(trace_id="tr_000000000001", **changes):
    row = {"trace_id": trace_id, "task_id": "tk1", "step_id": "tk1.s1", "created_utc": "2026-10-08T00:00:00+00:00",
           "mode": "balanced", "shadow": 0, "router_version": "v", "registry_version": "r1", "policy_version": "p",
           "domain": "tools", "taxonomy": "tool-api-v1", "route": "tool_lookup", "confidence": 0.9,
           "clef_source": "clef", "clef_ms": 400.0, "router_ms": 450.0, "requirement": 0.4,
           "text_sha256": "a" * 64, "text_chars": 30, "complexity": 0.1, "selected_model": "gpt-5.6-luna",
           "selected_tier": "small", "below_floor": 0, "attempt": 0, "candidates": "[]"}
    return {**row, **changes}


def test_traces_hold_versions_candidates_and_outcome_but_no_prompt_text():
    t = Telemetry(":memory:")
    t.log_decision(decision_row(candidates=json.dumps([{"model_id": "m"}])))
    t.log_outcome("tr_000000000001", exec_ms=900.0, input_tokens=10, output_tokens=5, cost_usd=0.001,
                  baseline_cost_usd=0.01, outcome="ok", judge_score=5)
    trace = t.trace("tr_000000000001")
    assert trace["router_version"] == "v" and trace["registry_version"] == "r1" and trace["candidates"] == [{"model_id": "m"}]
    assert trace["outcome"] == "ok" and trace["text_chars"] == 30 and len(trace["text_sha256"]) == 64
    assert not any(isinstance(v, str) and "weather" in v for v in trace.values())
    assert t.trace("tr_missing") is None


def test_outcome_updates_are_validated():
    t = Telemetry(":memory:")
    t.log_decision(decision_row())
    with pytest.raises(ValueError, match="Unknown outcome fields"):
        t.log_outcome("tr_000000000001", selected_model="x")
    with pytest.raises(KeyError):
        t.log_outcome("tr_nope", outcome="ok")
    t.log_outcome("tr_000000000001")


def test_aggregates_separate_shadow_runs_and_compute_savings():
    t = Telemetry(":memory:")
    live = dict(cost_usd=0.002, baseline_cost_usd=0.010, exec_ms=2000.0, outcome="ok", judge_score=4)
    for i, (model, tier) in enumerate([("gpt-5.6-luna", "small"), ("gpt-4.1", "mid")]):
        t.log_decision(decision_row(f"tr_00000000000{i}", selected_model=model, selected_tier=tier))
        t.log_outcome(f"tr_00000000000{i}", **live)
    t.log_decision(decision_row("tr_000000000009", shadow=1, selected_model="gpt-5.6-luna"))
    t.log_outcome("tr_000000000009", cost_usd=0.010, baseline_cost_usd=0.010, would_cost_usd=0.001, exec_ms=3000.0, outcome="ok")
    t.log_decision(decision_row("tr_00000000000a", attempt=1, clef_source="fallback", confidence=None, below_floor=1))
    agg = t.aggregates()
    assert agg["decisions"] == 4 and agg["executed_steps"] == 2
    assert agg["savings_pct"] == 80.0 and agg["cost_usd"] == 0.004
    assert agg["shadow"]["would_savings_pct"] == 90.0 and agg["shadow"]["steps"] == 1
    assert agg["escalation_rate_pct"] == pytest.approx(33.3, abs=0.1)
    assert agg["fallback_decisions"] == 1 and agg["below_floor"] == 1 and agg["judge_mean"] == 4
    assert agg["tier_mix"] == {"small": 1, "mid": 1} and agg["router_ms_p50"] == 450.0
    assert Telemetry(":memory:").aggregates() == {"decisions": 0}


# ------------------------------------------------------------------------------- support desk
def test_each_desk_is_isolated_and_deterministic():
    a, b = SupportDesk(), SupportDesk()
    assert a.execute("get_order", {"order_id": "a-1042"})["item_total"] == 129.0
    a.execute("issue_refund", {"order_id": "A-1042", "amount": 138.99, "reason": "damaged"})
    assert a.orders["A-1042"]["refunded"] == 138.99 and b.orders["A-1042"]["refunded"] == 0.0
    assert a.changes() == ["Refunded $138.99 on A-1042"]
    assert b.execute("get_customer", {"email_or_id": "ELENA.petrova@example.com"})["tier"] == "platinum"


def test_refund_and_credit_rules_are_enforced_by_the_desk():
    desk = SupportDesk()
    assert "approval" in desk.execute("issue_refund", {"order_id": "A-1075", "amount": 300, "reason": "x"})["error"]
    assert "error" in desk.execute("issue_refund", {"order_id": "A-1042", "amount": 500, "reason": "x"})
    assert "error" in desk.execute("issue_refund", {"order_id": "A-1042", "amount": 0, "reason": "x"})
    desk.execute("issue_refund", {"order_id": "A-1042", "amount": 138.99, "reason": "x"})
    assert "error" in desk.execute("issue_refund", {"order_id": "A-1042", "amount": 1, "reason": "again"})
    assert "approval" in desk.execute("add_store_credit", {"customer_id": "C-203", "amount": 151, "reason": "x"})["error"]
    assert desk.execute("add_store_credit", {"customer_id": "C-203", "amount": 109.8, "reason": "x"})["balance"] == 109.8
    large = SupportDesk()
    large.orders["A-1075"]["item_total"] = 900.0
    assert "approval" in large.execute("issue_refund", {"order_id": "A-1075", "amount": 250, "reason": "x"})["error"]


def test_return_window_and_order_state_rules():
    desk = SupportDesk()
    assert desk.execute("check_return_eligibility", {"order_id": "A-1042"})["eligible"] is True
    old = desk.execute("check_return_eligibility", {"order_id": "A-1061"})
    assert old["eligible"] is False and old["days_since_delivery"] == 41
    assert "error" in desk.execute("update_shipping_address", {"order_id": "A-1057", "new_address": "x"})
    assert "error" in desk.execute("cancel_order", {"order_id": "A-1057"})
    assert desk.execute("update_shipping_address", {"order_id": "A-1088", "new_address": " 14 Harbour Lane "})["address"] == "14 Harbour Lane"
    assert desk.execute("cancel_order", {"order_id": "A-1088"}) == {"cancelled": "A-1088"}
    assert desk.execute("track_shipment", {"order_id": "A-1075"})["eta"] == "unknown"
    assert desk.execute("search_help_center", {"query": "late delivery compensation"})["articles"][0]["id"] == "HC-2"


def test_bad_tool_calls_return_errors_instead_of_raising():
    desk = SupportDesk()
    assert "error" in desk.execute("nope", {}) and "error" in desk.execute("get_order", {})
    assert "error" in desk.execute("get_order", {"order_id": "Z-1"}) and "error" in desk.execute("get_customer", {"email_or_id": "x"})
    assert "error" in desk.execute("escalate_to_human", {"order_id": "A-1042", "summary": "s", "priority": "urgent"})
    assert desk.changes() == [] and all(not c["ok"] for c in desk.calls)


def test_outcome_checks_compare_arguments_loosely_and_support_absence():
    desk = SupportDesk()
    desk.execute("issue_refund", {"order_id": "A-1042", "amount": 138.99, "reason": "x"})
    desk.execute("update_shipping_address", {"order_id": "A-1088", "new_address": "14 Harbour Lane, Leeds"})
    checks = desk.check([
        {"label": "refund", "tool": "issue_refund", "args": {"order_id": "A-1042", "amount": 138.99}},
        {"label": "wrong amount", "tool": "issue_refund", "args": {"amount": 129.0}},
        {"label": "address", "tool": "update_shipping_address", "args": {"new_address": "harbour lane"}},
        {"label": "no cancel", "tool": "cancel_order", "absent": True},
        {"label": "no refund", "tool": "issue_refund", "absent": True},
    ])
    assert [c["passed"] for c in checks] == [True, False, True, True, False]
    failed = SupportDesk()
    failed.execute("issue_refund", {"order_id": "A-1042", "amount": 999, "reason": "x"})
    assert failed.check([{"label": "refund", "tool": "issue_refund"}])[0]["passed"] is False


def test_tool_rosters_agree_across_formats():
    api, trained = responses_tools(), router_tools()
    assert [t["name"] for t in api] == [t["name"] for t in trained]
    assert all(t["type"] == "function" for t in api) and all(t["parameters"]["type"] == "dict" for t in trained)
    assert set(trained[0]) == {"description", "name", "parameters"}


# ------------------------------------------------------------------------------- preferences
def test_preferences_validate_levels_and_recognize_presets():
    assert Preferences().preset() == "balanced"
    assert Preferences.from_dict(PRESETS["lean"]).preset() == "lean"
    assert Preferences(quality="average", latency="high", cost="medium").preset() is None
    for bad in ({"quality": "great"}, {"latency": "fast"}, {"cost": 3}, {"extra": "x"}):
        with pytest.raises(ValueError):
            Preferences.from_dict(bad)
    with pytest.raises(ValueError):
        Preferences.from_dict(["quality"])


def test_preferences_map_monotonically_onto_routing_parameters(policy):
    derived = {level: policy.derive(Preferences(quality=level)) for level in ("average", "good", "very_good")}
    floors = [derived[k].mode.quality_floor for k in ("average", "good", "very_good")]
    assert floors == sorted(floors) and len(set(floors)) == 3
    assert derived["very_good"].gate_tiers == ("small", "mid") and derived["average"].gate_tiers == ("small",)
    mu = [policy.derive(Preferences(latency=x)).mode.mu_latency for x in ("low", "medium", "high")]
    lam = [policy.derive(Preferences(cost=x)).mode.lambda_cost for x in ("low", "medium", "high")]
    assert mu == sorted(mu, reverse=True) and lam == sorted(lam, reverse=True)
    assert "70%" in derived["good"].notes["quality"] and derived["good"].mode.key == "custom"


def test_preferences_change_the_model_chosen_for_the_same_step(registry, policy):
    profile = policy.profile("tool-api-v1", "tool_workflow")

    def choice(prefs):
        mode = policy.derive(prefs).mode
        requirement = compute_requirement(policy, profile, "tool_workflow", assess_complexity("Look up an order."), 0.95, mode, 0)
        return select_model(policy, registry.snapshot(), profile, requirement, mode, 900, 200).selected

    lean = choice(Preferences(**PRESETS["lean"]))
    premium = choice(Preferences(**PRESETS["premium"]))
    assert registry.get(lean).tier != "frontier" and registry.get(premium).tier == "frontier"


def test_money_moving_and_customer_facing_steps_need_more_capability(policy):
    profile = policy.profile("tool-api-v1", "tool_workflow")
    mode = policy.derive(Preferences()).mode
    base = compute_requirement(policy, profile, "tool_workflow", assess_complexity("Look up order A-1042."), 0.95, mode, 0)
    moves = compute_requirement(policy, profile, "tool_workflow", assess_complexity("x"), 0.95, mode, 0, stakes=True)
    reply = compute_requirement(policy, profile, "tool_workflow", assess_complexity("x"), 0.95, mode, 0, customer_facing=True)
    assert moves.total > base.total and reply.total > base.total
    assert any("changes money" in name for name, _ in moves.parts) and any("customer-facing" in name for name, _ in reply.parts)
    for text, hit in [("Issue a refund of $138.99 on A-1042", True), ("Add store credit to C-203", True),
                      ("Update the shipping address on A-1088", True), ("Cancel order A-1088", True),
                      ("Search the help center for refund and compensation policies", False),
                      ("Get order and shipment details", False), ("Write the final reply", False)]:
        assert bool(policy.stakes.search(text)) is hit, text


def test_run_aggregates_report_measured_two_lane_totals():
    telemetry = Telemetry(":memory:")
    assert telemetry.run_aggregates() == {"runs": 0}

    def summary(task, clef_cost, base_cost):
        lane = lambda cost, tokens, score: {"cost_usd": cost, "tokens_total": tokens, "latency_ms": 1000.0, "reply_score": score,  # noqa: E731
                                            "models": {"gpt-4.1": {"steps": 2, "cost_usd": cost}}}
        return {"task_id": task, "preferences": Preferences().to_dict(),
                "lanes": {"clef": lane(clef_cost, 800, 5), "baseline": lane(base_cost, 1000, 4)},
                "saved": {"cost": {"pct": 50.0}, "tokens": {"pct": 20.0}, "latency": {"pct": 0.0}},
                "checks_passed": {"clef": [2, 2], "baseline": [1, 2]}}
    telemetry.log_run(summary("tk_a", 0.01, 0.02))
    telemetry.log_run(summary("tk_b", 0.03, 0.06))
    agg = telemetry.run_aggregates()
    assert agg["runs"] == 2 and agg["cost_usd"] == {"clef": 0.04, "baseline": 0.08, "saved_pct": 50.0}
    assert agg["tokens"]["saved_pct"] == 20.0 and agg["reply_score"] == {"clef": 5.0, "baseline": 4.0}
    assert agg["checks"] == {"clef": [4, 4], "baseline": [2, 4]} and agg["model_mix"]["gpt-4.1"]["steps"] == 4
    assert len(agg["recent"]) == 2 and "case" not in json.dumps(agg)


# --------------------------------------------------------------------------- Clef client
def test_describe_reports_confidence_margin_and_entropy():
    top, margin, entropy = describe({"a": 0.9, "b": 0.05, "c": 0.05})
    assert (top, margin) == (0.9, 0.85) and 0 < entropy < 0.5
    assert describe({"a": 0.5, "b": 0.5})[2] == 1.0 and describe({"a": 1.0, "b": 0.0})[2] == 0.0


def make_decider(backend):
    return ClefDecider("https://x.eastus2.inference.ml.azure.com/score", "flash-ft-gpu", "a" * 64,
                       {"code": "debugging", "tools": "tool_workflow"}, backends={"coding-v1": backend, "tool-api-v1": backend})


def test_decider_formats_the_state_exactly_as_the_model_was_trained():
    backend = Mock()
    backend.predict_state.return_value = SimpleNamespace(
        route="tool_lookup", probabilities={"tool_lookup": 0.9, "tool_action": 0.1}, input_tokens=123)
    decision = make_decider(backend).decide("tools", "What is the weather?", tools=router_tools()[:1])
    sent = backend.predict_state.call_args.args[0]
    assert sent.startswith("User request:\nWhat is the weather?\nAvailable context:\n")
    context = json.loads(sent.split("Available context:\n", 1)[1])
    assert set(context) == {"available_functions", "preceding_instructions"}
    assert decision.route == "tool_lookup" and decision.confidence == 0.9 and decision.source == "clef"
    code = make_decider(backend).decide("code", "Fix it")
    assert json.loads(backend.predict_state.call_args.args[0].split("Available context:\n", 1)[1]) == {
        "repository": "demo/app", "repository_content_supplied": False}
    assert code.taxonomy == "coding-v1"


def test_decider_never_raises_and_degrades_to_the_safe_default():
    backend = Mock()
    backend.predict_state.side_effect = ValueError("provenance differs")
    decision = make_decider(backend).decide("tools", "anything")
    assert decision.source == "fallback" and decision.route == "tool_workflow" and "provenance" in decision.error
    assert decision.confidence is None and decision.probabilities == {}


def test_decider_retries_throttling_but_not_other_errors(monkeypatch):
    monkeypatch.setattr(clef_decider.time, "sleep", lambda s: None)
    ok = SimpleNamespace(route="tool_lookup", probabilities={"tool_lookup": 1.0, "x": 0.0}, input_tokens=1)
    backend = Mock()
    backend.predict_state.side_effect = [RuntimeError("Clef endpoint returned HTTP 429"), ok]
    assert make_decider(backend).decide("tools", "x").source == "clef"
    assert backend.predict_state.call_count == 2
    backend.predict_state.reset_mock()
    backend.predict_state.side_effect = RuntimeError("Clef endpoint returned HTTP 500")
    assert make_decider(backend).decide("tools", "x").source == "fallback" and backend.predict_state.call_count == 1


def test_decider_reconnects_once_when_an_idle_connection_was_closed(monkeypatch):
    import requests

    monkeypatch.setattr(clef_decider.time, "sleep", lambda s: None)
    ok = SimpleNamespace(route="tool_lookup", probabilities={"tool_lookup": 1.0, "x": 0.0}, input_tokens=1)
    backend = Mock()
    backend.predict_state.side_effect = [requests.ConnectionError("Remote end closed connection"), ok]
    assert make_decider(backend).decide("tools", "x").source == "clef"
    backend.reset_connection.assert_called_once()
    backend.reset_mock()
    backend.predict_state.side_effect = requests.ConnectionError("down")
    decision = make_decider(backend).decide("tools", "x")
    assert decision.source == "fallback" and "ConnectionError" in decision.error and backend.reset_connection.call_count == 3


# ------------------------------------------------------------------------- model client
class FakeCredential:
    def __init__(self):
        self.calls = 0

    def get_token(self, scope):
        self.calls += 1
        assert scope == "https://ai.azure.com/.default"
        return SimpleNamespace(token="tok", expires_on=time.time() + 3600)


def response(status=200, payload=None):
    r = Mock()
    r.status_code, r.text = status, "body"
    r.json.return_value = payload or {}
    return r


def test_client_parses_text_usage_and_function_calls_and_caches_the_token():
    payload = {"id": "r1", "status": "completed", "usage": {"input_tokens": 10, "output_tokens": 4}, "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
        {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "Dubai"}'},
        {"type": "function_call", "call_id": "c2", "name": "x", "arguments": "not json"}]}
    session = Mock()
    session.post.return_value = response(200, payload)
    credential = FakeCredential()
    client = AzureResponsesClient("https://r.services.ai.azure.com/openai/v1/", credential=credential, session=session)
    first = client.complete("gpt-5.6-luna", system="s", user="u", tools=[{"type": "function"}], json_object=True)
    client.complete("gpt-5.6-luna", user="u")
    assert (first.text, first.input_tokens, first.output_tokens, first.response_id) == ("hello", 10, 4, "r1")
    assert first.function_calls == [{"call_id": "c1", "name": "get_weather", "arguments": {"city": "Dubai"}},
                                    {"call_id": "c2", "name": "x", "arguments": {}}]
    body = session.post.call_args_list[0].kwargs["json"]
    assert body["model"] == "gpt-5.6-luna" and body["text"] == {"format": {"type": "json_object"}} and body["tools"]
    assert body["input"][0] == {"role": "system", "content": "s"}
    assert session.post.call_args.kwargs["allow_redirects"] is False and credential.calls == 1


def test_client_retries_transient_failures_and_raises_otherwise(monkeypatch):
    monkeypatch.setattr("agent_demo.llm.time.sleep", lambda s: None)
    session = Mock()
    ok = response(200, {"id": "r", "status": "completed", "usage": {}, "output": []})
    session.post.side_effect = [response(429), requests.ConnectionError("x"), ok]
    client = AzureResponsesClient("https://r.openai.azure.com/openai/v1/", credential=FakeCredential(), session=session)
    assert client.complete("m", user="u").status == "completed" and session.post.call_count == 3
    session.post.side_effect = [response(400)]
    with pytest.raises(LLMError, match="HTTP 400"):
        client.complete("m", user="u")
    session.post.side_effect = [response(503)] * 3
    with pytest.raises(LLMError, match="after retries"):
        client.complete("m", user="u")


@pytest.mark.parametrize("url", ["http://r.openai.azure.com/openai/v1/", "https://evil.example.com/openai/v1/",
                                 "https://r.openai.azure.com/other/", "https://r.openai.azure.com.evil.com/openai/v1/"])
def test_client_only_talks_to_azure_endpoints(url):
    with pytest.raises(ValueError, match="Azure"):
        AzureResponsesClient(url)


