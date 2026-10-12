"""The benchmark runner and analysis, with scripted endpoints and no network."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from agent_demo.llm import LLMError, LLMResult
from capability_lab import analysis
from capability_lab.clients import BenchClient, load_models
from capability_lab.runner import load_runs, pending, pilot_subset, run_benchmark, run_one
from capability_lab.tasks import Task, build_tasks

TASKS = build_tasks(per_family=3, support_count=6)
BY_PROMPT = {t.prompt: t for t in TASKS}
MODELS = load_models()


class ScriptedClient:
    """Answers with the gold answer for the models named in `good`, and with nonsense for the rest."""

    def __init__(self, good: set[str], fail: set[str] | None = None) -> None:
        self.good, self.fail = good, fail or set()
        self.calls: list[tuple[str, str]] = []

    def _text(self, deployment: str, prompt: str) -> LLMResult:
        self.calls.append((deployment, prompt))
        if deployment in self.fail:
            raise LLMError(f"{deployment} is down")
        task = BY_PROMPT.get(prompt)
        text = task.gold if task and deployment in self.good else "I cannot help with that."
        return LLMResult(text, 100, 40, 500.0, "completed")

    def complete(self, deployment, *, system=None, user=None, tools=None, input_items=None, previous_response_id=None, max_output_tokens=0, **_):
        if tools is None:
            return self._text(deployment, user)
        # tool tasks: the "good" models issue the right store actions, the others do nothing
        self.calls.append((deployment, user or ""))
        if input_items is None and deployment in self.good:
            task = BY_PROMPT[user]
            actions = {"damaged-jacket": [("issue_refund", {"order_id": "A-1042", "amount": 138.99, "reason": "damaged"})],
                       "address-change": [("update_shipping_address", {"order_id": "A-1088", "new_address": "14 Harbour Lane, Leeds"})],
                       "late-vip-order": [("issue_refund", {"order_id": "A-1075", "amount": 24.99, "reason": "late"}),
                                          ("add_store_credit", {"customer_id": "C-203", "amount": 109.8, "reason": "late"})],
                       "warranty-claim": [("escalate_to_human", {"order_id": "A-1061", "summary": "warranty", "priority": "normal"})],
                       "where-is-my-order": [("track_shipment", {"order_id": "A-1057"})]}.get(task.meta["case"], [])
            calls = [{"call_id": f"c{i}", "name": n, "arguments": a} for i, (n, a) in enumerate(actions)]
            return LLMResult("", 200, 50, 800.0, "completed", "r1", calls)
        return LLMResult("Done.", 100, 20, 400.0, "completed", "r2")

    def chat(self, deployment, system, user, max_tokens=0):
        return self._text(deployment, user)


def test_a_correct_answer_passes_and_is_priced_from_the_endpoint_price():
    task = next(t for t in TASKS if t.family == "arithmetic_steps")
    record = run_one(ScriptedClient({"gpt-5.6-sol"}), MODELS["gpt-5.6-sol"], task)
    assert record["passed"] is True and record["error"] is None and record["tokens_in"] == 100 and record["tokens_out"] == 40
    assert record["cost_usd"] == pytest.approx(100 * 4.0 / 1e6 + 40 * 20.0 / 1e6)
    wrong = run_one(ScriptedClient(set()), MODELS["gpt-5.6-sol"], task)
    assert wrong["passed"] is False and wrong["error"] is None


def test_an_unreachable_endpoint_is_an_error_never_a_wrong_answer():
    task = TASKS[0]
    record = run_one(ScriptedClient(set(), fail={"gpt-5.6-sol"}), MODELS["gpt-5.6-sol"], task)
    assert record["error"] and record["passed"] is None


def test_phi_uses_chat_completions_and_is_not_asked_tool_tasks():
    client = ScriptedClient({"phi-4"})
    task = next(t for t in TASKS if t.family == "classify_intent")
    assert run_one(client, MODELS["phi-4"], task)["passed"] is True
    work = pending(TASKS, [MODELS["phi-4"], MODELS["gpt-5.6-luna"]], set())
    assert not any(t.uses_tools and m.id == "phi-4" for t, m, _ in work)
    assert sum(m.id == "gpt-5.6-luna" for _, m, _ in work) == len(TASKS)


def test_tool_tasks_are_graded_by_what_changed_in_the_store():
    task = next(t for t in TASKS if t.meta.get("case") == "damaged-jacket")
    good = run_one(ScriptedClient({"gpt-4.1"}), MODELS["gpt-4.1"], task)
    assert good["passed"] is True and good["detail"] == "ok"
    bad = run_one(ScriptedClient(set()), MODELS["gpt-4.1"], task)
    assert bad["passed"] is False and "refund" in bad["detail"].lower()


def test_a_run_can_be_stopped_and_resumed_without_repeating_work(tmp_path):
    out = tmp_path / "runs.jsonl"
    subset = pilot_subset(TASKS, 1)
    models = [MODELS["gpt-5.6-luna"], MODELS["gpt-5.6-sol"]]
    client = ScriptedClient({"gpt-5.6-sol"})
    first = run_benchmark(client, subset[:3], models, out, workers=2)
    assert first["attempted"] == 6 and first["errors"] == 0
    second = run_benchmark(client, subset, models, out, workers=2)
    assert second["skipped_already_done"] == 6 and second["attempted"] == len(pending(subset, models, set())) - 6
    rows = load_runs(out)
    assert len({(r["task_id"], r["model"], r["sample"]) for r in rows}) == len(rows)
    flaky = ScriptedClient({"gpt-5.6-sol"}, fail={"gpt-5.6-luna"})
    out2 = tmp_path / "flaky.jsonl"
    run_benchmark(flaky, subset[:2], models, out2, workers=1)
    retry = run_benchmark(client, subset[:2], models, out2, workers=1)
    assert retry["attempted"] == 2  # only the errored attempts are retried


def test_the_pilot_takes_a_few_tasks_from_every_family_across_levels():
    subset = pilot_subset(TASKS, 3)
    families = {t.family for t in subset}
    assert len(families) == 22 and all(sum(t.family == f for t in subset) <= 3 for f in families)
    assert {t.level for t in subset if t.family == "arithmetic_steps"} == {1, 2, 3}


def test_the_chat_path_reads_content_and_token_usage():
    session = Mock()
    session.post.return_value = SimpleNamespace(status_code=200, json=lambda: {"choices": [{"message": {"content": " pong "}}],
                                                                              "usage": {"prompt_tokens": 14, "completion_tokens": 3}})
    client = BenchClient(credential=Mock(get_token=Mock(return_value=SimpleNamespace(token="t", expires_on=9e12))), session=session)
    result = client.chat("phi-4", "sys", "ping")
    assert (result.text, result.input_tokens, result.output_tokens) == ("pong", 14, 3)
    assert session.post.call_args.args[0].endswith("chat/completions") and session.post.call_args.kwargs["json"]["max_tokens"] == 3000
    session.post.return_value = SimpleNamespace(status_code=400, text="no")
    with pytest.raises(LLMError):
        client.chat("phi-4", "sys", "ping")


# ---------------------------------------------------------------------------------------- analysis
def synthetic(patterns: dict[str, dict[str, bool]], cost: dict[str, float]) -> tuple[list[Task], list[dict]]:
    tasks = [Task(id=tid, family="f", skill="reasoning", level=1, prompt="p", verifier={"type": "exact", "value": "x"})
             for tid in patterns]
    runs = [{"task_id": tid, "model": m, "sample": 0, "passed": ok, "error": None, "cost_usd": cost[m], "latency_ms": 1000.0,
             "tokens_in": 10, "tokens_out": 10, "detail": "", "output": ""} for tid, row in patterns.items() for m, ok in row.items()]
    return tasks, runs


MODEL_COST = {"a": 0.001, "b": 0.002, "c": 0.01, "d": 0.02, "e": 0.05}


def row(*passing: str) -> dict[str, bool]:
    return {m: m in passing for m in MODEL_COST}


def test_tiers_come_from_the_share_of_endpoints_that_solve_a_task():
    tasks, runs = synthetic({"easy": row("a", "b", "c", "d", "e"), "mid": row("c", "d", "e"), "hard": row("e"), "impossible": row()}, MODEL_COST)
    labels = {label["task_id"]: label for label in analysis.tier_labels(tasks, runs)}
    assert {k: v["tier"] for k, v in labels.items()} == {"easy": "low", "mid": "medium", "hard": "high", "impossible": "high"}
    assert labels["easy"]["cheapest_passing"] == "a" and labels["mid"]["cheapest_passing"] == "c" and labels["hard"]["cheapest_passing"] == "e"
    assert labels["impossible"]["unsolved"] and labels["impossible"]["cheapest_passing"] is None and labels["mid"]["pass_fraction"] == 0.6
    assert analysis.classify(0.7) == "low" and analysis.classify(0.35) == "medium" and analysis.classify(0.34) == "high"
    assert analysis.classify(0.5, low_at_least=0.4) == "low"


def test_errors_are_missing_data_and_thinly_answered_tasks_get_no_label():
    tasks, runs = synthetic({"t": row("a", "b", "c", "d", "e")}, MODEL_COST)
    runs[0]["error"], runs[0]["passed"] = "boom", None
    assert analysis.pass_matrix(runs)["t"].keys() == {"b", "c", "d", "e"}
    assert analysis.tier_labels(tasks, runs, min_endpoints=5) == []
    assert analysis.tier_labels(tasks, runs, min_endpoints=4)[0]["pass_fraction"] == 1.0


def test_profiles_report_pass_rates_with_intervals_by_skill_and_tier():
    patterns = {f"e{i}": row("a", "b", "c", "d", "e") for i in range(8)} | {f"h{i}": row("e") for i in range(8)}
    tasks, runs = synthetic(patterns, MODEL_COST)
    profile = analysis.profiles(analysis.tier_labels(tasks, runs), runs)
    assert profile["a"]["reasoning"]["low"] == {"pass_rate": 1.0, "n": 8, "ci_low": profile["a"]["reasoning"]["low"]["ci_low"], "ci_high": 1.0}
    assert profile["a"]["reasoning"]["high"]["pass_rate"] == 0.0 and profile["e"]["reasoning"]["high"]["pass_rate"] == 1.0
    assert 0.6 < profile["a"]["reasoning"]["low"]["ci_low"] < 0.8 and analysis.wilson(0, 0) == (0.0, 1.0)


def test_a_selector_with_perfect_tiers_saves_money_without_losing_passes():
    patterns = {f"e{i}": row("a", "b", "c", "d", "e") for i in range(8)} | {f"h{i}": row("e") for i in range(8)}
    tasks, runs = synthetic(patterns, MODEL_COST)
    labels = analysis.tier_labels(tasks, runs)
    out = analysis.strategies(tasks, labels, runs)
    assert out["always"]["e"] == {"pass_rate": 1.0, "mean_cost_usd": 0.05, "tasks": 16}
    assert out["always"]["a"]["pass_rate"] == 0.5 and out["oracle"]["pass_rate"] == 1.0
    chosen = out["perfect_tier_selector"]
    assert chosen["pass_rate"] == 1.0 and chosen["mean_cost_usd"] == pytest.approx((0.001 + 0.05) / 2, abs=1e-6)
    assert chosen["models_used"] == {"a": 8, "e": 8}


def test_the_selector_does_not_peek_at_the_task_it_is_choosing_for():
    # One easy task that only the expensive model solves: its own result must not make "a" look good for it.
    patterns = {f"e{i}": row("a", "b", "c", "d", "e") for i in range(6)}
    patterns["odd"] = row("b", "c", "d", "e")
    tasks, runs = synthetic(patterns, MODEL_COST)
    labels = analysis.tier_labels(tasks, runs)
    result = analysis.simulate_selector(tasks, labels, runs, floor=1.0)
    assert result["tasks"] == 7 and result["pass_rate"] < 1.0  # "a" is chosen for "odd" because the others all passed
    named = {m: MODELS["gpt-5.6-luna"] for m in ("a",)}
    assert analysis.model_summary(runs, named)[0]["pass_rate"] == pytest.approx(6 / 7, abs=1e-3)
    json.dumps(result)
