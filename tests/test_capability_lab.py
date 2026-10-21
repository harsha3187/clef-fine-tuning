"""The benchmark's tasks and checkers: every gold answer passes, wrong answers fail, and the set is reproducible."""

from __future__ import annotations

import json
import random

import pytest

from agent_demo.support import SupportDesk
from capability_lab import verifiers
from capability_lab.generators import GENERATORS, MAX_LEVEL, POLICY, count_solutions, policy_total
from capability_lab.tasks import Task, build_tasks, load_tasks, save_tasks, tasks_hash
from capability_lab.verifiers import first_json, last_number, verify
from agent_demo.llm import LLMError, LLMResult
from capability_lab import analysis, evaluate
from capability_lab.clients import BenchClient, load_models
from capability_lab.runner import load_runs, pending, pilot_subset, run_benchmark, run_one
from types import SimpleNamespace
from unittest.mock import Mock


@pytest.mark.parametrize("family", list(GENERATORS))
@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_every_generated_gold_answer_passes_its_own_checker(family, level):
    skill, generate = GENERATORS[family]
    for seed in range(12):
        built = generate(random.Random(f"{family}:{level}:{seed}"), level)
        ok, detail = verify(built["verifier"], built["gold"])
        assert ok, (family, level, seed, detail, built["prompt"][:200])
        assert built["prompt"].strip() and skill in {"extraction", "reasoning", "coding", "tools", "writing"}


@pytest.mark.parametrize("family", [f for f in GENERATORS if f not in ("constrained_reply", "summarize_facts", "schedule_csp")])
def test_a_wrong_answer_fails_the_checker(family):
    _, generate = GENERATORS[family]
    built = generate(random.Random(f"{family}:wrong"), 2)
    for wrong in ("I don't know", "", "42424242", "[]", "{}", '{"calls": []}'):
        assert verify(built["verifier"], wrong)[0] is False or wrong in built["gold"]


def test_checks_and_csp_reject_violations():
    _, summarize = GENERATORS["summarize_facts"]
    built = summarize(random.Random(1), 3)
    assert verify(built["verifier"], "nothing relevant here")[0] is False
    padded = built["gold"] + " " + " ".join(["filler"] * 100)
    assert verify(built["verifier"], padded) == (False, verify(built["verifier"], padded)[1]) and "words" in verify(built["verifier"], padded)[1]
    _, reply = GENERATORS["constrained_reply"]
    spec = reply(random.Random(2), 3)
    assert verify(spec["verifier"], spec["gold"])[0] and not verify(spec["verifier"], spec["gold"].replace("within 5 business days", "soon"))[0]
    assert not verify(spec["verifier"], "Sorry! " + spec["gold"])[0]
    _, csp = GENERATORS["schedule_csp"]
    task = csp(random.Random(3), 2)
    gold = json.loads(task["gold"])
    broken = {**gold, next(iter(gold)): "Sun"}
    assert verify(task["verifier"], json.dumps(gold))[0] and not verify(task["verifier"], json.dumps(broken))[0]
    assert not verify(task["verifier"], json.dumps({"only": "Mon"}))[0]


@pytest.mark.parametrize("level", [1, 2, 3, 4])
def test_ordering_puzzles_have_exactly_one_answer(level):
    _, generate = GENERATORS["ordering_puzzle"]
    for seed in range(6):
        built = generate(random.Random(seed), level)
        people, rules = built["meta"]["people"], [tuple(r) for r in built["meta"]["rules"]]
        assert count_solutions(people, rules) == 1
        assert len(built["verifier"]["value"]) == len(people)


def test_the_policy_arithmetic_follows_the_stated_rules():
    assert POLICY.count("\n") >= 5
    assert policy_total(100, 10, 10, False, True, 0, "standard") == 85.0
    assert policy_total(100, 10, 10, False, False, 0, "standard") == 100.0
    assert policy_total(100, 10, 10, True, True, 0, "standard") == 110.0
    assert policy_total(100, 10, 40, False, True, 0, "standard") == 0.0
    assert policy_total(100, 10, 40, True, True, 0, "standard") == 110.0
    assert policy_total(100, 10, 10, False, True, 8, "gold") == 85.0 + 15.0
    assert policy_total(800, 10, 40, True, True, 12, "platinum") == 810.0 + 100.0
    assert policy_total(100, 10, 10, False, True, 7, "platinum") == 85.0


def test_the_checkers_are_forgiving_about_format_but_not_about_content():
    assert last_number("The total is $1,234.50 dollars.") == 1234.5 and last_number("none") is None
    assert first_json('Sure! ```json\n{"a": [1, 2]}\n``` Done.') == {"a": [1, 2]}
    assert first_json('text [1, "x"] more') == [1, "x"]
    with pytest.raises(ValueError):
        first_json("no json here")
    assert verify({"type": "number", "value": 12.5}, "Answer: 12.50")[0] and not verify({"type": "number", "value": 12.5}, "12.7")[0]
    assert verify({"type": "exact", "value": "refund_request"}, "Refund_Request.")[0]
    assert verify({"type": "integer", "value": 4}, "Line 4")[0] and not verify({"type": "integer", "value": 4}, "Line 5")[0]
    assert verify({"type": "json", "value": ["a", "b"]}, '["A", "b"]')[0] and not verify({"type": "json", "value": ["a", "b"]}, '["b", "a"]')[0]
    spec = {"type": "json", "value": {"x": 1.0, "y": "z"}, "keys": ["x"]}
    assert verify(spec, '{"x": 1, "y": "other"}')[0]
    calls = {"type": "calls", "value": [{"name": "a", "arguments": {"n": 1}}, {"name": "b", "arguments": {}}]}
    assert verify(calls, '{"calls": [{"name": "b", "arguments": {}}, {"name": "a", "arguments": {"n": 1}}]}')[0]
    assert not verify(calls, '{"calls": [{"name": "a", "arguments": {"n": 2}}, {"name": "b", "arguments": {}}]}')[0]
    assert not verify(calls, '{"calls": [{"name": "a", "arguments": {"n": 1}}]}')[0]
    assert verify({"type": "checks", "must_include": ["9800"]}, "About 9,800 customers")[0]
    assert not verifiers.verify({"type": "exact", "value": "x"}, "   ")[0]
    with pytest.raises(ValueError):
        verify({"type": "mystery"}, "x")


def test_the_task_set_is_reproducible_balanced_and_unique():
    tasks = build_tasks()
    assert tasks_hash(tasks) == tasks_hash(build_tasks())
    assert tasks_hash(tasks) != tasks_hash(build_tasks(seed=8))
    assert len({t.id for t in tasks}) == len(tasks)
    by_family: dict[str, list[Task]] = {}
    for t in tasks:
        by_family.setdefault(t.family, []).append(t)
    assert len(by_family) == 22 and {t.skill for t in tasks} == {"extraction", "reasoning", "coding", "tools", "writing"}
    for family, items in by_family.items():
        if family != "support_workflow":
            counts = [sum(t.level == level for t in items) for level in range(1, MAX_LEVEL.get(family, 4) + 1)]
            assert max(counts) - min(counts) <= 1 and len(items) == 24
    assert len({t.prompt for t in tasks}) == len(tasks)


def test_support_tasks_are_graded_by_what_changed_in_the_store():
    tasks = [t for t in build_tasks() if t.family == "support_workflow"]
    assert len(tasks) == 12 and all(t.uses_tools and t.verifier["type"] == "desk" and t.verifier["expect"] for t in tasks)
    jacket = next(t for t in tasks if t.meta["case"] == "damaged-jacket")
    desk = SupportDesk()
    assert not all(c["passed"] for c in desk.check(jacket.verifier["expect"]))
    desk.execute("issue_refund", {"order_id": "A-1042", "amount": 138.99, "reason": "damaged"})
    assert all(c["passed"] for c in desk.check(jacket.verifier["expect"]))
    assert {t.level for t in tasks} == {1, 2, 3}


def test_tasks_round_trip_and_duplicates_are_refused(tmp_path):
    tasks = build_tasks(per_family=3, support_count=3)
    path = tmp_path / "tasks.jsonl"
    digest = save_tasks(tasks, path)
    loaded = load_tasks(path)
    assert loaded == tasks and tasks_hash(loaded) == digest
    path.write_text(path.read_text(encoding="utf-8") + path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_tasks(path)


def test_the_pruned_solution_counter_agrees_with_brute_force():
    import itertools

    from capability_lab.generators import rule_holds

    rng = random.Random(11)
    people = ["Ana", "Ben", "Cleo", "Dev", "Eli"]
    for _ in range(40):
        rules = []
        for _ in range(rng.randint(0, 5)):
            kind = rng.choice(["next", "right", "notend", "at", "apart", "end"])
            a, b = rng.sample(people, 2)
            rules.append((kind, a, rng.randint(0, 4) if kind == "at" else b))
        brute = sum(all(rule_holds(r, perm) for r in rules) for perm in itertools.permutations(people))
        assert count_solutions(people, rules, limit=10**6) == brute
        assert count_solutions(people, rules, limit=2) == min(brute, 2)

def test_new_families_carry_independently_checkable_answers():
    _, calendar = GENERATORS["calendar_math"]
    built = calendar(random.Random(5), 1)
    assert built["gold"] in {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    _, bits = GENERATORS["bit_ops"]
    built = bits(random.Random(6), 3)
    expr = built["prompt"].split("\n")[1]
    assert str(eval(expr)) == built["gold"]  # noqa: S307 - expression generated by the test fixture itself
    _, knights = GENERATORS["knights_knaves"]
    built = knights(random.Random(7), 3)
    assert set(json.loads(built["gold"]).values()) <= {"knight", "knave"}
    _, tools = GENERATORS["tool_selection"]
    built = tools(random.Random(8), 2)
    assert f"- {built['gold']}:" in built["prompt"] or built["gold"] == "none"


def pred(tier_probs, skill="reasoning", **extra):
    top = max(tier_probs, key=tier_probs.get)
    return {"tier": top, "tier_probabilities": tier_probs, "skill": skill, "skill_probabilities": {skill: 1.0}, "confidence": tier_probs[top], **extra}


ONE = {"low": {"low": 0.9, "medium": 0.05, "high": 0.05}, "medium": {"low": 0.1, "medium": 0.8, "high": 0.1}, "high": {"low": 0.05, "medium": 0.05, "high": 0.9}}


def world(n=12):
    patterns = {f"easy{i}": row("a", "b", "c", "d", "e") for i in range(n)} | {f"hard{i}": row("e") for i in range(n)}
    tasks, runs = synthetic(patterns, MODEL_COST)
    return tasks, runs, analysis.tier_labels(tasks, runs)


def test_clef_is_asked_once_per_task_and_answers_are_cached(tmp_path):
    tasks, _, _ = world(3)

    class Router:
        calls = 0

        def classify(self, text):
            Router.calls += 1
            return SimpleNamespace(to_dict=lambda: pred(ONE["low"], latency_ms=5.0))
    out = tmp_path / "clef.jsonl"
    first = evaluate.ask_clef(Router(), tasks, out, workers=2)
    assert Router.calls == len(tasks) == len(first)
    again = evaluate.ask_clef(Router(), tasks, out, workers=2)
    assert Router.calls == len(tasks) and again.keys() == first.keys()
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(tasks)


def test_tier_metrics_count_direction_of_error_and_compare_with_the_majority_class():
    tasks, runs, labels = world(10)
    perfect = {label["task_id"]: pred(ONE[label["tier"]]) for label in labels}
    m = evaluate.tier_metrics(labels, perfect)
    assert m["tier_accuracy"] == 1.0 and m["macro_f1"] == pytest.approx(2 / 3, abs=0.001) and m["underestimates"] == m["overestimates"] == 0
    assert m["majority_class_accuracy"] == 0.5 and m["confusion"]["low"]["low"] == 10 and m["skill_accuracy"] == 1.0
    assert m["brier"] < 0.05 and 0 <= m["ece"] < 0.15
    always_low = {label["task_id"]: pred(ONE["low"]) for label in labels}
    wrong = evaluate.tier_metrics(labels, always_low)
    assert wrong["tier_accuracy"] == 0.5 and wrong["underestimates"] == 10 and wrong["overestimates"] == 0
    assert wrong["brier"] > m["brier"] and wrong["ece"] > m["ece"]


def test_measured_rates_leave_the_task_being_routed_out():
    tasks, runs, labels = world(6)
    cells = evaluate.Cells(labels, analysis.pass_matrix(runs))
    assert cells.rate("a", "reasoning", "low") == 1.0 and cells.rate("a", "reasoning", "high") == 0.0
    # five easy tasks remain once one is left out; a task alone in its cell falls back to nothing measured
    assert cells.rate("a", "reasoning", "low", leave_out="easy0") == 1.0
    sparse = evaluate.Cells(labels[:7] + labels[-1:], analysis.pass_matrix(runs))
    assert sparse.rate("e", "reasoning", "high", leave_out=labels[-1]["task_id"]) is None
    assert cells.rate("a", None, "low") == 1.0


def test_a_router_with_the_right_tiers_matches_the_oracle_and_a_wrong_one_loses_passes():
    tasks, runs, labels = world(12)
    good = {label["task_id"]: pred(ONE[label["tier"]]) for label in labels}
    bad = {label["task_id"]: pred(ONE["low"]) for label in labels}
    result = evaluate.evaluate_arms(tasks, labels, runs, good, floor=0.85)
    arms = result["arms"]
    assert arms["always e"]["pass_rate"] == 1.0 and arms["always a"]["pass_rate"] == 0.5 and result["best_value_endpoint"] == "e"
    assert arms["Clef (tier and skill)"]["pass_rate"] == 1.0 and arms["oracle cheapest passing"]["pass_rate"] == 1.0
    assert arms["true tier and skill (upper bound)"]["models_used"] == {"a": 12, "e": 12}
    saved = arms["always e"]["mean_cost_usd"] - arms["Clef (tier and skill)"]["mean_cost_usd"]
    assert saved > 0.02 and arms["Clef (tier and skill)"]["mean_cost_usd"] == pytest.approx((0.001 + 0.05) / 2 + evaluate.ROUTER_COST_USD, abs=1e-6)
    assert arms["true tier and skill (upper bound)"]["mean_cost_usd"] == pytest.approx((0.001 + 0.05) / 2, abs=1e-6)
    wrong = evaluate.evaluate_arms(tasks, labels, runs, bad, floor=0.85)["arms"]["Clef (tier only)"]
    assert wrong["pass_rate"] == 0.5 and wrong["models_used"] == {"a": 24}
    free = evaluate.evaluate_arms(tasks, labels, runs, good, router_cost=0.0)["arms"]["Clef (tier only)"]
    assert free["mean_cost_usd"] == pytest.approx(0.0255, abs=1e-6)
    json.dumps(result)


def test_uncertainty_is_handled_by_the_probabilities_not_by_the_argmax():
    tasks, runs, labels = world(12)
    unsure = {label["task_id"]: pred({"low": 0.5, "medium": 0.0, "high": 0.5}) for label in labels}
    arms = evaluate.evaluate_arms(tasks, labels, runs, unsure, floor=0.85)["arms"]
    # half the probability mass sits on "high", where only "e" passes, so the expected pass rate of cheap endpoints is 0.5
    assert arms["Clef (tier only)"]["models_used"] == {"e": 24} and arms["Clef (tier only)"]["pass_rate"] == 1.0
    relaxed = evaluate.evaluate_arms(tasks, labels, runs, unsure, floor=0.45)["arms"]["Clef (tier only)"]
    assert relaxed["models_used"] == {"a": 24} and relaxed["pass_rate"] == 0.5


def test_the_length_heuristic_is_a_real_baseline_that_ignores_difficulty():
    assert evaluate.heuristic_probs("x" * 100)["low"] == 1.0 and evaluate.heuristic_probs("x" * 500)["medium"] == 1.0
    assert evaluate.heuristic_probs("x" * 2000)["high"] == 1.0
    tasks, runs, labels = world(12)
    arms = evaluate.evaluate_arms(tasks, labels, runs, {l["task_id"]: pred(ONE[l["tier"]]) for l in labels})["arms"]  # noqa: E741
    assert arms["length heuristic"]["pass_rate"] == 0.5  # every synthetic prompt is short, so it always says "low"


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
