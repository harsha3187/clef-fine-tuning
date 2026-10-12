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
