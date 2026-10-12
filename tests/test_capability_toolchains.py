from __future__ import annotations

import pytest

from agent_demo.support import TOOLS, SupportDesk
from capability_lab.toolchains import tool_tasks

DEFAULTS = {"reason": "per policy", "summary": "Customer request needs a specialist.", "priority": "normal", "query": "returns",
            "new_address": "14 Harbour Lane, Leeds LS1 4AB"}


@pytest.fixture(scope="module")
def tasks():
    return tool_tasks(100)


def perform(desk, expectation):
    """What a correct agent would call to satisfy one expectation, using only the tool's own parameters."""
    wanted = next(t for t in TOOLS if t["name"] == expectation["tool"])["parameters"]["properties"]
    args = {**DEFAULTS, **expectation.get("args", {})}
    return desk.execute(expectation["tool"], {k: v for k, v in args.items() if k in wanted})

def test_tasks_are_unique_leveled_and_graded_by_desk_state(tasks):
    assert len(tasks) == 100 and len({t.prompt for t in tasks}) == 100
    assert {t.level for t in tasks} == {1, 2, 3, 4, 5} and all(t.uses_tools and t.skill == "tools" for t in tasks)
    assert all(t.verifier["type"] == "desk" and t.verifier["expect"] for t in tasks)
    assert tasks == tool_tasks(100)


def test_every_expected_outcome_is_achievable_and_every_forbidden_one_is_really_forbidden(tasks):
    for task in tasks:
        desk = SupportDesk()
        for item in task.verifier["expect"]:
            if not item.get("absent"):
                assert "error" not in perform(desk, item), (task.id, item["label"])
        assert all(r["passed"] for r in desk.check(task.verifier["expect"])), task.id


def test_forbidden_actions_are_either_rejected_by_policy_or_caught_by_the_check(tasks):
    cancelled_shipped = SupportDesk().execute("cancel_order", {"order_id": "A-1057"})
    assert "error" in cancelled_shipped
    desk = SupportDesk()
    desk.execute("issue_refund", {"order_id": "A-1061", "amount": 100.0, "reason": "warranty"})  # the trap: allowed by the desk
    warranty = next(t for t in tasks if any(e["label"].startswith("No refund outside") for e in t.verifier["expect"]))
    failed = [r for r in desk.check(warranty.verifier["expect"]) if not r["passed"]]
    assert failed and "No refund" in failed[0]["label"]


def test_harder_levels_ask_for_more_and_name_less(tasks):
    by_level = {level: [t for t in tasks if t.level == level] for level in range(1, 6)}
    assert all("order A-" in t.prompt for t in by_level[1]) and not any("(order A-" in t.prompt for t in by_level[2])
    size = {level: sum(len(t.verifier["expect"]) for t in rows) / len(rows) for level, rows in by_level.items()}
    assert size[3] > size[1] and size[4] >= size[3]
    assert all("Nair" not in t.prompt and "Webb" not in t.prompt and "Petrova" not in t.prompt for t in by_level[5])

def test_a_training_set_can_be_kept_apart_from_the_evaluation_set_and_from_one_customer(tasks):
    train = tool_tasks(60, 12, family="tool_chain2", customers=("priya", "marcus"), avoid={t.prompt for t in tasks})
    assert len(train) == 60 and {t.prompt for t in train}.isdisjoint({t.prompt for t in tasks})
    assert all(t.id.startswith("tool_chain2-") and t.family == "tool_chain2" for t in train)
    assert not any("Petrova" in t.prompt or "elena" in t.prompt for t in train)
    for task in train:
        desk = SupportDesk()
        for item in task.verifier["expect"]:
            if not item.get("absent"):
                assert "error" not in perform(desk, item), (task.id, item["label"])
        assert all(r["passed"] for r in desk.check(task.verifier["expect"]))
