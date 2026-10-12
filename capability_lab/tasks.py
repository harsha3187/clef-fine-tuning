"""The benchmark task set: build it deterministically, save it, and load it back with an integrity hash."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from capability_lab.generators import GENERATORS, MAX_LEVEL

DATA_DIR = Path(__file__).parent / "data"
SCENARIO_LEVEL = {"where-is-my-order": 1, "missing-details": 1, "address-change": 2, "damaged-jacket": 2,
                  "warranty-claim": 3, "late-vip-order": 3}
SUPPORT_PREFIXES = ["", "Hello support team,\n", "Hi there. "]


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    skill: str
    level: int
    prompt: str
    verifier: dict[str, Any]
    gold: str = ""
    uses_tools: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(**data)


def support_tasks(count: int, rng: random.Random) -> list[Task]:
    """Tool-use tasks on the simulated support desk: graded by what actually changed in the store."""
    from agent_demo.domains import SUPPORT

    cases = SUPPORT.scenarios()
    tasks = []
    for i in range(count):
        case = cases[i % len(cases)]
        prefix = SUPPORT_PREFIXES[(i // len(cases)) % len(SUPPORT_PREFIXES)]
        tasks.append(Task(
            id=f"support_workflow-{i + 1:03d}", family="support_workflow", skill="tools", level=SCENARIO_LEVEL[case["id"]],
            prompt=prefix + case["message"], verifier={"type": "desk", "expect": case["expect"]}, uses_tools=True,
            meta={"case": case["id"]}))
    return tasks


def build_tasks(per_family: int = 24, seed: int = 7, support_count: int = 12) -> list[Task]:
    """Equal numbers per generator, levels cycling up to each family's top level, so difficulty is spread evenly."""
    tasks: list[Task] = []
    for family, (skill, generate) in GENERATORS.items():
        rng = random.Random(f"{seed}:{family}")
        seen: set[str] = set()
        top = MAX_LEVEL.get(family, 4)
        for _ in range(per_family * 20):
            if len(seen) == per_family:
                break
            level = 1 + len(seen) % top
            built = generate(rng, level)
            if built["prompt"] in seen:
                continue
            seen.add(built["prompt"])
            tasks.append(Task(id=f"{family}-{len(seen):03d}", family=family, skill=skill, level=level, prompt=built["prompt"],
                              verifier=built["verifier"], gold=built["gold"], meta=built.get("meta", {})))
    tasks += support_tasks(support_count, random.Random(f"{seed}:support"))
    return tasks


def tasks_hash(tasks: list[Task]) -> str:
    return hashlib.sha256(json.dumps([t.to_dict() for t in tasks], sort_keys=True).encode()).hexdigest()[:12]


def save_tasks(tasks: list[Task], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t.to_dict(), sort_keys=True) + "\n" for t in tasks), encoding="utf-8")
    return tasks_hash(tasks)


def load_tasks(path: Path) -> list[Task]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    tasks = [Task.from_dict(r) for r in rows]
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task ids")
    return tasks
