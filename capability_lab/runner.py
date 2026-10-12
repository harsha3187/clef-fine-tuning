"""Run every task on every endpoint, grading each answer with the task's own checker, and keep every result."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agent_demo.domains import SUPPORT
from agent_demo.llm import LLMError
from agent_demo.support import SupportDesk
from capability_lab.clients import ModelSpec, ask, run_with_tools
from capability_lab.tasks import Task
from capability_lab.verifiers import verify

KEEP_OUTPUT = 3000


def run_one(client: Any, model: ModelSpec, task: Task, sample: int = 0) -> dict[str, Any]:
    """One (task, endpoint) attempt. Failures to answer are recorded as errors, never as wrong answers."""
    record: dict[str, Any] = {"task_id": task.id, "model": model.id, "sample": sample, "passed": None, "detail": "",
                              "output": "", "tokens_in": 0, "tokens_out": 0, "latency_ms": 0.0, "cost_usd": 0.0,
                              "error": None, "at": round(time.time(), 1)}
    try:
        if task.uses_tools:
            desk = SupportDesk()
            result, tin, tout, latency = run_with_tools(client, model, SUPPORT.tools_system, task.prompt, desk, SUPPORT.responses_tools)
            checks = desk.check(task.verifier["expect"])
            passed = all(c["passed"] for c in checks)
            detail = "; ".join(c["label"] for c in checks if not c["passed"]) or "ok"
            output = result.text or json.dumps([c["name"] for c in desk.calls])
        else:
            result = ask(client, model, task.prompt)
            tin, tout, latency, output = result.input_tokens, result.output_tokens, result.latency_ms, result.text
            passed, detail = verify(task.verifier, output)
    except LLMError as error:
        record["error"] = str(error)[:200]
        return record
    record.update(passed=bool(passed), detail=detail[:200], output=output[:KEEP_OUTPUT], tokens_in=tin, tokens_out=tout,
                  latency_ms=round(latency, 1), cost_usd=round(model.cost(tin, tout), 8))
    return record


def load_runs(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pending(tasks: list[Task], models: list[ModelSpec], done: set[tuple[str, str, int]], samples: int = 1) -> list[tuple[Task, ModelSpec, int]]:
    work = []
    for task in tasks:
        for model in models:
            if task.uses_tools and not model.tools:
                continue  # an endpoint without tool calling is not asked tool-use tasks
            for sample in range(samples):
                if (task.id, model.id, sample) not in done:
                    work.append((task, model, sample))
    return work


def pilot_subset(tasks: list[Task], per_family: int) -> list[Task]:
    """The first few tasks of each family; ids cycle through levels 1, 2, 3 so every level is represented."""
    seen: dict[str, int] = {}
    chosen = []
    for task in tasks:
        if seen.get(task.family, 0) < per_family:
            seen[task.family] = seen.get(task.family, 0) + 1
            chosen.append(task)
    return chosen


def run_benchmark(client: Any, tasks: list[Task], models: list[ModelSpec], out: Path, *, workers: int = 8, samples: int = 1,
                  on_progress: Callable[[int, int, dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Run whatever is not already in the results file; safe to stop and resume."""
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["task_id"], r["model"], r["sample"]) for r in load_runs(out) if r["error"] is None}
    work = pending(tasks, models, done, samples)
    lock = threading.Lock()
    finished = errors = 0
    spent = 0.0
    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, client, model, task, sample) for task, model, sample in work]
        for future in as_completed(futures):
            record = future.result()
            with lock:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                finished += 1
                errors += record["error"] is not None
                spent += record["cost_usd"]
            if on_progress:
                on_progress(finished, len(work), record)
    return {"attempted": len(work), "errors": errors, "spent_usd": round(spent, 4), "skipped_already_done": len(done)}
