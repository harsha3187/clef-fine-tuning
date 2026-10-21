"""Offline stand-ins for the Clef endpoint and the model API, so the agent can be tested without Azure."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable
from typing import Any

from agent_demo.clef_decider import ClefDecision
from agent_demo.llm import LLMError, LLMResult
from routing_lab.capability import CapabilityResult

TAXONOMY = {"code": "coding-v1", "tools": "tool-api-v1"}
FALLBACK = {"code": "debugging", "tools": "tool_workflow"}


class FakeDecider:
    """Decides a fixed route per domain (or via a callable on the text); can simulate an outage."""

    def __init__(self, routes: dict[str, Any] | None = None, confidence: float = 0.95, fail: bool = False) -> None:
        self.routes = routes or {"code": "debugging", "tools": "tool_lookup"}
        self.confidence, self.fail = confidence, fail
        self.calls: list[tuple[str, str]] = []
        self.last: dict[str, Any] | None = None

    def decide(self, domain: str, step_text: str, *, tools: Any = None, repository: str = "demo/app",
               preceding: tuple[str, ...] = ()) -> ClefDecision:
        self.calls.append((domain, step_text))
        self.preceding = preceding
        if self.fail:
            return ClefDecision(TAXONOMY[domain], FALLBACK[domain], source="fallback", error="endpoint down",
                                latency_ms=5.0)
        route = self.routes[domain]
        route = route(step_text) if callable(route) else route
        return ClefDecision(TAXONOMY[domain], route, {route: self.confidence, "other": round(1 - self.confidence, 4)},
                            self.confidence, round(2 * self.confidence - 1, 4), 0.1, 7.0, 100, "clef", None, "fake")


def plan_json(*steps: tuple[str, str, bool]) -> str:
    """Each step is (title, instruction, is_reply)."""
    return json.dumps({"steps": [{"title": t, "instruction": i, "reply": r} for t, i, r in steps]})


class ScriptedLLM:
    """Plays planner, executor and checker. `scores` feeds the checker; `fail_models` raise LLMError."""

    def __init__(self, plan: str | None = None, scores: list[int] | None = None, fail_models: set[str] | None = None,
                 plan_error: bool = False, tool_calls: list[dict[str, Any]] | None = None,
                 faults: list[str] | None = None, model_scores: dict[str, int] | None = None,
                 model_faults: dict[str, str] | None = None, reply_text: str | None = None,
                 barrier: Any = None) -> None:
        self.plan = plan or plan_json(("Look up the order", "Look up order A-1042.", False),
                                      ("Reply to the customer", "Write the reply.", True))
        self.scores = list(scores or [])
        self.fail_models = fail_models or set()
        self.plan_error = plan_error
        self.tool_calls = tool_calls
        self.faults = list(faults or [])
        self.model_scores = model_scores or {}
        self.model_faults = model_faults or {}
        self.reply_text = reply_text
        self.barrier = barrier
        self.calls: list[dict[str, Any]] = []

    def complete(self, deployment: str, *, system: str | None = None, user: str | None = None,
                 input_items: list[dict[str, Any]] | None = None, previous_response_id: str | None = None,
                 tools: list[dict[str, Any]] | None = None, json_object: bool = False,
                 max_output_tokens: int = 1200) -> LLMResult:
        self.calls.append({"deployment": deployment, "system": system, "user": user, "tools": bool(tools),
                           "previous": previous_response_id})
        if deployment in self.fail_models:
            raise LLMError(f"{deployment} is down")
        if json_object and system and system.startswith("You plan"):
            if self.plan_error:
                raise LLMError("planner down")
            return LLMResult(self.plan, 300, 80, 10.0, "completed", "resp-plan")
        if json_object and system and system.startswith("You grade"):
            score = self.scores.pop(0) if self.scores else 5
            fault = self.faults.pop(0) if self.faults else "this_output"
            # Lanes run in parallel, so scoring by the producing model keeps tests deterministic.
            producer = re.search(r"OUTPUT:\noutput from (\S+)", user or "")
            if producer and producer.group(1) in self.model_scores:
                score = self.model_scores[producer.group(1)]
                fault = self.model_faults.get(producer.group(1), "this_output")
            return LLMResult(json.dumps({"score": score, "reason": "scripted", "fault": fault}), 200, 20, 10.0,
                             "completed", "resp-judge")
        if self.barrier is not None and system and system.startswith("You are a travel research") and "Your step:\nResearch" in (user or ""):
            try:
                self.barrier.wait(timeout=5)
            except Exception as error:  # noqa: BLE001 - a broken barrier means the steps did not overlap
                raise LLMError("parallel steps did not overlap") from error
        if self.reply_text is not None and system and system.startswith("You write the final"):
            return LLMResult(self.reply_text, 200, 400, 10.0, "completed", "resp-reply")
        if tools and self.tool_calls and previous_response_id is None:
            calls = [{"call_id": f"c{i}", **c} for i, c in enumerate(self.tool_calls)]
            return LLMResult("", 120, 30, 10.0, "completed", "resp-tools", calls)
        return LLMResult(f"output from {deployment}", 100, 50, 10.0, "completed", f"resp-{len(self.calls)}")


def collect(into: list[dict[str, Any]]) -> Callable[[dict[str, Any]], None]:
    return into.append


class FakeCapabilityRouter:
    """Stands in for the capability router: fixed tier and skill probabilities, optional outage, thread-safe call log."""

    deployment = "fake-cap"

    def __init__(self, tier: dict[str, float] | None = None, fail: bool = False) -> None:
        self.tier = tier or {"low": 0.9, "medium": 0.08, "high": 0.02}
        self.fail = fail
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def classify(self, request: str, context: str = "") -> CapabilityResult:
        with self.lock:
            self.calls.append(request)
        if self.fail:
            raise RuntimeError("endpoint down")
        return CapabilityResult(self.tier, {"coding": 0.1, "reasoning": 0.2, "tools": 0.5, "writing": 0.1, "extraction": 0.1}, 40.0, 120, "fake-cap")
