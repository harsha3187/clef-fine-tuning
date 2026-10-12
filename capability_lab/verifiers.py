"""Programmatic answer checks for the capability benchmark.

Every task carries a verifier spec, so grading never depends on a model's opinion. A verifier receives the model's
raw text and returns (passed, detail). The task's gold answer must pass its own verifier (tested), and a perturbed
answer must fail.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
_FENCE = re.compile(r"```(?:json|python|text)?\s*(.*?)```", re.S | re.I)


def _norm(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().strip(".!").lower()


def last_number(text: str) -> float | None:
    found = _NUMBER.findall(text.replace("$", ""))
    for token in reversed(found):
        try:
            return float(token.replace(",", ""))
        except ValueError:
            continue
    return None


def first_json(text: str) -> Any:
    """The first JSON object or array in the text, ignoring code fences and chatter around it."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    decoder = json.JSONDecoder()
    for chunk in candidates:
        for start, char in enumerate(chunk):
            if char in "[{":
                try:
                    value, _ = decoder.raw_decode(chunk[start:])
                    return value
                except json.JSONDecodeError:
                    continue
    raise ValueError("No JSON found")


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) < 0.011
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b, strict=True))
    return _norm(a) == _norm(b)


def check_exact(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    fenced = _FENCE.search(output)
    body = fenced.group(1) if fenced else output
    got = _norm(body)
    want = _norm(spec["value"])
    return got == want, f"got {got[:60]!r}"


def check_number(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    got = last_number(output)
    if got is None:
        return False, "no number"
    return abs(got - float(spec["value"])) <= float(spec.get("tol", 0.01)), f"got {got}"


def check_integer(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    match = re.search(r"\d+", output)
    return bool(match) and int(match.group()) == int(spec["value"]), f"got {match.group() if match else None}"


def check_json(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    try:
        got = first_json(output)
    except ValueError:
        return False, "no JSON"
    want = spec["value"]
    keys = spec.get("keys")
    if keys and isinstance(got, dict):
        got, want = {k: got.get(k) for k in keys}, {k: want[k] for k in keys}
    return _same(got, want), "json differs"


def check_checks(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    text = output.strip()
    low = text.lower()
    nocommas = low.replace(",", "")  # "9,800" and "9800" are the same number
    words = len(re.findall(r"\b[\w'-]+\b", text))
    problems = []
    for needle in spec.get("must_include", []):
        if str(needle).lower().replace(",", "") not in nocommas:
            problems.append(f"missing {needle!r}")
    for needle in spec.get("must_not", []):
        if re.search(rf"\b{re.escape(str(needle).lower())}\b", low):
            problems.append(f"contains {needle!r}")
    if "max_words" in spec and words > spec["max_words"]:
        problems.append(f"{words} words > {spec['max_words']}")
    if "min_words" in spec and words < spec["min_words"]:
        problems.append(f"{words} words < {spec['min_words']}")
    if spec.get("starts_with") and not low.startswith(str(spec["starts_with"]).lower()):
        problems.append("wrong opening")
    if spec.get("ends_with") and not low.rstrip(" .!\n").endswith(str(spec["ends_with"]).lower().rstrip(" .!")):
        problems.append("wrong closing")
    return not problems, "; ".join(problems[:3]) or "ok"


def check_csp(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    """Any assignment satisfying every constraint is accepted; there may be many."""
    try:
        got = first_json(output)
    except ValueError:
        return False, "no JSON"
    tasks, slots = spec["tasks"], spec["slots"]
    if not isinstance(got, dict) or set(got) != set(tasks) or any(got[t] not in slots for t in tasks):
        return False, "not a full assignment"
    for kind, a, b in spec["constraints"]:
        sa = slots.index(got[a])
        sb = slots.index(got[b]) if kind in ("before", "apart", "together") else None
        if kind == "before" and not sa < sb:
            return False, f"{a} not before {b}"
        if kind == "apart" and sa == sb:
            return False, f"{a} shares a slot with {b}"
        if kind == "together" and sa != sb:
            return False, f"{a} not with {b}"
        if kind == "slot" and got[a] != b:
            return False, f"{a} not in {b}"
        if kind == "not_slot" and got[a] == b:
            return False, f"{a} in {b}"
    return True, "ok"


def check_calls(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    try:
        got = first_json(output)
    except ValueError:
        return False, "no JSON"
    calls = got.get("calls", got) if isinstance(got, dict) else got
    if isinstance(calls, dict):
        calls = [calls]
    want = spec["value"]
    if not isinstance(calls, list) or len(calls) != len(want):
        return False, "wrong number of calls"
    remaining = list(want)
    for call in calls:
        if not isinstance(call, dict):
            return False, "call is not an object"
        match = next((w for w in remaining if _norm(call.get("name")) == _norm(w["name"])
                      and _same(call.get("arguments", {}), w["arguments"])), None)
        if match is None:
            return False, f"unexpected call {str(call)[:80]}"
        remaining.remove(match)
    return True, "ok"


VERIFIERS: dict[str, Callable[[dict[str, Any], str], tuple[bool, str]]] = {
    "exact": check_exact, "number": check_number, "integer": check_integer, "json": check_json,
    "checks": check_checks, "csp": check_csp, "calls": check_calls,
}


def verify(spec: dict[str, Any], output: str) -> tuple[bool, str]:
    """Apply a task's verifier. Tool-use tasks are verified from the store's state, not from text, elsewhere."""
    kind = spec["type"]
    if kind not in VERIFIERS:
        raise ValueError(f"Unknown verifier {kind!r}")
    if not output or not output.strip():
        return False, "empty output"
    return VERIFIERS[kind](spec, output)
