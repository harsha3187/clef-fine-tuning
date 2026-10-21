"""Routing trace store: every decision with router and registry versions, candidates and outcome (BR-009).

Raw prompts are never stored: only a hash, a length and derived features (BRD data principle).
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from agent_demo.labels import LANE_LABELS

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
  trace_id TEXT PRIMARY KEY, task_id TEXT, step_id TEXT, created_utc TEXT, mode TEXT, shadow INTEGER,
  router_version TEXT, registry_version TEXT, policy_version TEXT,
  domain TEXT, taxonomy TEXT, route TEXT, confidence REAL, clef_source TEXT, clef_ms REAL, router_ms REAL,
  requirement REAL, text_sha256 TEXT, text_chars INTEGER, complexity REAL,
  selected_model TEXT, selected_tier TEXT, below_floor INTEGER, attempt INTEGER, candidates TEXT,
  exec_ms REAL, input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, baseline_cost_usd REAL,
  would_cost_usd REAL,
  judge_score INTEGER, judge_cost_usd REAL, outcome TEXT
);
CREATE INDEX IF NOT EXISTS decisions_created ON decisions (created_utc);
CREATE TABLE IF NOT EXISTS runs (
  task_id TEXT PRIMARY KEY, created_utc TEXT, quality TEXT, latency TEXT, cost TEXT, summary TEXT
);
"""
_OUTCOME_FIELDS = ("exec_ms", "input_tokens", "output_tokens", "cost_usd", "baseline_cost_usd", "would_cost_usd",
                   "judge_score", "judge_cost_usd", "outcome")


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(p * (len(ordered) - 1))))], 1)


class Telemetry:
    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)

    def log_decision(self, row: dict[str, Any]) -> None:
        columns = ", ".join(row)
        marks = ", ".join(f":{key}" for key in row)
        with self._lock:
            self._conn.execute(f"INSERT OR REPLACE INTO decisions ({columns}) VALUES ({marks})", row)
            self._conn.commit()

    def log_run(self, summary: dict[str, Any]) -> None:
        """Store one finished two-lane run. The summary holds measured totals only, never case text."""
        prefs = summary["preferences"]
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs (task_id, created_utc, quality, latency, cost, summary) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (summary["task_id"], datetime.now(timezone.utc).isoformat(), prefs["quality"], prefs["latency"],
                 prefs["cost"], json.dumps({**summary, "lane_labels": LANE_LABELS.copy()})))
            self._conn.commit()

    def run_aggregates(self, limit: int = 12, agent: str | None = None) -> dict[str, Any]:
        with self._lock:
            rows = self._conn.execute("SELECT created_utc, summary FROM runs ORDER BY created_utc DESC").fetchall()
        runs = [{"created_utc": r["created_utc"], **json.loads(r["summary"])} for r in rows]
        if agent is not None:
            runs = [r for r in runs if r.get("agent", "support") == agent]
        if not runs:
            return {"runs": 0}

        def total(lane: str, key: str) -> float:
            return sum(r["lanes"][lane][key] for r in runs)

        def mean(lane: str) -> float | None:
            scores = [r["lanes"][lane]["reply_score"] for r in runs if r["lanes"][lane]["reply_score"] is not None]
            return round(sum(scores) / len(scores), 2) if scores else None

        def pct(base: float, mine: float) -> float | None:
            return round(100 * (base - mine) / base, 1) if base else None

        passed = {lane: [sum(r["checks_passed"].get(lane, [0, 0])[0] for r in runs),
                         sum(r["checks_passed"].get(lane, [0, 0])[1] for r in runs)] for lane in ("clef", "baseline")}
        mix: dict[str, dict[str, float]] = {}
        for r in runs:
            for model, entry in r["lanes"]["clef"]["models"].items():
                slot = mix.setdefault(model, {"steps": 0, "cost_usd": 0.0})
                slot["steps"] += entry["steps"]
                slot["cost_usd"] = round(slot["cost_usd"] + entry["cost_usd"], 6)
        return {
            "runs": len(runs),
            "lane_labels": LANE_LABELS.copy(),
            "cost_usd": {"clef": round(total("clef", "cost_usd"), 6), "baseline": round(total("baseline", "cost_usd"), 6),
                         "saved_pct": pct(total("baseline", "cost_usd"), total("clef", "cost_usd"))},
            "tokens": {"clef": int(total("clef", "tokens_total")), "baseline": int(total("baseline", "tokens_total")),
                       "saved_pct": pct(total("baseline", "tokens_total"), total("clef", "tokens_total"))},
            "latency_ms": {"clef": round(total("clef", "latency_ms") / len(runs)),
                           "baseline": round(total("baseline", "latency_ms") / len(runs)),
                           "saved_pct": pct(total("baseline", "latency_ms"), total("clef", "latency_ms"))},
            "reply_score": {"clef": mean("clef"), "baseline": mean("baseline")},
            "checks": passed, "model_mix": mix,
            "recent": [{"task_id": r["task_id"], "created_utc": r["created_utc"], "preferences": r["preferences"],
                        "agent": r.get("agent", "support"),
                        "cost_saved_pct": r["saved"]["cost"]["pct"], "tokens_saved_pct": r["saved"]["tokens"]["pct"],
                        "latency_saved_pct": r["saved"]["latency"]["pct"],
                        "clef_score": r["lanes"]["clef"]["reply_score"],
                        "baseline_score": r["lanes"]["baseline"]["reply_score"]} for r in runs[:limit]],
        }

    def log_outcome(self, trace_id: str, **fields: Any) -> None:
        unknown = set(fields) - set(_OUTCOME_FIELDS)
        if unknown:
            raise ValueError(f"Unknown outcome fields: {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f"{key} = :{key}" for key in fields)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE decisions SET {assignments} WHERE trace_id = :trace_id", {**fields, "trace_id": trace_id})
            self._conn.commit()
        if cursor.rowcount == 0:
            raise KeyError(f"Unknown trace {trace_id}")

    def trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM decisions WHERE trace_id = ?", (trace_id,)).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["candidates"] = json.loads(data["candidates"] or "[]")
        return data

    def recent(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT trace_id, task_id, step_id, created_utc, mode, shadow, taxonomy, route, confidence, "
                "selected_model, selected_tier, attempt, exec_ms, cost_usd, baseline_cost_usd, judge_score, "
                "outcome, router_ms, clef_ms, clef_source, below_floor FROM decisions "
                "ORDER BY created_utc DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def aggregates(self) -> dict[str, Any]:
        with self._lock:
            rows = [dict(r) for r in self._conn.execute("SELECT * FROM decisions").fetchall()]
        if not rows:
            return {"decisions": 0}
        executed = [r for r in rows if r["cost_usd"] is not None]
        live = [r for r in executed if not r["shadow"]]
        shadow = [r for r in executed if r["shadow"] and r["would_cost_usd"] is not None]
        # Shadow rows ran the baseline model, so they are reported on their own, as what routing would have cost.
        cost = sum(r["cost_usd"] for r in live)
        baseline = sum(r["baseline_cost_usd"] or 0 for r in live)
        shadow_would = sum(r["would_cost_usd"] for r in shadow)
        shadow_baseline = sum(r["baseline_cost_usd"] or 0 for r in shadow)
        mix: dict[str, dict[str, float]] = {}
        for r in live:
            entry = mix.setdefault(r["selected_model"], {"steps": 0, "cost_usd": 0.0})
            entry["steps"] += 1
            entry["cost_usd"] += r["cost_usd"]
        routes: dict[str, int] = {}
        for r in rows:
            routes[r["route"]] = routes.get(r["route"], 0) + 1
        first_attempts = [r for r in rows if r["attempt"] == 0]
        judged = [r["judge_score"] for r in executed if r["judge_score"] is not None]
        tiers: dict[str, int] = {}
        for r in live:
            tiers[r["selected_tier"]] = tiers.get(r["selected_tier"], 0) + 1
        confidences = [r["confidence"] for r in rows if r["confidence"] is not None]
        return {
            "decisions": len(rows),
            "tasks": len({r["task_id"] for r in rows}),
            "executed_steps": len(live),
            "cost_usd": round(cost, 6),
            "baseline_cost_usd": round(baseline, 6),
            "savings_pct": round(100 * (1 - cost / baseline), 1) if baseline else None,
            "model_mix": {k: {"steps": v["steps"], "cost_usd": round(v["cost_usd"], 6)} for k, v in mix.items()},
            "tier_mix": tiers,
            "route_counts": routes,
            "router_ms_p50": _percentile([r["router_ms"] for r in rows if r["router_ms"] is not None], 0.5),
            "router_ms_p95": _percentile([r["router_ms"] for r in rows if r["router_ms"] is not None], 0.95),
            "clef_ms_p50": _percentile([r["clef_ms"] for r in rows if r["clef_source"] == "clef"], 0.5),
            "exec_ms_p50": _percentile([r["exec_ms"] for r in live if r["exec_ms"] is not None], 0.5),
            "exec_ms_p95": _percentile([r["exec_ms"] for r in live if r["exec_ms"] is not None], 0.95),
            "escalation_rate_pct": round(100 * sum(r["attempt"] > 0 for r in rows) / max(1, len(first_attempts)), 1),
            "fallback_decisions": sum(r["clef_source"] == "fallback" for r in rows),
            "below_floor": sum(bool(r["below_floor"]) for r in rows),
            "judge_mean": round(statistics.fmean(judged), 2) if judged else None,
            "confidence_mean": round(statistics.fmean(confidences), 3) if confidences else None,
            "shadow": {
                "steps": len(shadow), "would_cost_usd": round(shadow_would, 6),
                "baseline_cost_usd": round(shadow_baseline, 6),
                "would_savings_pct": round(100 * (1 - shadow_would / shadow_baseline), 1) if shadow_baseline else None,
            },
        }
