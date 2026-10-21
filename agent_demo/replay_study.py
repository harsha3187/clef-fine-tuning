"""Replay study: run harder tool-use tasks through the whole agent with the shadow selector and its replay.

Each task goes through the real planner, both lanes, the capability router, the selector and the replay on the
selector's chosen endpoints, and every run is checked against the task's expected store outcomes. The result says how
often the selector's choices reached the right outcome compared with the always-strongest lane, task by task.

    python -m agent_demo.replay_study --levels 3 4 5 --per-level 4 --approve-cost
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from agent_demo.preferences import Preferences
from agent_demo.labels import LANE_LABELS
from agent_demo.server import (
    CAPABILITY_ADAPTER_SHA256,
    CAPABILITY_DEPLOYMENT,
    FINETUNED_ADAPTER_SHA256,
    LLM_ENDPOINT,
    MEASURED_REGISTRY,
    ROOT,
    Engine,
)
from agent_demo.shadow import ShadowSelector
from capability_lab.toolchains import tool_tasks
from routing_lab.capability import CapabilityRouter
from routing_lab.selector import MeasuredRegistry


def pick(count_per_level: int, levels: list[int], seed: int) -> list[Any]:
    tasks = tool_tasks(200, seed)
    return [t for level in levels for t in [x for x in tasks if x.level == level][:count_per_level]]


def study(engine: Engine, tasks: list[Any], prefs: Preferences) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        events: list[dict[str, Any]] = []
        engine.agent.run(task.prompt, prefs=prefs, expectations=task.verifier["expect"], agent="support", emit=events.append)
        summary = next((e for e in events if e["type"] == "shadow_summary"), None)
        replay = summary["replay"] if summary else None
        done = next((e for e in events if e["type"] == "run_done"), {})
        passed = {lane: ok / total for lane, (ok, total) in done.get("checks_passed", {}).items() if total}
        if replay and replay["checks"].get("replay"):
            ok, total = replay["checks"]["replay"]
            passed["selector_replay"] = ok / total
        rows.append({
            "task": task.id, "level": task.level, "outcomes": passed,
            "lane_labels": {**LANE_LABELS, "selector_replay": "Selector replay"},
            "choices": sorted({r["decision"]["choice"] for r in summary["rows"] if r["decision"].get("choice")}) if summary else [],
            "tiers": [r["decision"]["tier"] for r in summary["rows"] if r["decision"].get("tier")] if summary else [],
            "replay_cost_usd": replay["cost_usd"] if replay else None, "strongest_cost_usd": replay["baseline_cost_usd"] if replay else None,
            "judge": [replay["mean_score"], replay["baseline_mean_score"]] if replay else None})
        shown = {rows[-1]["lane_labels"][key]: value for key, value in passed.items()}
        print(f"{task.id} L{task.level}: {shown}  choices {rows[-1]['choices']}", flush=True)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--levels", type=int, nargs="+", default=[3, 4, 5])
    parser.add_argument("--per-level", type=int, default=4)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--quality", default="good")
    parser.add_argument("--latency", default="medium")
    parser.add_argument("--cost", default="medium")
    parser.add_argument("--approve-cost", action="store_true", help="Each task costs about $0.05 to $0.15 of model calls")
    args = parser.parse_args()
    if not args.approve_cost:
        print("Add --approve-cost to run; each task costs about $0.05 to $0.15.")
        return 2
    uri = json.loads((ROOT / "azure" / "online" / "connection.json").read_text(encoding="utf-8-sig"))["scoring_uri"]
    runtime = ROOT / "outputs" / "replay_study"
    probe = Engine(runtime, uri, "flash-ft-gpu", FINETUNED_ADAPTER_SHA256, LLM_ENDPOINT)
    shadow = ShadowSelector(CapabilityRouter(uri, CAPABILITY_DEPLOYMENT, adapter_sha256=CAPABILITY_ADAPTER_SHA256),
                            MeasuredRegistry.load(MEASURED_REGISTRY), baseline=probe.policy.baseline_model,
                            router_cost_usd=probe.policy.router_cost_usd, log_path=runtime / "shadow.jsonl",
                            adapter_sha256=CAPABILITY_ADAPTER_SHA256)
    engine = Engine(runtime, uri, "flash-ft-gpu", FINETUNED_ADAPTER_SHA256, LLM_ENDPOINT, shadow=shadow, shadow_replay=True)
    rows = study(engine, pick(args.per_level, args.levels, args.seed), Preferences(args.quality, args.latency, args.cost))
    out = Path(runtime / "study.json")
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")

    def rate(key: str, items: list[dict[str, Any]]) -> str:
        values = [r["outcomes"][key] for r in items if key in r["outcomes"]]
        return f"{sum(values) / len(values):.0%} of expected outcomes over {len(values)} tasks" if values else "n/a"

    for key, label in (("baseline", LANE_LABELS["baseline"]), ("clef", LANE_LABELS["clef"] + " (legacy step router)"),
                       ("selector_replay", "Selector replay")):
        print(f"{label:30} {rate(key, rows)}")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
