"""Probe which endpoints can be trusted with the agent's own JSON calls (the planner and the reviewer).

The measured registry says how often an endpoint solves a task. It does not say whether the endpoint returns valid JSON when
asked for JSON, which the planner and the reviewer need. gpt-oss-120b, for one, answered a planning prompt with
`{"final{": 1, "steps": []}`. This probe asks every Responses-API endpoint for the same planner and reviewer output a number of
times and records how many were usable. The agent leaves an endpoint out of the planner and reviewer choice unless every
probe call was usable.

    python -m agent_demo.probe_structured --output agent_demo/data/structured_output.json   (or --output - for stdout)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_demo.agent import parse_plan
from agent_demo.domains import DOMAINS
from agent_demo.llm import LLMError
from agent_demo.shadow import display_name
from routing_lab.selector import MeasuredRegistry

ROOT = Path(__file__).resolve().parents[1]
PLAN_REQUESTS = {
    "support": ["Order A-1042 arrived broken and I want my money back.",
                "I'm a VIP and my order A-2210 is two weeks late. Please fix it and tell me what you did.",
                "Where is my order A-1007? Also, can I change the shipping address?"],
    "travel": ["Plan 4 nights in Lisbon for two adults from Dubai, 20 to 24 November 2026, mid-range budget, interests: food and culture.",
               "Weekend somewhere warm for a family of four, 6 to 8 December 2026, budget travel, interests: beaches and nature.",
               "Five days in Tokyo in March 2027, premium budget, interests: food and shopping."],
}
JUDGE_CASES = [
    ("Refund order A-1042, it arrived broken.", "Look up the order and refund it.",
     '[{"tool": "get_order", "args": {"order_id": "A-1042"}, "result": {"status": "delivered", "total": 59.0}}]',
     "The order A-1042 was delivered and is eligible. I could not refund it."),
    ("Where is order A-1007?", "Write the reply.",
     '[{"tool": "get_order", "args": {"order_id": "A-1007"}, "result": {"status": "shipped", "eta": "Friday"}}]',
     "Your order A-1007 has shipped and should arrive on Friday."),
    ("Cancel order A-3001.", "Cancel the order.",
     '[{"tool": "get_order", "args": {"order_id": "A-3001"}, "result": {"status": "shipped"}}]',
     "I cancelled order A-3001 and refunded you in full."),
]
REPEATS = 2


def usable_plan(text: str, domain_id: str) -> bool:
    try:
        parse_plan(text, DOMAINS[domain_id].max_steps, parallel=DOMAINS[domain_id].parallel)
    except ValueError:
        return False
    return True


def usable_judgement(text: str) -> bool:
    try:
        data = json.loads(text)
        return 1 <= int(data["score"]) <= 5 and isinstance(data.get("reason", ""), str)
    except (json.JSONDecodeError, KeyError, ValueError, TypeError):
        return False


def probe_endpoint(llm: Any, deployment: str) -> dict[str, Any]:
    plan_ok = plan_n = judge_ok = judge_n = 0
    errors: list[str] = []
    jobs = []
    for domain_id, requests in PLAN_REQUESTS.items():
        domain = DOMAINS[domain_id]
        system = domain.planner_system.format(max_steps=domain.max_steps, tools=", ".join(f"{t['name']} ({t['description']})" for t in domain.tools))
        jobs += [("plan", domain_id, system, request) for request in requests for _ in range(REPEATS)]
    judge_system = DOMAINS["support"].judge_system
    for request, step, evidence, output in JUDGE_CASES:
        user = f"REQUEST:\n{request}\n\nSTEP:\n{step}\n\nTOOL CALLS SO FAR IN THIS CASE:\n{evidence}\n\nOUTPUT:\n{output}"
        jobs += [("judge", "support", judge_system, user) for _ in range(REPEATS)]
    for kind, domain_id, system, user in jobs:
        try:
            result = llm.complete(deployment, system=system, user=user, json_object=True, max_output_tokens=4000 if kind == "plan" else 3000)
            ok = usable_plan(result.text, domain_id) if kind == "plan" else usable_judgement(result.text)
        except LLMError as error:
            ok = False
            errors.append(str(error)[:120])
        if kind == "plan":
            plan_n += 1
            plan_ok += ok
        else:
            judge_n += 1
            judge_ok += ok
    return {"planner": [plan_ok, plan_n], "judge": [judge_ok, judge_n], "errors": sorted(set(errors))[:3]}


def main(argv: list[str] | None = None) -> None:
    from agent_demo.llm import AzureResponsesClient
    from agent_demo.server import LLM_ENDPOINT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", type=Path, default=ROOT / "data" / "registry" / "measured.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    endpoints = [e for e in MeasuredRegistry.load(args.registry).endpoints if e.api == "responses"]
    llm = AzureResponsesClient(LLM_ENDPOINT)
    started = time.time()
    with ThreadPoolExecutor(max_workers=min(len(endpoints), 5)) as pool:
        rows = list(pool.map(lambda e: (e.id, probe_endpoint(llm, e.deployment)), endpoints))
    data = {"schema": "structured-output-v1", "probed": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
            "note": "Usable JSON from the planner and reviewer prompts, as [usable, asked].", "endpoints": dict(rows)}
    text = json.dumps(data, indent=2) + "\n"
    if str(args.output) == "-":  # a read-only container has nowhere to write; the caller redirects stdout
        sys.stdout.write(text)
    else:
        args.output.write_text(text, encoding="utf-8")
    for endpoint_id, row in rows:
        print(f"{display_name(endpoint_id):18} planner {row['planner'][0]}/{row['planner'][1]}  judge {row['judge'][0]}/{row['judge'][1]}  {row['errors'][:1]}", file=sys.stderr)
    print(f"{len(rows)} endpoints in {time.time() - started:.0f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
