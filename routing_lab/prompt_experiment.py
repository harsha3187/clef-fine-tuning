"""Compare versioned criteria on development probes without changing model weights.

Use run_prompt_comparison from the notebook with explicit remote-data approval.
All variants use identical records, instructions, option order and model revision.
"""

from __future__ import annotations

import json
from pathlib import Path

from routing_lab.backends import questions
from routing_lab.data import Contract, load_contracts, load_records
from routing_lab.endpoint import ClefEndpointBackend
from routing_lab.evaluation import run_experiment, sha256
from routing_lab.taxonomy import taxonomy_for_routes, taxonomy_routes


def expanded_contracts(
    contracts: dict[str, Contract], criteria_path: Path,
) -> dict[str, Contract]:
    """Replace descriptions only, preserving taxonomy, order and capabilities."""
    taxonomy_for_routes(contracts)
    criteria = json.loads(criteria_path.read_text(encoding="utf-8"))
    if not isinstance(criteria, dict) or set(criteria) != set(contracts):
        raise ValueError("Expanded criteria must contain exactly the catalog route names")
    if any(not isinstance(text, str) or not text.strip() or len(text) > 2048
           for text in criteria.values()):
        raise ValueError("Each criterion must be nonblank text of at most 2048 characters")
    return {
        route: Contract.model_validate({**contract.model_dump(), "description": criteria[route]})
        for route, contract in contracts.items()
    }


def paired_changes(baseline: list[dict], candidate: list[dict]) -> dict:
    """Pair predictions by identity and report corrections as well as regressions."""
    left = {row["id"]: row for row in baseline}
    right = {row["id"]: row for row in candidate}
    if not left or len(left) != len(baseline) or len(right) != len(candidate) or left.keys() != right.keys():
        raise ValueError("Paired predictions require matching nonempty unique IDs")
    rows = []
    taxonomies = {row.get("taxonomy_id", "coding-v1") for row in baseline + candidate}
    if len(taxonomies) != 1:
        raise ValueError("Paired predictions must use identical taxonomies")
    taxonomy_id = next(iter(taxonomies))
    routes = taxonomy_routes(taxonomy_id)
    for record_id, before in left.items():
        after = right[record_id]
        if (before["expected_route"] != after["expected_route"]
                or before["acceptable_routes"] != after["acceptable_routes"]):
            raise ValueError("Paired predictions must use identical labels")
        expected = before["expected_route"]
        old, new = before["decision"]["route"], after["decision"]["route"]
        for row in (before, after):
            if row["decision"].get("taxonomy_id", "coding-v1") != taxonomy_id:
                raise ValueError("Paired decision taxonomy must match record taxonomy")
            if (row["expected_route"] not in routes
                    or row["decision"]["route"] not in routes
                    or any(route not in routes for route in row["acceptable_routes"])):
                raise ValueError("Paired predictions contain unknown taxonomy labels")
        rows.append({
            "id": record_id, "expected_route": expected,
            "baseline": old, "expanded": new,
            "corrected": old != expected and new == expected,
            "regressed": old == expected and new != expected,
            "changed": old != new,
        })
    return {
        "label_basis": "unreviewed proposed expected_route; development only",
        "corrected": sum(row["corrected"] for row in rows),
        "regressed": sum(row["regressed"] for row in rows),
        "changed": sum(row["changed"] for row in rows),
        "records": rows,
    }


def run_prompt_comparison(
    *, contracts_path: Path, criteria_path: Path, data_path: Path,
    url: str, output: Path, allow_remote: bool = False,
    taxonomy_id: str = "coding-v1",
) -> dict:
    """Run a fixed two-arm development ablation and persist complete provenance.

    Failures propagate. Completed arms and partial predictions remain available,
    but the comparison stays incomplete. Existing experiment folders are never
    overwritten. No winner is automatically promoted to the default prompt.
    """
    if not allow_remote:
        raise ValueError("Set allow_remote only after approving data transmission to Azure")
    contracts = load_contracts(contracts_path, taxonomy_id)
    variants = {
        "baseline": contracts,
        "expanded_v1": expanded_contracts(contracts, criteria_path),
    }
    records = load_records(data_path, taxonomy_id=taxonomy_id)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "status": "incomplete",
        "taxonomy_id": taxonomy_id,
        "mode": "development_prompt_tuning_not_a_benchmark",
        "hypothesis": "Explicit scope, inclusions and exclusions improve initial-owner routing.",
        "limitations": [
            "Previously inspected synthetic examples with unreviewed proposed labels.",
            "Changing wording and length together cannot isolate description length as the cause.",
            "One serial pass per variant; latency is not a controlled performance benchmark.",
            "Any selected prompt needs a frozen, independently reviewed held-out evaluation.",
        ],
        "controls": {
            "data_sha256": sha256(data_path),
            "contracts_sha256": sha256(contracts_path),
            "criteria_sha256": sha256(criteria_path),
            "context_mode": "context",
            "variant_order": list(variants),
            "planned_questions": {name: questions(value) for name, value in variants.items()},
        },
        "results": {},
    }
    report_path = output / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    predictions = {}
    for name, variant in variants.items():
        backend = ClefEndpointBackend(variant, url=url, context_mode="context")
        summary = run_experiment(
            backend, records, output / name, probe=True,
            source_paths={
                "dataset": data_path, "contracts": contracts_path, "criteria": criteria_path,
            },
        )
        report["results"][name] = summary
        predictions[name] = [
            json.loads(line)
            for line in (output / name / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["paired"] = paired_changes(predictions["baseline"], predictions["expanded_v1"])
    before, after = report["results"]["baseline"], report["results"]["expanded_v1"]
    report["delta_expanded_minus_baseline"] = {
        key: after[key] - before[key]
        for key in ("strict_accuracy", "acceptable_accuracy", "macro_f1", "fallback_recall")
        if before[key] is not None and after[key] is not None
    }
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
