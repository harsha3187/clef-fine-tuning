"""Adapt public benchmarks into input-only candidates and unreviewed routing probes."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from routing_lab.backends import state
from routing_lab.benchmark_sources import REVISIONS, destination
from routing_lab.data import Record, Route, fingerprint, write_records
from routing_lab.evaluation import sha256

POLICY = "public-request-intent-v1"
MAX_ENDPOINT_STATE_CHARS = 32768


def jsonl(path: Path) -> list[dict]:
    """Read JSONL without splitting literal Unicode line-separator characters."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def first_request(row: dict) -> tuple[str, list[dict]]:
    """Use only the first user message; no future turns or assistant/tool answers."""
    question = row["question"]
    if not isinstance(question, list) or not question or not isinstance(question[0], list):
        raise ValueError("BFCL question must be a nonempty list of turns")
    preceding = []
    for message in question[0]:
        if not isinstance(message.get("content"), str):
            raise ValueError("Unsupported non-text BFCL message")
        if message["role"] == "user":
            if not message["content"].strip():
                raise ValueError("Empty first BFCL user request")
            return message["content"], preceding
        if message["role"] in {"system", "developer"}:
            preceding.append({"role": message["role"], "content": message["content"]})
        else:
            raise ValueError("Unexpected conversation history before the first BFCL user request")
    raise ValueError("No user request in BFCL first turn")


def proposed_route(query: str, source: str) -> tuple[Route | None, str]:
    """Propose conservative, explicitly non-gold labels; abstain on unclear coding intent."""
    if source == "swe-bench":
        if re.search(
            r"\b(error|exception|traceback|crash|fails?|failing|broken|incorrect|unexpected|"
            r"bug|regression|wrong|does not work|not working)\b", query, re.I
        ):
            return "debugging", "SWE issue contains an explicit failure cue"
        if re.search(r"\b(feature request|enhancement|add|implement|support|allow|enable|introduce|refactor|extend)\b",
                     query, re.I):
            return "code_generation", "SWE issue contains a change/feature cue without a detected failure cue"
        return None, "No sufficiently explicit issue intent; label abstained"
    if re.search(
        r"\b(code|coding|python|javascript|typescript|java|repository|repo|refactor|debug|"
        r"bug|unit tests?|pull request|stack ?trace|programming)\b", query, re.I
    ):
        return None, "Tool-use request mentions coding; automatic out-of-scope label abstained"
    return "fallback", "General tool-use request outside the current six coding-owner contracts"


def _candidate(source: str, category: str, split: str, native_id: str, query: str,
               context: dict, repository: str, flags: list[str]) -> dict:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Benchmark query must be nonempty text")
    normalized = " ".join(query.lower().split())
    group = "public-query-" + hashlib.sha256(normalized.encode()).hexdigest()
    route, reason = proposed_route(query, source)
    return {
        "id": f"{source}:{category}:{native_id}", "source": source, "category": category,
        "native_id": native_id, "native_split": split, "group_id": group,
        "repository": repository, "query": query,
        "context": json.dumps(context, ensure_ascii=False, sort_keys=True),
        "proposed_route": route, "label_reason": reason, "label_policy": POLICY,
        "review_status": "unreviewed", "flags": list(flags),
    }


def adapt_swe(row: dict, split: str) -> tuple[dict, dict]:
    """Exclude solutions, test names, hints and environment instructions from routing inputs."""
    candidate = _candidate(
        "swe-bench", split, split, row["instance_id"], row["problem_statement"],
        {"repository": row["repo"], "base_commit": row["base_commit"],
         "repository_content_supplied": False},
        row["repo"], ["repository_not_materialized", "hints_excluded"],
    )
    target = {"id": candidate["id"], "native_targets": {
        key: row[key] for key in ("patch", "test_patch", "FAIL_TO_PASS", "PASS_TO_PASS", "eval_script")
        if key in row
    }}
    return candidate, target


def adapt_bfcl(row: dict, category: str, answer: dict | None) -> tuple[dict, dict]:
    """Adapt initial routing only, excluding unsupported environment-dependent cases from scoring."""
    query, preceding = first_request(row)
    flags = []
    context = {"preceding_instructions": preceding}
    if "function" in row:
        if not isinstance(row["function"], list):
            raise ValueError("BFCL function schemas must be a list")
        context["available_functions"] = row["function"]
    else:
        flags.append("native_environment_not_reconstructed")
    if len(row["question"]) > 1:
        flags.append("later_turns_excluded")
    candidate = _candidate("bfcl", category, "evaluation", str(row["id"]), query, context,
                           "bfcl-tool-environment", flags)
    candidate["native_turn_count"] = len(row["question"])
    return candidate, {
        "id": candidate["id"], "native_answer": answer,
        "native_solution_path": row.get("path"),
    }


def adapt_toolbench(row: dict, category: str) -> tuple[dict, dict]:
    """Keep all offered API descriptions, not the gold 'relevant APIs' selection."""
    allowed = {"category_name", "tool_name", "api_name", "api_description",
               "required_parameters", "optional_parameters", "method"}
    context = {"available_apis": [
        {key: value for key, value in api.items() if key in allowed} for api in row["api_list"]
    ]}
    candidate = _candidate("toolbench", category, "official_example", str(row["query_id"]),
                           row["query"], context, "toolbench-tool-environment",
                           ["official_sample_not_evaluation_split"])
    return candidate, {"id": candidate["id"], "relevant_apis": row.get("relevant APIs")}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_snapshot(raw: Path) -> dict:
    """Validate the exact downloaded files before adapting or resuming experiments."""
    manifest = json.loads((raw / "download-manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete_available_sources_toolbench_full_missing":
        raise ValueError("Source acquisition is incomplete")
    if manifest["revisions"] != REVISIONS:
        raise ValueError("Unexpected benchmark source revisions")
    planned = {row["path"] for row in manifest["plan"]}
    actual = {row["path"] for row in manifest["files"]}
    if planned != actual or len(actual) != len(manifest["files"]):
        raise ValueError("Source inventory does not match its download plan")
    for row in manifest["files"]:
        if sha256(destination(raw, row["path"])) != row["content_sha256"]:
            raise ValueError(f"Source content changed: {row['path']}")
    return manifest


def build_dataset(raw: Path, output: Path) -> dict:
    """Write candidates, separate native targets, draft labels and endpoint-compatible probes."""
    import pyarrow.parquet as parquet

    downloads = verify_snapshot(raw)
    candidates, targets = [], []
    answers = {}
    for resource in downloads["files"]:
        if resource["source"] == "bfcl" and resource["role"] == "answer":
            for answer in jsonl(destination(raw, resource["path"])):
                if answer["id"] in answers:
                    raise ValueError("Duplicate BFCL native-answer identity")
                answers[answer["id"]] = answer
    for resource in downloads["files"]:
        if resource["role"] not in {"input", "sample"}:
            continue
        path = destination(raw, resource["path"])
        source = resource["source"]
        if source == "swe-bench":
            split = path.name.split("-")[0]
            if split not in {"dev", "test"}:
                raise ValueError(f"Unrecognized SWE-bench native split: {split}")
            pairs = [adapt_swe(row, split) for row in parquet.read_table(path).to_pylist()]
        elif source == "bfcl":
            category = path.stem.removeprefix("BFCL_v4_")
            pairs = [adapt_bfcl(row, category, answers.get(row["id"])) for row in jsonl(path)]
        elif source == "toolbench":
            pairs = [adapt_toolbench(row, path.stem) for row in json.loads(path.read_text(encoding="utf-8"))]
        else:
            raise ValueError(f"Unsupported benchmark source: {source}")
        for candidate, target in pairs:
            candidate["source_file"] = resource["path"]
            candidate["source_file_sha256"] = resource["content_sha256"]
            candidates.append(candidate)
            targets.append(target)
    if not candidates or len({row["id"] for row in candidates}) != len(candidates):
        raise ValueError("Empty candidates or duplicate native identities")
    records, excluded = [], []
    for candidate in candidates:
        reasons = [flag for flag in candidate["flags"] if flag in {
            "native_environment_not_reconstructed", "official_sample_not_evaluation_split",
        }]
        if candidate["proposed_route"] is None:
            reasons.append("routing_label_abstained")
        if reasons:
            excluded.append({"id": candidate["id"], "reasons": reasons})
            continue
        records.append(Record(
            id=candidate["id"], group_id=candidate["group_id"], repository=candidate["repository"],
            query=candidate["query"], context=candidate["context"],
            expected_route=candidate["proposed_route"], acceptable_routes=[candidate["proposed_route"]],
            tags=[candidate["source"], candidate["category"], candidate["native_split"], POLICY],
            provenance=json.dumps({
                "source": candidate["source"], "native_id": candidate["native_id"],
                "source_file": candidate["source_file"], "source_file_sha256": candidate["source_file_sha256"],
                "label_reason": candidate["label_reason"], "label_policy": POLICY,
            }, sort_keys=True),
            review_status="unreviewed", reviewer="",
        ))
    endpoint = [row for row in records if len(state(row, "context")) <= MAX_ENDPOINT_STATE_CHARS]
    by_split = {
        split: [row for row in candidates if row["source"] == "swe-bench" and row["native_split"] == split]
        for split in ("dev", "test")
    }
    manifest = {
        "status": "incomplete", "revisions": REVISIONS, "label_policy": POLICY,
        "adapter_sha256": sha256(Path(__file__)),
        "download_manifest_sha256": sha256(raw / "download-manifest.json"),
        "candidate_count": len(candidates), "draft_probe_count": len(records),
        "endpoint_compatible_count": len(endpoint),
        "endpoint_character_limit_excluded": len(records) - len(endpoint),
        "source_counts": dict(Counter(row["source"] for row in candidates)),
        "category_counts": dict(sorted(Counter(
            f"{row['source']}/{row['category']}" for row in candidates).items())),
        "draft_route_counts": dict(Counter(row.expected_route for row in records)),
        "draft_source_counts": dict(Counter(row.tags[0] for row in records)),
        "exclusion_reason_counts": dict(Counter(reason for row in excluded for reason in row["reasons"])),
        "duplicate_normalized_input_groups": sum(count > 1 for count in Counter(map(fingerprint, records)).values()),
        "swe_native_dev_test_repository_overlap": sorted(
            {row["repository"] for row in by_split["dev"]} & {row["repository"] for row in by_split["test"]}),
        "swe_native_dev_test_query_group_overlap": len(
            {row["group_id"] for row in by_split["dev"]} & {row["group_id"] for row in by_split["test"]}),
        "limitations": [
            "Automatic labels are unreviewed request-intent proposals, not native benchmark gold.",
            "Scores measure agreement with this heuristic, not SWE-bench resolution or BFCL/ToolBench tool correctness.",
            "The heuristic overlaps lexical baselines; rules-versus-label agreement is not independent evidence.",
            "This is not a balanced six-route benchmark; abstentions and unsupported environments are retained as candidates.",
            "Native dev/test/evaluation identities are preserved; no train split or repository-disjoint split is fabricated.",
            "Examples inspected or used for prompt tuning must not later be claimed as untouched held-out tests.",
            "Character-compatible does not mean token-compatible with Clef; inspect all truncation telemetry.",
            "ToolBench includes ten official examples only; the original full archive remains unavailable.",
            "SWE repository content is not materialized. Labels describe issue intent, not verified agent solvability.",
        ],
        "license_notes": {
            "swe-bench": "MIT repository license retained; dataset card does not separately license all issue/code content.",
            "bfcl": "Apache-2.0 LICENSE and upstream documentation retained.",
            "toolbench": "Apache-2.0 LICENSE retained; README also restricts use to research/education.",
        },
        "files": {},
    }
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_jsonl(output / "candidates.jsonl", candidates)
    _write_jsonl(output / "native-targets.jsonl", targets)
    _write_jsonl(output / "excluded-from-probes.jsonl", excluded)
    write_records(output / "draft-probes.jsonl", records)
    write_records(output / "endpoint-probes.jsonl", endpoint)
    for path in sorted(output.glob("*.jsonl")):
        manifest["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest["status"] = "complete_unreviewed_probes"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    """Build a new routing bundle from a verified source snapshot."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
