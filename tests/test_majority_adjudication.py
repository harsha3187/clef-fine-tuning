"""Majority-vote third pass: correct classification, budgeting/resume and tamper detection."""

import json
from pathlib import Path

import pytest

from routing_lab.annotation_judge import JudgeResponse
from routing_lab.batch_annotation import _read_json, _read_jsonl, export_consensus
from routing_lab.data import load_records
from routing_lab.majority_adjudication import (
    _majority, create_parser, export_majority, finalize_dataset, run_majority_vote,
)

from test_batch_annotation import make_plan as make_plan, metadata, run as run_batch

ROOT = Path(__file__).resolve().parents[1]


def _wire(verdict, route, quote):
    return {
        "verdict": verdict, "route": route if verdict == "label" else None, "alternative_route": None,
        "rationale": "Independent per-row assessment used only for these test fixtures.",
        "evidence": {"field": "query", "quote": quote},
        "missing_information": "Clarify the requested deliverable." if verdict == "insufficient_context" else None,
    }


class ScriptedJudge:
    """Returns pre-scripted per-row verdicts/routes, keyed by call index then item index."""

    def __init__(self, script, *, model="fixture-actual", prefix="fixture"):
        self.script = script
        self.model = model
        self.prefix = prefix
        self.calls = []

    def metadata(self):
        return metadata()

    def annotate(self, **kwargs):
        call = len(self.calls)
        self.calls.append(kwargs)
        overrides = self.script[call]
        result = {"annotations": {}}
        for key, row in kwargs["input_payload"]["items"].items():
            index = int(key.split("_")[1])
            verdict, route = overrides.get(index, ("label", "debugging"))
            result["annotations"][key] = _wire(verdict, route, row["query"])
        return JudgeResponse(payload=result, model=self.model, response_id=f"{self.prefix}-{call + 1}",
                             input_tokens=50, output_tokens=20, elapsed_ms=1.0)


# Nine coding-track rows. Item 0 agrees on both passes (consensus, excluded from
# adjudication). Items 1-8 disagree and are designed to cover every majority
# outcome: 2-of-3 consensus, a true split (no_majority), and double-abstention.
SOURCE_SCRIPT = [
    {1: ("label", "debugging"), 2: ("label", "debugging"), 3: ("label", "refactoring"),
     4: ("label", "debugging"), 5: ("ambiguous", None), 6: ("label", "debugging"),
     7: ("label", "debugging"), 8: ("ambiguous", None)},
    {1: ("label", "code_generation"), 2: ("label", "code_generation"), 3: ("label", "code_generation"),
     4: ("ambiguous", None), 5: ("insufficient_context", None), 6: ("label", "code_generation"),
     7: ("ambiguous", None), 8: ("label", "code_generation")},
]
# Candidate pool, sorted by id: native-1..native-8 -> batch positions 0..7.
THIRD_SCRIPT = [{
    0: ("label", "debugging"), 1: ("label", "code_generation"), 2: ("label", "debugging"),
    3: ("label", "debugging"), 4: ("label", "debugging"), 5: ("ambiguous", None),
    6: ("label", "code_generation"), 7: ("ambiguous", None),
}]
EXPECTED = {
    "native-1": ("majority_consensus", "debugging"),
    "native-2": ("majority_consensus", "code_generation"),
    "native-3": ("no_majority", None),
    "native-4": ("majority_consensus", "debugging"),
    "native-5": ("majority_abstain", None),
    "native-6": ("no_majority", None),
    "native-7": ("no_majority", None),
    "native-8": ("majority_abstain", None),
}


def build_source_run(make_plan, tmp_path, *, count=9, name="source"):
    plan = make_plan(count)
    source_run = tmp_path / name
    run_batch(plan, ScriptedJudge(SOURCE_SCRIPT), source_run)
    return plan, source_run


def test_majority_classification_matches_every_outcome_shape():
    for ann0_kind, ann1_kind, ann2_kind, expected in [
        (("label", "a"), ("label", "a"), ("label", "b"), ("majority_consensus", "a")),
        (("label", "a"), ("label", "b"), ("label", "b"), ("majority_consensus", "b")),
        (("label", "a"), ("label", "b"), ("label", "c"), ("no_majority", None)),
        (("ambiguous", None), ("ambiguous", None), ("label", "a"), ("majority_abstain", None)),
        (("ambiguous", None), ("label", "a"), ("label", "b"), ("no_majority", None)),
    ]:
        anns = [{"verdict": verdict, "route": route, "acceptable_routes": [route] if route else []}
                for verdict, route in (ann0_kind, ann1_kind, ann2_kind)]
        decision = _majority(*anns)
        assert (decision["status"], decision["route"]) == expected


def test_run_and_export_resolve_majority_and_isolate_the_rest(make_plan, tmp_path):
    plan, source_run = build_source_run(make_plan, tmp_path)
    majority_run = tmp_path / "majority"
    judge = ScriptedJudge(THIRD_SCRIPT, prefix="third")
    report = run_majority_vote(plan, judge, source_run, majority_run, max_requests=2, allow_remote=True)
    assert report["status"] == "complete_selected_rows"
    assert report["completed_rows"] == 8
    assert report["result_status_counts"] == {"majority_consensus": 3, "no_majority": 3, "majority_abstain": 2}
    assert len(judge.calls) == 1

    results = {row["id"]: row for row in _read_jsonl(majority_run / "results.jsonl")}
    for row_id, (status, route) in EXPECTED.items():
        assert (results[row_id]["status"], results[row_id]["route"]) == (status, route)
        assert len(results[row_id]["assessments"]) == 3

    export_dir = tmp_path / "export"
    exported = export_majority(plan, source_run, majority_run, export_dir)
    assert exported["status"] == "complete"
    assert exported["resolved_count"] == 3
    assert exported["candidate_count"] == 8
    assert exported["still_unresolved_count"] == 5
    records = {record.id: record for record in load_records(export_dir / "copilot-coding-v2.jsonl")}
    assert set(records) == {"native-1", "native-2", "native-4"}
    assert records["native-1"].expected_route == "debugging"
    assert records["native-2"].expected_route == "code_generation"
    assert all(record.review_status == "unreviewed" for record in records.values())
    assert all("majority_vote_third_pass" in record.tags for record in records.values())
    adjudication = _read_jsonl(export_dir / "adjudication.jsonl")
    assert {row["id"] for row in adjudication} == {"native-3", "native-5", "native-6", "native-7", "native-8"}
    assert all(len(row["assessments"]) == 3 for row in adjudication)

    consensus_dir = tmp_path / "consensus"
    export_consensus(plan, source_run, consensus_dir, allow_partial=True)
    final_dir = tmp_path / "final"
    finalized = finalize_dataset(consensus_dir, export_dir, final_dir)
    assert finalized["status"] == "complete"
    # native-0 (two-pass consensus) + native-1, native-2, native-4 (majority-resolved).
    assert finalized["total_rows"] == 4
    assert finalized["excluded_unresolved_count"] == 5
    final_records = {record.id: record for record in load_records(final_dir / "copilot-coding-v2.jsonl")}
    assert set(final_records) == {"native-0", "native-1", "native-2", "native-4"}
    excluded = _read_jsonl(final_dir / "excluded_unresolved.jsonl")
    assert {row["id"] for row in excluded} == {"native-3", "native-5", "native-6", "native-7", "native-8"}


def test_finalize_rejects_mismatched_or_tampered_exports(make_plan, tmp_path):
    plan, source_run = build_source_run(make_plan, tmp_path)
    majority_run = tmp_path / "majority"
    run_majority_vote(plan, ScriptedJudge(THIRD_SCRIPT, prefix="third"), source_run, majority_run,
                      max_requests=2, allow_remote=True)
    majority_dir = tmp_path / "majority-export"
    export_majority(plan, source_run, majority_run, majority_dir)
    consensus_dir = tmp_path / "consensus"
    export_consensus(plan, source_run, consensus_dir, allow_partial=True)

    majority_manifest_path = majority_dir / "manifest.json"
    original_majority_manifest = majority_manifest_path.read_text(encoding="utf-8")
    manifest = json.loads(original_majority_manifest)
    manifest["source_run_manifest_sha256"] = "a-different-run-entirely"
    majority_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="same source run"):
        finalize_dataset(consensus_dir, majority_dir, tmp_path / "bad1")
    majority_manifest_path.write_text(original_majority_manifest, encoding="utf-8")

    path = consensus_dir / "copilot-coding-v2.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed since it was exported"):
        finalize_dataset(consensus_dir, majority_dir, tmp_path / "bad2")


def test_remote_consent_and_positive_budget_are_required(make_plan, tmp_path):
    plan, source_run = build_source_run(make_plan, tmp_path)
    judge = ScriptedJudge(THIRD_SCRIPT)
    with pytest.raises(PermissionError):
        run_majority_vote(plan, judge, source_run, tmp_path / "m1", max_requests=1)
    with pytest.raises(ValueError, match="positive physical-request"):
        run_majority_vote(plan, judge, source_run, tmp_path / "m2", max_requests=0, allow_remote=True)
    with pytest.raises(ValueError, match="Invalid batch size"):
        run_majority_vote(plan, judge, source_run, tmp_path / "m3", max_requests=1,
                          allow_remote=True, batch_size=101)
    assert judge.calls == []


def test_incomplete_source_run_is_rejected(make_plan, tmp_path):
    plan = make_plan(9)
    source_run = tmp_path / "incomplete"
    run_batch(plan, ScriptedJudge(SOURCE_SCRIPT), source_run, max_requests=1)
    assert _read_json(source_run / "manifest.json")["status"] == "request_budget_reached"
    with pytest.raises(ValueError, match="fully complete"):
        run_majority_vote(plan, ScriptedJudge(THIRD_SCRIPT), source_run, tmp_path / "m",
                          max_requests=1, allow_remote=True)


def test_source_run_with_no_disagreements_is_rejected(make_plan, tmp_path):
    plan = make_plan(3)
    source_run = tmp_path / "agreeing"
    run_batch(plan, ScriptedJudge([{}, {}]), source_run)
    with pytest.raises(ValueError, match="no disagreements"):
        run_majority_vote(plan, ScriptedJudge([{}]), source_run, tmp_path / "m",
                          max_requests=1, allow_remote=True)


def test_budget_reached_then_resume_completes_remaining_batches(make_plan, tmp_path):
    plan, source_run = build_source_run(make_plan, tmp_path)
    majority_run = tmp_path / "majority"
    script = [
        {0: ("label", "debugging"), 1: ("label", "code_generation"), 2: ("label", "debugging"), 3: ("label", "debugging")},
        {0: ("label", "debugging"), 1: ("ambiguous", None), 2: ("label", "code_generation"), 3: ("ambiguous", None)},
    ]
    judge = ScriptedJudge(script, prefix="batched")
    report = run_majority_vote(plan, judge, source_run, majority_run, max_requests=1,
                               allow_remote=True, batch_size=4)
    assert report["status"] == "request_budget_reached"
    assert report["completed_rows"] == 4
    assert len(judge.calls) == 1

    judge2 = ScriptedJudge(script, prefix="batched")
    report = run_majority_vote(plan, judge2, source_run, majority_run, max_requests=2,
                               allow_remote=True, batch_size=4, resume=True)
    assert report["status"] == "complete_selected_rows"
    assert report["completed_rows"] == 8
    assert len(judge2.calls) == 1  # only the still-pending second batch is requested


def test_tampered_journal_is_detected(make_plan, tmp_path):
    plan, source_run = build_source_run(make_plan, tmp_path)
    majority_run = tmp_path / "majority"
    run_majority_vote(plan, ScriptedJudge(THIRD_SCRIPT, prefix="third"), source_run, majority_run,
                      max_requests=2, allow_remote=True)
    events = _read_jsonl(majority_run / "batch_events.jsonl")
    events[-1]["model"] = "tampered-model"
    path = majority_run / "batch_events.jsonl"
    path.write_bytes("".join(json.dumps(event) + "\n" for event in events).encode("utf-8"))
    with pytest.raises(ValueError, match="journal artifacts changed"):
        export_majority(plan, source_run, majority_run, tmp_path / "export")


def test_source_run_drift_after_creation_is_detected(make_plan, tmp_path):
    plan, source_run = build_source_run(make_plan, tmp_path)
    majority_run = tmp_path / "majority"
    run_majority_vote(plan, ScriptedJudge(THIRD_SCRIPT, prefix="third"), source_run, majority_run,
                      max_requests=2, allow_remote=True)
    identity_path = source_run / "identity.json"
    identity_path.write_text(identity_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="Source run changed"):
        export_majority(plan, source_run, majority_run, tmp_path / "export")


def test_cli_subcommands_are_registered():
    run_args = create_parser().parse_args([
        "run", "--plan", "p", "--judge-config", "j", "--source-run", "s", "--output", "o",
        "--max-requests", "5", "--allow-remote",
    ])
    assert run_args.command == "run" and run_args.batch_size == 100
    export_args = create_parser().parse_args([
        "export", "--plan", "p", "--source-run", "s", "--majority-run", "m", "--output", "o",
    ])
    assert export_args.command == "export" and export_args.allow_partial is False
