"""Annotation isolation, blind agreement, evidence checks, budget accounting and export gates."""

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from routing_lab.annotation import (
    POLICY_VERSION, TRACKS, export_consensus, jsonl, load_plan,
    prepare_plan, run_annotations, validate_annotation,
)
from routing_lab.annotation_judge import JudgeError, JudgeIncompleteError, JudgeRefusalError
from routing_lab.data import load_records
from routing_lab.evaluation import sha256

ROOT = Path(__file__).resolve().parents[1]


def candidate(identity, *, source="swe-bench", flags=None, group=None):
    return {
        "id": identity, "source": source, "category": "test", "native_split": "test",
        "native_id": identity, "group_id": group or identity, "repository": "example/repo",
        "query": "An existing function returns the wrong type.",
        "context": '{"repository":"example/repo"}',
        "source_file_sha256": "a" * 64, "flags": flags or [],
        "proposed_route": "OLD_LABEL_MUST_NOT_BE_REUSED", "label_reason": "OLD_REASON",
    }


@pytest.fixture
def plan(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rows = [
        candidate("bug"),
        candidate("duplicate-query", group="bug"),
        candidate("lookup", source="bfcl"),
        candidate("missing-env", source="bfcl", flags=["native_environment_not_reconstructed"]),
        candidate("sample", source="toolbench", flags=["official_sample_not_evaluation_split"]),
    ]
    (bundle / "candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(json.dumps({
        "status": "complete_unreviewed_probes", "revisions": {"fixture": "immutable"},
        "files": {"candidates.jsonl": {"sha256": sha256(bundle / "candidates.jsonl")}},
    }), encoding="utf-8")
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({
        "version": POLICY_VERSION, "track_assignment": TRACKS, "known_development_exposed_ids": ["bug"],
    }), encoding="utf-8")
    output = tmp_path / "plan"
    prepare_plan(bundle, ROOT / "contracts", policy, output)
    return output


def make_policy(path: Path, *, track_assignment: dict, exposed: list[str] | None = None) -> Path:
    path.write_text(json.dumps({
        "version": POLICY_VERSION, "track_assignment": track_assignment,
        "known_development_exposed_ids": exposed or [],
    }), encoding="utf-8")
    return path


def make_bundle(tmp_path: Path, rows: list[dict]) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows) + "", encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(json.dumps({
        "status": "complete_unreviewed_probes", "revisions": {"fixture": "immutable"},
        "files": {"candidates.jsonl": {"sha256": sha256(bundle / "candidates.jsonl")}},
    }), encoding="utf-8")
    return bundle


def test_prepare_plan_accepts_coding_v1_for_swe_bench(tmp_path):
    """A policy may select the 6-route coding-v1 taxonomy for swe-bench (fits the
    English model's token budget) without touching the tool/API track at all."""
    bundle = make_bundle(tmp_path, [candidate("bug"), candidate("lookup", source="bfcl")])
    policy = make_policy(tmp_path / "policy.json", track_assignment={
        "swe-bench": "coding-v1", "bfcl": "tool-api-v1", "toolbench": "tool-api-v1",
    })
    output = tmp_path / "plan"
    prepare_plan(bundle, ROOT / "contracts", policy, output)
    manifest, items, catalogs = load_plan(output)
    assert set(catalogs) == {"coding-v1", "tool-api-v1"}
    assert len(catalogs["coding-v1"]) == 6
    by_id = {item.id: item for item in items}
    assert by_id["bug"].taxonomy_id == "coding-v1"
    assert by_id["lookup"].taxonomy_id == "tool-api-v1"


def test_prepare_plan_rejects_cross_group_track_assignment(tmp_path):
    bundle = make_bundle(tmp_path, [candidate("bug")])
    policy = make_policy(tmp_path / "policy.json", track_assignment={
        "swe-bench": "tool-api-v1", "bfcl": "tool-api-v1", "toolbench": "tool-api-v1",
    })
    with pytest.raises(ValueError, match="Unsupported annotation policy"):
        prepare_plan(bundle, ROOT / "contracts", policy, tmp_path / "plan")


def test_prepare_plan_rejects_incomplete_track_assignment(tmp_path):
    bundle = make_bundle(tmp_path, [candidate("bug")])
    policy = make_policy(tmp_path / "policy.json", track_assignment={"swe-bench": "coding-v1"})
    with pytest.raises(ValueError, match="Unsupported annotation policy"):
        prepare_plan(bundle, ROOT / "contracts", policy, tmp_path / "plan")


def assessment(route="debugging"):
    return {
        "verdict": "label", "route": route, "acceptable_routes": [route],
        "rationale": "A functional wrong-result diagnosis belongs to this initial owner.",
        "evidence": [{"field": "query", "quote": "returns the wrong type"}],
        "missing_information": [],
    }


@dataclass
class Reply:
    payload: dict
    model: str = "test-judge-v1"
    response_id: str = "offline-fixture"
    input_tokens: int = 12
    output_tokens: int = 10
    elapsed_ms: float = 1.0


class FakeJudge:
    def __init__(self, replies=None):
        self.replies = replies
        self.calls = []

    def metadata(self):
        return {"provider": "offline-test", "configured_model": "test-judge"}

    def annotate(self, **kwargs):
        self.calls.append(kwargs)
        if self.replies is not None:
            reply = self.replies[len(self.calls) - 1]
            if isinstance(reply, (ValueError, JudgeError)):
                raise reply
            return Reply(reply)
        route = "debugging" if "debugging" in str(kwargs["schema"]) else "tool_lookup"
        return Reply(assessment(route))


def run(plan, judge, output, **kwargs):
    options = {"max_records": 1, "max_requests": 2, "allow_remote": True}
    options.update(kwargs)
    return run_annotations(plan, judge, output, **options)


def test_preparation_keeps_every_row_unlabeled_and_propagates_exposure(plan):
    manifest, items, _ = load_plan(plan)
    assert manifest["total_rows"] == 5
    assert manifest["labeled_rows"] == 0
    assert manifest["ready_rows"] == 3
    assert manifest["required_successful_requests_for_ready_rows"] == 6
    assert manifest["eligibility_counts"] == {"ready": 3, "blocked_context": 1, "example_only": 1}
    assert all(item.development_exposed for item in items[:2])
    assert "OLD_LABEL" not in (plan / "items.jsonl").read_text()
    assert items[0].taxonomy_id == "copilot-coding-v2"
    assert items[2].taxonomy_id == "tool-api-v1"


def test_frozen_catalog_edits_are_rejected(plan):
    path = plan / "copilot-coding-v2.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        load_plan(plan)


def test_remote_consent_is_required_before_a_judge_call(plan, tmp_path):
    judge = FakeJudge()
    with pytest.raises(PermissionError):
        run(plan, judge, tmp_path / "run", allow_remote=False)
    assert judge.calls == []
    assert not (tmp_path / "run").exists()


def test_blind_passes_exclude_old_labels_metadata_and_each_others_answers(plan, tmp_path):
    judge = FakeJudge()
    result = run(plan, judge, tmp_path / "run")
    assert result["status"] == "complete_selected_rows"
    assert result["result_status_counts"] == {"silver_consensus": 1}
    assert len(judge.calls) == 2
    assert judge.calls[0]["input_payload"] == judge.calls[1]["input_payload"]
    assert set(judge.calls[0]["input_payload"]) == {"query", "context"}
    assert judge.calls[0]["system_prompt"] != judge.calls[1]["system_prompt"]
    sent = json.dumps(judge.calls)
    assert "OLD_LABEL" not in sent and "OLD_REASON" not in sent
    assert assessment()["rationale"] not in sent
    assert "source_file_sha256" not in sent and "native_id" not in sent


def test_budget_counts_attempts_and_resume_does_not_repeat_completed_pass(plan, tmp_path):
    output = tmp_path / "run"
    first = FakeJudge()
    manifest = run(plan, first, output, max_requests=1)
    assert len(first.calls) == 1 and manifest["status"] == "request_budget_reached"
    second = FakeJudge()
    manifest = run(plan, second, output, max_requests=2, resume=True)
    assert len(second.calls) == 1
    assert manifest["requests_attempted"] == 2
    assert len(manifest["budget_approvals"]) == 2
    assert manifest["known_input_tokens"] == 24
    assert manifest["requests_without_valid_usage"] == 0
    third = FakeJudge()
    run(plan, third, output, resume=True)
    assert third.calls == []


def test_invalid_response_is_logged_as_failed_not_accepted(plan, tmp_path):
    output = tmp_path / "run"
    invalid = assessment()
    invalid["evidence"][0]["quote"] = "   "
    with pytest.raises(ValueError, match="empty"):
        run(plan, FakeJudge([invalid]), output)
    assert [event["event"] for event in jsonl(output / "events.jsonl")] == ["started", "failed"]
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    recovered = run(plan, FakeJudge(), output, max_requests=3, resume=True)
    assert recovered["requests_attempted"] == 3
    assert recovered["requests_without_valid_usage"] == 0
    assert recovered["known_input_tokens"] == 36


def test_unverified_evidence_quote_is_recorded_but_no_longer_checked_against_source_text(plan, tmp_path):
    """Row attribution is already guaranteed by ID mapping; quote text is unverified audit context."""
    output = tmp_path / "run"
    result = assessment()
    result["evidence"][0]["quote"] = "This exact sentence does not appear anywhere in the source input."
    manifest = run(plan, FakeJudge([result, result]), output)
    assert manifest["status"] == "complete_selected_rows"
    assert manifest["result_status_counts"] == {"silver_consensus": 1}
    stored = jsonl(output / "results.jsonl")
    assert stored[0]["assessments"][0]["evidence"][0]["quote"] == result["evidence"][0]["quote"]


@pytest.mark.parametrize("route", ["invented", "tool_action"])
def test_wrong_taxonomy_route_rejected(plan, route):
    _, items, _ = load_plan(plan)
    with pytest.raises(ValueError, match="outside"):
        validate_annotation(assessment(route), items[0])


def test_abstention_cannot_sneak_in_a_guessed_label(plan):
    _, items, _ = load_plan(plan)
    response = assessment()
    response["verdict"] = "ambiguous"
    with pytest.raises(ValueError, match="guessed"):
        validate_annotation(response, items[0])


def test_disagreements_are_not_resolved_by_a_silent_tiebreak(plan, tmp_path):
    output = tmp_path / "run"
    run(plan, FakeJudge([assessment(), assessment("code_generation")]), output)
    result = jsonl(output / "results.jsonl")[0]
    assert result["status"] == "needs_adjudication" and result["route"] is None
    with pytest.raises(ValueError, match="No consensus"):
        export_consensus(plan, output, tmp_path / "export", allow_partial=True)


def test_subset_export_is_explicit_and_never_human_reviewed(plan, tmp_path):
    output = tmp_path / "run"
    run(plan, FakeJudge(), output)
    with pytest.raises(ValueError, match="Not all"):
        export_consensus(plan, output, tmp_path / "rejected")
    exported = export_consensus(plan, output, tmp_path / "export", allow_partial=True)
    assert exported["unexported_ready_count"] == 2
    records = load_records(tmp_path / "export" / "copilot-coding-v2.jsonl")
    assert records[0].taxonomy_id == "copilot-coding-v2"
    assert records[0].review_status == "unreviewed" and records[0].reviewer == ""
    assert "development_exposed" in records[0].tags
    with pytest.raises(ValueError, match="reviewed"):
        load_records(tmp_path / "export" / "copilot-coding-v2.jsonl", reviewed_only=True)


def test_tracks_export_separately_and_context_blocked_rows_never_sent(plan, tmp_path):
    judge = FakeJudge()
    output = tmp_path / "run"
    run(plan, judge, output, max_records=100, max_requests=6)
    assert len(judge.calls) == 6
    exported = export_consensus(plan, output, tmp_path / "export")
    assert set(exported["files"]) == {"copilot-coding-v2.jsonl", "tool-api-v1.jsonl"}
    assert exported["accepted_count"] == 3


def test_tampered_completed_results_cannot_be_silently_replaced(plan, tmp_path):
    output = tmp_path / "run"
    run(plan, FakeJudge(), output)
    (output / "results.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="changed"):
        run(plan, FakeJudge(), output, resume=True)
    with pytest.raises(ValueError, match="changed"):
        export_consensus(plan, output, tmp_path / "export", allow_partial=True)


def test_character_budget_fails_before_any_remote_request(plan, tmp_path):
    judge = FakeJudge()
    with pytest.raises(ValueError, match="character budget"):
        run(plan, judge, tmp_path / "run", max_input_chars=1024)
    assert judge.calls == []


def test_transport_failure_usage_is_unknown_not_zero(plan, tmp_path):
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="transport"):
        run(plan, FakeJudge([ValueError("transport failed before a response")]), output)
    result = run(plan, FakeJudge(), output, max_requests=3, resume=True)
    assert result["requests_attempted"] == 3
    assert result["known_input_tokens"] == 24
    assert result["requests_without_valid_usage"] == 1


@pytest.mark.parametrize("error", [JudgeRefusalError("refused"), JudgeIncompleteError("incomplete")])
def test_refusal_and_incomplete_response_are_audited_failures(plan, tmp_path, error):
    output = tmp_path / "run"
    with pytest.raises(type(error)):
        run(plan, FakeJudge([error]), output)
    events = jsonl(output / "events.jsonl")
    assert events[-1]["event"] == "failed"
    assert events[-1]["error_type"] == type(error).__name__
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("limits", [{"max_records": 0}, {"max_requests": 0}, {"max_input_chars": 0}])
def test_invalid_budgets_never_send_a_request(plan, tmp_path, limits):
    judge = FakeJudge()
    with pytest.raises(ValueError):
        run(plan, judge, tmp_path / "run", **limits)
    assert judge.calls == []
