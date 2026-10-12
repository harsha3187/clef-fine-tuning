"""Offline physical-request, recovery, atomic batch validation and provenance tests."""

import json
from pathlib import Path

import pytest

from routing_lab.annotation import (
    POLICY_VERSION, TRACKS, digest, judge_prompt, load_plan, prepare_plan,
)
from routing_lab.annotation_judge import JudgeError, JudgeResponse
from routing_lab import batch_annotation
from routing_lab.batch_annotation import (
    _encoded_jsonl, _load_run, _read_json, _read_jsonl, create_parser,
    export_consensus, run_annotations, run_with_retries,
)
from routing_lab.evaluation import sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def make_plan(tmp_path):
    def make(count=23, *, tools=0, blocked=0, exposed=()):
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        rows = [{
            "id": f"native-{index}", "source": "bfcl" if index >= count else "swe-bench",
            "category": "HIDDEN_CATEGORY", "native_split": "test", "native_id": f"native-{index}",
            "group_id": f"group-{index}", "repository": "fixture/repo",
            "query": f"Row {index}: an existing function returns the wrong type.",
            "context": f'{{"repository":"fixture/repo","row":{index}}}',
            "source_file_sha256": "a" * 64,
            "flags": ["native_environment_not_reconstructed"] if index >= count + tools else [],
            "proposed_route": "OLD_LABEL", "label_reason": "OLD_REASON",
        } for index in range(count + tools + blocked)]
        candidates = bundle / "candidates.jsonl"
        candidates.write_bytes(_encoded_jsonl(rows))
        (bundle / "manifest.json").write_text(json.dumps({
            "status": "complete_unreviewed_probes", "revisions": {"fixture": "frozen"},
            "files": {"candidates.jsonl": {"sha256": sha256(candidates)}},
        }), encoding="utf-8")
        policy = tmp_path / "policy.json"
        policy.write_text(json.dumps({
            "version": POLICY_VERSION, "track_assignment": TRACKS,
            "known_development_exposed_ids": [f"native-{index}" for index in exposed],
        }), encoding="utf-8")
        plan = tmp_path / "plan"
        prepare_plan(bundle, ROOT / "contracts", policy, plan)
        return plan
    return make


def metadata(**overrides):
    return {
        "provider": "offline-test", "endpoint": "https://fixture.openai.azure.com/openai/v1/",
        "deployment": "fixture", "configured_model": "fixture",
        "authentication": "AzureCliCredential", "auth_scope": "https://ai.azure.com/.default",
        "max_retries": 0, "follow_redirects": False, "store": False,
        "timeout_seconds": 180.0, "max_output_tokens": 16000, "config_sha256": "fixture-batch",
        **overrides,
    }


def annotation(route="debugging", quote="returns the wrong type"):
    return {
        "verdict": "label", "route": route, "acceptable_routes": [route],
        "rationale": "The request explicitly describes a wrong result requiring diagnosis.",
        "evidence": [{"field": "query", "quote": quote}], "missing_information": [],
    }


def wire_annotation(route, query):
    value = annotation(route, query)
    value["alternative_route"] = None
    value.pop("acceptable_routes")
    value["evidence"] = value["evidence"][0]
    value["missing_information"] = None
    return value


class FakeJudge:
    def __init__(self, modify=None, *, config=None, model="fixture-actual", output=None):
        self.calls = []
        self.modify = modify
        self.config = config or metadata()
        self.model = model
        self.output = output

    def metadata(self):
        return self.config

    def annotate(self, **kwargs):
        if self.output is not None:
            events = _read_jsonl(self.output / "batch_events.jsonl")
            assert events[-1]["event"] == "started"
            checkpoint = _read_json(self.output / "manifest.json")
            assert checkpoint["requests_attempted"] == events[-1]["request_number"]
            assert checkpoint["status"] == "in_progress"
        self.calls.append(kwargs)
        route = "debugging" if "debugging" in str(kwargs["schema"]) else "tool_lookup"
        result = {"annotations": {
            key: wire_annotation(route, row["query"])
            for key, row in reversed(list(kwargs["input_payload"]["items"].items()))
        }}
        if self.modify:
            self.modify(result, len(self.calls))
        return JudgeResponse(
            payload=result, model=self.model, response_id=f"fixture-{len(self.calls)}",
            input_tokens=100, output_tokens=40, elapsed_ms=1.0,
        )


def run(plan, judge, output, **kwargs):
    # Fixtures below assert exact batch counts built around a size of 10; pin that
    # explicitly here so the production default (see test_default_batch_size_is_hundred)
    # can change independently of these scenario-specific assertions.
    kwargs.setdefault("batch_size", 10)
    return run_annotations(plan, judge, output, allow_remote=True, max_requests=kwargs.pop("max_requests", 100),
                           **kwargs)


def serial_run(plan, output, *, completed_rows=7, extra_failure=True, partial=False):
    _, all_items, catalogs = load_plan(plan)
    items = [item for item in all_items if item.eligibility == "ready"]
    output.mkdir()
    events = []
    selected = items[:completed_rows]
    if partial:
        selected = items[:1]
    for item in selected:
        for index in (0, 1):
            if partial and index == 1:
                continue
            base = {
                "request_number": len(events) // 2 + 1, "id": item.id, "pass_index": index,
                "input_sha256": item.input_sha256, "utc": "2026-10-01T00:00:00+00:00",
            }
            events.append({**base, "event": "started"})
            events.append({
                **base, "event": "completed", "annotation": annotation(), "model": "fixture-actual",
                "response_id": f"legacy-{base['request_number']}", "input_tokens": 10, "output_tokens": 5,
                "elapsed_ms": 1.0,
            })
    if extra_failure:
        item = items[completed_rows]
        base = {
            "request_number": len(events) // 2 + 1, "id": item.id, "pass_index": 0,
            "input_sha256": item.input_sha256, "utc": "2026-10-01T00:00:00+00:00",
        }
        events.extend([
            {**base, "event": "started"},
            {**base, "event": "failed", "error_type": "ValidationError", "model": "fixture-actual",
             "response_id": "legacy-invalid", "input_tokens": 10, "output_tokens": 5},
        ])
    (output / "events.jsonl").write_bytes(_encoded_jsonl(events))
    (output / "manifest.json").write_text(json.dumps({
        "policy_version": POLICY_VERSION, "plan_sha256": sha256(plan / "manifest.json"),
        "judge": metadata(timeout_seconds=90.0, max_output_tokens=1600, config_sha256="fixture-serial"),
        "selected_count": len(items), "selected_ids_sha256": digest([item.id for item in items]),
        "prompt_sha256": digest({taxonomy: [judge_prompt(contracts, index) for index in (0, 1)]
                                 for taxonomy, contracts in catalogs.items()}),
        "annotation_code_sha256": sha256(ROOT / "routing_lab" / "annotation.py"),
        "status": "failed", "requests_attempted": 2, "completed_assessments": 2,
        "known_input_tokens": 20, "known_output_tokens": 10,
    }), encoding="utf-8")
    return output


def test_twenty_three_rows_are_three_batches_six_requests(make_plan, tmp_path):
    plan = make_plan()
    output = tmp_path / "run"
    judge = FakeJudge(output=output)
    result = run(plan, judge, output)
    assert len(judge.calls) == 6
    assert [len(call["input_payload"]["items"]) for call in judge.calls] == [10, 10, 10, 10, 3, 3]
    assert result["new_batch_count"] == 3
    assert result["status"] == "complete_selected_rows"
    assert result["completed_assessments"] == 46
    assert result["completed_rows"] == 23
    assert result["known_input_tokens"] == 600
    assert result["known_output_tokens"] == 240
    assert result["requests_without_valid_usage"] == 0
    assert len(_read_jsonl(output / "batch_events.jsonl")) == 12
    assert len(_read_jsonl(output / "votes.jsonl")) == 46
    assert all("input_tokens" not in vote for vote in _read_jsonl(output / "votes.jsonl"))
    for first, second in zip(judge.calls[::2], judge.calls[1::2]):
        assert first["system_prompt"] != second["system_prompt"]
        assert list(first["input_payload"]["items"]) == list(reversed(second["input_payload"]["items"]))
        assert first["input_payload"] == second["input_payload"]
    for row in _read_jsonl(output / "results.jsonl"):
        assert row["assessments"][0]["evidence"][0]["quote"].startswith(
            f"Row {row['id'].removeprefix('native-')}:"
        )
    sent = json.dumps(judge.calls)
    assert all(value not in sent for value in ("native-", "HIDDEN_CATEGORY", "OLD_LABEL", "OLD_REASON"))


def test_tracks_are_separate_and_blocked_rows_excluded(make_plan, tmp_path):
    plan = make_plan(11, tools=3, blocked=2)
    judge = FakeJudge()
    result = run(plan, judge, tmp_path / "run")
    assert len(judge.calls) == 6 and result["completed_rows"] == 14
    results = _read_jsonl(tmp_path / "run" / "results.jsonl")
    assert all(row["route"] == "debugging" for row in results[:11])
    assert all(row["route"] == "tool_lookup" for row in results[11:])
    for batch in _read_json(tmp_path / "run" / "identity.json")["batches"]:
        assert batch["taxonomy_id"] in ("copilot-coding-v2", "tool-api-v1")


@pytest.mark.parametrize("invalid", ["missing", "extra", "empty_evidence"])
def test_invalid_batch_is_atomic_and_failure_checkpoint_is_current(make_plan, tmp_path, invalid):
    plan = make_plan(12)
    output = tmp_path / "run"

    def mutate(payload, call):
        if call != 3:
            return
        if invalid == "missing":
            del payload["annotations"]["item_000"]
        elif invalid == "extra":
            payload["annotations"]["item_099"] = payload["annotations"]["item_000"]
        else:
            payload["annotations"]["item_000"]["evidence"]["quote"] = "   "

    with pytest.raises(ValueError):
        run(plan, FakeJudge(mutate), output)
    manifest = _read_json(output / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["requests_attempted"] == 3
    assert manifest["completed_assessments"] == 20
    assert manifest["completed_rows"] == 10
    assert manifest["known_input_tokens"] == 300
    assert len(_read_jsonl(output / "results.jsonl")) == 10
    assert len(_read_jsonl(output / "votes.jsonl")) == 20
    assert _read_jsonl(output / "batch_events.jsonl")[-1]["event"] == "failed"
    judge = FakeJudge()
    result = run(plan, judge, output, resume=True, max_requests=5)
    assert len(judge.calls) == 2 and result["completed_rows"] == 12
    assert result["requests_attempted"] == 5


def test_budget_preserves_passes_and_cumulative_request_numbers(make_plan, tmp_path):
    plan = make_plan()
    output = tmp_path / "run"
    result = run(plan, FakeJudge(), output, max_requests=3)
    assert result["status"] == "request_budget_reached"
    assert result["completed_rows"] == 10 and result["completed_assessments"] == 30
    judge = FakeJudge()
    result = run(plan, judge, output, resume=True, max_requests=6)
    assert len(judge.calls) == 3 and result["completed_rows"] == 23
    assert len(result["budget_approvals"]) == 2
    assert [event["request_number"] for event in _read_jsonl(output / "batch_events.jsonl")
            if event["event"] == "started"] == list(range(1, 7))
    with pytest.raises(ValueError, match="below"):
        run(plan, FakeJudge(), output, resume=True, max_requests=5)
    no_calls = FakeJudge()
    run(plan, no_calls, output, resume=True, max_requests=6)
    assert no_calls.calls == []


def test_serial_recovery_counts_all_fifteen_attempts_skips_seven_rows(make_plan, tmp_path):
    plan = make_plan(23)
    legacy = serial_run(plan, tmp_path / "legacy")
    original = {path.name: path.read_bytes() for path in legacy.iterdir()}
    output = tmp_path / "run"
    judge = FakeJudge()
    result = run(plan, judge, output, import_serial_run=legacy, max_requests=19)
    assert len(judge.calls) == 4
    assert result["imported_requests_attempted"] == 15
    assert result["imported_completed_rows"] == 7
    assert result["new_batch_count"] == 2
    assert result["requests_attempted"] == 19
    assert result["known_input_tokens"] == 150 + 400
    assert result["known_output_tokens"] == 75 + 160
    assert result["completed_rows"] == 23
    sent = json.dumps(judge.calls)
    assert all(f"Row {index}:" not in sent for index in range(7))
    assert {path.name: path.read_bytes() for path in legacy.iterdir()} == original
    for name in ("manifest.json", "events.jsonl"):
        assert (output / "serial_import" / name).read_bytes() == original[name]
    votes = _read_jsonl(output / "votes.jsonl")
    assert all(vote["origin"]["kind"] == "serial" for vote in votes[:14])
    assert all(vote["origin"]["kind"] == "batch" for vote in votes[14:])
    assert all(vote["model"] == "fixture-actual" for vote in votes)
    starts = [event for event in _read_jsonl(output / "batch_events.jsonl") if event["event"] == "started"]
    assert starts[0]["request_number"] == 16
    run(plan, FakeJudge(), output, resume=True, max_requests=19)


def test_import_budget_alone_can_checkpoint_and_export_existing_rows(make_plan, tmp_path):
    plan = make_plan(9)
    legacy = serial_run(plan, tmp_path / "legacy")
    judge = FakeJudge()
    output = tmp_path / "run"
    result = run(plan, judge, output, import_serial_run=legacy, max_requests=15)
    assert judge.calls == [] and result["completed_rows"] == 7
    assert result["estimated_total_requests_without_new_failures"] == 17
    assert result["status"] == "request_budget_reached"
    with pytest.raises(ValueError, match="allow-partial"):
        export_consensus(plan, output, tmp_path / "denied")
    exported = export_consensus(plan, output, tmp_path / "export", allow_partial=True)
    assert exported["accepted_count"] == 7 and exported["partial_consensus"]
    with pytest.raises(ValueError, match="below imported"):
        run(plan, FakeJudge(), tmp_path / "too-low", import_serial_run=legacy, max_requests=14)
    assert not (tmp_path / "too-low").exists()


@pytest.mark.parametrize("field,value", [
    ("provider", "other"), ("endpoint", "https://elsewhere/"), ("deployment", "other"),
    ("configured_model", "other"), ("authentication", "other"), ("auth_scope", "other"),
    ("store", True),
])
def test_import_rejects_connection_or_auth_drift(make_plan, tmp_path, field, value):
    plan = make_plan(9)
    legacy = serial_run(plan, tmp_path / "legacy")
    with pytest.raises(ValueError, match="identical provider"):
        run(plan, FakeJudge(config=metadata(**{field: value})), tmp_path / "run", import_serial_run=legacy)


@pytest.mark.parametrize("change", ["partial", "hash", "route", "selection", "model"])
def test_import_rejects_partial_rows_and_invalid_provenance(make_plan, tmp_path, change):
    plan = make_plan(9)
    legacy = serial_run(plan, tmp_path / "legacy", partial=change == "partial")
    if change in {"hash", "selection"}:
        path = legacy / "manifest.json"
        manifest = _read_json(path)
        manifest["plan_sha256" if change == "hash" else "selected_ids_sha256"] = "altered"
        path.write_text(json.dumps(manifest), encoding="utf-8")
    elif change in {"route", "model"}:
        path = legacy / "events.jsonl"
        events = _read_jsonl(path)
        if change == "route":
            events[1]["annotation"]["acceptable_routes"] = ["not_a_real_route"]
        else:
            events[1]["model"] = ""
        path.write_bytes(_encoded_jsonl(events))
    with pytest.raises(ValueError):
        run(plan, FakeJudge(), tmp_path / "run", import_serial_run=legacy)
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("artifact", [
    "batch_events.jsonl", "results.jsonl", "votes.jsonl", "selected.jsonl",
    "identity.json", "snapshot/copilot-coding-v2.json", "manifest.json",
])
def test_tampering_is_rejected_on_resume_and_export(make_plan, tmp_path, artifact):
    plan = make_plan(3)
    output = tmp_path / "run"
    run(plan, FakeJudge(), output)
    path = output.joinpath(*artifact.split("/"))
    if artifact == "manifest.json":
        value = _read_json(path)
        value["known_input_tokens"] += 1
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.write_bytes(path.read_bytes() + b"\n")
    judge = FakeJudge()
    with pytest.raises(ValueError):
        run(plan, judge, output, resume=True)
    with pytest.raises(ValueError):
        export_consensus(plan, output, tmp_path / "export")
    assert judge.calls == [] and not (tmp_path / "export").exists()


def test_import_copy_tampering_is_rejected(make_plan, tmp_path):
    plan = make_plan(9)
    legacy = serial_run(plan, tmp_path / "legacy")
    output = tmp_path / "run"
    run(plan, FakeJudge(), output, import_serial_run=legacy, max_requests=15)
    path = output / "serial_import" / "events.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="provenance artifacts"):
        run(plan, FakeJudge(), output, resume=True)


def test_export_revalidates_and_keeps_unreviewed_separate_tracks(make_plan, tmp_path):
    plan = make_plan(3, tools=2, blocked=1, exposed=(0,))
    output = tmp_path / "run"
    run(plan, FakeJudge(), output)
    destination = tmp_path / "export"
    exported = export_consensus(plan, output, destination)
    assert exported["accepted_count"] == 5 and not exported["partial_consensus"]
    assert exported["eligibility_counts"]["blocked_context"] == 1
    for taxonomy, count in (("copilot-coding-v2", 3), ("tool-api-v1", 2)):
        records = _read_jsonl(destination / f"{taxonomy}.jsonl")
        assert len(records) == count
        assert all(row["taxonomy_id"] == taxonomy and row["review_status"] == "unreviewed"
                   and row["reviewer"] == "" for row in records)
        assert all("group:" + row["group_id"] in row["tags"] and "native_split:test" in row["tags"]
                   for row in records)
        assert all(json.loads(row["provenance"])["assessment_origins"][0]["kind"] == "batch" for row in records)
        assert sha256(destination / f"{taxonomy}.contracts.json") == sha256(plan / f"{taxonomy}.json")
    assert "development_exposed" in _read_jsonl(destination / "copilot-coding-v2.jsonl")[0]["tags"]
    with pytest.raises(FileExistsError):
        export_consensus(plan, output, destination)


def test_uncertain_disagreed_and_unfinished_rows_all_appear_in_adjudication(make_plan, tmp_path):
    plan = make_plan(12)
    output = tmp_path / "run"

    def modify(payload, call):
        if call == 2:
            payload["annotations"]["item_001"] = wire_annotation("code_generation", "returns the wrong type")
            row = payload["annotations"]["item_002"]
            row.update(verdict="ambiguous", route=None)

    run(plan, FakeJudge(modify), output, max_requests=3)
    with pytest.raises(ValueError, match="allow-partial"):
        export_consensus(plan, output, tmp_path / "denied")
    destination = tmp_path / "export"
    result = export_consensus(plan, output, destination, allow_partial=True)
    assert result["accepted_count"] == 8
    assert result["disagreement_or_uncertain_count"] == 2
    assert result["pending_count"] == 2
    audit = _read_jsonl(destination / "adjudication.jsonl")
    assert {row["id"] for row in audit} == {"native-1", "native-2", "native-10", "native-11"}
    assert all(row["review_status"] == "unreviewed" and row["reviewer"] == "" for row in audit)


@pytest.mark.parametrize("error", [JudgeError("refusal"), RuntimeError("unexpected"), KeyboardInterrupt()])
def test_transport_and_unexpected_failures_checkpoint_without_swallowing(make_plan, tmp_path, error):
    plan = make_plan(3)
    output = tmp_path / "run"

    def fail(payload, call):
        raise error

    with pytest.raises(type(error)):
        run(plan, FakeJudge(fail), output)
    manifest = _read_json(output / "manifest.json")
    assert manifest["status"] == "failed"
    assert manifest["requests_attempted"] == 1
    assert manifest["requests_without_valid_usage"] == 1
    assert manifest["completed_assessments"] == 0
    recovered = run(plan, FakeJudge(), output, resume=True, max_requests=3)
    assert recovered["requests_attempted"] == 3 and recovered["completed_rows"] == 3
    assert recovered["requests_without_valid_usage"] == 1


def test_model_drift_fails_but_charges_usage(make_plan, tmp_path):
    plan = make_plan(3)
    output = tmp_path / "run"
    run(plan, FakeJudge(), output, max_requests=1)
    with pytest.raises(ValueError, match="model version changed"):
        run(plan, FakeJudge(model="other-actual"), output, resume=True, max_requests=2)
    manifest = _read_json(output / "manifest.json")
    assert manifest["known_input_tokens"] == 200 and manifest["completed_assessments"] == 3


def test_consent_budget_nonoverwrite_lock_and_new_only_import(make_plan, tmp_path):
    plan = make_plan(3)
    output = tmp_path / "run"
    judge = FakeJudge()
    with pytest.raises(PermissionError):
        run_annotations(plan, judge, output, max_requests=2)
    with pytest.raises(ValueError):
        run(plan, judge, output, max_requests=0)
    assert not output.exists() and judge.calls == []
    run(plan, judge, output)
    with pytest.raises(FileExistsError):
        run(plan, FakeJudge(), output)
    lock = output / ".batch-annotation.lock"
    lock.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(plan, FakeJudge(), output, resume=True)
    with pytest.raises(FileExistsError):
        export_consensus(plan, output, tmp_path / "export")
    assert lock.read_text() == "occupied"
    lock.unlink()
    with pytest.raises(ValueError, match="NEW"):
        run(plan, FakeJudge(), output, resume=True, import_serial_run=output)
    with pytest.raises(SystemExit):
        create_parser().parse_args(["run", "--plan", str(plan), "--judge-config", "fake", "--output", "fake"])


def test_configuration_and_batching_drift_are_rejected(make_plan, tmp_path):
    plan = make_plan(3)
    output = tmp_path / "run"
    run(plan, FakeJudge(), output)
    with pytest.raises(ValueError, match="identical judge"):
        run(plan, FakeJudge(config=metadata(max_output_tokens=16001)), output, resume=True)
    with pytest.raises(ValueError, match="batching"):
        run(plan, FakeJudge(), output, resume=True, batch_size=5)
    _, _, _, _, _, _, _, _, votes, completed = _load_run(plan, output)
    assert len(votes) == 6 and len(completed) == 2


@pytest.mark.parametrize("event_count", [2, 4])
@pytest.mark.parametrize("interrupt_before", ["results.jsonl", "manifest.json"])
def test_interrupted_checkpoint_recovers_exact_journaled_passes(
    make_plan, tmp_path, monkeypatch, event_count, interrupt_before,
):
    plan = make_plan(3)
    output = tmp_path / "run"
    atomic = batch_annotation._atomic_bytes
    interrupted = False

    def interrupt(path, data):
        nonlocal interrupted
        journal = output / "batch_events.jsonl"
        if (not interrupted and path.name == interrupt_before and journal.exists()
                and len(_read_jsonl(journal)) == event_count):
            interrupted = True
            raise KeyboardInterrupt()
        atomic(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(batch_annotation, "_atomic_bytes", interrupt)
        with pytest.raises(KeyboardInterrupt):
            run(plan, FakeJudge(), output)
    assert interrupted
    judge = FakeJudge()
    result = run(plan, judge, output, resume=True)
    assert result["requests_attempted"] == 2
    assert result["completed_rows"] == 3
    assert len(judge.calls) == 2 - event_count // 2


def test_resume_rejects_changed_input_and_semantically_tampered_derived_results(make_plan, tmp_path):
    plan = make_plan(3)
    output = tmp_path / "run"
    run(plan, FakeJudge(), output)
    path = output / "results.jsonl"
    rows = _read_jsonl(path)
    rows[0]["route"] = "code_generation"
    path.write_bytes(_encoded_jsonl(rows))
    manifest_path = output / "manifest.json"
    manifest = _read_json(manifest_path)
    manifest["results_sha256"] = sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the audit"):
        export_consensus(plan, output, tmp_path / "export")
    path = plan / "items.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed"):
        run(plan, FakeJudge(), output, resume=True)


# --- Bounded retries for locally rejected batch annotations (merged from batch_worker) ---

def reject_evidence(payload, call):
    payload["annotations"]["item_000"]["evidence"]["quote"] = "   "


def failed_run(make_plan, tmp_path, *, count=2, modify=reject_evidence):
    plan = make_plan(count)
    output = tmp_path / "run"
    with pytest.raises(ValueError, match="empty"):
        run(plan, FakeJudge(modify=modify), output)
    return plan, output


def retry(plan, output, judge, *, cap=3, allowance=1):
    return run_with_retries(
        plan, judge, output, max_requests=cap, max_retry_requests=allowance, allow_remote=True,
    )


def test_rejected_response_retried_once_without_altering_evidence(make_plan, tmp_path):
    plan, output = failed_run(make_plan, tmp_path)
    original_failure = _read_jsonl(output / "batch_events.jsonl")[-1]
    judge = FakeJudge()
    report = retry(plan, output, judge)
    assert report["status"] == "complete_selected_rows"
    assert report["requests_attempted"] == 3
    assert report["completed_rows"] == 2
    assert len(judge.calls) == 2
    assert _read_jsonl(output / "batch_events.jsonl")[1] == original_failure
    assert _read_json(output / "retry_policy.json")["max_retry_requests"] == 1
    _load_run(plan, output, judge.metadata())


def test_successful_first_pass_is_never_repeated(make_plan, tmp_path):
    def reject_second(payload, call):
        if call == 2:
            reject_evidence(payload, call)
    plan, output = failed_run(make_plan, tmp_path, modify=reject_second)
    judge = FakeJudge()
    report = retry(plan, output, judge)
    assert report["completed_rows"] == 2
    assert len(judge.calls) == 1
    starts = [event["pass_index"] for event in _read_jsonl(output / "batch_events.jsonl")
              if event["event"] == "started"]
    assert starts == [0, 1, 1]


def test_second_rejection_stops_and_restart_cannot_reset_allowance(make_plan, tmp_path):
    plan, output = failed_run(make_plan, tmp_path)
    judge = FakeJudge(modify=reject_evidence)
    with pytest.raises(ValueError, match="single permitted retry"):
        retry(plan, output, judge)
    assert len(judge.calls) == 1
    next_judge = FakeJudge()
    with pytest.raises(ValueError, match="single permitted retry"):
        retry(plan, output, next_judge)
    assert not next_judge.calls
    assert _read_json(output / "manifest.json")["requests_attempted"] == 2


def test_total_retry_allowance_is_enforced_across_batches(make_plan, tmp_path):
    plan, output = failed_run(make_plan, tmp_path, count=11)
    def fail_next_batch(payload, call):
        if call == 3:
            reject_evidence(payload, call)
    judge = FakeJudge(modify=fail_next_batch)
    with pytest.raises(ValueError, match="retry allowance is exhausted"):
        retry(plan, output, judge, cap=5)
    assert len(judge.calls) == 3
    assert _read_json(output / "manifest.json")["completed_rows"] == 10


def test_service_error_is_not_automatically_retried(make_plan, tmp_path):
    plan, output = failed_run(make_plan, tmp_path)
    def unavailable(payload, call):
        raise JudgeError("service unavailable")
    judge = FakeJudge(modify=unavailable)
    with pytest.raises(JudgeError, match="unavailable"):
        retry(plan, output, judge)
    assert len(judge.calls) == 1
    next_judge = FakeJudge()
    with pytest.raises(ValueError, match="locally rejected"):
        retry(plan, output, next_judge)
    assert not next_judge.calls


def test_model_change_is_not_retried(make_plan, tmp_path):
    def reject_second(payload, call):
        if call == 2:
            reject_evidence(payload, call)
    plan, output = failed_run(make_plan, tmp_path, modify=reject_second)
    judge = FakeJudge(model="different-version")
    with pytest.raises(ValueError, match="Model-version changes"):
        retry(plan, output, judge)
    assert len(judge.calls) == 1


def test_integrity_failure_makes_no_requests(make_plan, tmp_path):
    plan, output = failed_run(make_plan, tmp_path)
    with (output / "votes.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    judge = FakeJudge()
    with pytest.raises(ValueError, match="artifacts changed"):
        retry(plan, output, judge)
    assert not judge.calls


def test_consent_and_physical_cap_are_enforced(make_plan, tmp_path):
    plan, output = failed_run(make_plan, tmp_path)
    judge = FakeJudge()
    with pytest.raises(PermissionError):
        run_with_retries(plan, judge, output, max_requests=3, max_retry_requests=1)
    with pytest.raises(ValueError, match="cap exceeds"):
        retry(plan, output, judge, cap=4)
    with pytest.raises(ValueError, match="no room"):
        retry(plan, output, judge, cap=1)
    assert not judge.calls


def test_policy_changes_require_explicit_migration(make_plan, tmp_path):
    plan, output = failed_run(make_plan, tmp_path)
    retry(plan, output, FakeJudge())
    judge = FakeJudge()
    with pytest.raises(ValueError, match="policy or worker implementation changed"):
        retry(plan, output, judge, cap=4, allowance=2)
    assert not judge.calls


def test_retry_subcommand_is_registered_on_the_shared_cli():
    args = create_parser().parse_args([
        "retry", "--plan", "p", "--judge-config", "j", "--output", "o",
        "--max-requests", "5", "--max-retry-requests", "1", "--allow-remote",
    ])
    assert args.command == "retry"
    assert args.max_retry_requests == 1


def test_default_batch_size_is_hundred(make_plan, tmp_path):
    """Production default; test fixtures above pin batch_size=10 explicitly instead."""
    import inspect
    assert inspect.signature(run_annotations).parameters["batch_size"].default == 100
    args = create_parser().parse_args([
        "run", "--plan", "p", "--judge-config", "j", "--output", "o", "--max-requests", "5",
    ])
    assert args.batch_size == 100
    plan = make_plan(23)
    output = tmp_path / "run"
    judge = FakeJudge(output=output)
    result = run_annotations(plan, judge, output, allow_remote=True, max_requests=2)
    assert result["new_batch_count"] == 1
    assert [len(call["input_payload"]["items"]) for call in judge.calls] == [23, 23]
    with pytest.raises(ValueError, match="Invalid batch size"):
        run_annotations(plan, judge, tmp_path / "invalid", allow_remote=True, max_requests=2, batch_size=101)


# --- One-off import of a prior BATCHED run's completed rows across a validator change ---

def completed_batch_run(make_plan, tmp_path, *, count=23, name="source"):
    plan = make_plan(count)
    source = tmp_path / name
    judge = FakeJudge(output=source)
    run(plan, judge, source, batch_size=10, max_requests=100)
    return plan, source


def test_import_batch_run_skips_its_rows_and_preserves_request_accounting(make_plan, tmp_path):
    plan, source = completed_batch_run(make_plan, tmp_path)
    destination = tmp_path / "destination"
    judge = FakeJudge()
    result = run(plan, judge, destination, batch_size=10, max_requests=6, import_batch_run=source)
    assert result["imported_requests_attempted"] == 6
    assert result["imported_completed_rows"] == 23
    assert result["new_batch_count"] == 0
    assert judge.calls == []
    assert result["status"] == "complete_selected_rows"
    assert result["completed_rows"] == 23
    imported = _read_jsonl(destination / "votes.jsonl")
    assert all(vote["origin"]["kind"] == "imported_batch" for vote in imported)
    assert (destination / "batch_import" / "manifest.json").exists()
    assert (destination / "batch_import" / "identity.json").exists()
    assert (destination / "batch_import" / "batch_events.jsonl").exists()


def test_import_batch_run_does_not_require_matching_implementation_hash(make_plan, tmp_path):
    """The whole point: importing across a deliberate, backward-compatible code change."""
    plan, source = completed_batch_run(make_plan, tmp_path)
    identity_path = source / "identity.json"
    identity = _read_json(identity_path)
    identity["implementation_sha256"] = {"annotation.py": "pretend-old-hash"}
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    destination = tmp_path / "destination"
    result = run(plan, FakeJudge(), destination, batch_size=10, max_requests=6, import_batch_run=source)
    assert result["imported_completed_rows"] == 23


def test_import_batch_run_rejects_locked_or_mismatched_metadata_source(make_plan, tmp_path):
    plan, source = completed_batch_run(make_plan, tmp_path)
    (source / ".batch-annotation.lock").write_text("123", encoding="utf-8")
    with pytest.raises(ValueError, match="locked"):
        run(plan, FakeJudge(), tmp_path / "d1", batch_size=10, max_requests=6, import_batch_run=source)
    (source / ".batch-annotation.lock").unlink()

    with pytest.raises(ValueError, match="provider, endpoint"):
        run(plan, FakeJudge(config=metadata(configured_model="different")), tmp_path / "d3",
            batch_size=10, max_requests=6, import_batch_run=source)


def test_import_batch_run_rejects_a_partially_completed_source(make_plan, tmp_path):
    plan = make_plan(11)
    source = tmp_path / "source"
    run(plan, FakeJudge(output=source), source, batch_size=10, max_requests=3)
    assert _read_json(source / "manifest.json")["completed_rows"] == 10  # row 11's second pass never ran
    with pytest.raises(ValueError, match="partially completed"):
        run(plan, FakeJudge(), tmp_path / "d0", batch_size=10, max_requests=6, import_batch_run=source)


def test_import_batch_run_rejects_a_different_plan(make_plan, tmp_path):
    plan, source = completed_batch_run(make_plan, tmp_path)
    identity_path = source / "identity.json"
    identity = _read_json(identity_path)
    identity["plan_sha256"] = "a-different-plan-hash"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    with pytest.raises(ValueError, match="identical annotation plan"):
        run(plan, FakeJudge(), tmp_path / "d2", batch_size=10, max_requests=6, import_batch_run=source)


def test_import_batch_run_rejects_a_chained_import(make_plan, tmp_path):
    plan, first = completed_batch_run(make_plan, tmp_path, count=11, name="first")
    chained = tmp_path / "chained"
    run(plan, FakeJudge(), chained, batch_size=10, max_requests=4, import_batch_run=first)
    with pytest.raises(ValueError, match="chained another import"):
        run(plan, FakeJudge(), tmp_path / "d4", batch_size=10, max_requests=4, import_batch_run=chained)


def test_import_batch_run_rejects_tampered_source_events(make_plan, tmp_path):
    plan, source = completed_batch_run(make_plan, tmp_path, count=11)
    destination = tmp_path / "d5"
    run(plan, FakeJudge(), destination, batch_size=10, max_requests=4, import_batch_run=source)
    tampered = _read_jsonl(source / "batch_events.jsonl")
    tampered[0]["model"] = "tampered-model"  # stale event_sha256 now mismatches recomputed content
    (source / "batch_events.jsonl").write_bytes(_encoded_jsonl(tampered))
    with pytest.raises(ValueError, match="hash chain"):
        run(plan, FakeJudge(), tmp_path / "d6", batch_size=10, max_requests=4, import_batch_run=source,
            resume=False)


def test_cannot_combine_imports_or_import_while_resuming(make_plan, tmp_path):
    plan, source = completed_batch_run(make_plan, tmp_path)
    with pytest.raises(ValueError, match="Only one of"):
        run(plan, FakeJudge(), tmp_path / "d7", batch_size=10, max_requests=6,
            import_batch_run=source, import_serial_run=source)
    with pytest.raises(ValueError, match="supported only for a NEW run"):
        run(plan, FakeJudge(), tmp_path / "d8", resume=True, import_batch_run=source)


def test_import_batch_subcommand_cli_argument_is_registered():
    args = create_parser().parse_args([
        "run", "--plan", "p", "--judge-config", "j", "--output", "o", "--max-requests", "5",
        "--import-batch-run", "some-prior-run",
    ])
    assert args.import_batch_run == Path("some-prior-run")
