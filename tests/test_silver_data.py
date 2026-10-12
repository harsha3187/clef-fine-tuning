"""Outcome attribution and weak-label safety using synthetic transcripts only."""

import hashlib
import json
from datetime import datetime, timezone

import pytest

from routing_lab.data import load_records
from routing_lab.evaluation import sha256
from routing_lab.session_data import ReviewCandidate
from routing_lab.silver_data import (
    build_silver_dataset, collect_outcome, infer_silver_label, load_silver_records,
)

HOST = "11111111-1111-4111-8111-111111111111"
SDK = "22222222-2222-4222-8222-222222222222"
CUTOFF = datetime(2026, 9, 30, tzinfo=timezone.utc)


def event(event_id, kind, day, **data):
    return {
        "id": event_id, "type": kind, "timestamp": f"2026-09-{day:02}T12:00:00Z",
        "data": data,
    }


def candidate():
    query = "Fix the concrete crash and add regression tests."
    return ReviewCandidate(
        id="chat-synthetic", session_id=HOST, event_id="u1",
        timestamp="2026-09-02T12:00:00Z", group_id="session-one", repository="workspace-one",
        query=query, context="PRE-DECISION CONTEXT", prior_event_ids=[],
        source_query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        flags=["attachments_not_imported"], development_only=False,
    )


def transcript():
    return [
        event("u1", "user.message", 2, content=candidate().query, interactionId="root"),
        event("a1", "assistant.message", 3, interactionId="root", model="model-a",
              content="I will investigate.", toolRequests=[
                  {"toolCallId": "call-a", "name": "task", "arguments": {"secret": "not copied"}},
                  {"toolCallId": "call-b", "name": "apply_patch", "arguments": {"patch": "not copied"}},
              ]),
        event("s1", "tool.execution_start", 3, toolCallId="call-a", toolName="task"),
        event("d1", "subagent.started", 3, toolCallId="call-a", agentName="debugging",
              model="model-child", executionMode="sync"),
        event("child", "assistant.message", 4, interactionId="child", model="DO NOT ATTRIBUTE",
              phase="final_answer", content="CHILD PRIVATE RESPONSE", toolRequests=[
                  {"toolCallId": "child-call", "name": "unrelated", "arguments": {}}]),
        event("c1", "tool.execution_complete", 5, toolCallId="call-a", success=False,
              result="SECRET TOOL RESULT"),
        event("foreign-agent", "subagent.started", 5, toolCallId="child-call",
              agentName="code-review", model="OTHER CHILD"),
        event("a2", "assistant.message", 6, interactionId="root", model="model-b",
              phase="final_answer", content="Fixed the crash. password=test-secret",
              reasoningText="SECRET REASONING"),
        event("late", "assistant.message", 30, interactionId="root", model="AFTER CUTOFF",
              phase="final_answer", content="AFTER CUTOFF RESPONSE"),
    ]


def test_attribution_joins_interaction_and_call_ids_not_time_alone():
    result = collect_outcome(candidate(), transcript(), CUTOFF)
    assert result["root_models"] == ["model-a", "model-b"]
    assert result["requested_tools"] == {"task": 1, "apply_patch": 1}
    assert result["started_tools"] == {"task": 1}
    assert result["failed_tool_call_ids"] == ["call-a"]
    assert result["delegated_agents"][0]["model"] == "model-child"
    assert len(result["delegated_agents"]) == 1
    assert len(result["final_responses"]) == 1
    assert result["task_success"] == "not_independently_verified"
    for text in ["CHILD PRIVATE", "SECRET TOOL", "SECRET REASONING", "test-secret",
                 "AFTER CUTOFF", "DO NOT ATTRIBUTE", "not copied"]:
        assert text not in str(result)


def outcome(response, *, agent=None, model="arbitrary"):
    return {
        "final_responses": [{"text": response}] if response else [],
        "root_models": [model], "delegated_agents": [{"agent": agent}] if agent else [],
    }


def test_fix_owns_regression_tests_and_model_name_is_not_a_signal():
    query = "Fix the crash and add regression tests."
    response = "Fixed the crash and added unit tests."
    first = infer_silver_label(query, outcome(response, model="fast-model"))
    second = infer_silver_label(query, outcome(response, model="large-model"))
    assert first == second
    assert first["route"] == "debugging" and first["status"] == "corroborated"
    assert not first["model_identity_used_as_label_signal"]


@pytest.mark.parametrize(("query", "response", "expected"), [
    ("Create an app and supporting tests.", "Built the app and added unit tests.", "code_generation"),
    ("Write unit tests for the parser.", "Added unit tests.", "test_generation"),
    ("Review the supplied code.", "No actionable findings.", "code_review"),
    ("Explain this function.", "The function returns the current value.", "repository_qa"),
    ("Research market opportunities.", "The research supports more market analysis.", "fallback"),
])
def test_query_and_response_corroboration(query, response, expected):
    result = infer_silver_label(query, outcome(response))
    assert result["route"] == expected
    assert result["status"] == "corroborated"


@pytest.mark.parametrize(("query", "response", "reason"), [
    ("Okay.", "Done.", "no_supported_route_signal"),
    ("Okay.", "Fixed the crash and built the app.", "conflicting_or_tied_signals"),
    ("Fix the crash.", "", "no_final_response_before_cutoff"),
])
def test_uncertainty_is_abstention_not_fallback(query, response, reason):
    result = infer_silver_label(query, outcome(response))
    assert result["route"] is None
    assert result["status"] == "abstained" and result["reason"] == reason


def test_unknown_agent_is_not_mapped_to_repository_qa():
    result = infer_silver_label("Okay.", outcome("Done.", agent="research"))
    assert result["route"] is None


def test_recorded_specialist_is_only_weak_evidence():
    result = infer_silver_label("Okay.", outcome("Done.", agent="code-review"))
    assert result["route"] == "code_review" and result["status"] == "tentative"


def test_source_mismatch_and_missing_interaction_fail_closed():
    data = transcript()
    data[0]["data"]["content"] = "Changed source"
    with pytest.raises(ValueError, match="original user"):
        collect_outcome(candidate(), data, CUTOFF)
    data = transcript()
    data[0]["data"].pop("interactionId")
    with pytest.raises(ValueError, match="interaction ID"):
        collect_outcome(candidate(), data, CUTOFF)


@pytest.fixture
def prepared(tmp_path):
    review = tmp_path / "review"
    sdk = tmp_path / "sdk" / SDK
    review.mkdir()
    sdk.mkdir(parents=True)
    original, dev = candidate(), candidate().model_copy(update={
        "id": "chat-dev", "event_id": "u-dev", "development_only": True,
    })
    unknown = candidate().model_copy(update={
        "id": "chat-unknown", "event_id": "u-unknown", "query": "Okay.",
        "source_query_sha256": hashlib.sha256(b"Okay.").hexdigest(),
    })
    queue = review / "review-queue.jsonl"
    queue.write_text("\n".join(row.model_dump_json() for row in [original, dev, unknown]) + "\n")
    events = transcript() + [
        event("u-dev", "user.message", 2, content=dev.query, interactionId="dev"),
        event("a-dev", "assistant.message", 3, interactionId="dev", phase="final_answer", content="Fixed."),
        event("u-unknown", "user.message", 2, content="Okay.", interactionId="unknown"),
        event("a-unknown", "assistant.message", 3, interactionId="unknown", phase="final_answer", content="Done."),
    ]
    path = sdk / "events.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in events) + "\n")
    (review / "manifest.json").write_text(json.dumps({
        "status": "complete", "review_queue_sha256": sha256(queue),
        "sessions": [{"host_id": HOST, "sdk_id": SDK, "events_sha256": sha256(path)}],
    }))
    return review, tmp_path / "sdk"


def test_automatic_dataset_keeps_evidence_out_of_features(prepared, tmp_path):
    review, sdk = prepared
    output = tmp_path / "silver"
    result = build_silver_dataset(review, sdk_root=sdk, output=output, as_of=CUTOFF)
    rows = load_silver_records(output)
    assert len(rows) == 1
    assert rows[0].query == candidate().query and rows[0].context == candidate().context
    assert rows[0].expected_route == "debugging"
    assert rows[0].review_status == "unreviewed" and rows[0].reviewer == ""
    assert "Fixed the crash" not in rows[0].context
    assert result["candidate_count"] == 3 and result["eligible_candidate_count"] == 2
    assert result["excluded_development_count"] == 1
    assert result["abstained_eligible_count"] == 1 and result["label_coverage"] == 0.5
    assert not result["remote_approved"] and result["remote_calls"] == 0
    evidence = [json.loads(line) for line in (output / "evidence.jsonl").read_text().splitlines()]
    assert len(evidence) == 3 and all("query" not in row and "context" not in row for row in evidence)
    with pytest.raises(ValueError, match="reviewed"):
        load_records(output / "silver-records.jsonl", reviewed_only=True)
    with pytest.raises(FileExistsError):
        build_silver_dataset(review, sdk_root=sdk, output=output, as_of=CUTOFF)
    with (output / "silver-records.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        load_silver_records(output)
