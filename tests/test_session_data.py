"""Synthetic transcripts exercise private import, temporal context and approval gates."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import nbformat
import pytest

from routing_lab.session_data import (
    ReviewCandidate, SessionSource, _events, export_reviewed, load_approved_records,
    load_review_queue, prepare_sessions, redact, input_digest,
)
from routing_lab.data import load_records, write_records

HOST = "11111111-1111-4111-8111-111111111111"
SDK = "22222222-2222-4222-8222-222222222222"
START = datetime(2026, 9, 1, tzinfo=timezone.utc)
END = datetime(2026, 10, 1, tzinfo=timezone.utc)


def event(event_id, kind, day, content, **extra):
    return {
        "id": event_id, "type": kind, "timestamp": f"2026-09-{day:02}T12:00:00Z",
        "data": {"content": content, "interactionId": "a", **extra},
    }


@pytest.fixture
def source(tmp_path):
    host = tmp_path / "host" / HOST
    sdk = tmp_path / "sdk" / SDK
    host.mkdir(parents=True)
    sdk.mkdir(parents=True)
    with sqlite3.connect(host / "session.db") as db:
        db.execute("CREATE TABLE session_metadata(key TEXT, value TEXT)")
        db.execute("CREATE TABLE turns(id TEXT, event_id TEXT)")
        db.execute("INSERT INTO session_metadata VALUES (?,?)",
                   ("defaultChatProviderData", json.dumps({"sdkSessionId": SDK})))
        db.executemany("INSERT INTO turns VALUES (?,?)", [("1", "u1"), ("2", "u2")])
    events = [
        event("u1", "user.message", 2, "Explain\u2028the scheduler.", attachments=[{"path": "not-read"}],
              transformedContent="DO NOT IMPORT SECRET SYSTEM INSTRUCTIONS", parentAgentTaskId="nonempty"),
        event("sub", "user.message", 3, "Delegated synthetic task", interactionId="child"),
        event("child", "assistant.message", 4, "CHILD RESPONSE", interactionId="child", phase="final_answer"),
        event("tool", "tool.execution_complete", 4, "SECRET TOOL OUTPUT"),
        event("a1", "assistant.message", 5, "Prior explanation.", phase="final_answer",
              reasoningText="SECRET REASONING"),
        event("u2", "user.message", 6, "Add a test.", interactionId="b"),
        event("a2", "assistant.message", 7, "FUTURE SOLUTION", interactionId="b", phase="final_answer"),
    ]
    (sdk / "events.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in events) + "\n",
        encoding="utf-8",
    )
    return dict(
        sources=[SessionSource(host_id=HOST, created_at=START, workspace="workspace-one")],
        host_root=tmp_path / "host", sdk_root=tmp_path / "sdk",
        start=START, end=END, output=tmp_path / "queue",
    )


def test_exact_host_turns_prior_context_only_and_no_source_mutations(source):
    db = source["host_root"] / HOST / "session.db"
    before = db.read_bytes()
    manifest = prepare_sessions(**source)
    rows = load_review_queue(source["output"] / "review-queue.jsonl")
    assert len(rows) == 2
    assert rows[0].context == ""
    assert rows[0].query == "Explain\u2028the scheduler."
    assert "Prior explanation." in rows[1].context
    assert rows[1].prior_event_ids == ["u1", "a1"]
    assert "FUTURE" not in str(rows)
    assert "SECRET" not in str(rows)
    assert "CHILD RESPONSE" not in str(rows)
    assert "Delegated" not in str(rows)
    assert rows[0].expected_route is None
    assert not rows[0].include and not rows[0].remote_approved
    assert rows[0].group_id == rows[1].group_id
    assert rows[0].repository == rows[1].repository
    assert "attachments_not_imported" in rows[0].flags
    assert manifest["sessions"][0]["ignored_non_host_user_events"] == 1
    assert manifest["remote_calls"] == 0
    assert before == db.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_sessions(**source)


def test_timestamp_cutoff_excludes_later_user_turn(source):
    source["end"] = datetime(2026, 9, 6, tzinfo=timezone.utc)
    assert prepare_sessions(**source)["candidates"] == 1


def test_timestamp_order_not_log_append_order(source):
    path = source["sdk_root"] / SDK / "events.jsonl"
    lines = path.read_text(encoding="utf-8").split("\n")
    path.write_text("\n".join(reversed(lines)), encoding="utf-8")
    prepare_sessions(**source)
    rows = load_review_queue(source["output"] / "review-queue.jsonl")
    assert "Prior explanation." in rows[1].context
    assert "FUTURE" not in rows[1].context


def test_equal_timestamp_reply_is_conservatively_excluded(source):
    path = source["sdk_root"] / SDK / "events.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    for item in events:
        if item["id"] == "a1":
            item["timestamp"] = "2026-09-06T12:00:00Z"
    path.write_text("\n".join(json.dumps(item) for item in events), encoding="utf-8")
    prepare_sessions(**source)
    rows = load_review_queue(source["output"] / "review-queue.jsonl")
    assert "Prior explanation." not in rows[1].context
    assert "prior_assistant_reply_unavailable" in rows[1].flags


def test_creation_window_and_duplicate_authorization(source):
    source["sources"] *= 2
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_sessions(**source)
    source["sources"] = source["sources"][:1]
    source["sources"][0].created_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="creation window"):
        prepare_sessions(**source)


def test_missing_index_event_fails_closed(source):
    with sqlite3.connect(source["host_root"] / HOST / "session.db") as db:
        db.execute("INSERT INTO turns VALUES ('3','missing')")
    with pytest.raises(ValueError, match="Missing indexed"):
        prepare_sessions(**source)
    assert json.loads((source["output"] / "manifest.json").read_text())["status"] == "incomplete"


def test_partial_tail_is_reported_but_middle_corruption_fails():
    valid = json.dumps(event("u", "user.message", 1, "hello")).encode()
    events, omitted = _events(valid + b'\n{"incomplete":')
    assert len(events) == 1 and omitted
    with pytest.raises(ValueError, match="Invalid event JSON"):
        _events(b'{"broken":\n' + valid + b"\n")


def test_common_secrets_and_identifiers_redacted():
    text, flags = redact(
        'password="test-secret" api_key=abc123 Bearer fake.auth.token '
        'https://example.test/path?sig=secret person@example.test '
        r'C:\Users\someone\project'
        '\n-----BEGIN PRIVATE KEY-----\nkeybody\n-----END PRIVATE KEY-----'
    )
    for value in ["test-secret", "abc123", "fake.auth.token", "sig=secret",
                  "person@example.test", "someone", "keybody"]:
        assert value not in text
    assert "redacted:credential" in flags


def candidate(**changes):
    return ReviewCandidate(
        id="chat-one", session_id=HOST, event_id="u1", timestamp="2026-09-02T12:00:00Z",
        group_id="session-one", repository="workspace-one", query="Explain this function.",
        context="def constant(): return 1", prior_event_ids=[], source_query_sha256="example",
        flags=[], development_only=False,
    ).model_copy(update=changes)


def write_queue(tmp_path, row):
    path = tmp_path / "review.jsonl"
    path.write_text(row.model_dump_json() + "\n", encoding="utf-8")
    return path


def approved():
    return dict(
        include=True, expected_route="repository_qa", acceptable_routes=["repository_qa"],
        review_status="reviewed", reviewer="Synthetic test reviewer", context_reviewed=True,
        remote_approved=True,
        approved_input_sha256=input_digest("Explain this function.", "def constant(): return 1"),
    )


@pytest.mark.parametrize("missing", [
    "expected_route", "reviewer", "context_reviewed", "remote_approved", "review_status",
])
def test_export_rejects_any_missing_approval(tmp_path, missing):
    fields = approved()
    fields.pop(missing)
    with pytest.raises(ValueError, match="needs label"):
        export_reviewed(write_queue(tmp_path, candidate(**fields)), tmp_path / "export")
    assert not (tmp_path / "export").exists()


def test_export_development_session_rejected(tmp_path):
    with pytest.raises(ValueError, match="development"):
        export_reviewed(
            write_queue(tmp_path, candidate(**approved(), development_only=True)),
            tmp_path / "export",
        )


def test_export_no_fake_labels_and_approved_hash_gate(tmp_path):
    queue = write_queue(tmp_path, candidate())
    with pytest.raises(ValueError, match="No included"):
        export_reviewed(queue, tmp_path / "empty")
    queue = write_queue(tmp_path, candidate(**approved()))
    output = tmp_path / "export"
    manifest = export_reviewed(queue, output)
    rows = load_approved_records(output)
    assert len(rows) == 1 and rows[0].review_status == "reviewed"
    assert manifest["remote_approved"] is True
    with (output / "records.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed"):
        load_approved_records(output)


def test_exported_unicode_line_separator_round_trip(tmp_path):
    approval = {**approved(), "approved_input_sha256": input_digest(
        "Explain\u2028this function.", "def constant(): return 1",
    )}
    queue = write_queue(tmp_path, candidate(**approval, query="Explain\u2028this function."))
    output = tmp_path / "export"
    export_reviewed(queue, output)
    rows = load_approved_records(output)
    assert rows[0].query == "Explain\u2028this function."
    path = tmp_path / "round-trip.jsonl"
    write_records(path, rows)
    assert load_records(path)[0] == rows[0]


def test_editing_review_input_invalidates_privacy_approval(tmp_path):
    queue = write_queue(tmp_path, candidate(**approved(), context="Changed after review."))
    with pytest.raises(ValueError, match="hash-bound privacy"):
        export_reviewed(queue, tmp_path / "export")


def test_notebook_reviewer_saves_and_invalidates_edited_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr("IPython.display.display", lambda *args, **kwargs: None)
    notebook = nbformat.read(
        Path(__file__).resolve().parents[1] / "notebooks" / "02_real_session_dataset.ipynb",
        as_version=4,
    )
    source = next(cell.source for cell in notebook.cells if cell.id == "real-review-widget")
    namespace = {
        "queue": [candidate()], "ROUTES": ("debugging", "repository_qa"),
        "PRIVATE_DIR": tmp_path, "review_pointer": tmp_path / "active-review.json",
        "ReviewCandidate": ReviewCandidate, "input_digest": input_digest,
        "uuid4": uuid4, "json": json,
    }
    exec(compile(source, "notebook-review-widget", "exec"), namespace)
    namespace["include"].value = True
    namespace["save_selection"](None)
    assert not (tmp_path / "active-review.json").exists()
    namespace["label"].value = "repository_qa"
    namespace["reviewer"].value = "Synthetic test reviewer"
    namespace["context_checked"].value = True
    namespace["remote_ok"].value = True
    namespace["save_selection"](None)
    path = namespace["REVIEW_QUEUE_PATH"]
    saved = load_review_queue(path)[0]
    assert saved.expected_route in saved.acceptable_routes
    assert saved.approved_input_sha256 == input_digest(saved.query, saved.context)
    namespace["query_editor"].value = "Changed after approval."
    assert not namespace["context_checked"].value
    assert not namespace["remote_ok"].value
    namespace["save_selection"](None)
    assert namespace["REVIEW_QUEUE_PATH"] == path
    assert load_review_queue(path)[0] == saved
