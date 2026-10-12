"""Prepare a private, unlabeled review queue from authorized VS Code sessions.

Read host databases and Copilot event logs locally; never call a model or modify
the sources. Export requires explicit label, context and remote-data approval.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from routing_lab.data import Record, Route, load_records, write_records
from routing_lab.evaluation import sha256


class SessionSource(BaseModel):
    """An explicitly authorized host session, not a broad history scan."""

    model_config = ConfigDict(extra="forbid")
    host_id: str
    created_at: datetime
    workspace: str
    development_only: bool = False


class ReviewCandidate(BaseModel):
    """Unlabeled until a human approves the exact sanitized decision-time text."""

    model_config = ConfigDict(extra="forbid", strict=True)
    id: str
    session_id: str
    event_id: str
    timestamp: str
    group_id: str
    repository: str
    query: str
    context: str
    prior_event_ids: list[str]
    source_query_sha256: str
    flags: list[str]
    development_only: bool
    include: bool = False
    expected_route: Route | None = None
    acceptable_routes: list[Route] = Field(default_factory=list)
    review_status: Literal["unreviewed", "reviewed"] = "unreviewed"
    context_reviewed: bool = False
    remote_approved: bool = False
    approved_input_sha256: str = ""
    reviewer: str = ""
    notes: str = ""


_REDACTIONS = (
    ("private_key", re.compile(
        r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.S)),
    ("credential", re.compile(
        r"""(?ix)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|accountkey|
        sharedaccesssignature|connectionstring)\b["']?\s*[:=]\s*
        (?:"[^"]*"|'[^']*'|[^\s,;}]+)""")),
    ("bearer", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-=]+")),
    ("token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]+|sk-[A-Za-z0-9_-]{20,})")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")),
    ("url", re.compile(r"https?://[^\s<>\"']+", re.I)),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("home", re.compile(r"(?i)(?:[a-z]:[\\/]Users[\\/][^\\/\s\"']+|/(?:home|Users)/[^/\s\"']+)")),
)


def redact(text: str) -> tuple[str, list[str]]:
    """Mask common identifiers/secrets; this is not a privacy certification."""
    flags = []
    for name, pattern in _REDACTIONS:
        text, count = pattern.subn(f"[REDACTED_{name.upper()}]", text)
        if count:
            flags.append(f"redacted:{name}")
    return text, flags


def _key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def input_digest(query: str, context: str) -> str:
    """Bind privacy approval to the exact edited input, not just a record ID."""
    return hashlib.sha256(
        json.dumps({"query": query, "context": context}, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _time(value: str | datetime) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if result.tzinfo is None:
        raise ValueError("Session windows and timestamps require a timezone")
    return result


def _events(raw: bytes) -> tuple[list[dict], bool]:
    """Ignore only an incomplete final append, reporting that omission explicitly."""
    lines = raw.decode("utf-8").split("\n")
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                return result, True
            raise ValueError(f"Invalid event JSON at line {index + 1}") from None
        if not isinstance(event, dict) or not {"id", "type", "timestamp", "data"} <= event.keys():
            raise ValueError(f"Unsupported event schema at line {index + 1}")
        result.append(event)
    return result, False


def _host_index(host_root: Path, host_id: str) -> tuple[str, set[str]]:
    database = host_root / str(UUID(host_id)) / "session.db"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as db:
        metadata = db.execute(
            "SELECT value FROM session_metadata WHERE key='defaultChatProviderData'"
        ).fetchone()
        if metadata is None:
            raise ValueError(f"No SDK mapping for host session {host_id}")
        sdk_id = str(UUID(json.loads(metadata[0])["sdkSessionId"]))
        # SDK logs also contain delegated user.message events; the host index excludes them.
        event_ids = {row[0] for row in db.execute(
            "SELECT event_id FROM turns WHERE event_id IS NOT NULL"
        )}
    return sdk_id, event_ids


def prepare_sessions(
    sources: list[SessionSource], *, host_root: Path, sdk_root: Path,
    start: datetime, end: datetime, output: Path,
) -> dict:
    """Snapshot authorized sessions into a local review queue with no proposed labels."""
    start, end = _time(start), _time(end)
    if start >= end or not sources:
        raise ValueError("Need authorized sessions and a nonempty time window")
    if len({source.host_id for source in sources}) != len(sources):
        raise ValueError("Duplicate authorized host sessions")
    if any(not start <= _time(source.created_at) < end for source in sources):
        raise ValueError("Inventory contains a session outside the authorized creation window")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "incomplete", "window_start": start.isoformat(), "window_end": end.isoformat(),
        "scope": "explicit VS Code session inventory, filtered by session creation and turn time",
        "context_policy": "up to two earlier host user turns and their earlier final replies; no tools or reasoning",
        "privacy": "heuristically redacted; human review and remote approval pending",
        "importer_sha256": sha256(Path(__file__)),
        "remote_calls": 0, "sessions": [],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    candidates = []
    for source in sources:
        sdk_id, root_ids = _host_index(host_root, source.host_id)
        raw = (sdk_root / sdk_id / "events.jsonl").read_bytes()
        events, partial_tail = _events(raw)
        indexed = {event["id"]: event for event in events}
        if len(indexed) != len(events):
            raise ValueError(f"Duplicate event IDs in session {source.host_id}")
        if root_ids - indexed.keys():
            raise ValueError(f"Missing indexed user events in {source.host_id}; retry after the session is idle")
        if any(indexed[event_id]["type"] != "user.message" for event_id in root_ids):
            raise ValueError("Host turn index did not resolve to user.message events")
        history: list[dict] = []
        session_count = 0
        for event in sorted(events, key=lambda event: _time(event["timestamp"])):
            when = _time(event["timestamp"])
            if when >= end:
                continue
            data = event["data"]
            if event["id"] in root_ids:
                content = data["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Indexed user turn has no usable text")
                query, flags = redact(content)
                context_parts, prior_ids = [], []
                if len(history) > 2:
                    flags.append("older_history_omitted")
                for previous in history[-2:]:
                    if previous["timestamp"] >= when:
                        flags.append("same_timestamp_context_omitted")
                        continue
                    context_parts.append(f"Earlier user request:\n{previous['query']}")
                    prior_ids.append(previous["event_id"])
                    flags.extend(previous["flags"])
                    if previous.get("reply") and previous["reply_timestamp"] < when:
                        context_parts.append(f"Earlier assistant reply:\n{previous['reply']}")
                        prior_ids.append(previous["reply_event_id"])
                    else:
                        flags.append("prior_assistant_reply_unavailable")
                context = "\n\n".join(context_parts)
                if len(context) > 10000:
                    context = context[-10000:]
                    flags.append("context_truncated_for_review")
                if data.get("attachments"):
                    flags.append("attachments_not_imported")
                if len(query.split()) < 8:
                    flags.append("short_followup_check_context")
                if source.development_only:
                    flags.append("routing_development_session")
                if when >= start:
                    candidates.append(ReviewCandidate(
                        id=f"chat-{_key(source.host_id + event['id'])}",
                        session_id=source.host_id, event_id=event["id"],
                        timestamp=event["timestamp"],
                        group_id=f"session-{_key(source.host_id)}",
                        repository=f"workspace-{_key(source.workspace.casefold().rstrip('/'))}",
                        query=query, context=context, prior_event_ids=prior_ids,
                        source_query_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                        flags=sorted(set(flags)), development_only=source.development_only,
                    ))
                    session_count += 1
                history.append({
                    "query": query, "event_id": event["id"],
                    "interaction_id": data.get("interactionId"), "flags": flags, "timestamp": when,
                })
            elif (event["type"] == "assistant.message" and data.get("interactionId")
                  and data.get("phase") in {"final", "final_answer"} and data.get("content")):
                previous = next((
                    item for item in reversed(history)
                    if item["interaction_id"] == data["interactionId"]
                ), None)
                if previous is None:
                    continue
                reply, reply_flags = redact(data["content"])
                if len(reply) > 5000:
                    reply = reply[:5000]
                    reply_flags.append("prior_reply_truncated_for_review")
                previous.update(reply=reply, reply_event_id=event["id"], reply_timestamp=when)
                previous["flags"] = sorted(set(previous["flags"] + reply_flags))
        manifest["sessions"].append({
            "host_id": source.host_id, "sdk_id": sdk_id, "candidate_count": session_count,
            "events_sha256": hashlib.sha256(raw).hexdigest(),
            "indexed_root_events": len(root_ids),
            "ignored_non_host_user_events": sum(
                event["type"] == "user.message" and event["id"] not in root_ids for event in events
            ),
            "partial_final_event_omitted": partial_tail,
            "development_only": source.development_only,
        })
    with (output / "review-queue.jsonl").open("x", encoding="utf-8") as stream:
        for candidate in candidates:
            stream.write(candidate.model_dump_json() + "\n")
    manifest.update(
        status="complete", candidates=len(candidates),
        evaluation_candidates=sum(not row.development_only for row in candidates),
        review_queue_sha256=sha256(output / "review-queue.jsonl"),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_review_queue(path: Path) -> list[ReviewCandidate]:
    """Load private candidates, rejecting duplicate identities."""
    rows = [
        ReviewCandidate.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]
    if not rows or len({row.id for row in rows}) != len(rows):
        raise ValueError("Need nonempty, unique review candidates")
    return rows


def export_reviewed(queue_path: Path, output: Path) -> dict:
    """Export only explicitly included, human-reviewed, Azure-approved records."""
    rows = load_review_queue(queue_path)
    selected = [row for row in rows if row.include]
    if not selected:
        raise ValueError("No included records; review and explicitly select candidates first")
    records = []
    for row in selected:
        if row.development_only:
            raise ValueError(f"{row.id}: routing-development turns cannot enter this evaluation export")
        if (row.review_status != "reviewed" or not row.reviewer.strip()
                or not row.context_reviewed or not row.remote_approved or row.expected_route is None):
            raise ValueError(f"{row.id}: needs label, named reviewer, context review and remote-data approval")
        if row.approved_input_sha256 != input_digest(row.query, row.context):
            raise ValueError(f"{row.id}: input changed or lacks hash-bound privacy approval; review it again")
        records.append(Record(
            id=row.id, group_id=row.group_id, repository=row.repository,
            query=row.query, context=row.context, expected_route=row.expected_route,
            acceptable_routes=row.acceptable_routes,
            tags=["real-session", "reviewed-pilot", *row.flags],
            provenance=f"vscode-user-turn:{row.session_id}:{row.event_id}",
            review_status="reviewed", reviewer=row.reviewer,
        ))
    output.mkdir(parents=True, exist_ok=False)
    dataset = output / "records.jsonl"
    write_records(dataset, records)
    manifest = {
        "status": "complete", "mode": "human_reviewed_real_session_pilot",
        "count": len(records), "remote_approved": True,
        "dataset_sha256": sha256(dataset), "review_queue_sha256": sha256(queue_path),
        "repositories": len({record.repository for record in records}),
        "sessions": len({record.group_id for record in records}),
        "excluded_candidates": len(rows) - len(selected),
    }
    (output / "approval.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_approved_records(directory: Path) -> list[Record]:
    """Reject unapproved or changed data before the notebook constructs a remote client."""
    manifest = json.loads((directory / "approval.json").read_text(encoding="utf-8"))
    path = directory / "records.jsonl"
    if (manifest.get("status") != "complete" or manifest.get("remote_approved") is not True
            or manifest.get("dataset_sha256") != sha256(path)):
        raise ValueError("Dataset is unapproved or changed since privacy review; export it again")
    return load_records(path, reviewed_only=True)
