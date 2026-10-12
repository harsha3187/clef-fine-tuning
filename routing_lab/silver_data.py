"""Build local silver routing labels from recorded session outcomes and telemetry.

Post-decision evidence is never copied into Record.query/context. Labels remain
unreviewed, uncertain cases abstain, and model identity is descriptive only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from uuid import UUID

from routing_lab.data import ROUTES, Record, load_records, write_records
from routing_lab.evaluation import sha256
from routing_lab.session_data import ReviewCandidate, _events, _time, load_review_queue, redact

LABELER_VERSION = "session-outcome-rules-v1"
AGENT_ROUTES = {
    "code-review": "code_review", "security-review": "code_review",
    "debugging": "debugging", "code-generation": "code_generation",
    "test-generation": "test_generation", "repository-qa": "repository_qa",
}
RESPONSE_PATTERNS = {
    "debugging": r"\b(fixed|repaired|root cause|resolved the (?:bug|error)|bug was|caused by)\b",
    "code_generation": (
        r"\b(added|built|created|implemented|updated|converted|changed)\b"
        r"[^\n]{0,180}\b(app|webapp|page|notebook|features?|function|script|widget|chart|graph|visual|currency|formatter)\b"
    ),
    "test_generation": r"\b(added|created|wrote|extended)\b[^\n]{0,100}\b(unit tests|test cases|test suite|test coverage)\b",
    "code_review": r"\b(review findings|actionable findings|no actionable (?:issues|findings)|reviewed the (?:code|diff))\b",
    "repository_qa": r"\b(function|class|method|module)\b[^\n]{0,80}\b(defined in|located in|calls|returns|implements)\b",
    "fallback": (
        r"\b(research|feasibility|market|trade data|imports|exports|supplier countries|"
        r"lottery|draws|predictions|forecast|pushed|deployed|provisioned|"
        r"missing (?:code|context)|need (?:the code|more context))\b"
    ),
}


def collect_outcome(candidate: ReviewCandidate, events: list[dict], as_of: datetime) -> dict:
    """Join root assistant/tool/agent events by interaction and tool-call IDs."""
    indexed = {event["id"]: event for event in events}
    if len(indexed) != len(events) or candidate.event_id not in indexed:
        raise ValueError("Missing or duplicate source events")
    user = indexed[candidate.event_id]
    if (user["type"] != "user.message"
            or user["timestamp"] != candidate.timestamp
            or hashlib.sha256(user["data"]["content"].encode("utf-8")).hexdigest()
            != candidate.source_query_sha256):
        raise ValueError("Candidate does not match its original user event")
    interaction = user["data"].get("interactionId")
    if not interaction:
        raise ValueError("Cannot attribute outcomes without an interaction ID")
    start, end = _time(candidate.timestamp), _time(as_of)
    if end <= start:
        raise ValueError("Outcome cutoff must follow the source user turn")
    window = [event for event in events if start <= _time(event["timestamp"]) < end]
    messages = sorted(
        [event for event in window if event["type"] == "assistant.message"
         and event["data"].get("interactionId") == interaction],
        key=lambda event: _time(event["timestamp"]),
    )
    requests = {
        request["toolCallId"]: request["name"]
        for event in messages for request in event["data"].get("toolRequests", [])
    }
    started = {
        event["data"]["toolCallId"]: event["data"]["toolName"]
        for event in window if event["type"] == "tool.execution_start"
        and event["data"].get("toolCallId") in requests
    }
    completed = [
        event for event in window if event["type"] == "tool.execution_complete"
        and event["data"].get("toolCallId") in requests
    ]
    delegated = []
    for event in window:
        data = event["data"]
        if event["type"] == "subagent.started" and data.get("toolCallId") in requests:
            delegated.append({
                "event_id": event["id"], "tool_call_id": data["toolCallId"],
                "agent": data.get("agentName"), "model": data.get("model"),
                "execution_mode": data.get("executionMode"),
            })
    final_responses = []
    for event in messages:
        data = event["data"]
        if data.get("phase") in {"final", "final_answer"} and data.get("content"):
            text, flags = redact(data["content"])
            final_responses.append({
                "event_id": event["id"], "timestamp": event["timestamp"],
                "text": text, "redaction_flags": flags,
            })
    return {
        "final_responses": final_responses,
        "root_models": sorted({
            event["data"]["model"] for event in messages
            if isinstance(event["data"].get("model"), str)
        }),
        "delegated_agents": delegated,
        "requested_tools": dict(Counter(requests.values())),
        "started_tools": dict(Counter(started.values())),
        "failed_tool_call_ids": [
            event["data"]["toolCallId"] for event in completed
            if event["data"].get("success") is False
        ],
        "tool_completion_event_ids": [event["id"] for event in completed],
        "task_success": "not_independently_verified",
    }


def _request_route(query: str) -> str | None:
    """Apply initial-owner precedence; supporting tests stay with fixes/features."""
    text = query.lower()
    if re.search(
        r"\b(fix|debug|crash\w*|exception|broken|fails?|failing|incorrect|not working)\b"
        r"|\bshows?\s+(?:raw\s+)?html\b|\bnot\s+(?:a\s+)?visual\b", text,
    ):
        return "debugging"
    if (re.search(r"\b(implement|refactor|build|create|add|extend)\b", text)
            and re.search(r"\b(feature[s]?|function|api|script|webapp|app|notebook|dashboard|graph|visual|interactive|page|component)\b", text)):
        return "code_generation"
    if re.search(r"\b(similar|another)\s+page\b|\b(show|display|convert|format)\b.*\b(currency|aed|usd|eur)\b", text):
        return "code_generation"
    if re.search(r"\b(write|add|create|extend)\b.*\btests?\b", text, re.S):
        return "test_generation"
    if re.search(r"\breview\b.*\b(code|diff|pull request|pr)\b", text, re.S):
        return "code_review"
    if (re.search(r"\b(explain|where|how|what|trace|find)\b", text)
            and re.search(r"\b(repository|code|function|class|module|method|implementation)\b", text)):
        return "repository_qa"
    if re.search(
        r"\b(research|market|opportunit\w*|trader|trade data|imports|exports|"
        r"scrape|predict|forecast|freight|certification|push|deploy|provision)\b"
        r"|\bsource the data\b", text,
    ):
        return "fallback"
    return None


def infer_silver_label(query: str, outcome: dict) -> dict:
    """Rank transparent weak signals, not calibrated correctness probabilities."""
    signals = []
    requested = _request_route(query)
    if requested:
        signals.append({"source": "request_intent", "route": requested, "weight": 5})
    response = "\n".join(row["text"] for row in outcome["final_responses"])
    for route, pattern in RESPONSE_PATTERNS.items():
        if re.search(pattern, response, re.I):
            signals.append({"source": "response_report", "route": route, "weight": 2})
    agent_routes = {
        AGENT_ROUTES[agent["agent"]]
        for agent in outcome["delegated_agents"] if agent["agent"] in AGENT_ROUTES
    }
    for route in sorted(agent_routes):
        signals.append({"source": "specialist_delegation", "route": route, "weight": 2})
    scores = {route: sum(s["weight"] for s in signals if s["route"] == route) for route in ROUTES}
    ranked = sorted(scores, key=lambda route: scores[route], reverse=True)
    top, second = ranked[:2]
    reason = None
    if not outcome["final_responses"]:
        reason = "no_final_response_before_cutoff"
    elif scores[top] == 0:
        reason = "no_supported_route_signal"
    elif scores[top] - scores[second] < 2:
        reason = "conflicting_or_tied_signals"
    supports = {signal["source"] for signal in signals if signal["route"] == top}
    return {
        "route": None if reason else top,
        "status": "abstained" if reason else ("corroborated" if len(supports) >= 2 else "tentative"),
        "reason": reason or "highest_weighted_weak_evidence",
        "signals": signals, "ranking_scores_not_probabilities": scores,
        "labeler_version": LABELER_VERSION,
        "model_identity_used_as_label_signal": False,
    }


def build_silver_dataset(
    review_directory: Path, *, sdk_root: Path, output: Path, as_of: datetime,
) -> dict:
    """Create automatically labeled records plus separate private outcome evidence."""
    source_manifest = json.loads((review_directory / "manifest.json").read_text(encoding="utf-8"))
    queue_path = review_directory / "review-queue.jsonl"
    if (source_manifest.get("status") != "complete"
            or source_manifest["review_queue_sha256"] != sha256(queue_path)):
        raise ValueError("Original extraction is incomplete or changed")
    candidates = load_review_queue(queue_path)
    end = _time(as_of)
    if end < max(_time(row.timestamp) for row in candidates):
        raise ValueError("Outcome cutoff predates candidates")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "incomplete", "mode": "automatic_silver_not_gold",
        "labeler_version": LABELER_VERSION, "outcome_as_of": end.isoformat(),
        "review_queue_sha256": sha256(queue_path),
        "extraction_manifest_sha256": sha256(review_directory / "manifest.json"),
        "labeler_source_sha256": sha256(Path(__file__)),
        "session_importer_source_sha256": sha256(Path(__file__).with_name("session_data.py")),
        "remote_approved": False, "remote_calls": 0, "sources": [],
        "limitations": [
            "Observed responses and agent choices can be wrong; these labels are unreviewed proxies.",
            "Model names are observed execution metadata, not optimal-model labels.",
            "Tool success and reported task completion do not establish independently verified task success.",
            "Missing attachments and truncated decision-time context remain unresolved.",
            "Selection/abstention coverage must accompany any agreement score.",
            "Post-decision evidence is stored separately and must not enter router features.",
        ],
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    evidence_rows, records = [], []
    for source in source_manifest["sessions"]:
        sdk_id = str(UUID(source["sdk_id"]))
        raw = (sdk_root / sdk_id / "events.jsonl").read_bytes()
        events, incomplete_tail = _events(raw)
        manifest["sources"].append({
            "host_id": source["host_id"], "sdk_id": sdk_id,
            "current_events_sha256": hashlib.sha256(raw).hexdigest(),
            "extraction_events_sha256": source["events_sha256"],
            "partial_final_event_omitted": incomplete_tail,
        })
        for candidate in candidates:
            if candidate.session_id != source["host_id"]:
                continue
            outcome = collect_outcome(candidate, events, end)
            label = infer_silver_label(candidate.query, outcome)
            included = not candidate.development_only and label["route"] is not None
            evidence_rows.append({
                "id": candidate.id, "source_user_event_id": candidate.event_id,
                "session_id": candidate.session_id, "label": label, "observed": outcome,
                "development_only": candidate.development_only,
                "included_in_silver_records": included, "input_flags": candidate.flags,
            })
            if included:
                records.append(Record(
                    id=candidate.id, group_id=candidate.group_id, repository=candidate.repository,
                    query=candidate.query, context=candidate.context,
                    expected_route=label["route"], acceptable_routes=[label["route"]],
                    tags=["real-session", "automatic-silver", label["status"], *candidate.flags],
                    provenance=f"{LABELER_VERSION}:{candidate.session_id}:{candidate.event_id}",
                    review_status="unreviewed", reviewer="",
                ))
    if len(evidence_rows) != len(candidates):
        raise ValueError("Extraction manifest does not cover every candidate exactly once")
    write_records(output / "silver-records.jsonl", records)
    with (output / "evidence.jsonl").open("x", encoding="utf-8") as stream:
        for row in evidence_rows:
            stream.write(json.dumps(row, ensure_ascii=True) + "\n")
    eligible = [row for row in evidence_rows if not row["development_only"]]
    manifest.update({
        "status": "complete", "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible), "silver_record_count": len(records),
        "label_coverage": len(records) / len(eligible) if eligible else None,
        "excluded_development_count": sum(row["development_only"] for row in evidence_rows),
        "abstained_eligible_count": sum(row["label"]["route"] is None for row in eligible),
        "route_counts": dict(Counter(record.expected_route for record in records)),
        "eligible_label_status_counts": dict(Counter(row["label"]["status"] for row in eligible)),
        "recorded_root_models": sorted({model for row in eligible for model in row["observed"]["root_models"]}),
        "recorded_delegated_agents": sorted({
            agent["agent"] for row in eligible for agent in row["observed"]["delegated_agents"]
            if isinstance(agent["agent"], str)
        }),
        "files": {
            name: sha256(output / name) for name in ("silver-records.jsonl", "evidence.jsonl")
        },
    })
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_silver_records(directory: Path) -> list[Record]:
    """Load hash-checked proxy targets; never imply human review or remote consent."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / "silver-records.jsonl"
    if (manifest.get("status") != "complete" or manifest.get("mode") != "automatic_silver_not_gold"
            or manifest["files"]["silver-records.jsonl"] != sha256(path)):
        raise ValueError("Silver dataset is incomplete or changed")
    records = load_records(path)
    if (len(records) != manifest["silver_record_count"]
            or any(record.review_status != "unreviewed" or record.reviewer
                   or "automatic-silver" not in record.tags for record in records)):
        raise ValueError("Silver records must retain explicit unreviewed provenance")
    return records
