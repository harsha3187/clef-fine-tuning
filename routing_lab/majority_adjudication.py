"""Third blind pass that majority-votes disagreements from a completed batch run.

Reuses the existing batched wire format, journal/checkpoint discipline and
file-level audit helpers from routing_lab.batch_annotation. Never overwrites
the source run; adjudication output is additive to the original consensus
export. A 2-of-3 agreement is required to resolve a row; a true three-way
split or a double abstention stays unresolved for human review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from routing_lab.annotation import AnnotationItem, Judge, digest, load_plan
from routing_lab.batch_annotation import (
    _append, _atomic_bytes, _atomic_json, _lock, _positive_int, _read_json, _read_jsonl,
    _response_metadata, _result_rows as _source_result_rows, _usage, _validate_response_metadata,
    _encoded_jsonl, _load_run as _load_source_run,
)
from routing_lab.batch_format import Batch, batch_payload, batch_prompt, batch_schema, build_batches, validate_batch
from routing_lab.data import Contract, Record, load_records, write_records
from routing_lab.evaluation import sha256

MAJORITY_POLICY_VERSION = "majority-vote-third-pass-v1"
_CODE_FILES = (
    "majority_adjudication.py", "batch_format.py", "annotation.py", "annotation_judge.py",
    "data.py", "taxonomy.py", "backends.py",
)


def _code_hashes() -> dict[str, str]:
    return {name: sha256(Path(__file__).with_name(name)) for name in _CODE_FILES}


def _majority(ann0: dict, ann1: dict, ann2: dict) -> dict:
    """At least 2 of 3 independent passes must agree; a true split stays unresolved."""
    votes = (ann0, ann1, ann2)
    labeled_routes = [vote["route"] for vote in votes if vote["verdict"] == "label"]
    abstentions = sum(1 for vote in votes if vote["verdict"] != "label")
    route_counts = Counter(labeled_routes)
    if route_counts:
        route, count = route_counts.most_common(1)[0]
        if count >= 2:
            acceptable = sorted({
                candidate for vote in votes if vote["verdict"] == "label" and vote["route"] == route
                for candidate in vote["acceptable_routes"]
            })
            return {"status": "majority_consensus", "route": route, "acceptable_routes": acceptable}
    if abstentions >= 2:
        return {"status": "majority_abstain", "route": None, "acceptable_routes": []}
    return {"status": "no_majority", "route": None, "acceptable_routes": []}


def _vote_rows(items: list[AnnotationItem], votes: dict) -> list[dict]:
    return [votes[item.id] for item in items if item.id in votes]


def _result_rows(items: list[AnnotationItem], votes: dict, source_results_by_id: dict) -> list[dict]:
    rows = []
    for item in items:
        if item.id not in votes:
            continue
        ann0, ann1 = source_results_by_id[item.id]["assessments"]
        ann2 = votes[item.id]["annotation"]
        decision = _majority(ann0, ann1, ann2)
        rows.append({
            "id": item.id, "input_sha256": item.input_sha256, "taxonomy_id": item.taxonomy_id,
            **decision, "assessments": [ann0, ann1, ann2], "review_status": "unreviewed",
        })
    return rows


def _majority_identity(plan: Path, items: list[AnnotationItem], catalogs: dict[str, dict[str, Contract]],
                       metadata: dict, batches: list[Batch], source_run_manifest_sha256: str,
                       source_run_identity_sha256: str, batch_size: int, max_input_chars: int) -> dict:
    return {
        "policy_version": MAJORITY_POLICY_VERSION,
        "plan_sha256": sha256(plan / "manifest.json"),
        "judge": metadata, "implementation_sha256": _code_hashes(),
        "batch_size": batch_size, "max_input_chars": max_input_chars,
        "source_run_manifest_sha256": source_run_manifest_sha256,
        "source_run_identity_sha256": source_run_identity_sha256,
        "selected_ids_sha256": digest([item.id for item in items]),
        "selected_count": len(items),
        "prompt_sha256": {taxonomy: digest(batch_prompt(contracts, 0)) for taxonomy, contracts in catalogs.items()},
        "batches": [{
            "id": batch.id, "taxonomy_id": batch.taxonomy_id, "ids": [item.id for item in batch.items],
            "schema_sha256": digest(batch_schema(batch)), "payload_sha256": digest(batch_payload(batch, 0)),
        } for batch in batches],
    }


def _replay_majority(events: list[dict], batches: list[Batch]) -> tuple[dict, set]:
    votes: dict[str, dict] = {}
    batch_map = {batch.id: batch for batch in batches}
    starts: dict[int, dict] = {}
    finished: set[int] = set()
    completed: set[str] = set()
    models: set[str] = set()
    previous = None
    for event in events:
        unhashed = {key: value for key, value in event.items() if key != "event_sha256"}
        if event.get("previous_sha256") != previous or event.get("event_sha256") != digest(unhashed):
            raise ValueError("Adjudication journal hash chain changed")
        previous = event["event_sha256"]
        number, batch_id = event["request_number"], event["batch_id"]
        if type(number) is not int or batch_id not in batch_map:
            raise ValueError("Invalid adjudication request identity")
        if event["event"] == "started":
            if number != len(starts) + 1 or batch_id in completed:
                raise ValueError("Invalid physical-request sequence or duplicate successful batch")
            starts[number] = event
        elif event["event"] in {"completed", "failed"}:
            if number not in starts or number in finished:
                raise ValueError("Missing request start or duplicate terminal response")
            start = starts[number]
            if event["batch_id"] != start["batch_id"]:
                raise ValueError("Adjudication response differs from its physical request")
            finished.add(number)
            _usage(event)
            if event["event"] == "failed":
                continue
            if batch_id in completed:
                raise ValueError("Duplicate completed batch")
            _validate_response_metadata(event)
            if models and event["model"] not in models:
                raise ValueError("Observed judge model version changed")
            models.add(event["model"])
            validated = validate_batch(event["payload"], batch_map[batch_id])
            for item in batch_map[batch_id].items:
                if item.id in votes:
                    raise ValueError("A row was assessed by multiple successful requests")
                votes[item.id] = {
                    "id": item.id, "input_sha256": item.input_sha256,
                    "annotation": validated[item.id].model_dump(),
                    "model": event["model"], "response_id": event["response_id"],
                    "origin": {
                        "kind": "third_pass", "request_number": number, "batch_id": batch_id,
                        "prompt_sha256": start["prompt_sha256"], "schema_sha256": start["schema_sha256"],
                        "payload_sha256": start["payload_sha256"],
                    },
                }
            completed.add(batch_id)
        else:
            raise ValueError("Unknown adjudication journal event")
    return votes, completed


def _summary(items: list[AnnotationItem], votes: dict, events: list[dict], batches: list[Batch],
            completed: set, source_results_by_id: dict) -> dict:
    attempted = sum(event["event"] == "started" for event in events)
    metered = [event for event in events if event["event"] in {"completed", "failed"} and _usage(event)]
    rows = _result_rows(items, votes, source_results_by_id)
    return {
        "requests_attempted": attempted,
        "new_batch_count": len(batches),
        "required_successful_requests": len(batches),
        "remaining_successful_requests": len(batches) - len(completed),
        "estimated_total_requests_without_new_failures": attempted + len(batches) - len(completed),
        "completed_batches": len(completed),
        "completed_rows": len(rows), "pending_rows": len(items) - len(rows),
        "result_status_counts": dict(Counter(row["status"] for row in rows)),
        "observed_judge_models": sorted({vote["model"] for vote in votes.values()}),
        "known_input_tokens": sum(event["input_tokens"] for event in metered),
        "known_output_tokens": sum(event["output_tokens"] for event in metered),
        "requests_without_valid_usage": attempted - len(metered),
    }


def _checkpoint(output: Path, manifest: dict, status: str, items: list[AnnotationItem], votes: dict,
               events: list[dict], batches: list[Batch], completed: set, source_results_by_id: dict) -> None:
    _atomic_bytes(output / "votes.jsonl", _encoded_jsonl(_vote_rows(items, votes)))
    _atomic_bytes(output / "results.jsonl", _encoded_jsonl(_result_rows(items, votes, source_results_by_id)))
    manifest.update(
        **_summary(items, votes, events, batches, completed, source_results_by_id),
        status=status, event_count=len(events),
        events_sha256=sha256(output / "batch_events.jsonl"),
        journal_head_sha256=events[-1]["event_sha256"] if events else None,
        votes_sha256=sha256(output / "votes.jsonl"), results_sha256=sha256(output / "results.jsonl"),
    )
    _atomic_json(output / "manifest.json", manifest)


def _verify_checkpoint(output: Path, manifest: dict, items: list[AnnotationItem], events: list[dict],
                       batches: list[Batch], source_results_by_id: dict) -> tuple[dict, set]:
    count = manifest["event_count"]
    if type(count) is not int or count < 0 or count > len(events):
        raise ValueError("Adjudication journal was truncated")
    prefix = events[:count]
    if hashlib.sha256(_encoded_jsonl(prefix)).hexdigest() != manifest["events_sha256"]:
        raise ValueError("Checkpointed adjudication journal artifacts changed")
    if (prefix[-1]["event_sha256"] if prefix else None) != manifest["journal_head_sha256"]:
        raise ValueError("Checkpoint journal head changed")
    if len(events) != count and manifest["status"] != "in_progress":
        raise ValueError("Completed adjudication journal artifacts changed")
    votes, completed = _replay_majority(prefix, batches)
    current_votes, current_completed = (
        _replay_majority(events, batches) if len(events) > count else (votes, completed)
    )
    for name, expected in (
        ("votes", _vote_rows(items, votes)), ("results", _result_rows(items, votes, source_results_by_id)),
    ):
        path = output / f"{name}.jsonl"
        actual = _read_jsonl(path)
        matches_checkpoint = sha256(path) == manifest[f"{name}_sha256"] and actual == expected
        current = (_vote_rows(items, current_votes) if name == "votes"
                   else _result_rows(items, current_votes, source_results_by_id))
        interrupted_checkpoint = (
            manifest["status"] == "in_progress" and len(events) > count and actual == current
        )
        if not matches_checkpoint and not interrupted_checkpoint:
            raise ValueError("Stored votes/results artifacts changed or differ from the audit journal")
    summary = _summary(items, votes, prefix, batches, completed, source_results_by_id)
    if any(manifest.get(key) != value for key, value in summary.items()):
        raise ValueError("Progress counters differ from the physical audit journal")
    if manifest["status"] == "complete_selected_rows" and len(votes) != len(items):
        raise ValueError("Adjudication run falsely claims completion")
    return current_votes, current_completed


def _load_majority_run(plan: Path, source_run: Path, output: Path, metadata: dict | None = None) -> tuple:
    manifest, identity = _read_json(output / "manifest.json"), _read_json(output / "identity.json")
    if sha256(output / "identity.json") != manifest["identity_sha256"]:
        raise ValueError("Frozen adjudication run identity changed")
    if manifest["policy_version"] != identity["policy_version"] or manifest["plan_sha256"] != identity["plan_sha256"]:
        raise ValueError("Manifest differs from the frozen adjudication identity")
    if manifest["status"] not in {"in_progress", "failed", "request_budget_reached", "complete_selected_rows"}:
        raise ValueError("Unsupported adjudication checkpoint status")
    approvals = manifest["budget_approvals"]
    if not isinstance(approvals, list) or not approvals or any(
        not _positive_int(approval.get("request_limit")) or not isinstance(approval.get("utc"), str)
        for approval in approvals
    ) or approvals[-1]["request_limit"] != manifest["approved_request_limit"]:
        raise ValueError("Invalid cumulative request budget approval history")
    if metadata is not None and metadata != identity["judge"]:
        raise ValueError("Resume requires the identical judge configuration")
    if identity["policy_version"] != MAJORITY_POLICY_VERSION:
        raise ValueError("Unsupported majority-vote adjudication policy")
    if sha256(source_run / "manifest.json") != identity["source_run_manifest_sha256"] or sha256(
        source_run / "identity.json"
    ) != identity["source_run_identity_sha256"]:
        raise ValueError("Source run changed since this adjudication pass was created")

    plan_manifest, all_items, catalogs = load_plan(plan)
    for name, expected in manifest["snapshot_sha256"].items():
        if Path(name).name != name or sha256(output / "snapshot" / name) != expected:
            raise ValueError("Frozen input/catalog snapshot changed")
    expected_snapshots = {
        "manifest.json": sha256(plan / "manifest.json"),
        **{name: expected for name, expected in plan_manifest["files"].items()},
    }
    if manifest["snapshot_sha256"] != expected_snapshots:
        raise ValueError("Frozen snapshot inventory differs from the annotation plan")

    source_loaded = _load_source_run(plan, source_run, identity["judge"])
    src_manifest, _, _, src_items, _, _, _, _, src_votes, _ = source_loaded
    if src_manifest["status"] != "complete_selected_rows":
        raise ValueError("Source run is no longer complete")
    source_results = _source_result_rows(src_items, src_votes)
    source_results_by_id = {row["id"]: row for row in source_results}
    recomputed_disagreement_ids = sorted(
        row["id"] for row in source_results if row["status"] == "needs_adjudication"
    )
    if digest(recomputed_disagreement_ids) != identity["selected_ids_sha256"]:
        raise ValueError("Source run's disagreement set changed since this adjudication pass was created")

    by_id = {item.id: item for item in src_items}
    items = [by_id[row_id] for row_id in recomputed_disagreement_ids]
    batches = build_batches(items, catalogs, batch_size=identity["batch_size"],
                            max_input_chars=identity["max_input_chars"])
    expected_identity = _majority_identity(
        plan, items, catalogs, identity["judge"], batches,
        identity["source_run_manifest_sha256"], identity["source_run_identity_sha256"],
        identity["batch_size"], identity["max_input_chars"],
    )
    if identity != expected_identity:
        raise ValueError("Resume/export requires identical plan, selection, prompts, schemas and implementation")
    if _read_jsonl(output / "selected.jsonl") != [item.model_dump() for item in items]:
        raise ValueError("Frozen selected input inventory changed")
    if sha256(output / "selected.jsonl") != manifest["selected_sha256"]:
        raise ValueError("Frozen selected input artifact changed")

    events = _read_jsonl(output / "batch_events.jsonl")
    votes, completed = _verify_checkpoint(output, manifest, items, events, batches, source_results_by_id)
    if not _positive_int(manifest["approved_request_limit"]) or manifest["approved_request_limit"] < sum(
        event["event"] == "started" for event in events
    ):
        raise ValueError("Recorded request budget is below attempted requests")
    return (manifest, identity, plan_manifest, items, catalogs, batches, events, votes, completed,
            source_results_by_id)


def run_majority_vote(plan: Path, judge: Judge, source_run: Path, output: Path, *, max_requests: int,
                      batch_size: int = 100, max_input_chars: int = 128000,
                      allow_remote: bool = False, resume: bool = False) -> dict:
    """One additional blind pass over disagreement rows; never touches the source run."""
    if not allow_remote:
        raise PermissionError("Remote annotation requires explicit --allow-remote consent")
    if not _positive_int(max_requests):
        raise ValueError("An explicit positive physical-request budget is required")
    if type(batch_size) is not int or not 1 <= batch_size <= 100 or type(max_input_chars) is not int or not (
        1024 <= max_input_chars <= 1_000_000
    ):
        raise ValueError("Invalid batch size or input character bound")
    from azure.core.exceptions import AzureError
    from openai import APIError

    from routing_lab.annotation_judge import JudgeError

    metadata = judge.metadata()
    if metadata.get("max_retries", 0) != 0 or metadata.get("follow_redirects", False) is not False:
        raise ValueError("Physical-request accounting requires disabled SDK retries and redirects")
    if not resume:
        source_loaded = _load_source_run(plan, source_run, metadata)
        src_manifest, _, plan_manifest, all_items, catalogs = source_loaded[:5]
        src_votes = source_loaded[8]
        if src_manifest["status"] != "complete_selected_rows":
            raise ValueError("Source run must be fully complete before majority-vote adjudication")
        source_results = _source_result_rows(all_items, src_votes)
        disagreement_ids = sorted(row["id"] for row in source_results if row["status"] == "needs_adjudication")
        if not disagreement_ids:
            raise ValueError("Source run has no disagreements to adjudicate")
        by_id = {item.id: item for item in all_items}
        items = [by_id[row_id] for row_id in disagreement_ids]
        batches = build_batches(items, catalogs, batch_size=batch_size, max_input_chars=max_input_chars)
        identity = _majority_identity(
            plan, items, catalogs, metadata, batches,
            sha256(source_run / "manifest.json"), sha256(source_run / "identity.json"),
            batch_size, max_input_chars,
        )
        output.mkdir(parents=True, exist_ok=False)
    with _lock(output):
        if not resume:
            snapshot = output / "snapshot"
            snapshot.mkdir()
            for name in ("manifest.json", *plan_manifest["files"]):
                shutil.copyfile(plan / name, snapshot / name)
            _atomic_json(output / "identity.json", identity)
            _atomic_bytes(output / "selected.jsonl", _encoded_jsonl([item.model_dump() for item in items]))
            _atomic_bytes(output / "batch_events.jsonl", b"")
            events, completed = [], set()
            manifest = {
                "policy_version": MAJORITY_POLICY_VERSION, "plan_sha256": identity["plan_sha256"],
                "identity_sha256": sha256(output / "identity.json"),
                "selected_sha256": sha256(output / "selected.jsonl"),
                "snapshot_sha256": {path.name: sha256(path) for path in snapshot.iterdir()},
                "approved_request_limit": max_requests, "budget_approvals": [{
                    "utc": datetime.now(timezone.utc).isoformat(), "request_limit": max_requests,
                }],
            }
            votes: dict[str, dict] = {}
            source_results_by_id = {row["id"]: row for row in source_results}
            _checkpoint(output, manifest, "in_progress", items, votes, events, batches, completed,
                       source_results_by_id)
        loaded = _load_majority_run(plan, source_run, output, metadata)
        manifest, identity, _, items, catalogs, batches, events, votes, completed, source_results_by_id = loaded
        if identity["batch_size"] != batch_size or identity["max_input_chars"] != max_input_chars:
            raise ValueError("Resume requires identical batching limits")
        spent = sum(event["event"] == "started" for event in events)
        if max_requests < spent:
            raise ValueError("Request budget is below cumulative physical requests already attempted")
        if not manifest["budget_approvals"] or max_requests != manifest["approved_request_limit"]:
            manifest["budget_approvals"].append({
                "utc": datetime.now(timezone.utc).isoformat(), "request_limit": max_requests,
            })
        manifest["approved_request_limit"] = max_requests
        _checkpoint(output, manifest, "in_progress", items, votes, events, batches, completed,
                   source_results_by_id)
        for batch in batches:
            if batch.id in completed:
                continue
            if spent >= max_requests:
                _checkpoint(output, manifest, "request_budget_reached", items, votes, events, batches,
                           completed, source_results_by_id)
                return manifest
            spent += 1
            prompt = batch_prompt(catalogs[batch.taxonomy_id], 0)
            payload, schema = batch_payload(batch, 0), batch_schema(batch)
            base = {"request_number": spent, "batch_id": batch.id}
            _append(output / "batch_events.jsonl", events, {
                **base, "event": "started", "prompt_sha256": digest(prompt),
                "schema_sha256": digest(schema), "payload_sha256": digest(payload),
            })
            _checkpoint(output, manifest, "in_progress", items, votes, events, batches, completed,
                       source_results_by_id)
            response = None
            accepted = False
            try:
                response = judge.annotate(system_prompt=prompt, input_payload=payload, schema=schema)
                response_metadata = _response_metadata(response)
                annotations = validate_batch(response.payload, batch)
                models = {vote["model"] for vote in votes.values()}
                if models and response.model not in models:
                    raise ValueError("Observed judge model version changed during the adjudication run")
                _append(output / "batch_events.jsonl", events, {
                    **base, **response_metadata, "event": "completed", "payload": response.payload,
                })
                start = events[-2]
                for item in batch.items:
                    votes[item.id] = {
                        "id": item.id, "input_sha256": item.input_sha256,
                        "annotation": annotations[item.id].model_dump(),
                        "model": response.model, "response_id": response.response_id,
                        "origin": {
                            "kind": "third_pass", "request_number": spent, "batch_id": batch.id,
                            "prompt_sha256": start["prompt_sha256"], "schema_sha256": start["schema_sha256"],
                            "payload_sha256": start["payload_sha256"],
                        },
                    }
                completed.add(batch.id)
                accepted = True
            except (ValueError, APIError, AzureError, JudgeError) as error:
                failure = {**base, "event": "failed", "error_type": type(error).__name__}
                if response is not None:
                    failure.update(_response_metadata(response))
                    failure["payload"] = response.payload
                _append(output / "batch_events.jsonl", events, failure)
                raise
            finally:
                _checkpoint(output, manifest, "in_progress" if accepted else "failed", items, votes, events,
                           batches, completed, source_results_by_id)
        _checkpoint(output, manifest, "complete_selected_rows", items, votes, events, batches, completed,
                   source_results_by_id)
        return manifest


def export_majority(plan: Path, source_run: Path, majority_run: Path, output: Path, *,
                    allow_partial: bool = False) -> dict:
    """Additive export: union with the original consensus export's per-taxonomy files."""
    with _lock(majority_run):
        loaded = _load_majority_run(plan, source_run, majority_run)
        (manifest, identity, plan_manifest, items, _, _, _, votes, _,
         source_results_by_id) = loaded
        if manifest["status"] not in {"complete_selected_rows", "request_budget_reached", "failed"}:
            raise ValueError("Interrupted adjudication run must be resumed/checkpointed before export")
        rows = _result_rows(items, votes, source_results_by_id)
        if not allow_partial and len(rows) != len(items):
            raise ValueError("Not every disagreement row has a third-pass vote; use explicit --allow-partial")
        if not rows:
            raise ValueError("No third-pass assessments to export")
        run_hash = sha256(majority_run / "manifest.json")
        item_by_id = {item.id: item for item in items}
        records: dict[str, list[Record]] = {taxonomy: [] for taxonomy in {item.taxonomy_id for item in items}}
        for row in rows:
            if row["status"] != "majority_consensus":
                continue
            item = item_by_id[row["id"]]
            records[item.taxonomy_id].append(Record.model_validate({
                "id": item.id, "taxonomy_id": item.taxonomy_id, "group_id": item.group_id,
                "repository": item.repository, "query": item.query, "context": item.context,
                "expected_route": row["route"], "acceptable_routes": row["acceptable_routes"],
                "tags": [
                    item.source, item.category, item.native_split, MAJORITY_POLICY_VERSION,
                    f"source:{item.source}", f"native_split:{item.native_split}", f"group:{item.group_id}",
                    "development_exposed" if item.development_exposed else "holdout_candidate_not_frozen",
                    "majority_vote_third_pass",
                ],
                "provenance": json.dumps({
                    "label_origin": "AI 3-pass majority vote (>=2 of 3 independent passes agree); not human gold",
                    "plan_sha256": identity["plan_sha256"],
                    "source_run_manifest_sha256": identity["source_run_manifest_sha256"],
                    "source_run_identity_sha256": identity["source_run_identity_sha256"],
                    "majority_run_manifest_sha256": run_hash,
                    "majority_run_identity_sha256": manifest["identity_sha256"],
                    "input_sha256": item.input_sha256, "native_id": item.native_id, "source": item.source,
                    "native_split": item.native_split, "development_exposed": item.development_exposed,
                    "assessments": row["assessments"],
                }, sort_keys=True),
                "review_status": "unreviewed", "reviewer": "",
            }))
        result_by_id = {row["id"]: row for row in rows}
        audit_rows = []
        for item in items:
            row = result_by_id.get(item.id)
            if row is not None and row["status"] == "majority_consensus":
                continue
            audit_rows.append({
                "id": item.id, "taxonomy_id": item.taxonomy_id, "input_sha256": item.input_sha256,
                "status": row["status"] if row is not None else "pending_third_pass",
                "review_status": "unreviewed", "reviewer": "",
                "assessments": row["assessments"] if row is not None else source_results_by_id[item.id]["assessments"],
            })
        output.mkdir(parents=True, exist_ok=False)
        exported = {
            "status": "incomplete", "mode": "majority_vote_third_pass", "allow_partial": allow_partial,
            "partial": len(rows) != len(items),
            "resolved_count": sum(1 for row in rows if row["status"] == "majority_consensus"),
            "candidate_count": len(items), "pending_count": len(items) - len(rows),
            "still_unresolved_count": sum(1 for row in rows if row["status"] != "majority_consensus"),
            "source_run_manifest_sha256": identity["source_run_manifest_sha256"],
            "majority_run_manifest_sha256": run_hash, "majority_run_identity_sha256": manifest["identity_sha256"],
            "plan_sha256": identity["plan_sha256"], "files": {}, "catalogs": {},
            "note": (
                "Additive to the original two-pass consensus export. Union this directory's "
                "per-taxonomy files with the original export's files for the complete resolved dataset."
            ),
        }
        _atomic_json(output / "manifest.json", exported)
        for taxonomy, taxonomy_rows in records.items():
            path = output / f"{taxonomy}.jsonl"
            write_records(path, taxonomy_rows)
            exported["files"][path.name] = {
                "sha256": sha256(path), "count": len(taxonomy_rows),
                "route_counts": dict(Counter(record.expected_route for record in taxonomy_rows)),
            }
            catalog = output / f"{taxonomy}.contracts.json"
            shutil.copyfile(majority_run / "snapshot" / plan_manifest["catalogs"][taxonomy], catalog)
            exported["catalogs"][taxonomy] = {"file": catalog.name, "sha256": sha256(catalog)}
        _atomic_bytes(output / "adjudication.jsonl", _encoded_jsonl(audit_rows))
        exported["files"]["adjudication.jsonl"] = {
            "sha256": sha256(output / "adjudication.jsonl"), "count": len(audit_rows),
        }
        exported["status"] = "complete"
        _atomic_json(output / "manifest.json", exported)
        return exported


def finalize_dataset(consensus_export: Path, majority_export: Path, output: Path) -> dict:
    """Merge the two-pass consensus export with the resolved majority-vote rows.

    Rows that never reached 2-of-3 agreement across three independent passes
    are intentionally excluded from the merged per-taxonomy dataset files, not
    silently dropped: their full audit trail is copied verbatim into
    excluded_unresolved.jsonl for later human review.
    """
    consensus_manifest = _read_json(consensus_export / "manifest.json")
    majority_manifest = _read_json(majority_export / "manifest.json")
    if consensus_manifest.get("status") != "complete" or majority_manifest.get("status") != "complete":
        raise ValueError("Both exports must be complete before finalizing")
    if consensus_manifest["run_manifest_sha256"] != majority_manifest["source_run_manifest_sha256"]:
        raise ValueError("Majority export does not derive from the same source run as the consensus export")
    taxonomies = sorted(set(consensus_manifest["catalogs"]) | set(majority_manifest["catalogs"]))
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict] = {}
    catalogs: dict[str, dict] = {}
    total = 0
    for taxonomy in taxonomies:
        combined: list[Record] = []
        combined_ids: set[str] = set()
        catalog_hash, chosen_catalog_path = None, None
        for export, manifest in ((consensus_export, consensus_manifest), (majority_export, majority_manifest)):
            info = manifest["files"].get(f"{taxonomy}.jsonl")
            if info is None:
                continue
            path = export / f"{taxonomy}.jsonl"
            if sha256(path) != info["sha256"]:
                raise ValueError(f"{export.name}/{taxonomy}.jsonl changed since it was exported")
            if info["count"] > 0:
                records = load_records(path, taxonomy_id=taxonomy)
                overlap = combined_ids & {record.id for record in records}
                if overlap:
                    raise ValueError(f"Duplicate row IDs between exports for {taxonomy}: {sorted(overlap)[:5]}")
                combined_ids.update(record.id for record in records)
                combined.extend(records)
            catalog_info = manifest["catalogs"][taxonomy]
            catalog_path = export / catalog_info["file"]
            if sha256(catalog_path) != catalog_info["sha256"]:
                raise ValueError(f"{export.name} catalog for {taxonomy} changed since it was exported")
            if catalog_hash is None:
                catalog_hash, chosen_catalog_path = catalog_info["sha256"], catalog_path
            elif catalog_hash != catalog_info["sha256"]:
                raise ValueError(f"Consensus and majority exports disagree on the {taxonomy} contract catalog")
        if not combined:
            continue
        out_path = output / f"{taxonomy}.jsonl"
        write_records(out_path, combined)
        files[out_path.name] = {
            "sha256": sha256(out_path), "count": len(combined),
            "route_counts": dict(Counter(record.expected_route for record in combined)),
        }
        total += len(combined)
        catalog_out = output / f"{taxonomy}.contracts.json"
        shutil.copyfile(chosen_catalog_path, catalog_out)
        catalogs[taxonomy] = {"file": catalog_out.name, "sha256": sha256(catalog_out)}
    if not files:
        raise ValueError("Nothing to finalize: both exports are empty")
    excluded_source = majority_export / "adjudication.jsonl"
    excluded_out = output / "excluded_unresolved.jsonl"
    shutil.copyfile(excluded_source, excluded_out)
    excluded_rows = _read_jsonl(excluded_out)
    manifest_out = {
        "status": "complete", "mode": "finalized_labeled_dataset",
        "consensus_export_manifest_sha256": sha256(consensus_export / "manifest.json"),
        "majority_export_manifest_sha256": sha256(majority_export / "manifest.json"),
        "total_rows": total, "excluded_unresolved_count": len(excluded_rows),
        "files": files, "catalogs": catalogs,
        "excluded_file": {
            "name": excluded_out.name, "sha256": sha256(excluded_out), "count": len(excluded_rows),
            "note": (
                "Rows that never reached 2-of-3 agreement across three independent passes. "
                "Deliberately excluded from the labeled dataset above; kept here only for "
                "human review/adjudication, not as usable labels."
            ),
        },
        "note": (
            "This directory is the complete usable AI-labeled dataset: the original two-pass "
            "consensus plus rows resolved by the third-pass majority vote. Still-unresolved rows "
            "are excluded from the per-taxonomy files above and isolated in excluded_unresolved.jsonl."
        ),
    }
    _atomic_json(output / "manifest.json", manifest_out)
    return manifest_out


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--judge-config", type=Path, required=True)
    run.add_argument("--source-run", type=Path, required=True,
                     help="A completed batch_annotation run whose disagreements will be adjudicated")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=100)
    run.add_argument("--max-input-chars", type=int, default=128000)
    run.add_argument("--max-requests", type=int, required=True)
    run.add_argument("--allow-remote", action="store_true")
    run.add_argument("--resume", action="store_true")
    export = commands.add_parser("export")
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--source-run", type=Path, required=True)
    export.add_argument("--majority-run", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--allow-partial", action="store_true")
    finalize = commands.add_parser("finalize",
                                   help="Merge the consensus and majority exports, excluding unresolved rows")
    finalize.add_argument("--consensus-export", type=Path, required=True)
    finalize.add_argument("--majority-export", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    if args.command == "run":
        from routing_lab.annotation_judge import AzureResponsesJudge, load_judge_config
        with AzureResponsesJudge(load_judge_config(args.judge_config)) as judge:
            result = run_majority_vote(
                args.plan, judge, args.source_run, args.output, batch_size=args.batch_size,
                max_requests=args.max_requests, max_input_chars=args.max_input_chars,
                allow_remote=args.allow_remote, resume=args.resume,
            )
    elif args.command == "export":
        result = export_majority(args.plan, args.source_run, args.majority_run, args.output,
                                 allow_partial=args.allow_partial)
    else:
        result = finalize_dataset(args.consensus_export, args.majority_export, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))



if __name__ == "__main__":
    main()
