"""Filter a frozen candidate bundle down to specific benchmark sources.

Used to label a subset of tracks under an alternate taxonomy policy (for
example coding-v1 instead of copilot-coding-v2 for swe-bench) without
re-labeling an unrelated, unchanged track and without inventing a cross-plan
import mechanism. Preserves exact row content and provenance for every
retained row; never edits, reorders beyond source filtering, or invents data.
"""

from __future__ import annotations

import json
from pathlib import Path

from routing_lab.benchmark_data import jsonl
from routing_lab.evaluation import sha256


def filter_bundle_by_source(bundle: Path, sources: set[str], output: Path) -> dict:
    """Keep only candidate rows whose source is in ``sources``, verbatim."""
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete_unreviewed_probes":
        raise ValueError("Candidate bundle is incomplete")
    candidates_path = bundle / "candidates.jsonl"
    if sha256(candidates_path) != manifest["files"]["candidates.jsonl"]["sha256"]:
        raise ValueError("Candidate bundle content changed")
    if not sources:
        raise ValueError("Provide at least one source to keep")
    rows = jsonl(candidates_path)
    kept = [row for row in rows if row["source"] in sources]
    if not kept:
        raise ValueError("No candidate rows matched the requested sources")
    output.mkdir(parents=True, exist_ok=False)
    out_candidates = output / "candidates.jsonl"
    with out_candidates.open("x", encoding="utf-8") as stream:
        for row in kept:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    out_manifest = {
        "status": "complete_unreviewed_probes",
        "revisions": manifest["revisions"],
        "filtered_from": {"bundle": str(bundle), "manifest_sha256": sha256(bundle / "manifest.json")},
        "sources": sorted(sources), "total_rows": len(kept),
        "files": {"candidates.jsonl": {"sha256": sha256(out_candidates)}},
    }
    (output / "manifest.json").write_text(json.dumps(out_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_manifest


def cap_oversized_queries_for_transport(rows: list[dict], *, max_query_chars: int) -> tuple[list[dict], list[dict]]:
    """Client-side-truncate only rows whose ``query`` exceeds a transport limit.

    Used when a managed endpoint's gateway hard-rejects oversized request bodies
    (observed: HTTP 424, independent of and well below the model's own much
    smaller token budget) before the request ever reaches the model container.
    Every other field is left untouched; only rows over the limit are changed,
    and each change is logged with its original/kept lengths for an honest
    benchmark report. Does not mutate the canonical dataset file on disk.
    """
    if max_query_chars < 1:
        raise ValueError("max_query_chars must be positive")
    adjusted = []
    log = []
    for row in rows:
        query = row["query"]
        if len(query) <= max_query_chars:
            adjusted.append(row)
            continue
        capped = dict(row)
        capped["query"] = query[:max_query_chars]
        adjusted.append(capped)
        log.append({
            "id": row["id"], "original_query_chars": len(query), "kept_query_chars": max_query_chars,
        })
    return adjusted, log
