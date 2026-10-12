"""Measure routing decisions and persist auditable, non-overwriting run artifacts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from routing_lab.backends import Backend, Decision
from routing_lab.data import Record, records_taxonomy
from routing_lab.taxonomy import abstention_routes, taxonomy_routes


def summarize(records: list[Record], decisions: list[Decision],
              latencies_ms: list[float]) -> dict:
    """Compute strict-label and acceptable-route metrics separately."""
    if not records or not (len(records) == len(decisions) == len(latencies_ms)):
        raise ValueError("Need matching nonempty records, decisions, and timings")
    taxonomy_id = records_taxonomy(records)
    routes = taxonomy_routes(taxonomy_id)
    abstentions = abstention_routes(taxonomy_id)
    if any(d.taxonomy_id != taxonomy_id for d in decisions):
        raise ValueError("Decision taxonomy must match record taxonomy")
    expected = [r.expected_route for r in records]
    expected_counts = Counter(expected)
    predicted = [d.route for d in decisions]
    accepted = np.array([d.route in r.acceptable_routes
                         for r, d in zip(records, decisions, strict=True)])
    correct = np.array([a == b for a, b in zip(expected, predicted, strict=True)])
    report = classification_report(
        expected, predicted, labels=list(routes), output_dict=True, zero_division=0
    )
    fallback_expected = np.array([r == "fallback" for r in expected])
    result = {
        "count": len(records),
        "taxonomy_id": taxonomy_id,
        "strict_accuracy": float(correct.mean()),
        "acceptable_accuracy": float(accepted.mean()),
        "macro_f1": report["macro avg"]["f1-score"],
        "macro_f1_label_basis": (
            "All taxonomy routes in route_order, including absent expected-label classes; "
            "zero_division=0. Compare only with the same taxonomy and label coverage."
        ),
        "per_route": {route: report[route] for route in routes},
        "confusion_matrix": confusion_matrix(expected, predicted, labels=list(routes)).tolist(),
        "route_order": list(routes),
        "expected_route_counts": {route: expected_counts[route] for route in routes},
        "observed_expected_routes": [route for route in routes if expected_counts[route] > 0],
        "absent_expected_routes": [route for route in routes if expected_counts[route] == 0],
        "fallback_rate": predicted.count("fallback") / len(predicted) if "fallback" in routes else None,
        "fallback_recall": (
            float(np.mean(np.array(predicted)[fallback_expected] == "fallback"))
            if fallback_expected.any() else None
        ),
        "abstention_routes": list(abstentions),
        "abstention_rate": sum(route in abstentions for route in predicted) / len(predicted),
        "abstention_recall": (
            float(np.mean([p in abstentions for e, p in zip(expected, predicted, strict=True)
                           if e in abstentions]))
            if any(e in abstentions for e in expected) else None
        ),
        "latency_ms": {
            "first_request": latencies_ms[0],
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
        },
        "serial_requests_per_second": (
            1000 * len(records) / sum(latencies_ms) if sum(latencies_ms) else None
        ),
        "input_tokens": sum(d.input_tokens or 0 for d in decisions),
        "output_tokens": sum(d.output_tokens or 0 for d in decisions),
        "token_accounting_complete": all(d.input_tokens is not None for d in decisions),
        "truncation_rate": (
            sum(d.state_tokens_dropped > 0 for d in decisions) / len(decisions)
            if all(d.state_tokens_dropped is not None for d in decisions) else None
        ),
        "slices": {
            tag: {
                "count": sum(tag in r.tags for r in records),
                "acceptable_accuracy": float(np.mean([
                    ok for r, ok in zip(records, accepted, strict=True) if tag in r.tags
                ])),
            }
            for tag in sorted({tag for r in records for tag in r.tags})
        },
    }
    clusters: dict[str, list[bool]] = {}
    for record, ok in zip(records, accepted, strict=True):
        clusters.setdefault(record.repository, []).append(bool(ok))
    if len(clusters) >= 2:
        rng = random.Random(42)
        values = list(clusters.values())
        samples = []
        for _ in range(1000):
            sample = [ok for block in rng.choices(values, k=len(values)) for ok in block]
            samples.append(sum(sample) / len(sample))
        result["acceptable_accuracy_repository_bootstrap_95ci"] = (
            np.percentile(samples, [2.5, 97.5]).tolist()
        )
    if all(d.probabilities is not None for d in decisions):
        probs = np.array([[d.probabilities[r] for r in routes] for d in decisions])
        confidence = probs.max(axis=1)
        targets = np.array([[int(r.expected_route == label) for label in routes]
                            for r in records])
        bins, ece = [], 0.0
        for i in range(10):
            mask = (confidence >= i / 10) & (
                confidence <= 1 if i == 9 else confidence < (i + 1) / 10
            )
            if mask.any():
                acc, conf = float(correct[mask].mean()), float(confidence[mask].mean())
                ece += float(mask.mean()) * abs(acc - conf)
                bins.append({"lower": i / 10, "upper": (i + 1) / 10,
                             "count": int(mask.sum()), "accuracy": acc, "confidence": conf})
        result["probability_metrics"] = {
            "label_basis": "single expected_route, not acceptable-route set",
            "multiclass_brier": float(np.mean(np.sum((probs - targets) ** 2, axis=1))),
            "ece_10_bins": ece,
            "reliability_bins": bins,
            "top2_acceptable_accuracy": float(np.mean([
                any(routes[j] in record.acceptable_routes for j in row.argsort()[-2:])
                for record, row in zip(records, probs, strict=True)
            ])),
            "risk_coverage": [],
        }
        for threshold in [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
            mask = (confidence >= threshold) & ~np.isin(predicted, abstentions)
            result["probability_metrics"]["risk_coverage"].append({
                "threshold": threshold, "coverage": float(mask.mean()),
                "selective_error": float(1 - accepted[mask].mean()) if mask.any() else None,
            })
    return result


def sha256(path: Path) -> str:
    """Hash exact bytes for reproducible provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_experiment(
    backend: Backend, records: list[Record], output: Path, *,
    probe: bool = False, source_paths: dict[str, Path] | None = None,
    initialization_seconds: float | None = None,
) -> dict:
    """Persist partial predictions on failure, but mark complete only on success.

    Raw requests, code, and credentials are not copied into output artifacts.
    Original data must still be retained securely to reproduce their hashes.
    """
    if not records:
        raise ValueError("Cannot evaluate an empty dataset")
    taxonomy_id = records_taxonomy(records)
    backend_metadata = backend.metadata()
    if backend_metadata.get("taxonomy_id", "coding-v1") != taxonomy_id:
        raise ValueError("Backend taxonomy must match record taxonomy")
    if not probe and any(r.review_status != "reviewed" for r in records):
        raise ValueError("Unreviewed labels are probe-only, not a benchmark")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "incomplete",
        "taxonomy_id": taxonomy_id,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "probe_unreviewed_not_a_benchmark" if probe else "reviewed_benchmark",
        "backend": backend_metadata,
        "initialization_seconds": initialization_seconds,
        "dataset_digest": hashlib.sha256(
            "\n".join(r.model_dump_json() for r in records).encode()
        ).hexdigest(),
        "files": {name: sha256(path) for name, path in (source_paths or {}).items()},
        "python": sys.version, "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "source_hashes": {
            path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
        },
        "packages": {
            name: importlib.metadata.version(name) for name in
            ("pydantic", "numpy", "scikit-learn")
        },
        "timing_scope": "serial predict wall time, includes tokenization and first request; excludes load",
    }
    for optional in ("torch", "transformers", "openai", "azure-identity", "requests"):
        try:
            manifest["packages"][optional] = importlib.metadata.version(optional)
        except importlib.metadata.PackageNotFoundError:
            continue
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    decisions, latencies = [], []
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for record in records:
            start = time.perf_counter()
            decision = backend.predict(record)
            if decision.taxonomy_id != taxonomy_id:
                raise ValueError("Decision taxonomy must match record taxonomy")
            latency = (time.perf_counter() - start) * 1000
            decisions.append(decision)
            latencies.append(latency)
            stream.write(json.dumps({
                "id": record.id, "taxonomy_id": taxonomy_id, "expected_route": record.expected_route,
                "acceptable_routes": record.acceptable_routes, "tags": record.tags,
                "decision": decision.model_dump(), "latency_ms": latency,
            }) + "\n")
            stream.flush()
    summary = summarize(records, decisions, latencies)
    summary["mode"] = manifest["mode"]
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest["status"] = "complete"
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return summary
