"""Known-answer metrics and durable experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from routing_lab import evaluation
from routing_lab.backends import ConstantBackend, Decision
from routing_lab.data import ROUTES, Record
from routing_lab.evaluation import run_experiment, summarize
from routing_lab.taxonomy import TAXONOMIES, TaxonomyId, taxonomy_routes


def probabilities(route: str, confidence: float = 1.0) -> dict[str, float]:
    return {key: confidence if key == route else (1 - confidence) / 5 for key in ROUTES}


@pytest.mark.parametrize(("offset", "accuracy", "brier", "ece"), [(0, 1, 0, 0), (1, 0, 2, 1)])
def test_perfect_and_completely_wrong_six_class_predictions(
    make_record: Callable[..., Record], offset: int, accuracy: float, brier: float, ece: float,
) -> None:
    records = [make_record(expected_route=route) for route in ROUTES]
    decisions = [
        Decision(route=ROUTES[(i + offset) % 6], probabilities=probabilities(ROUTES[(i + offset) % 6]))
        for i in range(6)
    ]
    summary = summarize(records, decisions, [10.0] * 6)
    assert summary["strict_accuracy"] == accuracy
    assert summary["acceptable_accuracy"] == accuracy
    assert summary["macro_f1"] == accuracy
    assert summary["fallback_rate"] == pytest.approx(1 / 6)
    assert summary["fallback_recall"] == accuracy
    assert summary["route_order"] == list(ROUTES)
    assert summary["absent_expected_routes"] == []
    assert summary["observed_expected_routes"] == list(ROUTES)
    assert summary["expected_route_counts"] == {route: 1 for route in ROUTES}
    assert summary["serial_requests_per_second"] == 100
    matrix = [[int(j == (i + offset) % 6) for j in range(6)] for i in range(6)]
    assert summary["confusion_matrix"] == matrix
    for route in ROUTES:
        assert summary["per_route"][route]["f1-score"] == accuracy
        assert summary["per_route"][route]["support"] == 1
    metrics = summary["probability_metrics"]
    assert metrics["multiclass_brier"] == brier
    assert metrics["ece_10_bins"] == ece
    for point in metrics["risk_coverage"]:
        assert point["coverage"] == pytest.approx(5 / 6)
        assert point["selective_error"] == 1 - accuracy


@pytest.mark.parametrize("taxonomy", TAXONOMIES)
@pytest.mark.parametrize("predict_absent", [False, True])
def test_absent_expected_classes_do_not_change_macro_f1_semantics(
    make_record: Callable[..., Record], taxonomy: TaxonomyId, predict_absent: bool,
) -> None:
    routes = taxonomy_routes(taxonomy)
    records = [
        make_record(taxonomy_id=taxonomy, expected_route=routes[0],
                    acceptable_routes=[routes[0], routes[1]])
        for _ in range(3)
    ]
    predicted = routes[1] if predict_absent else routes[0]
    decisions = [Decision(taxonomy_id=taxonomy, route=predicted) for _ in records]
    summary = summarize(records, decisions, [1.0] * len(records))
    assert summary["taxonomy_id"] == taxonomy
    assert summary["route_order"] == list(routes)
    assert summary["expected_route_counts"] == {
        route: 3 if route == routes[0] else 0 for route in routes
    }
    assert summary["observed_expected_routes"] == [routes[0]]
    assert summary["absent_expected_routes"] == list(routes[1:])
    assert summary["macro_f1"] == pytest.approx(0 if predict_absent else 1 / len(routes))
    assert summary["acceptable_accuracy"] == 1.0
    assert "including absent expected-label classes" in summary["macro_f1_label_basis"]
    assert "same taxonomy and label coverage" in summary["macro_f1_label_basis"]


def test_calibration_and_selective_risk_known_values(make_record: Callable[..., Record]) -> None:
    records = [make_record(expected_route="debugging"), make_record(expected_route="code_generation")]
    decisions = [
        Decision(route="debugging", probabilities=probabilities("debugging", confidence))
        for confidence in (0.8, 0.6)
    ]
    metrics = summarize(records, decisions, [1.0, 3.0])["probability_metrics"]
    assert metrics["multiclass_brier"] == pytest.approx((0.048 + 1.232) / 2)
    assert metrics["ece_10_bins"] == pytest.approx(0.4)
    assert sum(row["count"] for row in metrics["reliability_bins"]) == 2
    curve = {point["threshold"]: point for point in metrics["risk_coverage"]}
    assert curve[0] == {"threshold": 0, "coverage": 1.0, "selective_error": 0.5}
    assert curve[0.7] == {"threshold": 0.7, "coverage": 0.5, "selective_error": 0.0}
    assert curve[0.9] == {"threshold": 0.9, "coverage": 0.0, "selective_error": None}


def test_fallback_is_abstention_even_at_full_confidence(make_record: Callable[..., Record]) -> None:
    record = make_record(expected_route="fallback")
    decision = Decision(route="fallback", probabilities=probabilities("fallback"))
    summary = summarize([record], [decision], [1.0])
    assert summary["acceptable_accuracy"] == 1
    for point in summary["probability_metrics"]["risk_coverage"]:
        assert point["coverage"] == 0
        assert point["selective_error"] is None


def test_slices_acceptable_labels_usage_and_latency(make_record: Callable[..., Record]) -> None:
    records = [
        make_record(
            expected_route="debugging", acceptable_routes=["debugging", "code_review"],
            tags=["ambiguous", "shared"],
        ),
        make_record(expected_route="repository_qa", tags=["shared"]),
    ]
    decisions = [
        Decision(
            route="code_review", input_tokens=10, output_tokens=2,
            state_tokens_dropped=3,
        ),
        Decision(
            route="fallback", input_tokens=20, output_tokens=4,
            state_tokens_dropped=0,
        ),
    ]
    summary = summarize(records, decisions, [10.0, 30.0])
    assert summary["strict_accuracy"] == 0
    assert summary["acceptable_accuracy"] == 0.5
    assert summary["slices"] == {
        "ambiguous": {"count": 1, "acceptable_accuracy": 1.0},
        "shared": {"count": 2, "acceptable_accuracy": 0.5},
    }
    assert summary["fallback_recall"] is None
    assert summary["input_tokens"] == 30
    assert summary["output_tokens"] == 6
    assert summary["token_accounting_complete"] is True
    assert summary["truncation_rate"] == 0.5
    assert summary["latency_ms"] == {"first_request": 10.0, "p50": 20.0, "p95": 29.0}
    assert "probability_metrics" not in summary
    assert len(summary["acceptable_accuracy_repository_bootstrap_95ci"]) == 2


def test_unknown_usage_and_zero_timing_are_not_fabricated(make_record: Callable[..., Record]) -> None:
    summary = summarize([make_record()], [Decision(route="repository_qa")], [0.0])
    assert summary["token_accounting_complete"] is False
    assert summary["truncation_rate"] is None
    assert summary["serial_requests_per_second"] is None
    assert "acceptable_accuracy_repository_bootstrap_95ci" not in summary


def test_partial_probability_coverage_does_not_report_calibration(
    make_record: Callable[..., Record],
) -> None:
    summary = summarize(
        [make_record(), make_record()],
        [Decision(route="debugging", probabilities=probabilities("debugging")),
         Decision(route="fallback")],
        [1.0, 1.0],
    )
    assert "probability_metrics" not in summary


@pytest.mark.parametrize(("n_records", "n_decisions", "n_timings"), [(0, 0, 0), (1, 0, 1), (1, 1, 0)])
def test_summary_rejects_empty_or_misaligned_inputs(
    make_record: Callable[..., Record], n_records: int, n_decisions: int, n_timings: int,
) -> None:
    with pytest.raises(ValueError, match="matching nonempty"):
        summarize(
            [make_record() for _ in range(n_records)],
            [Decision(route="fallback") for _ in range(n_decisions)], [1.0] * n_timings,
        )


def test_run_persists_completed_artifacts_and_never_overwrites(
    make_record: Callable[..., Record], write_dataset: Callable[..., Path], tmp_path: Path,
) -> None:
    records = [make_record(query="Private synthetic query", context="Private synthetic context")]
    source = write_dataset(records)
    output = tmp_path / "run"
    summary = run_experiment(
        ConstantBackend(), records, output,
        source_paths={"dataset": source}, initialization_seconds=0.25,
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["mode"] == "reviewed_benchmark"
    assert manifest["initialization_seconds"] == 0.25
    assert manifest["files"]["dataset"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert "finished_utc" in manifest
    assert manifest["cpu_count"] == os.cpu_count()
    assert manifest["source_hashes"] == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(evaluation.__file__).parent.glob("*.py")
    }
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary
    prediction_text = (output / "predictions.jsonl").read_text(encoding="utf-8")
    prediction = json.loads(prediction_text)
    assert prediction["id"] == records[0].id
    assert prediction["decision"]["route"] == "fallback"
    assert prediction["latency_ms"] >= 0
    assert records[0].query not in prediction_text
    assert records[0].context not in prediction_text
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        run_experiment(ConstantBackend(), records, output)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_failed_backend_leaves_partial_predictions_and_incomplete_manifest(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    class FailingBackend(ConstantBackend):
        calls = 0

        def predict(self, record: Record) -> Decision:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic backend failure")
            return super().predict(record)

    records = [make_record(), make_record(), make_record()]
    output = tmp_path / "failed-run"
    with pytest.raises(RuntimeError, match="synthetic backend failure"):
        run_experiment(FailingBackend(), records, output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    assert "finished_utc" not in manifest
    assert not (output / "summary.json").exists()
    lines = (output / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["id"] == records[0].id


def test_experiment_draft_gate_and_explicit_probe(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    records = [make_record(review_status="unreviewed", reviewer="")]
    output = tmp_path / "draft-run"
    with pytest.raises(ValueError, match="probe-only"):
        run_experiment(ConstantBackend(), records, output)
    assert not output.exists()
    result = run_experiment(ConstantBackend(), records, output, probe=True)
    assert result["mode"] == "probe_unreviewed_not_a_benchmark"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == result["mode"]
    assert manifest["status"] == "complete"


def test_empty_experiment_does_not_create_output(tmp_path: Path) -> None:
    output = tmp_path / "empty-run"
    with pytest.raises(ValueError, match="empty"):
        run_experiment(ConstantBackend(), [], output)
    assert not output.exists()
