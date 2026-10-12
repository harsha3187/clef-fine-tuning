"""CLI opt-in and supervised leakage checks; all model constructors are fakes."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest

from routing_lab import cli
from routing_lab.backends import ConstantBackend
from routing_lab.data import Contract, Record


@pytest.fixture
def contract_path(tmp_path: Path, contracts: dict[str, Contract]) -> Path:
    path = tmp_path / "contracts.json"
    path.write_text(
        json.dumps({key: value.model_dump() for key, value in contracts.items()}), encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("backend", "constructor", "flag"),
    [("clef", "ClefBackend", "--allow-model-download"), ("azure", "AzureLLMBackend", "--allow-remote")],
)
def test_model_backends_require_explicit_opt_in_before_construction(
    monkeypatch: pytest.MonkeyPatch, make_record: Callable[..., Record],
    write_dataset: Callable[..., Path], tmp_path: Path,
    backend: str, constructor: str, flag: str,
) -> None:
    path = write_dataset([make_record()])
    factory = Mock(side_effect=AssertionError("No model or network calls allowed"))
    monkeypatch.setattr(cli, constructor, factory)
    args = cli.create_parser().parse_args([
        "run", "--data", str(path), "--output", str(tmp_path / "run"), "--backend", backend,
    ])
    with pytest.raises(ValueError, match=flag):
        cli.execute(args)
    factory.assert_not_called()
    assert not args.output.exists()


@pytest.mark.parametrize(
    ("backend", "constructor", "flag"),
    [("clef", "ClefBackend", "--allow-model-download"), ("azure", "AzureLLMBackend", "--allow-remote")],
)
def test_explicit_opt_in_dispatches_only_to_mocked_backend(
    monkeypatch: pytest.MonkeyPatch, make_record: Callable[..., Record],
    write_dataset: Callable[..., Path], contract_path: Path, tmp_path: Path,
    backend: str, constructor: str, flag: str,
) -> None:
    path = write_dataset([make_record()])
    factory = Mock(return_value=ConstantBackend())
    monkeypatch.setattr(cli, constructor, factory)
    args = cli.create_parser().parse_args([
        "run", "--data", str(path), "--contracts", str(contract_path),
        "--output", str(tmp_path / "run"), "--backend", backend, flag,
    ])
    cli.execute(args)
    factory.assert_called_once()
    assert (args.output / "summary.json").exists()


@pytest.mark.parametrize("backend", ["majority", "tfidf"])
@pytest.mark.parametrize("overlap", ["id", "group_id", "repository", "text"])
def test_supervised_cli_rejects_training_leakage_before_fit(
    monkeypatch: pytest.MonkeyPatch, make_record: Callable[..., Record],
    write_dataset: Callable[..., Path], contract_path: Path, tmp_path: Path,
    backend: str, overlap: str,
) -> None:
    evaluation = make_record()
    changes = (
        {"query": evaluation.query.upper(), "context": evaluation.context.upper()}
        if overlap == "text" else {overlap: getattr(evaluation, overlap)}
    )
    train_path = write_dataset([make_record(**changes)], "train.jsonl")
    eval_path = write_dataset([evaluation], "eval.jsonl")
    fit = Mock(side_effect=AssertionError("Leaked training must not fit"))
    monkeypatch.setattr(cli, "TfidfBackend" if backend == "tfidf" else "ConstantBackend", fit)
    args = cli.create_parser().parse_args([
        "run", "--data", str(eval_path), "--train", str(train_path),
        "--contracts", str(contract_path), "--output", str(tmp_path / "run"),
        "--backend", backend,
    ])
    with pytest.raises(ValueError, match="Data leakage"):
        cli.execute(args)
    fit.assert_not_called()
    assert not args.output.exists()


@pytest.mark.parametrize("backend", ["majority", "tfidf"])
def test_supervised_cli_requires_training_data(
    make_record: Callable[..., Record], write_dataset: Callable[..., Path],
    contract_path: Path, tmp_path: Path, backend: str,
) -> None:
    path = write_dataset([make_record()])
    args = cli.create_parser().parse_args([
        "run", "--data", str(path), "--contracts", str(contract_path),
        "--output", str(tmp_path / "run"), "--backend", backend,
    ])
    with pytest.raises(ValueError, match="require --train"):
        cli.execute(args)


def test_probe_does_not_allow_unreviewed_training_labels(
    make_record: Callable[..., Record], write_dataset: Callable[..., Path],
    contract_path: Path, tmp_path: Path,
) -> None:
    evaluation = write_dataset([make_record(review_status="unreviewed", reviewer="")], "eval.jsonl")
    training = write_dataset([make_record(review_status="unreviewed", reviewer="")], "train.jsonl")
    args = cli.create_parser().parse_args([
        "run", "--data", str(evaluation), "--train", str(training),
        "--contracts", str(contract_path), "--output", str(tmp_path / "run"),
        "--backend", "majority", "--probe",
    ])
    with pytest.raises(ValueError, match="reviewed"):
        cli.execute(args)


def test_majority_uses_disjoint_reviewed_training_labels(
    make_record: Callable[..., Record], write_dataset: Callable[..., Path],
    contract_path: Path, tmp_path: Path,
) -> None:
    evaluation = write_dataset([make_record(expected_route="debugging")], "eval.jsonl")
    training = write_dataset([
        make_record(expected_route="debugging"), make_record(expected_route="debugging"),
        make_record(expected_route="fallback"),
    ], "train.jsonl")
    args = cli.create_parser().parse_args([
        "run", "--data", str(evaluation), "--train", str(training),
        "--contracts", str(contract_path), "--output", str(tmp_path / "run"),
        "--backend", "majority",
    ])
    cli.execute(args)
    manifest = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"]["route"] == "debugging"
    assert "training" in manifest["files"]


def test_cli_existing_output_is_not_overwritten(
    make_record: Callable[..., Record], write_dataset: Callable[..., Path], tmp_path: Path,
) -> None:
    path = write_dataset([make_record()])
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    args = cli.create_parser().parse_args([
        "run", "--data", str(path), "--output", str(output),
    ])
    with pytest.raises(ValueError, match="already exists"):
        cli.execute(args)
    assert sentinel.read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize(("extra", "message"), [
    (["--calibrated"], "needs a fine-tuned export"),
    (["--snapshot", "export"], "only by the clef backend"),
    (["--calibrated", "--snapshot", "export"], "only by the clef backend"),
])
def test_snapshot_and_calibration_flags_are_validated_before_any_load(
    make_record: Callable[..., Record], write_dataset: Callable[..., Path], tmp_path: Path,
    extra: list[str], message: str,
) -> None:
    path = write_dataset([make_record()])
    backend = "clef" if "only by the clef" not in message else "rules"
    flags = ["--allow-model-download"] if backend == "clef" else []
    args = cli.create_parser().parse_args([
        "run", "--data", str(path), "--output", str(tmp_path / "run"), "--backend", backend, *flags, *extra,
    ])
    with pytest.raises(ValueError, match=message):
        cli.execute(args)
    assert not args.output.exists()
