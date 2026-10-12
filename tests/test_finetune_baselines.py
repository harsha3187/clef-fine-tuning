"""Comparator selection must use training labels, never validation labels."""

import json

import pytest

from routing_lab import finetune_baselines
from routing_lab.data import write_records


def test_majority_comes_from_training_and_preserves_silver(tmp_path, monkeypatch, make_record):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    training = [make_record(expected_route="debugging", review_status="unreviewed") for _ in range(3)]
    validation = [make_record(expected_route="code_generation", review_status="unreviewed")
                  for _ in range(2)]
    write_records(bundle / "coding-v1-train.jsonl", training)
    write_records(bundle / "coding-v1-validation.jsonl", validation)
    monkeypatch.setattr(finetune_baselines, "verify_bundle", lambda _: {})
    monkeypatch.setattr(finetune_baselines, "TRACKS", ("coding-v1",))
    output = tmp_path / "reports"
    results = finetune_baselines.evaluate_baselines(bundle, output)
    assert results["coding-v1"]["majority"]["strict_accuracy"] == 0.0
    manifest = json.loads((output / "coding-v1-majority" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"]["route"] == "debugging"
    assert manifest["mode"] == "probe_unreviewed_not_a_benchmark"
    assert all(record.review_status == "unreviewed" for record in validation)
    with pytest.raises(FileExistsError):
        finetune_baselines.evaluate_baselines(bundle, output)
