"""Offline trainer checks; no model packages, downloads, CUDA, or paid compute."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pydantic import ValidationError

import routing_lab.finetune as training
from routing_lab.evaluation import sha256
from routing_lab.finetune_data import PARTITIONS, TRACKS
from routing_lab import finetune_baselines
from routing_lab.data import write_records


def rewrite_mixed(bundle, transform):
    path = bundle / "mixed-train.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    transform(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][path.name]["sha256"] = sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_verified_native_bundle(bundle):
    manifest, _, records, examples = training.validate_inputs(bundle)
    assert manifest["partition_counts"] == dict.fromkeys(PARTITIONS, 2)
    assert all(len(records[p][t]) == 1 for p in PARTITIONS for t in TRACKS)
    assert {row["taxonomy_id"] for row in examples["train"]} == set(TRACKS)


@pytest.mark.parametrize("transform", [
    lambda rows: rows[0]["answers"].update(route="fallback"),
    lambda rows: rows[0].update(state="tampered input"),
    lambda rows: rows[0]["questions"]["route"].update(instructions="tampered catalog"),
    lambda rows: rows[0]["questions"]["route"].update(
        criteria=dict(reversed(list(rows[0]["questions"]["route"]["criteria"].items())))),
    lambda rows: rows[0].update(extra="unrecognized"),
    lambda rows: rows.__setitem__(1, rows[0]),
    lambda rows: rows.pop(),
    lambda rows: rows.clear(),
])
def test_rejects_hash_consistent_but_invalid_native_data(bundle, transform):
    rewrite_mixed(bundle, transform)
    with pytest.raises(ValueError, match="mismatch|Incomplete"):
        training.validate_inputs(bundle)


def test_rejects_hash_mismatch_and_unexpected_files(bundle):
    path = bundle / "mixed-train.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        training.validate_inputs(bundle)
    (bundle / "test10.jsonl").write_text("not permitted", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        training.validate_inputs(bundle)


def write_config(path, **overrides):
    path.write_text(json.dumps(overrides), encoding="utf-8")
    return path


@pytest.mark.parametrize("override", [
    {"extra": 1}, {"epochs": 0}, {"epochs": "1"}, {"micro_batch_size": True},
    {"gradient_accumulation_steps": 0}, {"lora_rank": 0}, {"lora_dropout": 1.0},
    {"lora_learning_rate": 0}, {"head_learning_rate": float("nan")}, {"label_smoothing": 0.5},
    {"max_length": 64}, {"max_length": 8192, "eval_max_length": 4096}, {"seed": -1},
    {"max_train_steps": 2}, {"validation_limit_per_track": 8}, {"calibration_limit_per_track": 8},
    {"smoke": True}, {"variant": "other"}, {"model_revision": "main"},
    {"variant": "clef", "model_revision": training.clef.RELEASES["flash"].revision},
])
def test_config_rejects_unsafe_or_ambiguous_values(override):
    with pytest.raises(ValidationError):
        training.TrainingConfig(**override)


def test_config_pins_the_release_revision():
    assert training.TrainingConfig().model_revision == training.clef.RELEASES["flash"].revision
    assert training.TrainingConfig(variant="clef").model_revision == training.clef.RELEASES["clef"].revision


def test_checked_in_configs():
    root = Path(__file__).resolve().parents[1] / "azure" / "training"
    full, smoke = (training.TrainingConfig.model_validate_json((root / name).read_text(encoding="utf-8"))
                   for name in ("config.json", "smoke-config.json"))
    assert (full.variant, full.epochs, full.max_length) == ("flash", 3, 4096)
    assert (full.micro_batch_size, full.gradient_accumulation_steps) == (4, 4)
    assert not full.smoke and full.max_train_steps is None
    assert smoke.smoke and smoke.epochs == 1 and smoke.max_train_steps == 4
    assert smoke.validation_limit_per_track == smoke.calibration_limit_per_track == 16
    assert full.lora_learning_rate == smoke.lora_learning_rate


def test_learning_rate_schedule_warms_up_then_decays_to_the_floor():
    scales = [training.lr_scale(step, 100, 10) for step in range(100)]
    assert scales[0] == pytest.approx(0.1) and scales[9] == pytest.approx(1.0)
    assert scales[10] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(scales[10:], scales[11:]))
    assert scales[-1] == pytest.approx(0.1, abs=0.01)
    assert training.lr_scale(5, 100, 0) <= 1.0


def test_planned_steps_account_for_partial_groups_and_smoke_limit():
    config = training.TrainingConfig(epochs=3, micro_batch_size=2, gradient_accumulation_steps=4)
    assert training.planned_steps(17, config) == (3, 9)
    assert training.planned_steps(3966, training.TrainingConfig()) == (248, 744)
    smoke = training.TrainingConfig(smoke=True, max_train_steps=4, micro_batch_size=1, gradient_accumulation_steps=4)
    assert training.planned_steps(100, smoke) == (25, 4)


def fake_rows(lengths):
    return [(SimpleNamespace(input_ids=(0,) * length), index) for index, length in enumerate(lengths)]


def test_micro_batches_cover_every_row_once_and_pad_little():
    import random

    lengths = [10, 900, 12, 880, 15, 890, 11, 905, 14, 870, 13, 895, 16, 885, 18, 875, 17, 910]
    rows = fake_rows(lengths)
    batches = training.micro_batches(rows, 4, random.Random(0), window_batches=2)
    assert sorted(item[1] for batch in batches for item in batch) == list(range(len(rows)))
    assert len(batches) == -(-len(rows) // 4)
    spread = [max(len(r[0].input_ids) for r in b) - min(len(r[0].input_ids) for r in b) for b in batches if len(b) == 4]
    assert max(spread) < 100
    assert training.micro_batches(rows, 4, random.Random(0)) == training.micro_batches(rows, 4, random.Random(0))


def test_smoke_ordering_is_longest_first_and_deterministic():
    batches = training.micro_batches(fake_rows([5, 50, 20, 40, 10]), 2, None)
    assert [[len(r[0].input_ids) for r in b] for b in batches] == [[50, 40], [20, 10], [5]]


def test_partial_accumulation_uses_actual_example_count():
    batches = [fake_rows([1, 1]), fake_rows([1, 1]), fake_rows([1])]
    groups = list(training.accumulation_groups(batches, 2))
    assert [[len(chunk) for chunk, _ in group] for group in groups] == [[2, 2], [1]]
    assert [[weight for _, weight in group] for group in groups] == [[0.5, 0.5], [1.0]]
    uneven = list(training.accumulation_groups([fake_rows([1, 1, 1]), fake_rows([1])], 2))
    assert [weight for _, weight in uneven[0]] == [0.75, 0.25]


def test_step_count_matches_the_batches_actually_produced():
    import random

    config = training.TrainingConfig(micro_batch_size=4, gradient_accumulation_steps=4)
    rows = fake_rows(range(1, 100))
    steps = len(list(training.accumulation_groups(
        training.micro_batches(rows, 4, random.Random(1)), 4)))
    assert steps == training.planned_steps(len(rows), config)[0]


def test_selection_uses_equal_track_mean_not_population_weighting():
    results = {"coding-v1": {"macro_f1": 0.2, "count": 1000}, "tool-api-v1": {"macro_f1": 0.8, "count": 10}}
    assert training.selection_score(results) == pytest.approx(0.5)
    results["tool-api-v1"]["macro_f1"] = math.nan
    with pytest.raises(FloatingPointError):
        training.selection_score(results)


class FakeModule:
    """Mimic the release module's option order: choice options are sorted by key."""

    @staticmethod
    def encode_record(tokenizer, record, max_length):
        criteria = record["questions"]["route"]["criteria"]
        question = SimpleNamespace(option_ids=tuple(sorted(criteria)))
        return SimpleNamespace(input_ids=(1, 2, 3), questions=(question,))


def test_label_index_follows_the_sorted_option_order_the_model_scores():
    row = {"id": "x", "state": "s", "answers": {"route": "debugging"},
           "questions": {"route": {"type": "choice", "instructions": "i",
                                   "criteria": {"fallback": "f", "debugging": "d", "code_review": "c"}}}}
    encoded, target = training.encode_example(FakeModule, None, row, 512)
    assert encoded.questions[0].option_ids == ("code_review", "debugging", "fallback")
    assert target == 1
    row["answers"]["route"] = "unoffered"
    with pytest.raises(ValueError, match="not an offered option"):
        training.encode_example(FakeModule, None, row, 512)


def test_choice_loss_and_parameter_groups():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([4.0, 0.0, 0.0])
    sharp = training.choice_loss(torch, logits, 0, 0.0, 0.0)
    assert sharp.item() == pytest.approx(torch.nn.functional.cross_entropy(logits[None], torch.tensor([0])).item())
    assert training.choice_loss(torch, logits, 0, 0.1, 0.0) > sharp
    probabilities = torch.softmax(logits, -1)
    brier = ((probabilities - torch.tensor([1.0, 0, 0])) ** 2).sum()
    assert training.choice_loss(torch, logits, 0, 0.0, 2.0).item() == pytest.approx(sharp.item() + 2 * brier.item())
    assert training.choice_loss(torch, logits, 2, 0.0, 1.0) > training.choice_loss(torch, logits, 0, 0.0, 1.0)

    model = torch.nn.ModuleDict({"head": torch.nn.Linear(2, 2), "language_model": torch.nn.Linear(2, 2)})
    model["language_model"].weight.requires_grad_(True)
    names = {"language_model.weight": "lora", "language_model.bias": "lora"}
    config = training.TrainingConfig(lora_learning_rate=1e-3, head_learning_rate=5e-5, weight_decay=0.1)
    groups = training._param_groups(torch, model, config)
    assert {(g["lr"], g["weight_decay"]) for g in groups} == {(1e-3, 0.1), (1e-3, 0.0), (5e-5, 0.1), (5e-5, 0.0)}
    assert names
    model["head"].requires_grad_(False)
    with pytest.raises(RuntimeError, match="Both LoRA and joint-head"):
        training._param_groups(torch, model, config)


def test_dry_run_imports_no_model_packages_or_writes(bundle, tmp_path):
    config = write_config(tmp_path / "config.json")
    code = f"""
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {{"torch", "transformers", "peft", "huggingface_hub", "safetensors"}}:
            raise AssertionError("model package imported: " + name)
sys.meta_path.insert(0, Block())
from routing_lab.finetune import main
main(["--bundle", {str(bundle)!r}, "--config", {str(config)!r}, "--dry-run"])
"""
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["status"] == "dry_run" and report["training_launched"] is False
    assert report["partition_counts"] == dict.fromkeys(PARTITIONS, 2)
    assert report["steps_per_epoch"] == 1 and report["planned_steps"] == 3
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("flags", [
    ["--dry-run", "--allow-training"], ["--dry-run", "--output", "x"], ["--output", "x"],
    ["--output", "x", "--allow-training"], [],
])
def test_cli_requires_exclusive_dry_run_or_complete_consent(bundle, tmp_path, flags):
    config = write_config(tmp_path / "config.json")
    with pytest.raises(SystemExit):
        training.main(["--bundle", str(bundle), "--config", str(config), *flags])


def test_training_consent_and_cuda_fail_before_outputs(bundle, tmp_path):
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="requires --allow-training"):
        training.train(bundle, training.TrainingConfig(), output)
    assert not output.exists()
    with pytest.raises(RuntimeError, match="packages|CUDA"):
        training.train(bundle, training.TrainingConfig(), output, allow_training=True, allow_model_download=True)
    assert not output.exists()


def test_training_failure_is_persisted_without_claiming_success(bundle, tmp_path, monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "peft", ModuleType("peft"))

    def explode(*args, **kwargs):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(training, "_train", explode)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="synthetic failure"):
        training.train(bundle, training.TrainingConfig(), output, allow_training=True, allow_model_download=True)
    report = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert report["status"] == "failed" and report["selected_full_model"] is False
    assert report["frozen_test_used"] is False and "synthetic failure" in report["error"]


def test_training_refuses_bundle_mutation_or_existing_outputs(bundle, tmp_path):
    with pytest.raises(ValueError, match="must not modify"):
        training.train(bundle, training.TrainingConfig(), bundle / "run",
                       allow_training=True, allow_model_download=True)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "old.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        training.train(bundle, training.TrainingConfig(), occupied,
                       allow_training=True, allow_model_download=True)
    assert (occupied / "old.txt").read_text(encoding="utf-8") == "keep"


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
