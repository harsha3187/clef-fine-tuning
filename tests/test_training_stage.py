"""Staging tests: only allowlisted, verified files are uploaded, and consent gates hold."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from routing_lab.evaluation import sha256

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "azure" / "training"


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"training_{name}", TRAINING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stage():
    return load_script("stage")


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"epochs": 1}), encoding="utf-8")
    return path


def staged_files(destination):
    return sorted(p.relative_to(destination).as_posix() for p in destination.rglob("*") if p.is_file())


def test_training_stage_copies_allowlist_not_workspace(stage, bundle, config, tmp_path):
    job = stage.stage_training(bundle, config, tmp_path / "upload")
    destination = job.parent
    files = staged_files(destination)
    assert job.name == "job.yml"
    assert {"config.json", "conda.yml", "job.yml", "upload-manifest.json", "bundle/manifest.json"} <= set(files)
    assert all(name.endswith(".py") for name in files if name.startswith("routing_lab/"))
    assert "routing_lab/finetune.py" in files and "routing_lab/clef.py" in files
    assert not any("test10" in name or name.startswith("data/") for name in files)
    hashes = json.loads((destination / "upload-manifest.json").read_text(encoding="utf-8"))
    assert hashes["config.json"] == sha256(destination / "config.json")
    assert "upload-manifest.json" not in hashes


def test_training_stage_rejects_test_data_and_bad_hashes_before_creating_output(stage, bundle, config, tmp_path):
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["coding-v1-test10.jsonl"] = {"sha256": "c" * 64}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        stage.stage_training(bundle, config, tmp_path / "upload")
    assert not (tmp_path / "upload").exists()


def test_training_stage_validates_the_config_first(stage, bundle, tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"epochs": 0}), encoding="utf-8")
    with pytest.raises(ValueError):
        stage.stage_training(bundle, bad, tmp_path / "upload")
    assert not (tmp_path / "upload").exists()


def prepare_source(bundle, tmp_path):
    """Write test files and record their hashes in the bundle manifest, as real preparation does."""
    source = tmp_path / "source"
    source.mkdir()
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for track, reference in manifest["frozen_test10"].items():
        path = source / reference["file"]
        path.write_text(f"synthetic frozen rows for {track}\n", encoding="utf-8")
        reference["sha256"] = sha256(path)
        manifest["source"]["files"][reference["file"]]["sha256"] = reference["sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source


def test_evaluation_stage_copies_exactly_the_hash_matched_test_files(stage, bundle, tmp_path):
    source = prepare_source(bundle, tmp_path)
    job = stage.stage_evaluation(bundle, source, tmp_path / "eval")
    files = staged_files(job.parent)
    assert job.name == "eval-job.yml"
    assert {"data/coding-v1-test10.jsonl", "data/tool-api-v1-test10.jsonl",
            "data/coding-v1.contracts.json", "data/tool-api-v1.contracts.json"} <= set(files)
    assert not any(name.startswith("bundle/") or "train" in name for name in files)


def test_evaluation_stage_refuses_a_test_file_that_changed(stage, bundle, tmp_path):
    source = prepare_source(bundle, tmp_path)
    (source / "coding-v1-test10.jsonl").write_text("edited after freezing\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the hash recorded"):
        stage.stage_evaluation(bundle, source, tmp_path / "eval")


def test_staged_package_imports_in_isolation(stage, bundle, config, tmp_path):
    staged = stage.stage_training(bundle, config, tmp_path / "upload").parent
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import routing_lab.finetune, routing_lab.cli; "
            "assert routing_lab.finetune.__file__.startswith(sys.argv[1]); assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-I", "-c", code, str(staged)], check=True, cwd=tmp_path,
                   capture_output=True, text=True)


@pytest.mark.parametrize("extra", [[], ["--approve-cost"], ["--approve-upload"]])
def test_submission_requires_both_consents(stage, extra):
    args = stage.create_parser().parse_args(
        ["train", "--bundle", "b", "--config", "c", "--output", "o", "--submit", *extra])
    with pytest.raises(ValueError, match="requires"):
        stage._require_consent(args)


@pytest.mark.parametrize("extra", [[], ["--approve-test-use"], ["--model-job", "job-1"]])
def test_test_evaluation_needs_explicit_test_approval_and_a_model(stage, extra):
    args = stage.create_parser().parse_args(
        ["eval", "--bundle", "b", "--source", "s", "--output", "o", "--submit",
         "--approve-cost", "--approve-upload", *extra])
    with pytest.raises(ValueError, match="requires"):
        stage._require_consent(args)


def test_submission_sets_plain_compute_name_and_model_input(stage, bundle, config, tmp_path, monkeypatch):
    pytest.importorskip("azure.ai.ml")
    job = SimpleNamespace(compute="azureml:clef-gpu-train", inputs={"model": SimpleNamespace(path="x")})
    client = Mock()
    client.jobs.create_or_update.return_value = SimpleNamespace(name="submitted-1")
    monkeypatch.setattr("azure.ai.ml.MLClient", Mock(return_value=client))
    monkeypatch.setattr("azure.ai.ml.load_job", Mock(return_value=job))
    monkeypatch.setattr("azure.identity.AzureCliCredential", Mock())
    source = prepare_source(bundle, tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "stage", "eval", "--bundle", str(bundle), "--source", str(source), "--output", str(tmp_path / "e"),
        "--submit", "--approve-cost", "--approve-upload", "--approve-test-use", "--model-job", "train-job-7",
        "--compute", "clef-gpu-train", "--subscription", "sub", "--resource-group", "rg", "--workspace", "ws"])
    assert stage.main() == 0
    assert job.compute == "clef-gpu-train"
    assert job.inputs["model"].path == "azureml://datastores/workspaceblobstore/paths/azureml/train-job-7/results/"
    client.jobs.create_or_update.assert_called_once_with(job)


def test_provisioning_changes_nothing_without_approval(monkeypatch):
    provision = load_script("provision")
    monkeypatch.setattr(sys, "argv", ["provision"])
    with pytest.raises(ValueError, match="--approve"):
        provision.main()


def test_job_specs_use_the_training_environment_and_cluster():
    for name in ("job.yml", "eval-job.yml"):
        text = (TRAINING / name).read_text(encoding="utf-8")
        assert "compute: azureml:clef-gpu-train" in text and "conda_file: conda.yml" in text
    assert "--allow-training" in (TRAINING / "job.yml").read_text(encoding="utf-8")
    assert "--calibrated" in (TRAINING / "eval-job.yml").read_text(encoding="utf-8")
    conda = (TRAINING / "conda.yml").read_text(encoding="utf-8")
    for pin in ("torch==2.11.0+cu126", "transformers==5.10.2", "peft==0.21.1"):
        assert pin in conda
