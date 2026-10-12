"""Loader branching and temperature handling for fine-tuned exports, without any model weights."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from routing_lab import clef


def make_export(tmp_path, **changes):
    meta = {"format": 1, "base": clef.model_info("flash"), "temperature": 0.6, **changes}
    (tmp_path / clef.FINETUNE_META).write_text(json.dumps(meta), encoding="utf-8")
    return tmp_path


def test_calibration_is_only_valid_for_a_finetuned_export(tmp_path):
    with pytest.raises(ValueError, match="only for a fine-tuned export"):
        clef.load("flash", snapshot=tmp_path, calibrated=True)
    with pytest.raises(ValueError, match="only for a fine-tuned export"):
        clef.load("flash", calibrated=True)


def test_export_trained_from_another_base_is_refused(tmp_path):
    make_export(tmp_path, base=clef.model_info("clef"))
    with pytest.raises(ValueError, match="different base release"):
        clef.load("flash", snapshot=tmp_path)
    tampered = clef.model_info("flash") | {"revision": "0" * 40}
    make_export(tmp_path, base=tampered)
    with pytest.raises(ValueError, match="different base release"):
        clef.load("flash", snapshot=tmp_path)


@pytest.mark.parametrize("temperature", [None, 0, -1.0, "warm"])
def test_calibrated_load_needs_a_fitted_temperature(tmp_path, temperature):
    make_export(tmp_path, temperature=temperature)
    with pytest.raises(ValueError, match="no fitted temperature"):
        clef.load("flash", snapshot=tmp_path, calibrated=True)


def test_malformed_metadata_is_rejected(tmp_path):
    (tmp_path / clef.FINETUNE_META).write_text(json.dumps({"base": "flash"}), encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        clef.read_finetune_meta(tmp_path)


def test_decide_applies_the_temperature_without_changing_the_choice():
    torch = pytest.importorskip("torch")
    option_ids = ("a", "b", "c")
    encoded = SimpleNamespace(
        input_ids=(1, 2, 3), questions=(SimpleNamespace(question_id="route", option_ids=option_ids),))
    module = SimpleNamespace(
        render=lambda value: value,
        encode_record=lambda tokenizer, record, max_length, max_state_tokens=None: encoded,
        collate_records=lambda records, pad, device: {},
        systemone_answer=lambda question, probabilities: {
            "choice": max(probabilities, key=probabilities.get), "probabilities": probabilities},
    )
    def tokenizer(text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[1])

    tokenizer.pad_token_id = 0
    logits = torch.tensor([2.0, 1.0, 0.0])

    class Model:
        def parameters(self):
            return iter([torch.zeros(1)])

        def __call__(self, batch):
            return [[logits]]

    questions = {"route": {"type": "choice", "criteria": dict.fromkeys(option_ids, "x")}}
    plain = clef.ClefRuntime(Model(), SimpleNamespace(tokenizer=tokenizer), module, "flash")
    warm = clef.ClefRuntime(Model(), SimpleNamespace(tokenizer=tokenizer), module, "flash", temperature=4.0)
    sharp = clef.ClefRuntime(Model(), SimpleNamespace(tokenizer=tokenizer), module, "flash", temperature=0.25)
    answers = {name: clef.decide(runtime, "state", questions)["answers"]["route"]
               for name, runtime in (("plain", plain), ("warm", warm), ("sharp", sharp))}
    assert {a["choice"] for a in answers.values()} == {"a"}
    top = {name: a["probabilities"]["a"] for name, a in answers.items()}
    assert top["warm"] < top["plain"] < top["sharp"]


def write_export(root, adapter=b"adapter", head=b"head"):
    export = root / "export"
    (export / clef.ADAPTER_DIR).mkdir(parents=True)
    (export / clef.ADAPTER_DIR / "adapter_model.safetensors").write_bytes(adapter)
    (export / clef.HEAD_FILE).write_bytes(head)
    make_export(export, best_epoch=3, bundle_sha256="b" * 64)
    return export


def test_find_export_handles_the_extra_folder_azure_may_add(tmp_path):
    export = write_export(tmp_path / "AZUREML_MODEL_DIR" / "clef-flash-router-ft")
    assert clef.find_export(tmp_path / "AZUREML_MODEL_DIR") == export


@pytest.mark.parametrize("copies", [0, 2])
def test_find_export_requires_exactly_one_export(tmp_path, copies):
    for index in range(copies):
        write_export(tmp_path / f"copy-{index}")
    with pytest.raises(RuntimeError, match="exactly one"):
        clef.find_export(tmp_path)


def test_provenance_identifies_the_exact_weights(tmp_path):
    import hashlib

    export = write_export(tmp_path, adapter=b"A" * 3_000_000, head=b"H")
    provenance = clef.finetune_provenance(export)
    assert provenance == {
        "adapter_sha256": hashlib.sha256(b"A" * 3_000_000).hexdigest(),
        "head_sha256": hashlib.sha256(b"H").hexdigest(),
        "temperature": 0.6, "best_epoch": 3, "bundle_sha256": "b" * 64}
    (export / clef.ADAPTER_DIR / "adapter_model.safetensors").write_bytes(b"A" * 3_000_001)
    assert clef.finetune_provenance(export)["adapter_sha256"] != provenance["adapter_sha256"]
