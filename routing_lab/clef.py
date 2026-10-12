"""Pinned Cloudflare Clef runtime shared by the local GPU backend, scorer and trainer.

Nothing heavy is imported at module load. ``load`` downloads the pinned snapshot and
imports the release's own ``joint_schema_model.py`` only after verifying its SHA-256,
because that file is executed as code.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

ClefVariant = Literal["flash", "clef"]
RELEASE_MODULE = "joint_schema_model.py"
FINETUNE_META = "clef_finetune.json"
ADAPTER_DIR = "adapter"
HEAD_FILE = "joint_head.safetensors"
# Identical in Clef and Clef-Flash at the pinned revisions below.
RELEASE_MODULE_SHA256 = "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3"
DEFAULT_MAX_LENGTH = 16_384
MAX_LENGTH_CEILING = 32_768


@dataclass(frozen=True)
class ClefRelease:
    variant: ClefVariant
    model_id: str
    revision: str
    base_model: str
    hidden_size: int


RELEASES: Mapping[str, ClefRelease] = MappingProxyType({
    "flash": ClefRelease(
        "flash", "Cloudflare/clef-flash", "17f0b0ad64efb65d273590632833508766b2aae6",
        "Qwen/Qwen3.5-9B", 4096,
    ),
    "clef": ClefRelease(
        "clef", "Cloudflare/clef", "2f3de3dd85f379784083b0814d997ab627200f0c",
        "Qwen/Qwen3.8-27B", 5120,
    ),
})


def release(variant: str) -> ClefRelease:
    if variant not in RELEASES:
        raise ValueError(f"Unknown Clef variant {variant!r}; choose one of {sorted(RELEASES)}")
    return RELEASES[variant]


def model_info(variant: str) -> dict[str, str]:
    """Provenance echoed by the endpoint and verified by every client."""
    pinned = release(variant)
    return {"checkpoint": pinned.model_id, "revision": pinned.revision, "variant": pinned.variant}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def import_release_module(snapshot: Path) -> Any:
    """Import the verified release module without leaving an unverified copy on sys.path."""
    source = snapshot / RELEASE_MODULE
    if _sha256(source) != RELEASE_MODULE_SHA256:
        raise RuntimeError("Clef release module hash differs from the audited revision")
    spec = importlib.util.spec_from_file_location("joint_schema_model", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules["joint_schema_model"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class ClefRuntime:
    """Loaded model, processor and the verified release module."""

    model: Any
    processor: Any
    module: Any
    variant: ClefVariant
    temperature: float = 1.0
    finetune: dict[str, Any] | None = None


def _base_snapshot(pinned: ClefRelease) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(pinned.model_id, revision=pinned.revision))


def read_finetune_meta(export: Path) -> dict[str, Any]:
    """Read and sanity-check the metadata that marks a directory as a fine-tuned export."""
    meta = json.loads((export / FINETUNE_META).read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or not isinstance(meta.get("base"), dict):
        raise ValueError("Fine-tuned export metadata is malformed")
    return meta


def find_export(root: Path) -> Path:
    """Locate the single fine-tuned export under a model mount; Azure may add a folder level."""
    found = [path.parent for path in root.rglob(FINETUNE_META)]
    if len(found) != 1:
        raise RuntimeError(f"Expected exactly one fine-tuned export under the model directory, found {len(found)}")
    return found[0]


def finetune_provenance(export: Path) -> dict[str, Any]:
    """Identify exactly which weights an export holds; clients can pin the adapter hash."""
    meta = read_finetune_meta(export)
    return {
        "adapter_sha256": _sha256(export / ADAPTER_DIR / "adapter_model.safetensors"),
        "head_sha256": _sha256(export / HEAD_FILE),
        "temperature": meta.get("temperature"),
        "best_epoch": meta.get("best_epoch"),
        "bundle_sha256": meta.get("bundle_sha256"),
    }


def _load_finetuned(
    pinned: ClefRelease, export: Path, device: str, calibrated: bool,
) -> ClefRuntime:
    """Pinned base, then its LoRA adapter merged in, then the fine-tuned joint head."""
    meta = read_finetune_meta(export)
    if meta["base"] != model_info(pinned.variant):
        raise ValueError("Fine-tuned export was trained from a different base release than requested")
    temperature = 1.0
    if calibrated:
        value = meta.get("temperature")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError("Export has no fitted temperature; load it without calibration")
        temperature = float(value)
    from peft import PeftModel
    from safetensors.torch import load_file

    base = _base_snapshot(pinned)
    module = import_release_module(base)
    model, processor = module.load_release_model(base, device=device)
    model.language_model = PeftModel.from_pretrained(
        model.language_model, str(export / ADAPTER_DIR),
    ).merge_and_unload()
    model.head.load_state_dict(load_file(str(export / HEAD_FILE)), strict=True)
    return ClefRuntime(model.eval(), processor, module, pinned.variant, temperature, meta)


def load(
    variant: str = "flash", *, device: str = "cuda", snapshot: str | Path | None = None,
    calibrated: bool = False,
) -> ClefRuntime:
    """Load a pinned release, or a fine-tuned export directory made by ``routing_lab.finetune``.

    ``snapshot`` may also be a full release-layout directory. ``calibrated`` applies the export's
    fitted temperature and is only valid for a fine-tuned export.
    """
    pinned = release(variant)
    if snapshot is not None and (Path(snapshot) / FINETUNE_META).is_file():
        return _load_finetuned(pinned, Path(snapshot), device, calibrated)
    if calibrated:
        raise ValueError("Calibration is available only for a fine-tuned export")
    path = Path(snapshot) if snapshot is not None else _base_snapshot(pinned)
    module = import_release_module(path)
    model, processor = module.load_release_model(path, device=device)
    return ClefRuntime(model, processor, module, pinned.variant)


def _state_budget(runtime: ClefRuntime, record: dict[str, Any], max_length: int) -> tuple[int, int]:
    """Return (state tokens, tokens available to state) for truncation telemetry."""
    tokenizer = runtime.processor.tokenizer
    module = runtime.module
    state_tokens = len(tokenizer(module.render(record["state"]), add_special_tokens=False).input_ids)
    empty = module.encode_record(tokenizer, {**record, "state": ""}, max_length=max_length)
    return state_tokens, max_length - len(empty.input_ids)


def decide(
    runtime: ClefRuntime, state: str, questions: dict[str, Any], *,
    max_length: int = DEFAULT_MAX_LENGTH, max_state_tokens: int | None = None,
) -> dict[str, Any]:
    """Answer one text-only SystemOne request, with native probabilities and usage."""
    import torch

    module = runtime.module
    record = {"state": state, "questions": questions}
    state_tokens, available = _state_budget(runtime, record, max_length)
    kept = min(state_tokens, available, max_state_tokens if max_state_tokens is not None else state_tokens)
    encoded = module.encode_record(
        runtime.processor.tokenizer, record, max_length=max_length, max_state_tokens=max_state_tokens,
    )
    device = next(runtime.model.parameters()).device
    batch = module.collate_records([encoded], runtime.processor.tokenizer.pad_token_id, device)
    with torch.inference_mode():
        logits = runtime.model(batch)[0]
        if runtime.temperature != 1.0:
            logits = [item / runtime.temperature for item in logits]
    answers = {
        question.question_id: module.systemone_answer(
            questions[question.question_id],
            dict(zip(question.option_ids, item.float().softmax(-1).tolist(), strict=True)),
        )
        for question, item in zip(encoded.questions, logits, strict=True)
    }
    return {
        "answers": answers,
        "usage": {
            "input_tokens": len(encoded.input_ids), "output_tokens": 0,
            "state_tokens_dropped": state_tokens - kept,
        },
    }
