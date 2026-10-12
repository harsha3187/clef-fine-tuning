from __future__ import annotations

import json
import shutil
import struct
from pathlib import Path

import pytest

from routing_lab import clef
from routing_lab.verify_model import LFS_MAGIC, main, read_sums, safetensors_summary, sha256_of, verify, write_sums

REAL = Path(__file__).resolve().parents[1] / "models" / "clef-flash-capability-ft-v3"


def write_safetensors(path: Path, tensors: dict[str, tuple[list[int], str]], *, extra: int = 0, drop: int = 0) -> None:
    header, offset = {}, 0
    for name, (shape, dtype) in tensors.items():
        size = 4
        for dim in shape:
            size *= dim
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(offset + extra - drop))


@pytest.fixture
def export(tmp_path):
    folder = tmp_path / "m"
    (folder / "adapter").mkdir(parents=True)
    write_safetensors(folder / "adapter" / "adapter_model.safetensors",
                      {"a.lora_A.weight": ([4, 16], "F32"), "a.lora_B.weight": ([16, 4], "F32")})
    (folder / "adapter" / "adapter_config.json").write_text(json.dumps({"r": 4}), encoding="utf-8")
    write_safetensors(folder / "joint_head.safetensors", {"w": ([2, 3], "F32")})
    (folder / "clef_finetune.json").write_text(json.dumps({"format": 1, "base": clef.model_info("flash"), "temperature": 1.5, "best_epoch": 2}), encoding="utf-8")
    write_sums(folder, ["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "joint_head.safetensors", "clef_finetune.json"])
    return folder


def test_an_intact_export_verifies_and_the_command_reports_it(export, capsys):
    assert verify(export) == []
    assert main([str(export)]) == 0 and "4 files verified" in capsys.readouterr().out
    assert list(read_sums(export)) == sorted(read_sums(export))


def test_a_git_lfs_pointer_is_called_out_instead_of_failing_obscurely(export):
    weights = export / "joint_head.safetensors"
    weights.write_bytes(LFS_MAGIC + b" v1\noid sha256:abc\nsize 487063576\n")
    problems = verify(export)
    assert any("Git LFS pointer" in p and "git lfs pull" in p for p in problems)
    with pytest.raises(ValueError, match="LFS pointer"):
        safetensors_summary(weights)


def test_a_changed_truncated_or_missing_file_is_detected(export):
    head = export / "joint_head.safetensors"
    original = head.read_bytes()
    head.write_bytes(original[:-4])
    assert any("hash differs" in p for p in verify(export))
    write_sums(export, ["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "joint_head.safetensors", "clef_finetune.json"])
    assert any("truncated" in p for p in verify(export))  # a consistent listing does not hide a short file
    head.unlink()
    assert any("missing" in p for p in verify(export))


def test_the_adapter_must_match_its_configured_rank(export):
    (export / "adapter" / "adapter_config.json").write_text(json.dumps({"r": 8}), encoding="utf-8")
    write_sums(export, ["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "joint_head.safetensors", "clef_finetune.json"])
    assert any("LoRA rank" in p for p in verify(export))


def test_a_different_base_model_or_a_bad_temperature_is_refused(export):
    meta = json.loads((export / "clef_finetune.json").read_text(encoding="utf-8"))
    (export / "clef_finetune.json").write_text(json.dumps({**meta, "base": {**meta["base"], "revision": "other"}}), encoding="utf-8")
    write_sums(export, ["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "joint_head.safetensors", "clef_finetune.json"])
    assert any("different base model" in p for p in verify(export))
    (export / "clef_finetune.json").write_text(json.dumps({**meta, "temperature": None}), encoding="utf-8")
    write_sums(export, ["adapter/adapter_model.safetensors", "adapter/adapter_config.json", "joint_head.safetensors", "clef_finetune.json"])
    assert any("temperature" in p for p in verify(export))


def test_a_folder_without_a_listing_fails(tmp_path, capsys):
    assert verify(tmp_path) and main([str(tmp_path)]) == 1 and "MODEL CHECK FAILED" in capsys.readouterr().err


def test_the_shipped_model_matches_the_deployed_endpoint_values_and_its_listing():
    if not REAL.exists():
        pytest.skip("models/ not present")
    assert verify(REAL) == []
    meta = json.loads((REAL / "clef_finetune.json").read_text(encoding="utf-8"))
    assert sha256_of(REAL / "adapter" / "adapter_model.safetensors") == "3651988bb592eda963ccf96afe38ec89068840c26464094b1f5f09e481ab6beb"
    assert meta["best_epoch"] == 3 and abs(meta["temperature"] - 1.9436062027063157) < 1e-12
    card = (REAL / "MODEL_CARD.md").read_text(encoding="utf-8") if (REAL / "MODEL_CARD.md").exists() else ""
    assert not card or "3651988b" in card


def test_copying_the_folder_keeps_it_valid(export, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(export, copy)
    assert verify(copy) == []
