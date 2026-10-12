"""Check a fine-tuned Clef export on disk: hashes, file structure and the base model it was trained from.

    python -m routing_lab.verify_model models/clef-flash-capability-ft-v3

It needs no model libraries. It catches the failures that happen when weights travel through git and containers:
a Git LFS pointer file where the weights should be, a truncated download, and a file that is not the one that was listed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

from routing_lab import clef

SUMS = "SHA256SUMS"
LFS_MAGIC = b"version https://git-lfs"
DTYPE_BYTES = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sums(folder: Path) -> dict[str, str]:
    """`SHA256SUMS` in the usual `<hash>  <relative path>` form."""
    sums = {}
    for line in (folder / SUMS).read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, name = line.partition("  ")
            sums[name.strip()] = digest.strip()
    return sums


def write_sums(folder: Path, names: list[str]) -> None:
    lines = [f"{sha256_of(folder / name)}  {name}" for name in sorted(names)]
    (folder / SUMS).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def safetensors_summary(path: Path) -> dict[str, object]:
    """Parse the safetensors header and check the tensor offsets exactly cover the file. Raises ValueError otherwise."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        first = handle.read(64)
        if first.startswith(LFS_MAGIC):
            raise ValueError(f"{path.name} is a Git LFS pointer, not the weights: run `git lfs pull`")
        if len(first) < 8:
            raise ValueError(f"{path.name} is too short to be a safetensors file")
        (header_len,) = struct.unpack("<Q", first[:8])
        if header_len <= 0 or 8 + header_len > size:
            raise ValueError(f"{path.name} has a corrupt header length")
        handle.seek(8)
        header = json.loads(handle.read(header_len))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    end = 0
    for name, info in tensors.items():
        begin, stop = info["data_offsets"]
        count = 1
        for dim in info["shape"]:
            count *= dim
        if info["dtype"] not in DTYPE_BYTES or stop - begin != count * DTYPE_BYTES[info["dtype"]] or begin < 0:
            raise ValueError(f"{path.name}: tensor {name} has inconsistent shape, dtype or offsets")
        end = max(end, stop)
    if 8 + header_len + end != size:
        raise ValueError(f"{path.name} is truncated or has extra bytes (expected {8 + header_len + end}, found {size})")
    return {"tensors": len(tensors), "dtypes": sorted({t["dtype"] for t in tensors.values()}), "shapes": tensors}


def verify(folder: Path) -> list[str]:
    """Return the list of problems; an empty list means the export is intact and matches its listing."""
    problems: list[str] = []
    if not (folder / SUMS).is_file():
        return [f"{SUMS} is missing in {folder}"]
    sums = read_sums(folder)
    for name, expected in sums.items():
        path = folder / name
        if not path.is_file():
            problems.append(f"listed file is missing: {name}")
            continue
        with path.open("rb") as handle:
            if handle.read(64).startswith(LFS_MAGIC):
                problems.append(f"{name} is a Git LFS pointer, not the weights: run `git lfs pull`")
                continue
        if sha256_of(path) != expected:
            problems.append(f"hash differs from {SUMS}: {name}")
    for name in sums:
        if name.endswith(".safetensors") and (folder / name).is_file() and not any(name in p for p in problems):
            try:
                summary = safetensors_summary(folder / name)
            except (ValueError, KeyError, json.JSONDecodeError) as error:
                problems.append(str(error))
                continue
            if name.endswith("adapter_model.safetensors"):
                config = json.loads((folder / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
                ranks = {min(t["shape"]) for n, t in summary["shapes"].items() if "lora_A" in n or "lora_B" in n}
                if not ranks or ranks != {config["r"]}:
                    problems.append(f"adapter tensors do not match the configured LoRA rank {config['r']}")
    try:
        meta = clef.read_finetune_meta(folder)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [*problems, f"clef_finetune.json is unreadable: {error}"]
    if meta["base"] != clef.model_info(meta["base"].get("variant", "flash")):
        problems.append("the export was trained from a different base model than this code pins")
    temperature = meta.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or temperature <= 0:
        problems.append("the export has no usable calibration temperature")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args(argv)
    problems = verify(args.folder)
    if problems:
        print("MODEL CHECK FAILED", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    meta = clef.read_finetune_meta(args.folder)
    print(f"ok: {args.folder} ({len(read_sums(args.folder))} files verified); base {meta['base']['checkpoint']}@{meta['base']['revision'][:8]}, "
          f"temperature {meta['temperature']:.3f}, best epoch {meta.get('best_epoch')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
