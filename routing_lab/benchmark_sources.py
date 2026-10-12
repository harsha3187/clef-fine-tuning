"""Download pinned public benchmark data without executing benchmark code or APIs.

The original ToolBench full archive is probed separately. Its repository examples
are explicitly samples, never a substitute advertised as the full benchmark.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from routing_lab.evaluation import sha256

REVISIONS = {
    "swe_dataset": "c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2",
    "swe_code": "02e7a74ffd0b707aab73d203fe87bdc7c76afc8e",
    "bfcl": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
    "toolbench": "d56fdd89faf8c91fa135090b212bb9057ee5cfc2",
}
BFCL_ROOT = "berkeley-function-call-leaderboard/bfcl_eval/data/"
MAX_FILE_BYTES = 128 * 1024 * 1024


@dataclass(frozen=True)
class Resource:
    """One immutable source file and its upstream integrity metadata."""

    path: str
    url: str
    source: str
    role: str
    size: int | None = None
    git_blob_sha1: str | None = None
    sha256: str | None = None


def destination(root: Path, name: str) -> Path:
    """Reject absolute, traversal and Windows drive paths from upstream metadata."""
    parts = PurePosixPath(name).parts
    if not parts or PurePosixPath(name).is_absolute() or "\\" in name or any(
        part in {".", ".."} or ":" in part for part in parts
    ):
        raise ValueError("Unsafe upstream file path")
    return root.joinpath(*parts)


def _json(session, url: str):
    with session.get(url, timeout=(15, 90)) as response:
        response.raise_for_status()
        return response.json()


def _tree(session, repository: str, revision: str) -> list[dict]:
    tree = _json(session, f"https://api.github.com/repos/{repository}/git/trees/{revision}?recursive=1")
    if tree.get("truncated"):
        raise ValueError(f"GitHub tree is truncated for {repository}; cannot claim a full inventory")
    return [row for row in tree["tree"] if row["type"] == "blob"]


def make_plan(session) -> list[Resource]:
    """Inventory full SWE-bench and active BFCL V4 data plus ToolBench examples."""
    plan = []
    revision = REVISIONS["swe_dataset"]
    hf_root = "https://huggingface.co/datasets/SWE-bench/SWE-bench"
    files = _json(session, f"https://huggingface.co/api/datasets/SWE-bench/SWE-bench/tree/{revision}/data")
    for row in files:
        if row["type"] == "file" and row["path"].endswith(".parquet"):
            plan.append(Resource(
                f"swe-bench/{row['path']}", f"{hf_root}/resolve/{revision}/{row['path']}",
                "swe-bench", "input", size=row["size"], sha256=row.get("lfs", {}).get("oid"),
            ))
    if not plan:
        raise ValueError("No SWE-bench Parquet files found")
    plan.append(Resource("swe-bench/dataset-card.md", f"{hf_root}/resolve/{revision}/README.md",
                         "swe-bench", "attribution"))
    for repo, revision, source in (
        ("SWE-bench/SWE-bench", REVISIONS["swe_code"], "swe-bench"),
        ("ShishirPatil/gorilla", REVISIONS["bfcl"], "bfcl"),
        ("OpenBMB/ToolBench", REVISIONS["toolbench"], "toolbench"),
    ):
        rows = _tree(session, repo, revision)
        for row in rows:
            path = row["path"]
            relative, role = None, None
            if path in {"LICENSE", "NOTICE", "CITATION.cff"}:
                relative, role = path, "attribution"
            elif path == "README.md" and source != "bfcl":
                relative, role = "project-README.md", "attribution"
            elif source == "bfcl" and path == "berkeley-function-call-leaderboard/README.md":
                relative, role = "project-README.md", "attribution"
            elif source == "bfcl" and path.startswith(BFCL_ROOT):
                relative = path[len(BFCL_ROOT):]
                if relative.startswith("unused_datasets/"):
                    continue
                if relative.startswith("possible_answer/"):
                    role = "answer"
                elif relative.startswith(("multi_turn_func_doc/", "memory_prereq_conversation/")):
                    role = "context_resource"
                elif relative == "BFCL_v4_format_sensitivity.json":
                    role = "index"
                elif relative.startswith("BFCL_v4_") and relative.endswith(".json"):
                    role = "input"
                elif relative == "README.md":
                    role = "attribution"
            elif source == "toolbench" and path in {
                "data_example/instruction/G1_query.json",
                "data_example/instruction/G2_query.json",
                "data_example/instruction/G3_query.json",
            }:
                relative, role = path, "sample"
            if relative is not None and role is not None:
                plan.append(Resource(
                    f"{source}/{relative}", f"https://raw.githubusercontent.com/{repo}/{revision}/{path}",
                    source, role, row["size"], row["sha"],
                ))
    if len({resource.path for resource in plan}) != len(plan):
        raise ValueError("Duplicate download destinations")
    counts = Counter((resource.source, resource.role) for resource in plan)
    for key, count in {
        ("swe-bench", "input"): 2, ("bfcl", "input"): 19,
        ("bfcl", "answer"): 16, ("bfcl", "context_resource"): 17,
        ("toolbench", "sample"): 3,
    }.items():
        if counts[key] != count:
            raise ValueError(f"Pinned source inventory changed or is incomplete: {key}")
    return plan


def download_resource(session, resource: Resource, root: Path) -> dict:
    """Stream a bounded download, verify it, then publish it without overwriting."""
    target = destination(root, resource.path)
    if target.exists():
        raise FileExistsError(target)
    if resource.size is not None and resource.size > MAX_FILE_BYTES:
        raise ValueError("Source file exceeds the configured download bound")
    target.parent.mkdir(parents=True, exist_ok=True)
    with session.get(resource.url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", "").lower():
            raise ValueError("Source returned an HTML page instead of a data/attribution file")
        fd, name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
        partial = Path(name)
        try:
            count = 0
            with os.fdopen(fd, "wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    count += len(chunk)
                    if count > MAX_FILE_BYTES:
                        raise ValueError("Download exceeded its byte bound")
                    stream.write(chunk)
            if not count or (resource.size is not None and count != resource.size):
                raise ValueError("Download length does not match upstream metadata")
            digest = sha256(partial)
            if resource.sha256 is not None and digest != resource.sha256:
                raise ValueError("Download SHA-256 does not match upstream LFS metadata")
            if resource.git_blob_sha1 is not None:
                blob = hashlib.sha1(f"blob {count}\0".encode())
                with partial.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        blob.update(chunk)
                if blob.hexdigest() != resource.git_blob_sha1:
                    raise ValueError("Download does not match the pinned Git blob")
            partial.rename(target)
            return {**asdict(resource), "downloaded_bytes": count, "content_sha256": digest}
        finally:
            partial.unlink(missing_ok=True)


def probe_toolbench_archives(session) -> list[dict]:
    """Read bounded public responses; do not authenticate or download an unknown-size archive."""
    import requests

    urls = [
        "https://drive.google.com/drive/folders/1TysbSWYpP8EioFu9xPJtpbJZMLLmwAmL",
        "https://drive.google.com/uc?export=download&id=1XFjDxVZdUY7TXYF2yvzx3pJlS2fy78jk",
        "https://cloud.tsinghua.edu.cn/f/c9e50625743b40bfbe10/?dl=1",
    ]
    result = []
    for url in urls:
        try:
            with session.get(url, stream=True, timeout=(15, 45)) as response:
                prefix = next(response.iter_content(chunk_size=8192), b"")
                archive = prefix.startswith(b"PK\x03\x04")
                expired = b"Link does not exist" in prefix
                result.append({
                    "url": url, "http_status": response.status_code,
                    "content_type": response.headers.get("Content-Type"),
                    "zip_prefix_detected": archive, "expired_link_message": expired,
                    "status": "archive_detected_not_downloaded" if archive else "no_archive_retrieved",
                })
        except requests.RequestException as error:
            result.append({
                "url": url, "status": "network_error", "zip_prefix_detected": False,
                "error": str(error),
            })
    return result


def download_snapshot(output: Path, *, resume: bool = False) -> dict:
    """Download official pinned sources with resumable per-file hash validation."""
    import requests

    manifest_path = output / "download-manifest.json"
    with requests.Session() as session:
        session.headers["User-Agent"] = "routing-lab-public-data/1"
        if resume:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["revisions"] != REVISIONS:
                raise ValueError("Resume revisions differ from this download implementation")
            plan = [Resource(**row) for row in manifest["plan"]]
        else:
            plan = make_plan(session)
            output.mkdir(parents=True, exist_ok=False)
            manifest = {
                "status": "incomplete", "started_utc": datetime.now(timezone.utc).isoformat(),
                "revisions": dict(REVISIONS), "plan": [asdict(row) for row in plan], "files": [],
                "scope": "full current SWE-bench; all active BFCL V4 data resources; original ToolBench samples",
                "toolbench_full_status": "not_downloaded",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        completed = {row["path"]: row for row in manifest["files"]}
        for resource in plan:
            if resource.path in completed:
                if sha256(destination(output, resource.path)) != completed[resource.path]["content_sha256"]:
                    raise ValueError("A completed download changed; refusing to silently replace it")
                continue
            entry = download_resource(session, resource, output)
            manifest["files"].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"Downloaded {resource.path}: {entry['downloaded_bytes']} bytes", flush=True)
        if manifest["status"] == "complete_available_sources_toolbench_full_missing":
            return manifest
        manifest["toolbench_archive_checks"] = probe_toolbench_archives(session)
        detected = any(row["zip_prefix_detected"] for row in manifest["toolbench_archive_checks"])
        manifest.update(
            status="complete_available_sources_toolbench_full_missing",
            toolbench_full_status="archive_detected_requires_separate_acquisition" if detected else "blocked_official_links",
            finished_utc=datetime.now(timezone.utc).isoformat(),
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest


def main() -> None:
    """Download or resume a pinned source snapshot."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    download_snapshot(args.output, resume=args.resume)


if __name__ == "__main__":
    main()
