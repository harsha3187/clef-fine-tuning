---
title: Public benchmark routing dataset
description: Download pinned SWE-bench and BFCL data, track the ToolBench blocker, and build input-only routing probes.
---

## What is available

| Source | Acquired task rows | Scope |
|--------|-------------------|-------|
| SWE-bench | 2,519 | Current full distribution: 225 dev and 2,294 test rows |
| BFCL | 4,696 | All 19 active V4 task categories, with available native answers and context resources |
| Original ToolBench | 10 | Official repository examples only, not the full dataset or evaluation split |

The acquisition retains licenses, source documentation, immutable revisions,
upstream sizes and hashes. The download manifest records exact URLs, downloaded
bytes, SHA-256 checksums and ToolBench availability checks. No native benchmark
code, environment setup script, evaluator or external tool API is executed.

Open [the public benchmark notebook](../notebooks/04_public_benchmarks.ipynb).
It verifies the snapshot, inspects source coverage, shows an exact Clef payload
and runs a small local smoke test. Azure inference is disabled by default.

The built snapshot contains 7,225 candidates, 5,489 automatically labeled draft
probes, and 5,484 probes within the endpoint's character limit. The draft labels
cover 1,559 debugging, 351 code-generation and 3,579 fallback requests. They do
not cover test generation, code review or repository QA.

All six notebook code cells have been executed and saved in the
[executed notebook](../data/public-routing/public-benchmarks-20260930/04_public_benchmarks.executed.ipynb).
The 20-record local smoke test completed without Azure calls.

## Dataset locations

These generated directories are ignored by version control:

* [Raw snapshot](../data/upstream/public-benchmarks-20260930/):
  downloaded source files and `download-manifest.json`.
* [Routing bundle](../data/public-routing/public-benchmarks-20260930/):
  normalized candidates, native targets, unreviewed probe records and `manifest.json`.

The bundle contains:

| Artifact | Purpose |
|----------|---------|
| `candidates.jsonl` | Every normalized initial request, source identity, native split, proposed label or abstention, and omission flags |
| `native-targets.jsonl` | Native patches, tests, expected calls or relevant-API labels, kept separate from router inputs |
| `draft-probes.jsonl` | Records with automatic proposed routes and supported input schemas |
| `endpoint-probes.jsonl` | Draft records whose full context-mode state fits the endpoint's 32,768-character bound |
| `excluded-from-probes.jsonl` | Candidate IDs excluded for ambiguous labels, unavailable native environments or sample-only status |
| `manifest.json` | Exact counts, artifact hashes, adapter hash, source revisions, split overlap and interpretation limits |

Exclusion-reason counts can overlap when a candidate has more than one reason.
The endpoint subset is not guaranteed to fit Clef's `max_length` budget. Inspect
`truncation_rate` alongside scores. Full inputs are retained rather than silently shortened.

## These are not six-route gold benchmarks

The source benchmarks measure issue resolution or tool use, not our six-agent
ownership decision. Their native gold answers cannot be treated as routing labels.

The versioned `public-request-intent-v1` policy proposes:

* `debugging` for SWE issues with explicit failure cues.
* `code_generation` for SWE issues with change or feature cues and no detected
  failure cue.
* `fallback` for general tool-use requests outside the coding contracts.
* An abstention when SWE intent is unclear or a tool-use request mentions coding.

These lexical proposals can be wrong. They remain `unreviewed`, with no reviewer,
and work only in probe mode. Agreement with them is not verified correctness.
The label policy overlaps lexical baselines, so a rules smoke test is not an
independent evaluation. The resulting dataset is imbalanced and does not provide
complete coverage of all six routes.

BFCL's native *irrelevance* category does not automatically imply our fallback
route. Available tool-call languages also do not imply a code-generation request.
The proposal policy examines the user request, not native expected calls or
category-based tool relevance.

Human review or an independently validated labeling process is still required
before freezing gold evaluation data or making a fine-tuning decision.

## Input and split boundaries

SWE inputs include the problem statement and repository/base-commit identity.
Patches, test patches, test names, evaluation scripts and hints are excluded.
Repository content has not been fetched: proposed labels describe issue intent,
not demonstrated downstream solvability.

BFCL inputs include the first user message, preceding system/developer messages
and explicitly supplied function schemas. Later messages, solution paths and
possible answers do not enter the router input. Multi-turn, memory and web-search
tasks without explicit function schemas are retained as candidates but excluded
from scored probes because their native environments are not reconstructed.

ToolBench examples include the query and all offered API descriptions. Gold
`relevant APIs` selections and response templates are excluded from inputs.
These examples are excluded from scored probes.

Native IDs, categories, revisions and splits remain attached to candidates.
Identical normalized queries share a group across categories. Native SWE dev/test
repository overlap and exact-input duplication are reported; no repository-disjoint
claim is made. The current SWE distribution has no train split. Legacy training
data, overlapping Lite/Verified subsets and inactive BFCL datasets are not silently
merged into this snapshot.

The notebook samples SWE native-dev records and BFCL evaluation records for
development. Any inspected or prompt-tuning records become development-exposed.
Do not later claim those records as untouched held-out tests.

## Reproduce the download and adaptation

Use the existing project environment and install the optional benchmark extra:

```powershell
uv sync --frozen --all-extras
uv run --no-sync python -m routing_lab.benchmark_sources --output data\upstream\public-benchmarks-20260930
uv run --no-sync python -m routing_lab.benchmark_data --raw data\upstream\public-benchmarks-20260930 --output data\public-routing\public-benchmarks-20260930
```

Each output directory must be new. To resume interrupted acquisition, repeat the
download command with `--resume`. Completed files are hash-verified rather than
silently replaced. Adaptation refuses incomplete or modified snapshots.

No dataset API key is needed. Do not install or provide RapidAPI, ToolBench,
OpenAI or search-provider credentials for this static routing preparation.
The existing Azure endpoint and Azure CLI sign-in are sufficient for optional
Clef inference; no additional Azure resource is needed.

## Original ToolBench archive blocker

The official Google Drive folder and archive link returned 404 during source
verification; one subsequent archive check timed out. The Tsinghua mirror
returned an HTML page saying "Link does not exist." The persisted download
manifest records the acquisition-time checks, including network failures.

The ten repository examples are not a replacement for the full archive.
A working, authorized full archive or official mirror is needed to complete
that acquisition. No community repack or similarly named benchmark was substituted.

## Sources and usage terms

* [SWE-bench dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench),
  revision `c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2`.
  The MIT repository license is retained, but the dataset card does not
  independently license every upstream issue and code fragment.
* [BFCL V4 source](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard),
  revision `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.
  Apache-2.0 license and documentation are retained. The older V3 Hugging Face
  mirror is not mixed into V4.
* [Original ToolBench](https://github.com/OpenBMB/ToolBench/tree/d56fdd89faf8c91fa135090b212bb9057ee5cfc2),
  revision `d56fdd89faf8c91fa135090b212bb9057ee5cfc2`.
  The Apache-2.0 license is retained. Its README additionally restricts usage
  to research and education.

Keep upstream attribution and review applicable terms before redistribution,
fine-tuning or a different deployment use. Public benchmark contamination in
pretraining is unknown; these probes alone cannot establish generalization.
