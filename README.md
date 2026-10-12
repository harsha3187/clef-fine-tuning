---
title: Coding-agent routing experiment
description: Evaluate Cloudflare's Clef decision model before deciding whether to fine-tune it.
---

## Start here

The experiment selects one initial owning agent from five coding roles plus fallback.
It does not generate patches, execute repository code, or grant permissions.
Clef-Flash has been LoRA fine-tuned on the frozen split and scored once on the test files: 95.3% on coding and 95.7% on tool/API, up from 80.8% and 73.6%. See [baseline results](docs/baseline-results.md) for the caveats.
The model is [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) (9B), with Clef (27B) selectable.
Clef runs on an Azure GPU, not this workstation. Local tests use deterministic baselines and
mocked model responses.

For the local notebook calling Clef over HTTPS, use the
[managed endpoint setup](docs/endpoint.md). It keeps all model weights and inference
on Azure; the local client needs only the `endpoint` dependency extra and Azure CLI
sign-in. The endpoint is prepared for East US 2 but deployed in East US 2 on one H100 and
verified live (59/60 on the synthetic probe). It bills until deleted. See the endpoint guide for status, sizing and costs.

1. Read the [agent contracts](contracts/agents.json).
2. Open the [experiment notebook](notebooks/01_routing_experiment.ipynb).
3. Provision the minimum resources in the [Azure guide](docs/azure.md).
4. Run the first 60-example off-the-shelf probe on Azure.
5. Compare original and expanded criteria in the notebook's
   [prompt-tuning experiment](docs/prompt-tuning.md), retaining corrections and regressions.
6. Follow the [review and evaluation protocol](docs/evaluation.md) before interpreting
   scores as a benchmark or deciding to fine-tune.

The included examples are original, agent-authored synthetic probes with proposed
labels. None is represented as human-reviewed gold data.

For genuine coding requests, use the [private session dataset workflow](docs/session-data.md).
It provides a local reviewer and explicit privacy/label gates before Azure evaluation.
Keep real chat data and executed reviewer notebooks under ignored private storage,
not in the shareable notebook template.

For a no-manual-labeling start, the [automatic session dataset](docs/session-data.md#automatic-silver-labeling-workflow)
uses recorded responses and agent telemetry to infer unreviewed silver labels.
Scores against these labels measure agreement, not verified task-routing correctness.

For public source data, use the [public benchmark workflow](docs/public-benchmarks.md)
and [inspection notebook](notebooks/04_public_benchmarks.ipynb). They download pinned
full SWE-bench and active BFCL V4 data, keep native answers separate, and export
unreviewed routing probes. Original ToolBench's full archive is currently blocked;
its ten available official examples remain separate from scored probes.

For the next stage, use the [expanded routing and annotation workflow](docs/annotation.md)
and [annotation notebook](notebooks/05_annotation.ipynb).
They separate a 14-role coding-agent track from a seven-role tool/API track and
prepare evidence-backed, blind two-pass GPT-5.6 Sol annotations using Microsoft
Entra authentication. Old keyword labels are not reused. Remote annotation and actual router evaluation are separate,
explicitly gated steps.

Current full-dataset labeling uses [strict-JSON batches](docs/annotation.md#current-execution-strict-json-batches)
of up to ten rows per request, with two blind passes and preserved prior results.

For the shared coding/tool router experiment, use the
[fine-tuning guide](docs/fine-tuning.md) and
[fine-tuning data notebook](notebooks/06_shared_router_finetuning.ipynb).
Training-derived partitions preserve the original test files and silver-label
provenance. Training bills a GPU, so it needs explicit consent flags and starts with a smoke job.

For a customer support agent and a travel planner that compare routing with and without the fine-tuned model, see the
[support desk proof of concept](docs/agent-demo.md).

For the capability router work (what a model-routing layer should ask Clef, the ten-endpoint benchmark and the
measured results), see [router architecture](docs/router-architecture.md). For turning four public routing datasets into one
canonical format and a fine-tuning bundle, see [routing datasets](docs/routing-datasets.md).
To run the app and the routing API in containers, or to call the router from another application, see
[running in containers](docs/docker.md); the fine-tuned Clef weights are in [models/](models/README.md). To work on the code in VS Code,
open the folder in the [dev container](docs/devcontainer.md).

## Environment

Use Python 3.11 and uv. The core environment includes notebook tools, a conventional
classifier and tests, but no Clef or PyTorch installation.

```text
uv sync --frozen
uv run --no-sync python -m routing_lab.cli validate --data data/probes.jsonl
uv run --no-sync python -m routing_lab.cli run --backend rules --probe --output outputs/rules-001
uv run --no-sync python -m pytest
uv run --no-sync python -m ruff check routing_lab tests
```

Each run directory must be new. Existing results are never overwritten. Select
`.venv` as the notebook kernel. The notebook's default path runs without network
access or credentials.

The lock file was resolved through the organization's anonymous
`packagefeedproxy.microsoft.io` package mirror because direct package downloads
failed TLS negotiation on this workstation. The mirror is explicitly configured in
[pyproject.toml](pyproject.toml), not hidden in shell state. Permit it on Azure.
If your approved environment uses PyPI instead, change the default index, regenerate
the lock with `uv lock`, and retain that new lock as part of the experiment version.
Do not silently mix package sources between comparison runs.

## What is implemented

| Surface | Capability |
|---------|------------|
| Contracts and data | Six agent contracts and 60 unreviewed synthetic probes |
| Data validation | Strict schema, reviewer identity requirement, duplicate ID rejection |
| Split preparation | Connected repository/task/exact-duplicate grouping |
| Baselines | Rules, constant fallback, training-set majority, TF-IDF/logistic regression |
| Model adapters | Pinned Clef/Clef-Flash checkpoint on a GPU; local-to-Azure Clef endpoint client; opt-in Azure OpenAI |
| Evaluation | Accuracy, macro-F1, per-route results, confusion matrix, slices, calibration diagnostics, coverage/error curves, timing and truncation |
| Prompt tuning | Versioned richer criteria, fixed baseline comparison, paired corrections/regressions, state truncation |
| Real session data | Read-only host-indexed extraction, prior-only context, redaction, private reviewer and hash-bound approval gates |
| Provenance | Dataset/config hashes, model revisions, environment versions, partial-run status |
| Downstream comparison | Import paired, independently graded task results and compare success/cost/time |
| Fine-tuning | LoRA plus joint-head trainer, validation-based selection, temperature calibration, export and reload check; staged Azure GPU jobs and a one-time frozen-test evaluation, all run once |
| Azure | GPU probe job and gated GPU managed online endpoint |
| Notebook | Interactive probes, custom requests, plots, reviewed splits and baseline comparisons |

The Clef managed endpoint has passed live inference verification. A fine-tuned deployment (`flash-ft-gpu`) runs beside it at 0% traffic; both bill until deleted. The optional Azure
OpenAI adapter still requires its own live integration verification.
The downstream importer is not an agent runner. Actual solving agents, isolated
execution environments, a grading harness, reviewed data and a GPU training run
remain later gated stages.

## Run artifacts

Every completed run writes:

* `manifest.json`: inputs, revisions, package versions, timing scope and completion status
* `predictions.jsonl`: record IDs, proposed labels, predictions and per-request timings
* `summary.json`: aggregate and slice metrics

Raw requests, code and credentials are not copied into those artifacts. Record IDs
and labels can still be sensitive. Keep the original input dataset securely because
its hash alone is not enough to reproduce a run. Run failures remain marked
`incomplete`, with any completed predictions retained for diagnosis.

Compare only equivalent datasets, contracts, input conditions and execution budgets.
There are no default claims about production accuracy, calibrated confidence,
Azure cost savings or fine-tuning benefits.
