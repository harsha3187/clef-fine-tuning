---
title: Azure-hosted Clef with a local notebook
description: Prepare an authenticated GPU endpoint for Cloudflare Clef-Flash and call it locally without downloading model weights.
---

## Status

**Deployed and verified on 2026-10-06; a fine-tuned deployment was added on 2026-10-07 (below).** Endpoint `clef-routing-exp`, deployment `flash-gpu`, runs
`Cloudflare/clef-flash` on one `Standard_NC40ads_H100_v5` in `aml-clef-exp-eus2` (`eastus2`), with 100%
traffic and `aad_token` auth. Scoring URI:

```text
https://clef-routing-exp.eastus2.inference.ml.azure.com/score
```

**It bills about USD 6.98/hour (about USD 168/day) until the deployment is deleted.** See
[Delete when finished](#delete-when-finished).

The earlier deploy attempts failed on storage/datastore settings, then on H100 quota: the family quota
is subscription-wide, and a Laya smoke job on another workspace's cluster held 80 of the 100 vCPUs. It
succeeded once that job completed. Provisioning took about 30 minutes including the image build and the
19 GB model download.

## First live probe (2026-10-06)

All 60 requests of the synthetic probe completed against the live endpoint
([run](../outputs/clef-flash-endpoint-001/summary.json)). Compared with the Laya endpoint's first probe:

| Measurement | Laya English (UAE North, CPU) | Clef-Flash (East US 2, H100) |
|-------------|-------------------------------|------------------------------|
| Agreement with proposed labels | 30/60 (50%) | 59/60 (98.3%) |
| Macro-F1 | 0.4601 | 0.9833 |
| Client p50 latency | 4.321 s (before the later fixes) | 0.333 s |
| Client p95 latency | 5.368 s | 0.357 s |
| First request | not recorded | 6.4 s (token fetch and connection setup) |
| State truncation | 0% | 0% |
| ECE (10 bins) | not recorded | 0.185 |

These are unreviewed, agent-authored labels on a small synthetic set, not a gold benchmark. The labels
are easy by design, so 98% does not predict performance on the real split. The meaningful comparison is
the held-out 90/10 split in [baseline-results.md](baseline-results.md), which is the next run. ECE of
0.185 with 98% accuracy means the probabilities are under-confident; check calibration before setting
escalation thresholds. Latencies are not like-for-like: the Laya figure predates its latency fixes.

## Fine-tuned deployment (2026-10-07)

A second deployment, **`flash-ft-gpu`**, serves the fine-tuned Clef-Flash on the same endpoint and URL, on its
own `Standard_NC40ads_H100_v5`. It was created at **0% traffic**: the default route still serves the
off-the-shelf model in `flash-gpu`, so nothing changed for existing callers.

| | `flash-gpu` (default, 100% traffic) | `flash-ft-gpu` (0% traffic) |
|---|---|---|
| Model | `Cloudflare/clef-flash`, pinned | Same base plus the registered model `clef-flash-router-ft:1`: LoRA adapter, joint head and fitted temperature 0.937, merged at startup |
| Reported provenance | `{checkpoint, revision, variant}` | The same, plus `fine_tuned: {adapter_sha256, head_sha256, temperature, best_epoch, bundle_sha256}` |
| Offline test accuracy | 80.8% coding, 73.6% tool/API | 95.3% coding, 95.7% tool/API ([results](baseline-results.md#clef-flash-fine-tuned-on-the-same-test10-split-2026-10-07)) |

Verified on the live deployment: 40 sampled test rows (20 per track) returned the same choice as the offline
evaluation in every case, with no measurable probability difference, and each response carried the pinned adapter
hash. This is a parity check, not a new evaluation. The model asset points at the training job's export in
workspace storage (tags record the training job, evaluation job, bundle hash and adapter hash), and the deployment
sets `CLEF_FINETUNED=true`, so it refuses to start rather than silently serve the base model.

**Both deployments bill now: about USD 14/hour together** (USD 6.98 each). Target the fine-tuned one explicitly:

```powershell
uv run --no-sync python -m routing_lab.cli run --backend clef-endpoint --allow-remote --probe `
  --deployment flash-ft-gpu --adapter-sha256 bd6eb1be473a9b9f0f6c91bbc1f835d350351cd5695a5c3fd04ba21c67954e1f `
  --data data\probes.jsonl --output outputs\clef-ft-endpoint-001
```

The client sends the `azureml-model-deployment` header, and with `--adapter-sha256` it rejects any response that
is not the pinned fine-tuned model. To move traffic, or to stop paying for one deployment:

```powershell
.\.venv\Scripts\python.exe azure\online\manage.py deploy --approve-cost --deployment-name flash-ft-gpu `
  --deployment-config azure\online\deployment-finetuned.yml --set-traffic-percent 100
az ml online-deployment delete --name flash-gpu --endpoint-name clef-routing-exp --resource-group rg-clef-exp-eus2 --workspace-name aml-clef-exp-eus2
```

Register another export with `manage.py register-model --model-path azureml://datastores/... --tag KEY=VALUE`. The
H100 quota still has room for both (80 of 100 vCPUs), but another team's cluster in this subscription can take it.

## East US 2 workspace

| Item | Value |
|------|-------|
| Resource group / workspace | `rg-clef-exp-eus2` / `aml-clef-exp-eus2` (`eastus2`) |
| Storage | `amlclefestorage36c4c93ba`, public access `SecuredByPerimeter`, shared keys off |
| Network Security Perimeter | `nsp-clef-exp`: inbound rules for the egress IPs of the machines that call it and the same subscription; storage associated in `Enforced` mode |
| VNet | `vnet-clef-exp` / `subnet-compute`, with private endpoints and private DNS for storage blob and file |
| Image build | CPU cluster `clef-image-build` (`Standard_DS3_v2`, min 0, max 1) set as the workspace `image_build_compute`; its identity has Blob Data Contributor on storage and AcrPush on the registry |
| Datastores | All four system datastores switched to identity-based auth, because shared keys are disabled |

Why it looks like this: a subscription policy (`MCAPSGov Deploy and Modify Policies`) forces storage
`publicNetworkAccess` back to `Disabled`, so ordinary IP rules do not stick. The perimeter is the
sanctioned alternative, and it is what the earlier Laya workspace used. If this workstation's egress
IP changes, update the `allow-workstation-ip` access rule on the NSP, or uploads fail with
`AuthorizationFailure`. Environment registration can also return a transient
`DatabaseOverCapacity` 503 in East US 2: retry.

## What Clef is

[Clef](https://huggingface.co/Cloudflare/clef) and [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash)
are open-weight (Apache-2.0) decision models from Cloudflare, announced in
[Introducing Clef](https://blog.cloudflare.com/clef-decision-models). Given a `state` and a schema
of typed questions, they return a probability for every allowed option in one forward pass, with
no text generation and no output parsing. The API is compatible with Typesafe's Jev / SystemOne
`POST /v1/systemone` body, which is also the shape this project already used for Laya.

| | Clef-Flash (first test) | Clef |
|---|-------------------------|------|
| Base model | Qwen3.5-9B | Qwen3.8-27B |
| Parameters / weights | 9.4B, about 18.8 GB bf16 | 27B, about 54 GB bf16 |
| Reported median latency | 38.8 ms | 209 ms |
| Context window | 64k per the blog; the release encoder defaults to 16,384 tokens | same |
| Modalities | text, images, video (this project sends text only) | same |
| Pinned revision | `17f0b0ad64efb65d273590632833508766b2aae6` | `2f3de3dd85f379784083b0814d997ab627200f0c` |

Architecture: a frozen Qwen backbone with rank-256 LoRA adapters, plus a "joint schema head" that
routes evidence from the state to each option, lets all fields attend to each other, and scores
every option. Post-training used label-smoothed cross-entropy plus a Brier loss for calibration,
then RLCD, a reinforcement stage that gives partial credit to adjacent ordinal options.
Cloudflare's own Decision Index puts Laya far behind both Clef models (for example BANKING77
macro-F1 14.3 vs 94.2 for Clef), so a large gain over the Laya baseline is expected. Those are
Cloudflare's numbers, not ours; the benchmark here is still the project's own split.

Practical differences from Laya that shaped the code:

* **GPU only.** There is no CPU serving path. The endpoint is configured for one H100 (see
  [GPU sizing](azure.md#gpu-sizing)); the CPU `Standard_DS3_v2` deployments are gone.
* **Options are sorted.** `encode_record` sorts `choice` options by key, so option order cannot be
  varied. `--reverse-options` now applies only to the Azure OpenAI backend.
* **Truncation is silent upstream.** `encode_record` drops state tokens without reporting it. Our
  [clef.py](../routing_lab/clef.py) recomputes the dropped count so `truncation_rate` stays honest.
* **Custom code is pinned.** The release ships `joint_schema_model.py`, which we execute. `clef.load`
  checks its SHA-256 against the audited revision before importing it.
* **`torchvision` is mandatory** even for text: `AutoProcessor` fails without it. The environment
  pins `torch` 2.11.0 and `torchvision` 0.26.0 on CUDA 12.6 wheels, with `transformers` 5.10.2.

## Architecture

```text
Local VS Code / notebook
  -> Azure CLI sign-in / Microsoft Entra token
  -> Azure ML managed online endpoint (HTTPS)
  -> pinned Clef-Flash on one dedicated Azure GPU
  -> typed route probabilities and token/truncation telemetry
```

The endpoint is `clef-routing-exp`, deployment `flash-gpu`, in workspace `aml-clef-exp-eus2`
(`rg-clef-exp-eus2`, `eastus2`). Connection details live in
`azure/online/connection.json` (not committed: copy [connection.example.json](../azure/online/connection.example.json) and fill it in; set `AZURE_SUBSCRIPTION_ID` for `manage.py` and the training scripts); `scoring_uri` is filled in by `manage.py`
after a successful deployment.

## Request and response

Request (the scorer validates a single `route` choice over a complete taxonomy catalog):

```json
{
  "state": "User request: ...",
  "questions": {"route": {"type": "choice", "instructions": "...", "criteria": {"debugging": "...", "fallback": "..."}}},
  "max_length": 16384
}
```

`max_length` is optional (512 to 32,768) and defaults to the release's 16,384. No `images`,
`videos`, `max_state_tokens` or extra fields are accepted.

Response:

```json
{
  "answers": {"route": {"type": "choice", "choice": "debugging", "confidence": 0.91, "probabilities": {"debugging": 0.91}}},
  "usage": {"input_tokens": 412, "output_tokens": 0, "state_tokens_dropped": 0},
  "experiment_model": {"checkpoint": "Cloudflare/clef-flash", "revision": "17f0b0ad...", "variant": "flash"}
}
```

`experiment_model` is added by our scorer. The client rejects any response whose provenance does not
match the pinned release for the variant it expects.

## Verification done without Azure

* Unit tests cover the scorer, the endpoint client, the local backend and the CLI offline.
* The real `joint_schema_model.py`, real tokenizer and `transformers` 5.10.2 were run end to end on
  CPU against a tiny randomly initialised Qwen3.5 model with a random head. This confirmed loading,
  encoding, joint scoring, the SystemOne answer shape and the truncation telemetry. It says
  nothing about accuracy or GPU memory.
* **Not yet verified:** real weights, GPU memory at 16k tokens on the chosen SKU, latency, quota,
  and the CUDA 12.6 wheel against the host driver.

## Deploy (billable, needs approval)

Cost: `Standard_NC40ads_H100_v5` (the configured SKU, 94 GB) retails around USD 6.98/hour in
`eastus2`, about USD 168 per day. It fits Clef-Flash with room to spare and is the only available SKU
that can also hold Clef 27B. A10 and A100 quota in East US 2 is zero. There is no scale-to-zero: delete the deployment
when the experiment ends. Budget alerts do not stop resources.

Startup downloads about 19 GB from Hugging Face, so allow several minutes (the probes wait up to
about 27 minutes; Azure caps the failure threshold below 120). The endpoint uses `aad_token`, public network access, and a system-assigned
identity, as before.

```powershell
az login
az account set --subscription <your-subscription-id>
uv sync --frozen --all-extras
.\.venv\Scripts\python.exe azure\online\manage.py deploy --approve-cost
.\.venv\Scripts\python.exe azure\online\manage.py status
.\.venv\Scripts\python.exe -X utf8 azure\online\manage.py logs
```

`manage.py` packages only `score.py`, `routing_lab/__init__.py`, `taxonomy.py` and `clef.py`; no
data is uploaded. It registers the `clef-routing-gpu:2` environment, creates the endpoint and
deployment, routes 100% of traffic after success, and writes the scoring URI to `connection.json`.
It deletes a deployment stuck in `Failed` before recreating it, and does not grant roles.

The Azure CLI `ml` extension cannot install on this workstation, so use the SDK helper above.

To test Clef (27B) side by side, add a second deployment with `CLEF_VARIANT: "clef"` and an
H100 instance at 0% traffic using `--deployment-name` and `--deployment-config`.

## Verify after deploying

```powershell
$env:CLEF_ENDPOINT_URL = (Get-Content azure\online\connection.json | ConvertFrom-Json).scoring_uri
uv run --no-sync python -m routing_lab.cli run --backend clef-endpoint --allow-remote --probe --output outputs\clef-endpoint-001
```

Or select `clef-endpoint` in [the notebook](../notebooks/01_routing_experiment.ipynb) and set
`ALLOW_CLEF_ENDPOINT = True`. Start with the 60 synthetic probes, then the held-out split in
[baseline-results.md](baseline-results.md) so Clef is compared with Laya on identical rows.
Client timings include token acquisition, network, queueing and inference.

The local caller needs `Microsoft.MachineLearningServices/workspaces/onlineEndpoints/score/action`
(AzureML Data Scientist includes it). The client reuses one HTTP session, caches the Entra token
until five minutes before expiry, and retries the gateway's intermittent HTTP 424 up to eight times.
Other errors fail immediately and never echo response bodies.

## Delete when finished

```powershell
az ml online-deployment delete --name flash-gpu --endpoint-name clef-routing-exp --resource-group rg-clef-exp-eus2 --workspace-name aml-clef-exp-eus2
az ml online-endpoint delete --name clef-routing-exp --resource-group rg-clef-exp-eus2 --workspace-name aml-clef-exp-eus2
```

Deleting the deployment stops its compute billing. Storage and registry charges can remain.
