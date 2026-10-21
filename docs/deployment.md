---
title: Azure, endpoint, containers and dev container
description: Provision Azure resources, deploy and call the Clef endpoint, run the service in containers and work on the code in the dev container.
ms.date: 2026-10-20
---

The sections run from provisioning Azure resources, through the hosted Clef endpoint, to running the service in containers and developing in a dev container.

* [Azure resources for the first model probe](#azure-resources-for-the-first-model-probe)
* [Azure-hosted Clef with a local notebook](#azure-hosted-clef-with-a-local-notebook)
* [Running the Adaptive Intelligence Engine in containers](#running-the-adaptive-intelligence-engine-in-containers)
* [Dev container](#dev-container)

## Azure resources for the first model probe

Provision Azure infrastructure for notebooks and reproducible Clef evaluation on a GPU.

### Provision now

If you want to keep the notebook here and call Clef as an Azure-hosted API, follow
the [managed endpoint guide](#azure-hosted-clef-with-a-local-notebook) instead of the interactive-compute path.
The endpoint uses dedicated GPU serving compute, not `compute-laya-exp`. Its serving
charges continue independently of the notebook compute instance.

Clef is a transformer with 9B (Clef-Flash) or 27B (Clef) parameters, so model inference needs a GPU.
The notebook compute instance and the optional `router-cpu` cluster stay CPU-only; they run the
rules baseline, data preparation and the endpoint client. These are starting allocations, not
measured sizing guarantees.

| Resource | Requested setup | Purpose |
|----------|-----------------|---------|
| Resource group | One experiment-only resource group in your approved region | Cost and access boundary |
| Azure Machine Learning workspace | One workspace in the same region | Notebook compute and experiment jobs |
| Workspace dependencies | Standard storage account, Key Vault and Application Insights; let the workspace creation workflow create approved defaults | Data, identity integration and logs |
| CPU compute instance | Linux, 4 vCPU, 16 GB RAM; `Standard_D4s_v5` or an available equivalent; at least 64 GB usable disk | Interactive notebook and first model probe |
| CPU compute cluster, optional | Name `router-cpu`, same CPU class, min nodes 0, max nodes 1, idle scale-down about 120 seconds | Run the supplied reproducible job instead of an interactive session |
| Container registry | Workspace-associated ACR when required for custom environment builds | Azure ML environment image storage |
| Budget | Resource-group budget and alerts; compute-instance shutdown schedule | Avoid idle spend; alerts do not stop compute automatically |

You can start with the compute instance alone. The command-job template specifically
expects a cluster named `router-cpu`; provision the cluster only if using that path.
Stop the interactive instance when finished. Cluster scale-to-zero does not eliminate
storage, registry or monitoring charges.

Verify the selected SKU is supported by Azure ML in your region and that your
subscription has enough regional and VM-family vCPU quota. No Azure OpenAI resource
is needed for the first Clef probe.

### Access and network prerequisites

* Give the experiment operator appropriate Azure ML workspace access, such as
  AzureML Data Scientist, plus only the separate permissions needed to create
  compute. Resource provisioning permissions and model-use permissions differ.
* Give the executing identity access to the workspace datastore, typically Storage
  Blob Data Contributor scoped to the experiment storage where needed.
* Permit approved outbound package/model downloads: the configured package mirror,
  Hugging Face and its resolved artifact/CDN hosts,
  PyTorch's wheel host, and the Microsoft container registry.
* For restricted networks, have the platform team mirror the pinned artifacts into
  approved storage and build a reviewed environment image. Do not disable TLS checks.
* Do not place credentials in notebooks, YAML, model state, or committed files.
* Use synthetic/public approved inputs first. Authorize internal code separately.

The project upload excludes the original Word document, local environment, secrets,
model caches and private reviewed datasets through `.amlignore`. Inspect the upload
contents when adding files. Register reviewed data separately as an access-controlled,
versioned Azure ML data asset.

### Interactive GPU notebook path

Use an Azure ML GPU compute instance (see [GPU sizing](#gpu-sizing)). Copy the project to it,
preserving its directory structure. Run these commands from the project root in the
instance terminal:

```text
python -m pip install uv==0.12.18
uv sync --frozen --python 3.11
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.11.0 torchvision==0.26.0
uv pip install --group clef
uv run --no-sync python -m ipykernel install --user --name routing-lab --display-name "Routing lab (Azure)"
uv run --no-sync python -m routing_lab.cli validate --data data/probes.jsonl
```

Open [the notebook](../notebooks/01_routing_experiment.ipynb) in Azure ML Studio or
VS Code connected to that compute. Select `Routing lab (Azure)`. In the backend
cell set `BACKEND = 'clef'`, `RUNNING_ON_AZURE = True` and `DEVICE = 'cuda'`.
Run all cells; the reviewed-data and remote-LLM sections remain disabled.

The first load downloads the pinned public checkpoint (about 19 GB for Clef-Flash, 54 GB for
Clef). Allow disk space and time for packages, weights and caches. The core lock excludes
model dependencies on purpose. The separate Clef requirements pin Torch, torchvision and
Transformers; other model-only transitive dependencies are not fully locked yet. Each run records
critical installed versions. After the first validated Azure run, freeze the full
model environment before claiming cross-machine reproducibility.

Use `--no-sync` after installing model extras: an exact `uv sync` may remove packages
that are not part of the core lock. Reinstall the model requirements after resyncing.

Equivalent CLI probe:

```text
uv run --no-sync python -m routing_lab.cli run --backend clef --probe --allow-model-download --device cuda --output outputs/clef-flash-001
uv run --no-sync python -m routing_lab.cli run --backend clef --probe --allow-model-download --variant clef --output outputs/clef-27b-001
uv run --no-sync python -m routing_lab.cli run --backend clef --probe --allow-model-download --context-mode query --output outputs/clef-flash-query-001
```

### Azure ML command-job path

Use Azure CLI with its `ml` extension and authenticate to your subscription.
From the project root, submit:

```powershell
az ml job create --file azure\probe-job.yml --resource-group <resource-group> --workspace-name <workspace>
```

The [job template](../azure/probe-job.yml) runs the Clef-Flash probe on a GPU cluster named `clef-gpu`
(create it first, minimum nodes 0, maximum 1) and the rules baseline.
It stores their output folders in the Azure ML job's `results` output. Download and
inspect the manifests before treating the job as successful. A failed adapter call
fails the job rather than substituting a fake model result.

The image tag is currently `latest` for initial environment bootstrapping. Once the
first job succeeds, register and reuse that built Azure ML environment version or
pin its image digest. Schema validation does not prove that your subscription can
build the image, download dependencies or allocate compute.

### GPU sizing

Weights are bfloat16: Clef-Flash is about 18.8 GB and Clef about 54 GB, before activations. The
pinned stack (`torch` 2.11, `transformers` 5.10.2) is the one the model authors tested on an H200.

| SKU (East US 2) | GPU | Fits Clef-Flash | Fits Clef | Retail price | Quota |
|-----------------|-----|-----------------|-----------|--------------|-------|
| `Standard_NC40ads_H100_v5` | 1 x H100, 94 GB | Yes | Yes | about USD 6.98/hour | 100 vCPUs, shared subscription-wide |
| `Standard_NV36ads_A10_v5` | 1 x A10, 24 GB | Short inputs only | No | about USD 4.58/hour | 0 |
| A100 SKUs | | | | | 0 |

Quota is for the whole subscription: another team's H100 cluster counts against the same 100 vCPUs
(`Standard_NC80adis_H100_v5` alone takes 80), so check `az ml` usage or attempt a deployment before
planning around it. UAE North has no usable GPU quota.

The inference environment installs CUDA 12.6 wheels, so it needs a host driver that supports CUDA 12.6.
Select `DEVICE = 'cuda'` in the notebook or `--device cuda` in the CLI. No automatic CPU fallback is
used when CUDA fails.

Transformers prints that the "fast path" for Qwen3.5's linear-attention layers is unavailable
unless `flash-linear-attention` and `causal-conv1d` are installed. The torch fallback is correct
but slower and uses more memory on long inputs. Treat those packages as a later latency
optimization; they need a CUDA toolchain to build and are not in the pinned environment.

### Optional strong-LLM comparison

Provision an approved Azure OpenAI deployment that supports Chat Completions JSON
mode. Record the deployed model version, region, quotas and pricing separately.
Assign the executing identity Cognitive Services OpenAI User at the resource scope.
The adapter uses Entra ID, not a stored API key, and disables automatic retries so
failed requests are visible.

```text
uv sync --frozen --extra llm
```

Set these environment variables in the Azure session:

* `AZURE_OPENAI_ENDPOINT`: your resource endpoint
* `AZURE_OPENAI_DEPLOYMENT`: the deployment name, not an assumed catalog model ID
* `AZURE_OPENAI_API_VERSION`: an API version supported by your deployment

The explicit opt-in command is:

```text
uv run --no-sync python -m routing_lab.cli run --backend azure --allow-remote --probe --output outputs/azure-llm-001
```

Only submit approved data. This is an Azure OpenAI reference router, not Jev.
No Jev integration or external third-party data transmission is configured.

### Information needed to start the live run

Provide the subscription ID, resource group, workspace name, region, compute
name/SKU and whether access is via Azure ML Studio or VS Code remote. These are
resource identifiers, not credentials. Also confirm permitted download endpoints.
Do not send keys, tokens or connection strings.

### References

* [Azure ML command-job schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2)
* [Azure ML compute instances](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance?view=azureml-api-2)
* [Azure ML compute clusters](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?view=azureml-api-2)
* [Clef-Flash on Hugging Face](https://huggingface.co/Cloudflare/clef-flash)
* [Clef on Hugging Face](https://huggingface.co/Cloudflare/clef)
* [Cloudflare announcement](https://blog.cloudflare.com/clef-decision-models)

## Azure-hosted Clef with a local notebook

Prepare an authenticated GPU endpoint for Cloudflare Clef-Flash and call it locally without downloading model weights.

### Status

**Deployed and verified on 2026-10-06; a fine-tuned deployment was added on 2026-10-07 (below).** Endpoint `<endpoint-name>`, deployment `flash-gpu`, runs
`Cloudflare/clef-flash` on one `Standard_NC40ads_H100_v5` in `<workspace>` (`eastus2`), with 100%
traffic and `aad_token` auth. Scoring URI:

```text
https://<endpoint-name>.<region>.inference.ml.azure.com/score
```

**It bills about USD 6.98/hour (about USD 168/day) until the deployment is deleted.** See
[Delete when finished](#delete-when-finished).

The earlier deploy attempts failed on storage/datastore settings, then on H100 quota: the family quota
is subscription-wide, and a Laya smoke job on another workspace's cluster held 80 of the 100 vCPUs. It
succeeded once that job completed. Provisioning took about 30 minutes including the image build and the
19 GB model download.

### First live probe (2026-10-06)

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
the held-out 90/10 split in [baseline results](data-and-training.md#off-the-shelf-baseline-results-before-fine-tuning), which is the next run. ECE of
0.185 with 98% accuracy means the probabilities are under-confident; check calibration before setting
escalation thresholds. Latencies are not like-for-like: the Laya figure predates its latency fixes.

### Fine-tuned deployment (2026-10-07)

A second deployment, **`flash-ft-gpu`**, serves the fine-tuned Clef-Flash on the same endpoint and URL, on its
own `Standard_NC40ads_H100_v5`. It was created at **0% traffic**: the default route still serves the
off-the-shelf model in `flash-gpu`, so nothing changed for existing callers.

| | `flash-gpu` (default, 100% traffic) | `flash-ft-gpu` (0% traffic) |
|---|---|---|
| Model | `Cloudflare/clef-flash`, pinned | Same base plus the registered model `clef-flash-router-ft:1`: LoRA adapter, joint head and fitted temperature 0.937, merged at startup |
| Reported provenance | `{checkpoint, revision, variant}` | The same, plus `fine_tuned: {adapter_sha256, head_sha256, temperature, best_epoch, bundle_sha256}` |
| Offline test accuracy | 80.8% coding, 73.6% tool/API | 95.3% coding, 95.7% tool/API ([results](data-and-training.md#clef-flash-fine-tuned-on-the-same-test10-split-2026-10-07)) |

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
az ml online-deployment delete --name flash-gpu --endpoint-name <endpoint-name> --resource-group <resource-group> --workspace-name <workspace>
```

Register another export with `manage.py register-model --model-path azureml://datastores/... --tag KEY=VALUE`. The
H100 quota still has room for both (80 of 100 vCPUs), but another team's cluster in this subscription can take it.

### East US 2 workspace

| Item | Value |
|------|-------|
| Resource group / workspace | `<resource-group>` / `<workspace>` (`eastus2`) |
| Storage | `amlclefestorage36c4c93ba`, public access `SecuredByPerimeter`, shared keys off |
| Network Security Perimeter | `<perimeter-name>`: inbound rules for the egress IPs of the machines that call it and the same subscription; storage associated in `Enforced` mode |
| VNet | `vnet-clef-exp` / `subnet-compute`, with private endpoints and private DNS for storage blob and file |
| Image build | CPU cluster `clef-image-build` (`Standard_DS3_v2`, min 0, max 1) set as the workspace `image_build_compute`; its identity has Blob Data Contributor on storage and AcrPush on the registry |
| Datastores | All four system datastores switched to identity-based auth, because shared keys are disabled |

Why it looks like this: a subscription policy (`MCAPSGov Deploy and Modify Policies`) forces storage
`publicNetworkAccess` back to `Disabled`, so ordinary IP rules do not stick. The perimeter is the
sanctioned alternative, and it is what the earlier Laya workspace used. If this workstation's egress
IP changes, update the `allow-workstation-ip` access rule on the NSP, or uploads fail with
`AuthorizationFailure`. Environment registration can also return a transient
`DatabaseOverCapacity` 503 in East US 2: retry.

### What Clef is

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
  [GPU sizing](#gpu-sizing)); the CPU `Standard_DS3_v2` deployments are gone.
* **Options are sorted.** `encode_record` sorts `choice` options by key, so option order cannot be
  varied. `--reverse-options` now applies only to the Azure OpenAI backend.
* **Truncation is silent upstream.** `encode_record` drops state tokens without reporting it. Our
  [clef.py](../routing_lab/clef.py) recomputes the dropped count so `truncation_rate` stays honest.
* **Custom code is pinned.** The release ships `joint_schema_model.py`, which we execute. `clef.load`
  checks its SHA-256 against the audited revision before importing it.
* **`torchvision` is mandatory** even for text: `AutoProcessor` fails without it. The environment
  pins `torch` 2.11.0 and `torchvision` 0.26.0 on CUDA 12.6 wheels, with `transformers` 5.10.2.

### Architecture

```text
Local VS Code / notebook
  -> Azure CLI sign-in / Microsoft Entra token
  -> Azure ML managed online endpoint (HTTPS)
  -> pinned Clef-Flash on one dedicated Azure GPU
  -> typed route probabilities and token/truncation telemetry
```

The endpoint is `<endpoint-name>`, deployment `flash-gpu`, in workspace `<workspace>`
(`<resource-group>`, `eastus2`). Connection details live in
`azure/online/connection.json` (not committed: copy [connection.example.json](../azure/online/connection.example.json) and fill it in; set `AZURE_SUBSCRIPTION_ID` for `manage.py` and the training scripts); `scoring_uri` is filled in by `manage.py`
after a successful deployment.

### Request and response

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

### Verification done without Azure

* Unit tests cover the scorer, the endpoint client, the local backend and the CLI offline.
* The real `joint_schema_model.py`, real tokenizer and `transformers` 5.10.2 were run end to end on
  CPU against a tiny randomly initialised Qwen3.5 model with a random head. This confirmed loading,
  encoding, joint scoring, the SystemOne answer shape and the truncation telemetry. It says
  nothing about accuracy or GPU memory.
* **Not yet verified:** real weights, GPU memory at 16k tokens on the chosen SKU, latency, quota,
  and the CUDA 12.6 wheel against the host driver.

### Deploy (billable, needs approval)

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

### Verify after deploying

```powershell
$env:CLEF_ENDPOINT_URL = (Get-Content azure\online\connection.json | ConvertFrom-Json).scoring_uri
uv run --no-sync python -m routing_lab.cli run --backend clef-endpoint --allow-remote --probe --output outputs\clef-endpoint-001
```

Or select `clef-endpoint` in [the notebook](../notebooks/01_routing_experiment.ipynb) and set
`ALLOW_CLEF_ENDPOINT = True`. Start with the 60 synthetic probes, then the held-out split in
[baseline results](data-and-training.md#off-the-shelf-baseline-results-before-fine-tuning) so Clef is compared with Laya on identical rows.
Client timings include token acquisition, network, queueing and inference.

The local caller needs `Microsoft.MachineLearningServices/workspaces/onlineEndpoints/score/action`
(AzureML Data Scientist includes it). The client reuses one HTTP session, caches the Entra token
until five minutes before expiry, and retries the gateway's intermittent HTTP 424 up to eight times.
Other errors fail immediately and never echo response bodies.

### Delete when finished

```powershell
az ml online-deployment delete --name flash-gpu --endpoint-name <endpoint-name> --resource-group <resource-group> --workspace-name <workspace>
az ml online-endpoint delete --name <endpoint-name> --resource-group <resource-group> --workspace-name <workspace>
```

Deleting the deployment stops its compute billing. Storage and registry charges can remain.

## Running the Adaptive Intelligence Engine in containers

Two images, one compose file:

| Image | What it is | Needs |
|---|---|---|
| `adaptive-intelligence-engine` (`Dockerfile`) | The web app (support desk and travel planner), the routing API and the shadow selector. 1.15 GB with the Azure CLI, 264 MB without (--build-arg WITH_AZ_CLI=0, for a service principal or managed identity). | Docker with Compose 2.24 or newer, an Azure sign-in |
| `clef-server` (`docker/clef-server.Dockerfile`, optional) | The fine-tuned Clef from `models/` served locally with the same scorer and HTTP server Azure ML runs. | An NVIDIA GPU, 18 GB for the base model |

The app does not contain a model. It calls Clef on the Azure ML endpoint in `azure/online/connection.json` (not committed: copy `connection.example.json`, or set `CLEF_ENDPOINT_URL`) or the local `clef-server`
and your Azure OpenAI deployments. So the app container needs an Azure identity and network access to both.

### Start it

```sh
docker compose up --build -d           # builds the app image and starts it on http://localhost:8765
docker compose exec app az login --use-device-code    # sign in once; the session is kept in a volume
```

Open <http://localhost:8765>. Until you sign in, the page loads but every model call fails and the header shows the capability router
as "not answering". The container's first log lines say which sign-in it found:

```sh
docker compose logs app
```

#### Signing in when device-code login is blocked

Many tenants block the device-code flow, so `az login --use-device-code` inside the container fails. Use the host's own `az login` instead,
through a small token broker that runs on the host:

```sh
az login                                              # on the host, as you normally do
python -m routing_lab.token_broker --write-env .env   # on the host; leave it running
docker compose up -d                                  # the container picks the settings up from .env
```

The broker speaks the managed-identity protocol that every Azure SDK already understands, so the container needs no login and no secret
file: it asks `host.docker.internal:8788` for a token whenever it needs one, and the broker answers with `az account get-access-token`. Tokens
are fetched on demand and refreshed automatically for as long as the broker runs and the host stays signed in.

What it will and will not do:
- It listens on **loopback only**, answers only requests carrying a random per-run secret (written to `.env`, which git ignores), and hands out
  tokens only for the Azure ML endpoint (`https://ml.azure.com`) and the Azure OpenAI/Foundry API (`https://ai.azure.com`). Anything else, such as the
  management API, is refused. Add a resource only on purpose, with `--allow`.
- Anything that can reach that port with the secret can act as you on those two resources until you stop the broker. Run it only on a
  machine you control, and stop it (Ctrl+C) when you are done. Restarting it creates a new secret, so run `docker compose up -d` again to pick it up.
- This is for a developer machine. On Azure, use a real managed identity (`AIE_AZURE_AUTH=managed-identity`); for a server or CI, use a service principal.
 (a server, CI, another machine): create a `.env` from `.env.example` with `AZURE_TENANT_ID`,
`AZURE_CLIENT_ID` and `AZURE_CLIENT_SECRET` for a service principal that can call the Azure ML endpoint and the Azure OpenAI resource. On
Azure (Container Apps, AKS, a VM) use a managed identity instead and set `AIE_AZURE_AUTH=managed-identity`. `AIE_AZURE_AUTH` is
`cli` (the default outside compose), `default` (the standard chain: service principal, managed identity, then the CLI session; the
compose default) or `managed-identity`.

#### If every call fails with "not authorized by network security perimeter"

The Clef workspace only accepts calls from allowed IP addresses (`<perimeter-name>`, rule `allow-workstation-ip`). A container reaches Azure
from its host's public address, and a home or office address changes. Add the address, from a shell that is signed in to Azure:

```sh
IP=$(curl -s https://api.ipify.org)
# then add "$IP/32" to the addressPrefixes of the rule (az rest PUT, same as the existing three entries)
```

Another host in the same subscription is already allowed by the `allow-same-subscription` rule.

### What is in the app image

Only what the service reads at run time: `agent_demo/` (server, agents, UI), `routing_lab/` (Clef client, selector), `capability_lab/`, `contracts/`,
`data/registry/measured.json` (the measured endpoint pass rates the selector uses) and `azure/online/connection.json`. Runtime dependencies
are `pydantic`, `requests` and `azure-identity` (the `app` group in `pyproject.toml`). The image runs as a non-root user, with a read-only root filesystem,
no Linux capabilities and `no-new-privileges`; only `/data` (telemetry, shadow log, saved preferences) and `~/.azure` are writable, as named volumes.

Check an image: `docker build --target test -t aie-test . && docker run --rm aie-test` runs 335 tests inside a clean container, with
no GPU and no cloud access.

### Using it from another application

The service is also an API. It listens on `127.0.0.1:8765` of the host, never on the network; to let another container call it, put both on a compose
network, add the caller's name to the allowed hosts, and keep the port unpublished:

```yaml
services:
  app:
    environment:
      AIE_ALLOWED_HOSTS: my-other-app
  my-other-app:
    # reaches the router at http://app:8765
```

**Which endpoint would serve this request?**

```sh
curl -s http://localhost:8765/v1/select -H "Content-Type: application/json" -d '{
  "request": "A customer wants a refund 45 days after delivery, but the product has a 12-month warranty. What do we owe them?",
  "needs_tools": true,
  "preferences": {"quality": "good", "latency": "medium", "cost": "low"},
  "allow": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"]
}'
```

The reply carries Clef's probabilities (`tier`, `tier_probabilities`), the `choice`, its `expected_success`, the runner-up endpoints with estimated cost and
latency, the endpoints ruled out and why, whether the router was unsure, and the model, prompt, registry and selector versions. Fields: `request`
(required), `preferences` (`quality` average, good or very_good; `latency` and `cost` low, medium or high), `needs_tools`, `kind` (plan, step or reply),
`allow` (endpoint ids to consider), `input_tokens` and `output_tokens` (to estimate cost). A `503` means Clef did not answer.

The measured registry (`data/registry/measured.json`) decides which endpoints exist and how well each does at each difficulty. It describes this
project's ten endpoints; for yours, build one with `python -m capability_lab registry` after benchmarking them.

From Python, without the service, use the client the service itself uses:

```python
from routing_lab.capability import CapabilityRouter
router = CapabilityRouter("https://<endpoint>.<region>.inference.ml.azure.com/score", "flash-cap-ft3",
                          adapter_sha256="3651988bb592eda963ccf96afe38ec89068840c26464094b1f5f09e481ab6beb")
print(router.classify("What is the molar mass of Ba(NO3)2?").to_dict())
```

The client checks every reply against the pinned base model and adapter hash, and refuses answers from any other model.

### Run Clef itself in a container (optional)

```sh
docker compose --profile local-clef up --build clef-server        # needs an NVIDIA GPU and the NVIDIA Container Toolkit
```

then in `.env` set `AIE_SHADOW_LOCAL=1` and `AIE_SHADOW_URL=http://clef-server:5001/score`, and restart `app`. `clef-server` checks
the files in `models/clef-flash-capability-ft-v3` against `SHA256SUMS` before it starts, downloads the 18 GB base model (Apache-2.0,
pinned revision) into a volume on first use, and serves `POST /score` exactly like the Azure endpoint. The app sends no Azure token to it. To
bake the model into the image instead of mounting it: `docker build -f docker/clef-server.Dockerfile --target server-baked -t clef-server:baked .`
once the weights are in `models/clef-flash-capability-ft-v3` (see `models/README.md`).

Any other client can call it directly. The request must carry Clef's question text (instructions and tier descriptions) exactly as the model
was trained on it; `CapabilityRouter` builds that for you, and `routing_lab/backends.py` is the source.

What is and is not verified for this image is in [the verification notes](#what-was-verified) below.

### Settings

Every command-line option of `python -m agent_demo.server` has an `AIE_*` environment variable:

| Variable | Default | Meaning |
|---|---|---|
| `AIE_HOST`, `AIE_PORT` | `0.0.0.0` in the image, `8765` | Where it listens inside the container |
| `AIE_ALLOW_NON_LOOPBACK` | `1` in the image | Required to bind beyond loopback. Without it the server refuses |
| `AIE_ALLOWED_HOSTS` | empty | Extra `Host` names accepted, for other containers. Any other Host gets 403 |
| `AIE_AZURE_AUTH` | `default` in compose | `cli`, `default` or `managed-identity` |
| `CLEF_ENDPOINT_URL` | from `connection.json` | Azure ML scoring URL for Clef |
| `AIE_SHADOW_DEPLOYMENT`, `AIE_SHADOW_ADAPTER_SHA256` | `flash-cap-ft3`, the v3 adapter hash | Which Clef deployment the shadow router calls and which weights it must report |
| `AIE_SHADOW_LOCAL`, `AIE_SHADOW_URL` | off | Use the local `clef-server` |
| `AIE_APPLY_SELECTOR` | `1` | Adaptive routing runs each step on the endpoint the capability router and selector choose. `0` keeps it on the old step-type router (whose deployment no longer exists, so every step falls back to the strongest model) |
| `AIE_ROUTE_PLANNER`, `AIE_ROUTE_JUDGE` | `1` | The decision model also selects the planner and the reviewer for Adaptive routing (`0`: the fixed GPT-4.1). Always strongest retains the fixed reviewer. See "Routing the planner and the reviewer" in the router architecture |
| `AIE_NO_SHADOW`, `AIE_NO_REPLAY`, `AIE_SHADOW_USE_SKILL` | off | Turn the shadow selector or its replay off, or use the skill head |
| `AIE_LLM_ENDPOINT` | none: set it in `.env` | Your Azure AI resource, `https://<name>.services.ai.azure.com/openai/v1/`. The agents and the probe scripts call it |
| `AIE_RUNTIME_DIR` | `/data/runtime` | Where telemetry and the shadow log are written |

### Security notes

The service holds cloud credentials and can spend money (a travel plan costs roughly $0.4 to $1). It was built for localhost, and the container keeps that:
the compose file publishes the port on `127.0.0.1` only, the Host header is checked, and state-changing requests must be JSON. Do not publish it on
`0.0.0.0` or the internet without an authenticating reverse proxy in front: there are no user accounts.

### What was verified

On this machine (Windows on ARM, Docker 28, no GPU, not signed in to Azure from the container):

**App image**
- The test image builds and 335 tests pass in a clean container (`docker run --rm aie-test`), which also proves `pydantic`, `requests` and `azure-identity`
  are the complete runtime dependency set.
- The service image builds (1.15 GB; 264 MB without the Azure CLI, which also runs as a service in managed-identity mode) and a code-only rebuild takes about 16 seconds.
- Started with compose: the container reports healthy and the page, static files and API answer on `localhost:8765`.
- A wrong `Host` header gets 403, a non-JSON POST gets 415, and a call with no Azure sign-in returns a clean 503 without internals.
- The port is not reachable from the machine's LAN address.
- The root filesystem is read-only, the process runs as uid 10001 with no capabilities, and `az login --use-device-code` starts inside it.
- The header shows the capability router and, separately, a red "Step router down" chip, because the old step-type router's deployment no longer exists.

**Running against Azure, through the token broker**
- Inside the container, the real Azure SDK gets valid tokens for the Azure ML endpoint and the Azure OpenAI API from the host broker, and a request for the management API is refused.
- `POST /v1/select` reaches the Azure ML endpoint from the container: Clef answers with the v3 adapter hash verified, in about 0.4 s, and the selector chooses an endpoint.
- A support-desk case and a travel plan both ran end to end through the container with no errors: both lanes completed every step, the capability router
  answered for every step, the replay on the selector's choices matched the strongest model on the expected-outcome checks (2 of 2 and 4 of 4), and the header badge turned green.

**Model files and the Clef server image**
- The files in `models/` verify against `SHA256SUMS` and match the adapter hash, head hash, temperature, best epoch and training-bundle hash that the live Azure endpoint reports for the deployed model.
- The Clef server image builds (3.2 GB). Its packages resolve to the same versions as the Azure environment (`torch 2.11.0+cu126`, `transformers 5.10.2`, `peft 0.21.1`, `azureml-inference-server-http 1.5.0`).
- Started with `models/` mounted, it verifies the four files (24 seconds through a Windows bind mount), starts the Azure inference server with the scorer and one worker, and with no GPU stops in seconds
  with a clear message instead of downloading the 18 GB base model.
- With Git LFS pointer files, or a missing or changed file, it refuses to start and says which file and what to do.
- The scorer under the real inference server library returns sanitized 400s for bad requests and a 503 `not_ready` when no model is loaded.

**Not verified here**, because they need a signed-in Azure session or an NVIDIA GPU:
- Loading the model in `clef-server` and answering a real request. The Azure endpoint serves these exact files, but this image has not run on a GPU.
- The `amd64` build. This machine builds `arm64`; the Dockerfiles use only multi-arch base images and the PyTorch CUDA 12.6 index, so `docker build --platform linux/amd64` should work, but it was not run.

## Dev container

A VS Code dev container for working on this repository: Python 3.11, the service and test dependencies, ruff, Git LFS and the Azure CLI. It has no GPU and no model weights.
Clef itself runs on the Azure endpoint (or the optional `clef-server` container, see [running in containers](#running-the-adaptive-intelligence-engine-in-containers)).

### Open it

1. Install Docker and the VS Code Dev Containers extension.
2. Open the repository folder and choose **Reopen in Container** (or run `devcontainer up --workspace-folder .` with the `@devcontainers/cli` package).
3. The first build takes a few minutes. It installs the `app`, `test` and `devcontainer` dependency groups from `pyproject.toml` into the image.

### Use it

```text
python -m pytest -q                  the whole suite (about 1,260 tests, under two minutes)
python -m ruff check agent_demo routing_lab capability_lab routing_data azure tests     lint the code (notebooks are left out)
python -m agent_demo.server          the app on http://localhost:8765 (VS Code forwards the port)
```

If another process already uses port 8765 on your machine (for example the compose service), VS Code forwards the dev container's port to a different local port and tells you which.

### Signing in to Azure

Device-code login can be blocked by a tenant, and a Windows `az` session cannot be shared with a Linux container. Use the host token broker, the same way as for the app container:

```text
python -m routing_lab.token_broker --write-env .env      on the host, leave it running
```

The container reaches it at `host.docker.internal` (the container is started with `--add-host=host.docker.internal:host-gateway`) and every new terminal reads `.env`. Open a new terminal after
creating `.env`. `az login` inside the container also works where your tenant allows it.

### Notes

- The repository is bind-mounted, so edits are live and `.env` (gitignored) is readable inside. Python packages live in the image, not in a `.venv`; the host's `.venv` is ignored.
- Docker Desktop's file sharing can fail with "Cannot allocate memory" when the VM is short of memory, most often in the minutes after a rebuild. Three clean runs in a row followed; if a test run fails with that
  error, give Docker more memory or stop other containers and rerun.
- The Azure CLI comes from the dev container feature and can be older than the latest release.
