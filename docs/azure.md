---
title: Azure resources for the first model probe
description: Provision Azure infrastructure for notebooks and reproducible Clef evaluation on a GPU.
---

## Provision now

If you want to keep the notebook here and call Clef as an Azure-hosted API, follow
the [managed endpoint guide](endpoint.md) instead of the interactive-compute path.
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

## Access and network prerequisites

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

## Interactive GPU notebook path

Use an Azure ML GPU compute instance (see [GPU sizing](#gpu-sizing)). Copy the project to it,
preserving its directory structure. Run these commands from the project root in the
instance terminal:

```text
python -m pip install uv==0.12.18
uv sync --frozen --python 3.11
uv pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.11.0 torchvision==0.26.0
uv pip install -r requirements-clef.txt
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

## Azure ML command-job path

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

## GPU sizing

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

## Optional strong-LLM comparison

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

## Information needed to start the live run

Provide the subscription ID, resource group, workspace name, region, compute
name/SKU and whether access is via Azure ML Studio or VS Code remote. These are
resource identifiers, not credentials. Also confirm permitted download endpoints.
Do not send keys, tokens or connection strings.

## References

* [Azure ML command-job schema](https://learn.microsoft.com/en-us/azure/machine-learning/reference-yaml-job-command?view=azureml-api-2)
* [Azure ML compute instances](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance?view=azureml-api-2)
* [Azure ML compute clusters](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-attach-compute-cluster?view=azureml-api-2)
* [Clef-Flash on Hugging Face](https://huggingface.co/Cloudflare/clef-flash)
* [Clef on Hugging Face](https://huggingface.co/Cloudflare/clef)
* [Cloudflare announcement](https://blog.cloudflare.com/clef-decision-models)
