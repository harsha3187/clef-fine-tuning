---
title: Adaptive routing
description: Send each step of an AI agent workflow to the cheapest model that can do the job, using a fine-tuned Clef decision model, a measured model registry and a preference-driven selector.
---

Most agent workflows send every step to the strongest, most expensive model. Adaptive routing asks a small decision model how much capability each step needs, then picks the cheapest model that is expected to succeed, within your quality, latency and cost preferences.

The repository contains the router, a selector that turns the router's answer into a concrete model, a demo app that compares adaptive routing with always using the strongest model, and the tooling to deploy and fine-tune the decision model.

## How it works

```text
agent step -> decision model -> selector -> chosen model -> result -> quality check
              (Clef or          (measured     (cheapest that            |
               Decision-1)       profiles +    is expected to           v
               tier + skill      preferences)  succeed)          retry on a stronger
               probabilities                                     model if the score is low
```

1. The decision model reads one step and returns a probability for each capability tier (low, medium, high) and for each kind of work (coding, reasoning, tools, writing, extraction). It never sees the available models and never generates an answer.
2. The selector combines those probabilities with measured pass rates per model, cost and latency estimates, and your preferences. It removes models that are unhealthy, not allowed, unable to call tools, or not verified for JSON output, then ranks the rest.
3. The step runs on the chosen model. A reviewer scores the result, and a weak score moves the step to a stronger model.
4. Every decision is stored with its inputs, so the choice can be explained and audited.

Two decision models are supported. Set `AIE_ROUTER_PROVIDER=clef` for the fine-tuned Clef-Flash served from Azure ML (the default), or `decision-1` for Microsoft Decision-1 on Microsoft Foundry.

### What the demo app compares

The app runs each case twice from one shared plan, and compares the two runs on cost, tokens, time, answer quality and whether the right actions happened.

| Lane | Behavior |
|---|---|
| Adaptive routing | The decision model and selector choose a model for every step, the planner and the reviewer |
| Always strongest | Every step goes to the strongest model with every tool attached |

Two agents ship with the app: a customer support desk and a travel planner with parallel sub-agents. Web search, weather and destination guides are real. Flight fares, hotel rates and booking holds are simulated and labeled as such.

In the first runs against live endpoints, adaptive routing cost 65% to 68% less on two small cases. A later run of a larger travel case was cheaper and faster but used more tokens and scored lower with the reviewer. A handful of runs is a demonstration, not a benchmark. See [docs/router.md](docs/router.md) for the first measurements and their limits.

## Repository layout

| Path | Contents |
|---|---|
| `agent_demo/` | The app: HTTP server, agents, policy, shadow selector, telemetry, web UI |
| `routing_lab/` | The decision-model clients, selector, evaluation, annotation and fine-tuning code |
| `capability_lab/` | Benchmark that measures each model's pass rate per tier and skill |
| `routing_data/` | Converts four public routing datasets into one canonical format |
| `azure/` | Azure ML endpoint, deployment and training job definitions and scripts |
| `models/` | The model card and checksums of the fine-tuned Clef router (the weights are not stored in git) |
| `data/` | Measured model registry and probe set (large datasets are generated, not committed) |
| `contracts/` | Routing taxonomies and criteria |
| `notebooks/` | Experiment notebooks |
| `docs/` | The three guides listed at the end of this file |
| `tests/` | The test suite |

## Quick start

### 1. Open the dev container

You need Docker and Visual Studio Code with the Dev Containers extension.

```text
git clone git@github.com:harsha3187/adaptive-routing.git
cd adaptive-routing
code .
```

In VS Code, open the Command Palette and run Dev Containers: Reopen in Container. Without VS Code, run `devcontainer up --workspace-folder .` with the `@devcontainers/cli` package.

The first build takes a few minutes. It installs Python 3.11, the `app`, `test` and `devcontainer` dependency groups from `pyproject.toml` and the Azure CLI. The container has no GPU and no model weights. Clef runs on an Azure endpoint.

Check the setup:

```text
python -m pytest -q
python -m ruff check agent_demo routing_lab capability_lab routing_data azure tests
```

Without a dev container, use Python 3.11 and uv: run `uv sync --frozen`, then `uv run --no-sync python -m pytest -q`.

### 2. Configure

```text
cp .env.example .env
cp azure/online/connection.example.json azure/online/connection.json
```

Both files are ignored by git. Edit `.env`:

| Variable | Purpose |
|---|---|
| `AIE_LLM_ENDPOINT` | Required. Your Azure AI resource URL, for example `https://<name>.services.ai.azure.com/openai/v1/` |
| `CLEF_ENDPOINT_URL` | Scoring URL of the Clef endpoint. Alternatively, fill in `scoring_uri` in `azure/online/connection.json` |
| `AIE_ROUTER_PROVIDER` | `clef` (default) or `decision-1` |
| `AIE_DECISION_DEPLOYMENT` | The Foundry deployment name of Decision-1, when that provider is selected |

The execution models are called with your Azure identity, so the app needs an Azure sign-in. Pick one:

* On the host, run `az login`, then `python -m routing_lab.token_broker --write-env .env` and leave it running. The container reads `.env` in every new terminal, so open a new one afterward. Use this when your tenant blocks device-code login.
* In the container, run `az login --use-device-code` and leave `AIE_AZURE_AUTH` unset.
* On a server or in CI, set `AZURE_TENANT_ID`, `AZURE_CLIENT_ID` and `AZURE_CLIENT_SECRET`, and set `AIE_AZURE_AUTH=default`.

### 3. Run the agent app

```text
python -m agent_demo.server
```

Open <http://localhost:8765>. VS Code forwards the port from the dev container. The server binds to loopback only, because it holds cloud credentials. Each run calls paid models and costs from a few cents (support desk) to about one dollar (travel planner with web search).

In the app:

1. Open Preferences and choose Lean, Balanced or Premium, or set quality, latency and cost yourself. The page shows which model each kind of step would get.
2. Open Support desk or Travel planner, pick an example case and run it. Both lanes run side by side with a live event stream.
3. Read the comparison table, then the step-by-step view, which shows each decision model assessment, the chosen model, the reason, the reviewer score and any escalation.
4. Open Models to mark a model unhealthy or disallowed and see the selector adapt. Open Insights for totals across runs.

Useful settings, all environment variables that can also be passed as flags:

| Variable | Default | Effect |
|---|---|---|
| `AIE_PORT` | `8765` | Port to listen on |
| `AIE_APPLY_SELECTOR` | `1` | Run adaptive routing on the models the selector chooses. `0` restores the older step-type router |
| `AIE_ROUTE_PLANNER` | `1` | Let the selector choose the planner model |
| `AIE_ROUTE_JUDGE` | `1` | Let the selector choose the reviewer for adaptive routing |
| `AIE_RUNTIME_DIR` | `outputs/agent_demo` | Telemetry, saved preferences and the registry copy |
| `AIE_CAPTURE_ROUTER_PROMPTS` | off | Store the exact decision-model requests (includes user text) |

Delete the runtime directory to reset saved preferences and insights.

### Run it in containers instead

```text
docker compose up --build -d
docker compose exec app az login --use-device-code
```

The service listens on `http://localhost:8765`, published on loopback only. Compose reads `.env`. To serve the Clef weights from your own GPU rather than Azure, first place the weights in `models/clef-flash-capability-ft-v3` (see [models/README.md](models/README.md)), then run `docker compose --profile local-clef up --build clef-server`. That needs an NVIDIA GPU and the NVIDIA Container Toolkit, and you also set `AIE_SHADOW_LOCAL=1` and `AIE_SHADOW_URL=http://clef-server:5001/score`. See [docs/deployment.md](docs/deployment.md).

## Use the router from another application

The service exposes the decision model and selector as an API:

```text
curl -s http://localhost:8765/v1/select \
  -H "Content-Type: application/json" \
  -d '{"request": "What is the molar mass of NaCl?", "needs_tools": false,
       "preferences": {"quality": "good", "latency": "medium", "cost": "low"}}'
```

The response contains the chosen model (`choice`), the decision model's tier and skill probabilities, the ranked `alternatives`, the models that were `eliminated` and why, and the expected success rate. Optional fields are `kind` (`plan`, `step` or `reply`), `needs_json`, `allow` (a list of model ids), `input_tokens` and `output_tokens`. Preference levels are `average`, `good` or `very_good` for quality, and `low`, `medium` or `high` for latency and cost. `POST /v1/route` is the older step-type router, and `POST /v1/outcome` reports how a decision turned out.

## Deploy the Clef endpoint

The decision model runs on an Azure ML managed online endpoint with one H100 GPU. It bills until you delete it, so every action that creates compute needs an explicit `--approve-cost`.

```text
export AZURE_SUBSCRIPTION_ID=<subscription-id>
export AZURE_RESOURCE_GROUP=<resource-group>
export AZURE_ML_WORKSPACE=<workspace>
export AZURE_ML_ENDPOINT=<online-endpoint-name>

python azure/online/manage.py deploy --approve-cost --deployment-name flash-cap-ft3 \
  --deployment-config azure/online/deployment-capability-v3.yml
python azure/online/manage.py status
```

The deployment YAML references a registered model, so register the fine-tuned weights first (see the next section). The endpoint uses Microsoft Entra authentication. `azure/online/manage.py` writes the scoring URL to `azure/online/connection.json`. Sizing, costs, verification and deletion are in [docs/deployment.md](docs/deployment.md).

## Fine-tune the Clef router

Fine-tuning is optional. The Azure endpoint (or any Clef endpoint you point `CLEF_ENDPOINT_URL` at) is enough to run everything above. Train your own router when you have labeled routing data for your domain. Training runs on an Azure H100 (the shipped model took about 160 minutes, roughly $19), and each submit needs `--approve-cost` and `--approve-upload`.

1. Build the training data. Either convert the four public routing datasets, or measure your own models on generated tasks:

   ```text
   python -m routing_data.fetch
   python -m routing_data convert
   python -m routing_data export --name routing-ext-v2

   python -m capability_lab build
   python -m capability_lab run --approve-cost
   python -m capability_lab report
   python -m capability_lab bundle
   ```

   Bundles are written under `data/finetuning/`. The held-out test files are never uploaded.

2. Create the GPU cluster once, with the Azure variables from the previous section set:

   ```text
   pip install "azure-ai-ml>=1.35.1"
   PYTHONPATH=. python azure/training/provision.py --approve
   ```

3. Stage the training package and inspect it. Without `--submit` nothing is uploaded:

   ```text
   PYTHONPATH=. python azure/training/stage.py train --kind capability \
     --bundle data/finetuning/routing-ext-v2 \
     --config azure/training/routing-ext-config.json \
     --output outputs/stage-ext
   ```

4. Run a four-step smoke job first, using `azure/training/capability-smoke-config.json` as the config, then the full job. Add the consent flags to submit:

   ```text
   PYTHONPATH=. python azure/training/stage.py train --kind capability \
     --bundle data/finetuning/routing-ext-v2 \
     --config azure/training/routing-ext-config.json \
     --output outputs/stage-ext --submit --approve-cost --approve-upload
   ```

5. Register the exported model, point `azure/online/deployment-capability-v3.yml` at the new version, and deploy it at 0% traffic until you have checked it:

   ```text
   python azure/online/manage.py register-model --model-name clef-flash-capability-ft \
     --model-version <n> --model-path azureml://datastores/<datastore>/paths/<job-output>/ --tag source=<job-name>
   ```

6. Evaluate on the frozen test split once, after model selection is final:

   ```text
   python -m capability_lab evaluate --split test --deployment <deployment-name> --adapter-sha256 <adapter-hash>
   ```

The data formats, the label policy, the frozen-split rules and the results of the three fine-tuning rounds are in [docs/data-and-training.md](docs/data-and-training.md). The shipped model and its limits are described in [its model card](models/clef-flash-capability-ft-v3/MODEL_CARD.md).

## Secrets and configuration

No credentials are stored in this repository.

* `.env` and `azure/online/connection.json` are ignored by git. Keep your keys, tokens, subscription ID and endpoint names there, never in tracked files.
* The Azure subscription, resource group, workspace and endpoint come from environment variables.
* The token broker listens on loopback, answers only requests that carry a random per-run secret, and hands out tokens for two allow-listed resources. Stop it when you finish.
* The telemetry database stores hashes and lengths of requests, not request text. The decision log keeps request bodies only if you turn on `AIE_CAPTURE_ROUTER_PROMPTS`.

## Package index

`pyproject.toml` sets the default uv package index to a mirror that was reachable where this project was built. If you cannot reach it, change the `[[tool.uv.index]]` URL to PyPI, run `uv lock` and keep the new lock file. Do not mix package sources between runs you intend to compare.

## Documentation

* [docs/router.md](docs/router.md): router architecture, selector and policy, measurements, and the demo app
* [docs/data-and-training.md](docs/data-and-training.md): datasets, labeling, evaluation, prompt tuning, fine-tuning and baseline results
* [docs/deployment.md](docs/deployment.md): Azure resources, the Clef endpoint, containers and the dev container
