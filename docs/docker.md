# Running the Adaptive Intelligence Engine in containers

Two images, one compose file:

| Image | What it is | Needs |
|---|---|---|
| `adaptive-intelligence-engine` (`Dockerfile`) | The web app (support desk and travel planner), the routing API and the shadow selector. 1.15 GB with the Azure CLI, 264 MB without (--build-arg WITH_AZ_CLI=0, for a service principal or managed identity). | Docker with Compose 2.24 or newer, an Azure sign-in |
| `clef-server` (`docker/clef-server.Dockerfile`, optional) | The fine-tuned Clef from `models/` served locally with the same scorer and HTTP server Azure ML runs. | An NVIDIA GPU, 18 GB for the base model |

The app does not contain a model. It calls Clef on the Azure ML endpoint in `azure/online/connection.json` (not committed: copy `connection.example.json`, or set `CLEF_ENDPOINT_URL`) or the local `clef-server`
and your Azure OpenAI deployments. So the app container needs an Azure identity and network access to both.

## Start it

```sh
git lfs install && git lfs pull        # fetch the model weights (only needed for the local Clef image)
docker compose up --build -d           # builds the app image and starts it on http://localhost:8765
docker compose exec app az login --use-device-code    # sign in once; the session is kept in a volume
```

Open <http://localhost:8765>. Until you sign in, the page loads but every model call fails and the header shows the capability router
as "not answering". The container's first log lines say which sign-in it found:

```sh
docker compose logs app
```

### Signing in when device-code login is blocked

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

### If every call fails with "not authorized by network security perimeter"

The Clef workspace only accepts calls from allowed IP addresses (`nsp-clef-exp`, rule `allow-workstation-ip`). A container reaches Azure
from its host's public address, and a home or office address changes. Add the address, from a shell that is signed in to Azure:

```sh
IP=$(curl -s https://api.ipify.org)
# then add "$IP/32" to the addressPrefixes of the rule (az rest PUT, same as the existing three entries)
```

Another host in the same subscription is already allowed by the `allow-same-subscription` rule.

## What is in the app image

Only what the service reads at run time: `agent_demo/` (server, agents, UI), `routing_lab/` (Clef client, selector), `capability_lab/`, `contracts/`,
`data/registry/measured.json` (the measured endpoint pass rates the selector uses) and `azure/online/connection.json`. Runtime dependencies
are `pydantic`, `requests` and `azure-identity` (`requirements-app.txt`). The image runs as a non-root user, with a read-only root filesystem,
no Linux capabilities and `no-new-privileges`; only `/data` (telemetry, shadow log, saved preferences) and `~/.azure` are writable, as named volumes.

Check an image: `docker build --target test -t aie-test . && docker run --rm aie-test` runs 335 tests inside a clean container, with
no GPU and no cloud access.

## Using it from another application

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

## Run Clef itself in a container (optional)

```sh
docker compose --profile local-clef up --build clef-server        # needs an NVIDIA GPU and the NVIDIA Container Toolkit
```

then in `.env` set `AIE_SHADOW_LOCAL=1` and `AIE_SHADOW_URL=http://clef-server:5001/score`, and restart `app`. `clef-server` checks
the files in `models/clef-flash-capability-ft-v3` against `SHA256SUMS` before it starts, downloads the 18 GB base model (Apache-2.0,
pinned revision) into a volume on first use, and serves `POST /score` exactly like the Azure endpoint. The app sends no Azure token to it. To
bake the model into the image instead of mounting it: `docker build -f docker/clef-server.Dockerfile --target server-baked -t clef-server:baked .`
after `git lfs pull`.

Any other client can call it directly. The request must carry Clef's question text (instructions and tier descriptions) exactly as the model
was trained on it; `CapabilityRouter` builds that for you, and `routing_lab/backends.py` is the source.

What is and is not verified for this image is in [the verification notes](#what-was-verified) below.

## Settings

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
| `AIE_APPLY_SELECTOR` | `1` | The With Clef lane runs each step on the endpoint the capability router and selector choose. `0` keeps it on the old step-type router (whose deployment no longer exists, so every step falls back to the strongest model) |
| `AIE_ROUTE_PLANNER`, `AIE_ROUTE_JUDGE` | `1` | Clef also chooses the planner model and the model that checks the With Clef lane (`0`: the fixed GPT-4.1). See "Routing the planner and the reviewer" in the router architecture |
| `AIE_NO_SHADOW`, `AIE_NO_REPLAY`, `AIE_SHADOW_USE_SKILL` | off | Turn the shadow selector or its replay off, or use the skill head |
| `AIE_LLM_ENDPOINT` | none: set it in `.env` | Your Azure AI resource, `https://<name>.services.ai.azure.com/openai/v1/`. The agents and the probe scripts call it |
| `AIE_RUNTIME_DIR` | `/data/runtime` | Where telemetry and the shadow log are written |

## Security notes

The service holds cloud credentials and can spend money (a travel plan costs roughly $0.4 to $1). It was built for localhost, and the container keeps that:
the compose file publishes the port on `127.0.0.1` only, the Host header is checked, and state-changing requests must be JSON. Do not publish it on
`0.0.0.0` or the internet without an authenticating reverse proxy in front: there are no user accounts.

## What was verified

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