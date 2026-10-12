# Routing proof of concept: support desk and travel planner

Two agents that show what routing with the fine-tuned Clef-Flash model changes: a customer support agent for a
fictional online store (Alder & Co.) and a travel planner. Every case runs twice from one shared plan, and the two
runs are compared on cost, tokens, time, answer quality and whether the right actions happened.

## What is compared

| Lane | What it does |
|---|---|
| With Clef | Clef classifies each step, then a policy shaped by the user's preferences picks the model. A quality check can retry a weak result on a stronger model. |
| Without Clef | The common default: every step goes to the strongest model (`gpt-5.6-sol`) with every tool attached. |

Both lanes use separate copies of the store, so neither can affect the other. A separate AI reviewer
(`gpt-4.1`) scores every output from 1 to 5 against the store policy and the tool results, and the example
cases also check that the right actions happened (for example a refund of exactly $138.99).

## How a case flows

1. The planner (`gpt-4.1`) splits the customer message into steps. The last step writes the reply.
2. With Clef, each step goes to the fine-tuned Clef deployment together with the store's tool list. Clef
   returns a step type (look up, take an action, several tool calls, ask for missing details, no tool
   needed) and a confidence. It does not rate difficulty or compare models.
3. The policy turns step type, risk signals and the user's preferences into the capability a model needs,
   removes ineligible models, and picks the best-value model above the quality bar.
4. Steps that move money or change an order, and steps written for a customer, need more capability.
5. For the tiers named by the quality setting, the check scores the result. A result under the threshold
   moves to a stronger model, unless the step already changed the store or the check blames earlier work.

## Travel planner

Give departure and return dates, where you start, and optionally a destination, travelers, a budget level,
interests and notes. With no destination, the agent chooses one by ranking candidate cities on your interests
and the weather for your dates.

The planner marks research steps that can run at the same time. They run as parallel sub-agents in both lanes
(flights, hotels, sights, dining, weather), followed by a booking step and the final itinerary. Time counts the
slowest member of a parallel group, not the sum. The itinerary is JSON, rendered as flights, hotel, weather,
day by day, budget, tips and sources.

| Tool | Data |
|---|---|
| `search_web` | Real open web search with citations (the Azure model's built-in search tool) |
| `get_destination_overview` | Real: Wikivoyage guide text (Wikipedia fallback) and Open-Meteo geocoding |
| `get_weather` | Real: Open-Meteo forecast within 15 days, otherwise the same dates last year |
| `suggest_destinations` | Real weather for a curated shortlist, ranked by interests, weather and distance |
| `find_hotels` | Real hotel names and star ratings from OpenStreetMap; nightly rates are estimates |
| `search_flights` | Simulated fares and times, deterministic, labelled illustrative |
| `hold_flight`, `hold_hotel` | Simulated holds. Nothing is booked or charged |

No free public booking service exists, so fares, rates and holds are simulated and labelled in the output. Both
lanes see identical tool results in a run (web results are shared), so differences come from the models. Web
search is paid once, reported on its own line, and left out of the comparison. A lane is limited to 8 searches.
Public services are contacted only from a fixed host list, with caching and Nominatim's one request per second.

Clef is shown the earlier sequential steps as `preceding_instructions`, which it was trained on. Without them it
read "check the weather for the chosen destination" as a request for missing details.

A trip run costs about $0.4 to $1 including web search, against a few cents for a support case.

## Preferences

The Preferences screen sets three levels. The mapping to routing parameters lives in
[policy.json](../agent_demo/data/policy.json) under `preferences`.

| Setting | Levels | Effect |
|---|---|---|
| Response quality | Average, Good enough, Very good | Quality bar (55%, 70%, 85%), check threshold, which tiers are checked |
| Latency | Low, Medium, High | Weight on speed. Low penalizes slow models heavily |
| Cost | Low, Medium, High | Weight on token spend. Low prefers cheap models |

Presets Lean, Balanced and Premium set all three at once. The screen previews which model each kind of step
would get. Saved preferences persist in `outputs/agent_demo/preferences.json`.

## Run it

Needs the `flash-ft-gpu` deployment (see [endpoint.md](endpoint.md)) and `az login` with access to the
Azure OpenAI resource set by `--llm-endpoint`.

```powershell
.\.venv\Scripts\python.exe -m agent_demo.server --port 8765
```

Open `http://127.0.0.1:8765`. The server binds to loopback only because it holds cloud credentials.
Each run costs a few cents. Delete `outputs/agent_demo` to reset preferences, the registry copy and insights.

| Flag | Default |
|---|---|
| `--deployment` | `flash-ft-gpu` |
| `--adapter-sha256` | the pinned fine-tuned adapter digest |
| `--llm-endpoint` | `AIE_LLM_ENDPOINT` (required: your Azure AI resource `/openai/v1/` URL) |
| `--runtime-dir` | `outputs/agent_demo` |

## Screens

| Screen | Purpose |
|---|---|
| Support desk | Pick or write a customer message, run it both ways, read the comparison, the two replies and the step-by-step savings |
| Travel planner | Enter dates (and a destination if you have one), run it both ways, and read the two itineraries with the same comparison |
| Preferences | Set quality, latency and cost, and preview the routing they produce |
| Models | Registry with health and allowlist toggles, and model onboarding |
| Insights | Measured totals over all runs: cost, tokens, time, quality, model mix |
| How it works | The flow and the caveats in plain language |

## Shadow selector (now applied)

The With Clef lane applies the capability router and selector by default: see "The With Clef lane applies the selector" in [router architecture](router-architecture.md).
The rest of this section describes the shadow and replay behaviour, which is what runs with `--no-apply-selector`.

Beside the two lanes, every model call (the plan and each step) also goes to the capability router `flash-cap-ft`, and
the selector in `routing_lab/selector.py` turns its tier and skill probabilities and the measured endpoint registry into
the endpoint it would choose. This runs on a background thread, executes nothing and changes nothing: it appears as a
"Shadow selector" section under the step list, in Insights, in `GET /api/shadow`, and in
`outputs/agent_demo/shadow.jsonl`. The cost shown for it is a counterfactual (the chosen endpoint priced on the tokens the
always-strongest lane used), and its quality is expected success from benchmark pass rates, not a re-run. Unless `--no-replay` is given, the agent also replays the plan on the endpoints the selector chose (fresh desk, no retry), judges
each step with the same reviewer and checks the same expected outcomes, so the screen shows whether the choices held up.
`python -m agent_demo.replay_study --approve-cost` runs generated tool tasks through the whole pipeline for a task-by-task
comparison. Start the
server with `--no-shadow` to turn it off, or `--shadow-deployment` and `--shadow-adapter-sha256` to point it elsewhere.
Design, rules and measured results are in [router architecture](router-architecture.md).
## HTTP API

| Endpoint | Purpose |
|---|---|
| `POST /api/run` | Start a case: `agent` (`support` default, or `travel`), `message` (support) or `form` (travel), optional `scenario_id`, optional `preferences`. Follow with `GET /api/run/{id}?since=n` or `/stream` (SSE) |
| `GET /api/preferences/preview?quality=&latency=&cost=` | Derived parameters and model per step type |
| `POST /api/preferences` | Save preferences |
| `GET /api/state`, `/api/insights?agent=`, `/api/trace/{id}` | UI state (including both agents' cases and tools), measured totals, one routing trace |
| `GET /api/shadow` | Totals over every logged shadow decision: endpoints it would pick, tiers seen, unsure and thin-evidence counts |
| `POST /api/registry`, `POST /api/registry/{id}` | Onboard a model, or update healthy, allowed, live |
| `POST /v1/route`, `POST /v1/outcome` | The router as a service for any agent |

Run events carry a `lane` (`clef` or `baseline`). `step_compare` carries the per-step savings and
`run_done` the measured totals.

## What the numbers include

- With Clef cost: model execution, quality checks, and an estimated Clef decision cost
  (`router_cost_per_decision_usd`: one H100 at $6.98/h over about 185 ms of server time, assuming the GPU is
  kept busy). Planning is shared by both lanes and left out.
- Tokens: billed model tokens, including checks. Clef's own tokens run on the hosted model and are listed
  separately.
- Time: the sum of step times, including the Clef call and checks for the With Clef lane.

## Caveats

- Model skill scores are assumptions. Latency priors come from a small sample of measured calls. Results
  depend on both.
- A single run of one case is noisy: the same model can make different mistakes on different runs, and
  gateway latency varies. Compare totals across runs on the Insights screen.
- Six support cases and four trips are a demonstration, not a benchmark.
- The reviewer is a model, not a human, and `gpt-4.1` also executes some steps.
- The 95% Clef figures are agreement with AI-generated labels on SWE-bench and BFCL-style data, not on
  support conversations. Clef is called with a 12 s limit and falls back to a safe default step type.
- A router call takes about 0.6 s on the single-request H100 endpoint, against the BRD target of under 50 ms.
- Not built: dynamic budgets across steps and Clef cascading. Real flight and hotel booking (no free public API).

## Costs

Both H100 deployments (`flash-gpu` and `flash-ft-gpu`) bill about $14/hr together while they exist.
The Azure OpenAI deployments bill per token only.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_agent_demo_core.py tests/test_agent_demo_router_agent.py tests/test_agent_demo_server.py -q
```

They use a fake decider and a scripted model, so no Azure calls are made.
