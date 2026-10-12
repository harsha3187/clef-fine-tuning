# Router architecture: Clef as a capability router

Status (2026-10-08): phases 1 and 2 are built and measured. Phases 3 to 5 wait on the decisions at the end.

## Where things stand

| Phase | State | Result |
|---|---|---|
| 0 | Done | The agent demo is a pipeline demo, not evidence about Clef |
| 1 | Done | `capability_lab` benchmark, now 1,577 tasks in 21 families plus the support-desk tool tasks, run on 10 endpoints (15,986 attempts, $10.4) |
| 2 | Done | Capability catalog (low, medium, high) with an optional `skill` question in one forward pass |
| 3 | Done | Fine-tuned on measured labels: job `musing_yam_jn6hxfnd22`, registered as `clef-flash-capability-ft:1`, deployed as `flash-cap-ft` (0% traffic) |
| 4 | Shadow mode done | Selector (`routing_lab/selector.py`) runs beside both agents, logs every decision, changes nothing. Applying it is open |
| 5 | Replay done | Selector choices replayed on their endpoints, judged and checked against expected outcomes; harder tool tasks added. Applying it to live steps is open |
| 6 | Done | Second fine-tune with the hard tool tasks: better on the kinds it saw, not on unseen ones (see above) |
| 7 | Open | More varied tool tasks (new policies, new stores), a third fine-tune, then a canary only if the unseen-situation score improves |

Two deployments are live and billing about $7 an hour each: `flash-ft-gpu` (step-type router, still serves the agent
demo) and `flash-cap-ft2` (capability router version 2, used by shadow mode; the version 1 deployment was deleted). The base capability deployment `flash-gpu` was deleted to
free GPU quota for training; its predictions on the test tasks are cached in `outputs/capability/`.

### The benchmark

300 tasks were too easy and too few: 70% were "low", and difficulty barely changed with the level knob (levels 3 and 4
looked alike). It was rebuilt with nine more families whose difficulty is computed rather than judged: JSON transforms,
calendar maths, SQL results (run on in-memory SQLite), knights and knaves, unit chains, regex prediction, bit
operations, tool selection and list formatting. The gold answer always comes from trusted code (datetime, `re`,
`sqlite3`, brute force), never from a model. Tier labels are measured: a task is low when 70% or more of the ten
endpoints solve it, high when under 35% do, otherwise medium.

Labels: 943 low, 573 medium, 61 high, 11 unsolved by every endpoint. Endpoint pass rates: gpt-5.6-sol 97%, terra 97%,
gpt-oss-120b 96%, luna 95%, gpt-4.1 64%, deepseek-v4-pro 62%, deepseek-v4-flash 59%, nano 49%, llama-4-maverick 47%,
phi-4 44%. So "medium" means "only the strong four solve it", which is the useful small-versus-capable boundary.

### Fine-tuning

| | |
|---|---|
| Method | LoRA r=32 on the language backbone plus the joint head; loss is the route loss plus 0.5 times the skill loss |
| Data | 804 train, 115 validation, 116 calibration rows, natural class mix, no oversampling |
| Held out | `test_id`: 114 unseen tasks from trained families. `test_ood`: 428 tasks from 6 families never trained on (schedule_csp, summarize_facts, code_bug_line, tool_selection, json_transform, bit_ops) |
| Run | 4 epochs, best epoch 4, 22.7 minutes on one H100 NVL, about $3 |
| Calibration | Temperature 0.93 fitted on the calibration split, route head only |

Results on the held-out tasks, base Clef against fine-tuned. The tier column also shows what always guessing "low"
would score.

| Set | Measure | Base | Fine-tuned |
|---|---|---|---|
| test_id (114) | Tier accuracy (majority class 57.9%) | 55.3% | 90.4% |
| | Macro F1 | 0.40 | 0.62 |
| | Brier / ECE | 0.58 / 0.10 | 0.13 / 0.03 |
| | Skill accuracy | 80.7% | 100% |
| test_ood (428) | Tier accuracy (majority class 65.9%) | 36.0% | 66.8% |
| | Macro F1 | 0.27 | 0.44 |
| | Brier / ECE | 0.72 / 0.31 | 0.50 / 0.17 |
| | Overestimates / underestimates | 203 / 71 | 81 / 61 |
| | Skill accuracy | 77.8% | 79.4% |

How to read it:

- On task types it was trained on, the router is good: 90% tier accuracy, well calibrated, skill almost perfect.
- On task types it has never seen, tier accuracy is 66.8%, no better than always guessing "low" (65.9%). It improved on
  the base a lot (fewer false alarms, much better calibrated) but has not learned a general sense of difficulty. Per
  held-out family: tool_selection 80/80, summarize_facts 62/80, json_transform 62/80, bit_ops 45/80,
  schedule_csp 27/80, code_bug_line 10/28.
- Skill accuracy on unseen families (79%) is a lesson too: a skill head trained on 15 families does not extend cleanly.
  Do not use it to route tools until it is trained on more families.
- The in-distribution score is optimistic because tasks in a family share a template. Treat 90% as a ceiling and the
  out-of-distribution number as the honest one.

### What routing saves on the 542 test tasks

Cost per task includes an estimated $0.0004 per Clef decision. The selector picks the cheapest endpoint whose measured
pass rate (for the predicted tier and skill) meets a 0.85 floor.

| Arm | Pass rate | Cost per task |
|---|---|---|
| Always gpt-5.6-sol | 94.8% | $0.00225 |
| Always gpt-5.6-luna | 94.3% | $0.00017 |
| Length heuristic as the tier source | 79.3% | $0.00020 |
| Base Clef, tier and skill | 94.7% | $0.00124 |
| Fine-tuned Clef, tier and skill | 95.4% | $0.00071 |
| Fine-tuned Clef, tier only | 95.0% | $0.00058 |
| True tier and skill (upper bound) | 94.8% | $0.00018 |
| Oracle, cheapest passing endpoint | 99.8% | $0.00012 |

- Against always using the strongest model, fine-tuned Clef saves about 68% at the same pass rate (95.4% against 94.8%).
  Base Clef saved 45%.
- It still does not beat always using `gpt-5.6-luna`, which is four times cheaper. One cheap endpoint that is within a
  point of the best leaves nothing to route. Routing pays when the alternative is an expensive default.
- Roughly half of the fine-tuned arm's cost is the router itself ($0.0004 of $0.0007). On short tasks like these the
  decision costs more than a Luna call; on agent steps costing $0.005 to $0.03 it is noise.
- Most of the remaining gap to the upper bound is out-of-distribution tier errors, so more task diversity is the lever.

### Caveats

- All tasks are synthetic and short. Real agent traffic will differ.
- Labels come from one sample per endpoint, so some are noisy.
- Endpoint prices for gpt-4.1, deepseek-v4-pro and deepseek-v4-flash are unverified.
- The Clef decision cost of $0.0004 is an estimate from GPU price and measured latency, not a billed figure.
- The selector's pass-rate table includes the held-out families' endpoint results (it never sees Clef's answers); only
  Clef's tier and skill predictions are held out.

### The selector in shadow mode

Every model call in both agents (the plan and each step) now also goes to the capability router `flash-cap-ft2`; the
selector turns Clef's tier and skill probabilities and the measured registry into an endpoint choice. It runs on a
background thread beside the two existing lanes, so it cannot change what they do or delay them. Nothing is
executed on the chosen endpoint: the screen shows what it would have picked and priced.

How a decision is made (`routing_lab/selector.py`, one function used both live and in the offline evaluation):

1. Hard filters: allow-list, health, tool support, context window. An endpoint that was never measured for the kind
   of request is not selectable.
2. Expected success = sum over tiers of P(tier) times the endpoint's measured pass rate for that skill and tier,
   weighted by P(skill). A cell with fewer than 4 benchmark tasks falls back to the endpoint's pooled rate for that tier.
3. Endpoints below the user's quality floor (0.55, 0.70 or 0.85) are dropped. If Clef's top tier has under 55%
   probability, or 25% or more of the probability is on "high", the floor is raised by 0.05.
4. The rest are ranked by `expected success - cost weight x relative cost - latency weight x relative latency`, with
   the weights from the same preference screen. If nothing reaches the floor, the most reliable endpoint is chosen and
   flagged "below floor".
5. Estimated cost uses the endpoint's price, the request size, and how many output tokens that endpoint writes
   compared with the others on the benchmark (so a verbose reasoning model is priced for its verbosity).

The measured registry (`data/registry/measured.json`, built by `python -m capability_lab registry`) holds pass counts by
skill and tier with sample sizes, a latency fit and the price, with a flag for prices not yet verified.

Each decision is appended to `outputs/agent_demo/shadow.jsonl` with the registry, selector and router versions and the
adapter hash, so outcomes can later be joined to it for calibration. The screen shows Clef's probabilities, the
choice, the runner-up endpoints with their expected success and cost, what was ruled out and why, and how much of
the evidence came from benchmark tasks of exactly that kind versus pooled results.

Offline, on the 542 held-out tasks (cached Clef answers, measured costs), the same selector gives:

| Preset | Pass rate | Cost per task | Against always-Sol cost |
|---|---|---|---|
| Lean (floor 0.55) | 94.3% | $0.00058 | 74% lower |
| Balanced (floor 0.70) | 94.1% | $0.00063 | 72% lower |
| Premium (floor 0.85) | 95.8% | $0.00091 | 60% lower |
| Always gpt-5.6-sol | 94.8% | $0.00225 | |
| Always gpt-5.6-luna | 94.3% | $0.00017 | |
| True tier and skill (upper bound), balanced | 95.6% | $0.00019 | |

Two live support-desk runs (9 decisions, no failures) showed what to watch:

- **The selector chose gpt-oss-120b for every call.** It is cheap and its measured pass rate is as high as Sol's on the
  tool-use tasks, so under these registry numbers there is nothing to route between. That is a finding about the
  registry, not evidence of routing skill.
- **Tool-use evidence is thin.** All 172 `tools` benchmark tasks are low tier, so for medium and high difficulty the
  selector leans on pooled results across skills. Those decisions are flagged "thin evidence" on the screen (3 of 9).
- **Clef's skill head says "tools" for nearly every agent step**, because the step text describes tool work, even for
  the plan and the final reply. The skill answer is not a useful discriminator on this traffic yet.
- **The counterfactual cost is not a quality claim.** It prices the chosen endpoint on the tokens the always-strongest lane
  used. A 90% cost reduction against Sol follows mostly from gpt-oss-120b costing 27 to 33 times less per token. Whether it
  does the step as well is what phase 5 has to measure.
### Verifying the selector: harder tool tasks and replay

The shadow findings above (gpt-oss-120b chosen everywhere, thin tool evidence) called for two things, both done.

**Harder tool-use benchmark.** `capability_lab/toolchains.py` generates 60 support-desk tasks with five levels: one request
with the order number, one without it, two requests, two plus a policy question and chatter, and the same identified only
by email. Expected outcomes follow the store's policy articles and are graded by what the tool calls did to the store; a test
checks every expected outcome is achievable on the real desk. All nine tool-capable endpoints ran them
(543 attempts, $5.2). Tiers: 29 low, 31 medium.

| Endpoint | Old easy tool tasks | New harder tool tasks |
|---|---|---|
| gpt-5.6-sol | 100% | 100% |
| gpt-5.6-luna | 99% | 97% |
| gpt-5.6-terra | 100% | 92% |
| deepseek-v4-pro | 99% | 90% |
| deepseek-v4-flash | 98% | 72% |
| gpt-5.4-nano, gpt-4.1 | 91%, 99% | 60%, 60% |
| gpt-oss-120b | 100% | 47% |
| llama-4-maverick | 76% | 20% |

So the earlier "gpt-oss-120b is as good as Sol at tools" was an artifact of easy tasks. The measured registry now merges both
sets (`python -m capability_lab registry`), and its medium-tier tool cells are real (gpt-oss-120b: 19%).

**Clef cannot see tool-use difficulty.** On these 60 tasks the fine-tuned capability router scored 46.7% tier accuracy
(always "low" would score 51.7%): it called 30 of the 31 medium tasks "low". The skill answer was 100% ("tools"). It never
saw multi-request, policy-trap tasks, so it reads them as routine. This is the clearest case for a second fine-tune that
includes these tasks.

**Replay.** With replay on (default; `--no-replay` turns it off) the agent also runs the plan on the endpoints the selector
chose, on a fresh desk, with no retry, judges each step with the same reviewer, and checks the same expected outcomes. It
runs beside the lanes and changes nothing in them. `python -m agent_demo.replay_study --approve-cost` runs generated
tool tasks through the whole pipeline and compares outcomes:

| Arm (12 tasks, levels 3 to 5) | Expected outcomes reached |
|---|---|
| Always strongest (gpt-5.6-sol) | 81% |
| Clef lane (old step-type router) | 86% |
| Selector's choices, replayed | 85% |

Replay cost $0.05 against $0.69 for the strongest model (93% lower, router cost not included). The selector used
gpt-oss-120b on 12 calls, luna on 9 and deepseek-v4-pro on 3. The six built-in support cases (replayed live) gave 10 of 10
expected outcomes against 9 of 10 for the strongest model.

What this does and does not show: with 12 tasks the three outcome rates are not distinguishable, so the honest claim is
"no measurable loss in outcomes at about a fourteenth of the cost on this sample", not "better". The reviewer scores were similar
(replay 4.7, strongest 4.7) but the reviewer is blind to some policy errors, so outcome checks matter more. Clef still
labelled 51 of 59 calls "low".

### Second fine-tune: adding the hard tool tasks

Version 2 (`clef-flash-capability-ft:2`, job `keen_card_bz0r2fpq8q`, deployment `flash-cap-ft2`) trains from the base model on
everything in version 1 plus 60 new hard tool tasks (`tools2`: another seed, Priya's and Marcus's requests only, no message
shared with the evaluation set), labelled the same way ($4.9; tiers 34 low, 26 medium). The training bundle grew from 804 to
846 rows; the same six families stay held out. The original 60 tool tasks became an evaluation-only set. The run took
22.8 minutes on one H100 (about $3), best epoch 2, temperature 0.74.

| Set | Measure | Version 1 | Version 2 |
|---|---|---|---|
| Seen kinds, held-out tasks (114) | Tier accuracy | 90.4% | 92.1% |
| Six unseen families (428) | Tier accuracy (majority 65.9%) | 66.8% | 64.0% |
| Hard tool tasks, Priya and Marcus (38) | Tier accuracy (majority 65.8%) | 63.2% | 81.6% |
| Hard tool tasks, Elena (22, unseen policy situations) | Tier accuracy (majority 81.8%) | 18.2% | 22.7% |
| All 60 hard tool tasks | Tier accuracy (majority 51.7%) | 46.7% | 60.0% |
| | Medium tasks recognised | 1 of 31 | 13 of 31 |

What it shows:

- Clef learned the kind of tool difficulty it was shown: for the same customers' requests (different messages) it now
  recognises most medium tasks, with fewer under-estimates (3 against 12).
- It did not generalise to a situation it had not seen. Elena's late-delivery and over-limit cases, which need a different policy
  reading, are still called "low" almost every time (22.7% accuracy). Six unseen families on the main benchmark moved from
  66.8% to 64.0%, no better than always guessing "low".
- So version 2 is better where it was trained and not better elsewhere. Task diversity, not more of the same tasks, is the
  missing ingredient.

Routing results barely changed: on the 542 held-out tasks the balanced selector reached 94.1% at $0.00057 per task (version 1:
94.1% at $0.00063); on the hard tool tasks luna was chosen for all 60 at 96.7% (the upper bound with perfect labels is the
same, because luna passes nearly everything there). In the replay study (same 12 tasks) Clef called 20 of 61 calls medium (version 1:
8), the selector spread over luna 12, gpt-oss-120b 10 and phi-4 2, and reached 83% of expected outcomes against 87%
for the always-strongest lane (earlier run: 85% against 81%). The two runs of the same tasks differ by as much as the arms
differ, so 12 tasks cannot rank them.

Shadow mode now uses version 2. Version 1 stays registered as `clef-flash-capability-ft:1`; its deployment was deleted to free
GPU quota for training.

### Prompt version p2: ask for analysis, longer tier definitions

The capability prompt (`CAPABILITY_INSTRUCTIONS` in `routing_lab/backends.py`, tier descriptions in
`contracts/capability-v1.json`) was rewritten as version p2. It now tells Clef to first decide whether serving the request
needs deep analysis (dependent steps, rules to apply or reconcile, exact arithmetic or logic, things to look up before acting,
consequences if wrong), then to pick the smallest tier that can serve the purpose, judging the work and not the wording. Each tier
definition is longer, with typical work and signs. The shadow router in the app still calls the fine-tuned version 2 model
(`flash-cap-ft2`); the version is recorded as `prompt: p2` on every decision. That model was trained on the shorter p1 wording, so p2
is a change of input it has not seen. Measured on the same held-out sets:

| Set | p1 wording | p2 wording |
|---|---|---|
| Main held-out, 542 tasks: tier accuracy | 69.9% | 65.9% |
| Main held-out: macro F1, over-estimates | 0.464, 93 | 0.440, 116 |
| Hard tool tasks, 60: tier accuracy | 60.0% | 61.7% |
| Hard tool tasks: medium recognised | 13 of 31 | 14 of 31 |

So p2 costs about four points on the main set (more over-estimates) and gives about two on the tool tasks, with this model. The gain needs
a fine-tune on p2 to show: new training bundles pick up p2 automatically.
### Third fine-tune: the four external datasets, and the endpoint rebuild

Version 3 (`clef-flash-capability-ft:3`, job `hungry_eye_8t7gqzr69g`, deployment `flash-cap-ft3`) was trained from the base model
on the cleaned external bundle `routing-ext-v2` only (5,444 rows from LLMRouterBench, xRouteBench, SCOPE-60K and R2-Bench; see
[routing-datasets.md](routing-datasets.md)) with the p2 prompt, 3 epochs, 160 minutes on one H100 (about $19). Validation on its own
labels: 63.7% accuracy against 41.7% for the majority class, macro F1 0.629, temperature 1.94. No skill labels were available, so
its skill head is untrained and the app does not use it.

On this project's own held-out sets it is clearly worse than version 2:

| Set | Version 2 (p2 wording) | Version 3 |
|---|---|---|
| Main held-out, 542 tasks: tier accuracy | 65.9% | 43.9% |
| Main held-out: over-estimates | 116 | 236 |
| Hard tool tasks, 60: tier accuracy | 61.7% | 53.3% (calls 55 of 60 "medium") |

The external labels are relative to each source's models and carry more "medium" and "high" than this project's measured labels, so the
router learned a more cautious scale and over-estimates our easy tasks. Use it for routing only with that in mind; the expected cost saving
is lower than with version 2. Training on the project's own labels plus the external set together is the obvious next step.

Because the skill head is untrained, the shadow selector takes the skill from what the step is for (tool steps are `tools`, the final answer is
`writing`), as the router contract allows, and logs that as `skill_hint`.

Deployment notes: a first `flash-cap-ft3` deployment sat in `CREATING` for more than an hour and a second one stuck in `DELETING`, so the
whole endpoint `clef-routing-exp` was deleted and rebuilt; the rebuilt deployment reached `Succeeded` in about 20 minutes (traffic 100%,
`aad_token`, same scoring URI). The model artifacts and scorer were not at fault (the same adapter loads and answers). The pattern matches
Azure H100 capacity waits seen earlier in East US 2.
### The With Clef lane applies the selector

Previously the With Clef lane used the old step-type router, whose deployment was deleted, so every step silently fell back to the strongest model and both lanes
used Sol. The capability router and selector only ran in shadow mode. Now the With Clef lane applies them (`--apply-selector`, on by default; `--no-apply-selector` restores the old
behaviour):

- Each model call is sent to the capability router once; the selector turns its probabilities, your preferences and the measured pass rates into an endpoint; the step runs there.
  The same Clef answer is used for the screen, so Clef is not asked twice.
- Only endpoints on the Responses API are eligible (Phi-4 needs the chat API). The quality gate and escalation work as before: a weak reviewer score moves the step to a more
  reliable endpoint with a raised floor, never the same one twice, and a failing endpoint is skipped.
- If Clef does not answer, the step runs on the strongest model and the screen says "Clef did not answer".
- The screen names the real models, shows Clef's probabilities, why the endpoint was chosen, and all ten endpoints with expected quality and estimated cost. The shadow panel and the
  replay are off in this mode, since the choices are executed for real.

First runs against the real endpoints (balanced preferences), With Clef against Sol on every step:

| Case | With Clef models | Cost | Reviewer | Expected outcomes |
|---|---|---|---|---|
| Support: late VIP order | Luna x3, Sol for the reply | $0.022 against $0.063 (65% less) | 5 and 5 | 2 of 2 and 2 of 2 |
| Travel: Lisbon weekend | Luna x6, DeepSeek V4 Pro for the itinerary | $0.071 against $0.220 (68% less) | itinerary 5 against 4 | 4 of 4 and 4 of 4 |

Costs include Clef's estimated $0.0004 per call and the quality checks. Time and tokens did not improve on every case: the support case was 40% slower (Luna is slower than Sol on short steps, plus the
router and checks) and the travel plan used 69% more tokens (cheaper per token, more tokens) while finishing 21% faster. Two runs are a demonstration, not a measurement.

```powershell
python -m capability_lab build
python -m capability_lab run --approve-cost --workers 28
python -m capability_lab report
python -m capability_lab bundle
python -m capability_lab build --set tools --count 60
python -m capability_lab run --set tools --approve-cost --workers 12
python -m capability_lab build --set tools2 --count 60 --seed 12
python -m capability_lab run --set tools2 --approve-cost --workers 12
python -m capability_lab registry
python -m capability_lab bundle --with-tools
python -m agent_demo.replay_study --approve-cost
python -m capability_lab evaluate --split test --deployment flash-cap-ft --adapter-sha256 <adapter hash> --preset balanced
python azure\training\stage.py train --kind capability --bundle data\finetuning\capability-v1 `
    --config azure\training\capability-config.json --output outputs\stage-cap --submit --approve-cost --approve-upload
python -m capability_lab evaluate --split test --deployment flash-cap-ft --adapter-sha256 <adapter hash> --workers 1
python -m routing_lab.capability "Reformat this date as YYYY-MM-DD: 3rd of March 2026"
```

Results are written to `outputs/capability/`, keyed by the task-set hash so a rebuilt set never mixes with old answers.
The bundle holds only train, validation and calibration rows; the test splits live in
`capability_lab/data/splits-<hash>.json` and are never uploaded.

### Routing the planner and the reviewer

The planner (which writes the plan both lanes run) and the reviewer (which scores each step) are model calls too. With `--route-planner` and `--route-judge`
(environment `AIE_ROUTE_PLANNER`, `AIE_ROUTE_JUDGE`; both need `--apply-selector`) Clef and the selector choose them. The fixed models in `policy.json` (GPT-4.1) stay as the fallback.

- **Planner.** Clef reads "split this request into ordered steps" plus the request. The selector picks the endpoint as reasoning work at your quality floor. The plan is shared, so
  both lanes get it, and its cost stays out of the comparison. A routed plan that is not valid JSON, or that has no working step before the reply, is replaced by one from GPT-4.1 and the screen says so.
- **Reviewer.** Only the With Clef lane's reviewer is routed; the Without Clef lane keeps the fixed reviewer, so that lane is still measured the same way as before. The reviewer is chosen from
  the Clef answer already given for the step being checked (no extra Clef call): reasoning work at the step's difficulty, with a quality floor 0.10 higher than the work's, because a weak reviewer lets weak work through. If its pick
  fails or gives no score, the fixed reviewer scores it.
- **JSON probe.** The measured registry says how often an endpoint solves a task, not whether it answers in JSON. `python -m agent_demo.probe_structured` asks every endpoint for planner and
  reviewer output (18 calls each) and writes `agent_demo/data/structured_output.json`. An endpoint that missed any call is left out of both choices. gpt-oss-120B (planner 8 of 12, reviewer 0 of 6) and
  Llama 4 Maverick (0 of 18) are out; the other seven were 18 of 18.

First A/B on five example cases (one run each, so a direction, not a measurement), With Clef lane cost against Without Clef:

| | Routing off (fixed GPT-4.1) | Planner and reviewer routed |
|---|---|---|
| With Clef lane cost, five cases | $0.140 against $0.375 (63% less) | $0.351 against $0.443 (21% less) |
| Planner cost, five cases | $0.016 | $0.032 |
| Travel case, time | 45 s against 59 s | 146 s against 82 s |
| Cases where a lane missed an expected outcome | 2 (one lane each) | 1 (both lanes: the plan from Luna skipped a needed action) |

Why it is worse today: the registry scores GPT-4.1 low on reasoning (it measures solving hard problems, not checking an answer), so the reviewer is picked from the GPT-5.6 family and is often Sol, which costs
more than GPT-4.1 and scores more harshly, which triggers more escalations. Choosing a planner or reviewer needs its own measurement (plan quality, agreement with a trusted reviewer), which does not exist yet.


The demo does not use Clef the way the intended design does.

1. **Clef answers a different question.** The fine-tuned Clef classifies a step into seven tool-use types
   (look up, take an action, several tool calls, and so on). It was trained for that catalog on SWE-bench and
   BFCL-style data. It was never asked, or trained, to say how much model capability a request needs.
2. **Clef barely influences the choice.** In the traced example all three steps got the same label
   (`tool_workflow`). The model choice came from my hand-written policy: a table of assumed values per step type,
   regex risk rules, and assumed per-model skill scores. Savings cannot be attributed to Clef.
3. **The comparison proves little.** It compares a routed run with an always-strongest run on four deployments
   from one model family. There is no rules-only arm, no always-cheapest arm, no ground truth for which tier a
   task needed, and quality is judged by a model that also executes some steps.
4. **Registry values are assumptions.** Skill scores are guesses, not measurements.

What the demo does show is that the plumbing works: parallel sub-agents, preferences, escalation, two-lane
accounting. That part is reusable. The routing decision is not yet a test of Clef.

## Target architecture

```
                      AI APPLICATION / AGENT
        (planner, step executors, reply writer, tool users)
                              |
            every model call passes through the router
                              v
                  +------------------------+
                  |  ROUTER API            |   POST /route  {request, skill hint?, constraints}
                  +-----------+------------+
                              v
                  +------------------------+
                  |  SYSTEM-1 ROUTER       |   Clef, two questions in one call:
                  |  (Clef)                |   1. capability tier needed: low / medium / high
                  |                        |   2. primary skill needed: coding, reasoning, tools,
                  |  Sees only the request |      writing, extraction
                  +-----------+------------+
                              |  returns probabilities, not a label
                              v
                  +------------------------+
                  |  MODEL INTELLIGENCE    |   per endpoint, measured:
                  |  (registry)            |   pass rate by skill and tier, price, latency p50/p95,
                  |                        |   capabilities, health, region, allow-list
                  +-----------+------------+
                              v
                  +------------------------+
                  |  SELECTOR              |   deterministic, no text inspection:
                  |  (policy)              |   hard filters, expected success, preferences
                  +-----------+------------+
                              v
        +---------------------+---------------------+
        v                     v                     v
   small endpoints      medium endpoints      reasoning endpoints
        |                     |                     |
        +---------------------+---------------------+
                              v
                  +------------------------+
                  |  OUTCOME + FEEDBACK    |   result, cost, latency, optional verifier;
                  |                        |   escalate on failure; log for calibration
                  +------------------------+
```

Tool selection stays with the model that runs the step in this phase. Clef is a router only. Once routing shows
measured savings, a later phase can let Clef shortlist tools.

## Request lifecycle

1. The agent has a model call to make (any call: plan, step, reply, summary).
2. It sends only the request text, the intended skill if known, and hard constraints to `/route`.
3. Clef returns a distribution over tiers and over skills, with a calibrated confidence.
4. The selector computes, for each eligible endpoint:
   `expected success = sum over tiers of P(tier) x measured pass rate(endpoint, skill, tier)`.
   It keeps endpoints above the user's quality floor, then maximizes
   `expected success - cost weight x cost - latency weight x latency`.
5. If Clef is unsure (low confidence, or noticeable probability on the high tier), the selector moves up a tier
   rather than guessing.
6. The agent calls the chosen endpoint. A failed check moves the call up one tier and logs the miss.
7. The decision, its inputs and the outcome are stored. Misses become training and calibration data.

The router is called at each model-call boundary. The planner is routed too; today it is fixed to one model.

## Contracts

Router request:

```json
{"request": "text", "skill_hint": "optional", "constraints": {"allow": ["endpoint ids"], "region": "eastus2", "needs_tools": true}}
```

Router response:

```json
{"tier": {"low": 0.08, "medium": 0.31, "high": 0.61}, "skill": {"reasoning": 0.7, "coding": 0.2},
 "confidence": 0.61, "router_ms": 41, "versions": {"router": "...", "registry": "...", "policy": "..."},
 "choice": "endpoint id", "expected_success": 0.93, "alternatives": [{"endpoint": "...", "expected_success": 0.88, "cost": 0.002}]}
```

## Design rules that prevent the earlier mistakes

1. **One signal source.** Complexity comes only from Clef. Text heuristics and regex rules may exist only as a
   named comparison arm, never inside the Clef arm.
2. **Distributions, not labels.** The selector uses the full probabilities, so uncertainty is handled by
   arithmetic, not by an if-statement.
3. **Measured registry only.** A model profile with an assumed value is marked unmeasured and cannot be selected
   by the router. Pass rates come from an eval harness and carry a sample size and a date.
4. **Ground truth from outcomes.** The tier label comes from how many of the endpoints solve a task: 70% or more is
   low, under 35% is high, and the rest is medium. Nothing assumes which endpoint is small. The oracle is the
   cheapest passing endpoint. Every check in the current set is programmatic; rubric models are not used.
5. **Comparison arms, same tasks, held out:** always-largest, always-smallest, heuristic router, Clef router,
   and an oracle that picks the best tier per task. Clef is credited only when it beats the heuristic router at
   equal quality, and its regret against the oracle is reported.
6. **The reviewer is independent.** Quality is judged by a fixed model that is never routed, and by programmatic
   checks first.
7. **Every decision explains itself.** The screen shows Clef's tier probabilities, the measured pass rates used,
   the chosen endpoint, and what the heuristic arm would have chosen.
8. **Change control.** Each architectural choice is written down and confirmed before it is built.

## Evaluation protocol

| Item | Plan |
|---|---|
| Task set | 300 generated tasks in 13 families over five skills, four difficulty levels, with gold answers that pass their own checkers (tested) |
| Runs | Every task on every endpoint, graded by verifier: 2,988 attempts, one sample each |
| Outputs | Measured pass rate by endpoint, skill and tier; oracle tier label per task; the cost and latency frontier |
| Clef tests | Zero-shot first, then a LoRA fine-tune on the oracle labels using the existing training pipeline |
| Metrics | Tier accuracy and calibration; cost saved at equal quality; regret against the oracle; router overhead |
| Splits | Train, validation, calibration and a test set touched once, as in the earlier fine-tuning |

## What has to change in the code

| Area | Change |
|---|---|
| Endpoint scorer | Today it accepts exactly one question named `route` over a known step-type catalog. Add a `capability-v1` catalog (tier and skill questions) and allow two questions per call. Needs a redeploy. |
| Router module | Replace the step-type base table and regex rules with the selector above. Keep the old policy as the heuristic arm. |
| Registry | Add measured pass rates with sample sizes; flag unmeasured entries. |
| Agent | Route every model call, including the planner. Remove the always-strongest comparison as the only test. |
| UI | Show the four-arm comparison and per-decision attribution. |

## Phases

| Phase | Work | Cost and risk (original estimate) |
|---|---|---|
| 0 | Label the current demo as a pipeline demo, not evidence about Clef | None |
| 1 | Build the task set, verifiers and eval harness; measure all endpoints; produce oracle labels | Model calls only, about $15 to $50 |
| 2 | Add the `capability-v1` catalog; redeploy the scorer; test zero-shot Clef against the oracle | Redeploy time; the H100 quota is nearly full with two deployments, so one must go first |
| 3 | If zero-shot is weak, fine-tune on oracle labels, then calibrate | About $20 of GPU, as before |
| 4 | Build the selector and the four-arm evaluation; run in shadow mode inside both agents | Model calls only |
| 5 | Optional: let Clef shortlist tools, once phase 4 shows savings | Later |

## Proposed endpoint set

Chosen: deploy a wider set on the existing Azure resource. Quota in East US 2 was checked and is available for
the models below. All bill per token only, with nothing charged while idle.

| Tier | Endpoint | Family | Status |
|---|---|---|---|
| Small | gpt-5.6-luna | OpenAI | deployed |
| Small | gpt-5.4-nano | OpenAI | deployed 2026-10-08 |
| Small | Phi-4 | Microsoft | deployed 2026-10-08 |
| Mid | gpt-4.1 | OpenAI | deployed |
| Mid | gpt-5.6-terra | OpenAI | deployed |
| Mid | Llama-4-Maverick | Meta | deployed 2026-10-08 |
| Mid | DeepSeek-V4-Flash | DeepSeek | deployed 2026-10-08 |
| Large | gpt-5.6-sol | OpenAI | deployed |
| Large | DeepSeek-V4-Pro | DeepSeek | deployed 2026-10-08 |
| Large | gpt-oss-120b | OpenAI open weights | deployed 2026-10-08 |

Not included yet: the Anthropic and xAI models. They are in the catalog, but Anthropic models usually bill through
a separate marketplace agreement, so they need a billing decision first.

Smoke test results (2026-10-08, one tiny call per capability):

| Endpoint | Responses API | Tool calling | JSON mode | Notes |
|---|---|---|---|---|
| gpt-5.4-nano | yes | yes | yes | Chat Completions needs `max_completion_tokens` |
| Phi-4 | no | no | yes (Chat Completions only) | Text and JSON steps only; needs a Chat Completions adapter |
| Llama-4-Maverick | yes | yes | yes | |
| DeepSeek-V4-Flash | yes | yes | yes | |
| DeepSeek-V4-Pro | yes | yes | yes | |
| gpt-oss-120b | yes | yes | yes | |

Five of the six work with the current client unchanged. Phi-4 needs a small Chat Completions adapter and a
"no tools" capability flag. The six are deployed but not yet in the app's registry: their values are unmeasured, and
the design rules forbid routing to an unmeasured endpoint. Prices and pass rates are added in phase 1. Capabilities are recorded per endpoint
(tools, JSON mode, context), so the selector never routes a tool step to a model that cannot call tools.

## Open decisions

1. Endpoint set: decided in principle (a wider multi-family set); the exact list above needs approval.
2. Should tiers be three (low, medium, high) or should the skill question be part of phase 1?
3. Should routing happen per request, per step, or both?
4. Do you accept deriving tier labels from measured outcomes, including some model-graded rubrics?
5. Approval to redeploy the scorer and to free H100 quota for it.
