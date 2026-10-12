"use strict";

/* ---------------------------------------------------------------- helpers */
const $ = (selector, root = document) => root.querySelector(selector);

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "text") el.textContent = value;
    else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "vars") for (const [name, v] of Object.entries(value)) el.style.setProperty(name, v);
    else if (key in el && key !== "list" && key !== "form") el[key] = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

const clearEl = (el) => { while (el.firstChild) el.removeChild(el.firstChild); };
const usd = (x) => (x === null || x === undefined ? "n/a" : x === 0 ? "$0" : Math.abs(x) < 0.01 ? `$${x.toFixed(5)}` : `$${x.toFixed(4)}`);
const int = (x) => (x === null || x === undefined ? "n/a" : Math.round(x).toLocaleString("en-US"));
const secs = (x) => (x === null || x === undefined ? "n/a" : x >= 1000 ? `${(x / 1000).toFixed(1)} s` : `${Math.round(x)} ms`);
const pct = (x, d = 0) => (x === null || x === undefined ? "n/a" : `${x.toFixed(d)}%`);
const fraction = (x) => (x === null || x === undefined ? "n/a" : `${Math.round(x * 100)}%`);

async function api(path, body) {
  const options = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  let response;
  try { response = await fetch(path, options); } catch (_) { throw new Error("The demo server is not reachable. Check that it is still running."); }
  let data = null;
  try { data = await response.json(); } catch (_) { /* non-JSON error body */ }
  if (!response.ok) throw new Error((data && data.error) || `The request failed (${response.status}).`);
  return data;
}

let toastTimer = null;
function toast(text) {
  const el = $("#toast");
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { el.hidden = true; }, 3200);
}

const ROUTE_LABEL = {
  tool_lookup: "Look something up", tool_action: "Take an action", tool_workflow: "Several tool calls",
  web_research: "Research", memory_retrieval: "Recall context", clarification: "Ask for missing details",
  no_matching_tool: "No tool needed", fallback: "Out of scope",
  low: "Low capability needed", medium: "Medium capability needed", high: "High capability needed",
};
const LEVEL_NAME = { average: "Average", good: "Good enough", very_good: "Very good", low: "Low", medium: "Medium", high: "High" };
const PREF_COPY = {
  quality: ["Response quality", "How good does each response need to be?"],
  latency: ["Latency", "How quickly do you need a response? Low means as fast as possible."],
  cost: ["Cost", "How much token spend is acceptable? Low keeps spend as low as possible."],
};

/* ------------------------------------------------------------------ state */
const app = {
  info: null, view: "desk", draft: null, saved: null, preview: null, previewState: "idle", previewSeq: 0,
  previewTimer: null, frame: 0, justSaved: false, agent: "support", insightsAgent: "all",
  desks: { support: { run: null, caseId: null }, travel: { run: null, caseId: null, itin: "clef" } },
};
// The visible agent's run and example case, so shared screens read them without knowing which agent is showing.
Object.defineProperties(app, {
  run: { get: () => app.desks[app.agent].run, set: (v) => { app.desks[app.agent].run = v; } },
  caseId: { get: () => app.desks[app.agent].caseId, set: (v) => { app.desks[app.agent].caseId = v; } },
});
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

const modelOf = (id) => (app.info.models || []).find((m) => m.id === id) || ((app.info.shadow || {}).endpoints || []).find((m) => m.id === id);
const modelChip = (id) => {
  const m = modelOf(id);
  return h("span", { class: `tier-${m ? m.tier : ""}`, title: m ? `${m.tier} tier` : "" }, m ? m.display : id || "n/a");
};

/* ------------------------------------------------------------------ status */
function renderStatus() {
  const info = app.info, root = $("#status");
  clearEl(root);
  const shadow = info.shadow || {}, v = shadow.versions || {}, last = shadow.last;
  if (shadow.enabled) {
    root.append(h("span", { class: "chip", title: `Capability router: Clef deployment ${v.router}, adapter ${v.adapter_sha256}, prompt ${v.prompt}` },
      h("span", { class: `dot ${last ? (last.ok ? "ok" : "bad") : ""}` }),
      `Capability router${last ? (last.ok ? ` · ${secs(last.latency_ms)}` : " · not answering") : " · warming up"}`));
  } else {
    root.append(h("span", { class: "chip", title: "Started with the shadow selector turned off" }, h("span", { class: "dot" }), "Capability router off"));
  }
  const legacy = info.clef.last;
  if (legacy && !legacy.ok && !shadow.applied) {  // the step-type router the With-Clef lane uses; it falls back to the strongest model when it is down
    root.append(h("span", { class: "chip", title: `The step-type router (${info.clef.deployment}) did not answer, so the With Clef lane sends steps to the strongest model` },
      h("span", { class: "dot bad" }), "Step router down"));
  }
  $("#registry-version").textContent = `registry ${info.registry_version}`;
}

async function refreshState() {
  app.info = await api("/api/state");
  renderStatus();
}

/* ------------------------------------------------------------- navigation */
const VIEWS = ["desk", "prefs", "models", "insights", "about"];
function setView(view, { focus = true, agent = null } = {}) {
  if (view === "travel") { view = "desk"; agent = "travel"; }
  if (!VIEWS.includes(view)) view = "desk";
  if (view === "desk" && agent && app.info.agents[agent]) app.agent = agent;
  app.view = view;
  for (const name of VIEWS) $(`#view-${name}`).hidden = name !== view;
  for (const tab of document.querySelectorAll(".tab")) {
    const current = tab.dataset.view === view && (!tab.dataset.agent || tab.dataset.agent === app.agent);
    if (current) tab.setAttribute("aria-current", "page"); else tab.removeAttribute("aria-current");
  }
  history.replaceState(null, "", view === "desk" ? (app.agent === "travel" ? "#travel" : "#desk") : `#${view}`);
  if (view === "desk") renderDeskHeader();
  if (view === "prefs") renderPrefs();
  if (view === "models") renderModels();
  if (view === "insights") loadInsights();
  if (view === "about") renderAbout();
  if (focus) window.scrollTo({ top: 0 });
}

/* ------------------------------------------------------------ preferences */
const prefsEqual = (a, b) => ["quality", "latency", "cost"].every((k) => a[k] === b[k]);

function schedulePreview() {
  clearTimeout(app.previewTimer);
  app.previewState = "loading";
  renderPreview();
  app.previewTimer = setTimeout(loadPreview, 180);
}

async function loadPreview() {
  const seq = ++app.previewSeq;
  const query = new URLSearchParams(app.draft).toString();
  try {
    const data = await api(`/api/preferences/preview?${query}`);
    if (seq !== app.previewSeq) return;
    app.preview = data;
    app.previewState = "ready";
  } catch (error) {
    if (seq !== app.previewSeq) return;
    app.previewState = "error";
    app.previewError = error.message;
  }
  renderPreview();
}

function renderPrefs() {
  const form = $("#prefs-form");
  clearEl(form);
  const options = app.info.preferences.options;
  const presets = app.info.preferences.presets;
  const matching = Object.entries(presets).find(([, v]) => prefsEqual(v, app.draft));
  form.append(h("div", { class: "presets", role: "group", "aria-label": "Presets" },
    h("span", { class: "label", text: "Start from" }),
    Object.entries(presets).map(([name, levels]) => h("button", {
      type: "button", class: "btn", "aria-pressed": String(!!matching && matching[0] === name), text: name[0].toUpperCase() + name.slice(1),
      title: `${LEVEL_NAME[levels.quality]} quality, ${LEVEL_NAME[levels.latency]} latency, ${LEVEL_NAME[levels.cost]} cost`,
      onclick: () => { app.draft = { ...levels }; renderPrefs(); schedulePreview(); },
    }))));
  for (const field of ["quality", "latency", "cost"]) {
    const [title, question] = PREF_COPY[field];
    const selected = options[field].find((o) => o.id === app.draft[field]);
    const hint = h("p", { class: "selected-hint", id: `hint-${field}`, text: selected ? selected.hint : "" });
    form.append(h("fieldset", {},
      h("legend", { text: title }), h("p", { class: "q", text: question }),
      h("div", { class: "seg", role: "radiogroup", "aria-label": title, "aria-describedby": `hint-${field}` },
        options[field].map((o) => h("label", {},
          h("input", { type: "radio", name: field, value: o.id, checked: o.id === app.draft[field], onchange: () => { app.draft[field] = o.id; app.justSaved = false; renderPrefs(); schedulePreview(); } }),
          h("span", { text: o.label })))),
      hint));
  }
  const dirty = !prefsEqual(app.draft, app.saved);
  form.append(h("div", { class: "form-actions" },
    h("button", { type: "button", id: "save-prefs", class: "btn primary", disabled: !dirty && app.info.preferences.saved, text: app.info.preferences.saved ? "Save preferences" : "Save and continue", onclick: savePrefs }),
    dirty ? h("button", { type: "button", class: "btn quiet", text: "Discard changes", onclick: () => { app.draft = { ...app.saved }; renderPrefs(); schedulePreview(); } }) : null,
    dirty ? h("span", { class: "dirty", text: "Unsaved changes" }) : null,
    !dirty && app.info.preferences.saved ? h("button", { type: "button", class: "btn", text: "Go to support desk", onclick: () => setView("desk", { agent: "support" }) }) : null));
  if (!app.preview || !prefsEqual(app.preview.preferences, app.draft)) schedulePreview(); else renderPreview();
}

async function savePrefs() {
  const button = $("#save-prefs");
  button.disabled = true;
  try {
    const data = await api("/api/preferences", app.draft);
    app.saved = { ...app.draft };
    app.info.preferences.saved = true;
    app.info.preferences.value = { ...app.draft };
    app.preview = data;
    app.previewState = "ready";
    toast("Preferences saved. The next run will use them.");
    renderPrefs();
    renderDeskHeader();
  } catch (error) {
    button.disabled = false;
    toast(error.message);
  }
}

function renderPreview() {
  const root = $("#prefs-preview");
  clearEl(root);
  root.append(h("h2", { text: "What this means" }),
    h("p", { class: "hint", text: "Based on your choices, here is how the router reads them and which model it would pick for a typical step." }));
  if (app.previewState === "error") {
    root.append(h("div", { class: "alert bad", role: "alert" }, h("div", { class: "grow" }, h("b", { text: "The preview could not load." }), h("div", { class: "hint", text: app.previewError })),
      h("button", { type: "button", class: "btn", text: "Try again", onclick: schedulePreview })));
    return;
  }
  const data = app.preview && prefsEqual(app.preview.preferences, app.draft) ? app.preview : null;
  if (!data) {
    root.append(h("div", { class: "reading" }, [1, 2, 3].map(() => h("div", {}, h("span", { class: "skel" }), h("span", { class: "skel", vars: { width: "90%" } })))));
    return;
  }
  root.append(h("div", { class: "reading" },
    ["quality", "latency", "cost"].map((k) => h("div", {}, h("b", { text: PREF_COPY[k][0] }), h("span", { text: data.notes[k] })))));
  root.append(h("h3", { text: "Model by kind of step" }),
    h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", { text: "Step" }), h("th", { class: "r", text: "Capability needed" }), h("th", { text: "Model" }), h("th", { class: "r", text: "Typical cost" }))),
      h("tbody", {}, data.rows.filter((r) => !["clarification", "web_research", "memory_retrieval"].includes(r.route)).map((r) => h("tr", {},
        h("td", { text: r.customer_facing ? "Write the final answer" : (r.route === "no_matching_tool" ? "Answer without tools" : ROUTE_LABEL[r.route] || r.route) }),
        h("td", { class: "r tnum", text: fraction(r.needs) }),
        h("td", {}, modelChip(r.model)),
        h("td", { class: "r tnum", text: usd(r.est_cost_usd) })))))));
  root.append(h("p", { class: "hint", text: "Capability needed is the router's estimate for a typical step. Steps that move money, change an order or booking, or write the final answer need more." }));
}

/* ------------------------------------------------------------------ desk */
const AGENT_COPY = {
  support: {
    title: "Support desk", run: "Run both ways", running: "Running both ways…", quality: "Reply quality",
    replyTitle: "What the customer receives", actions: "Store changes",
    lede: "Run a customer message two ways on the same plan: routed by Clef, and sent to the strongest model every time. Compare cost, tokens, speed and quality.",
  },
  travel: {
    title: "Travel planner", run: "Plan both ways", running: "Planning both ways…", quality: "Itinerary quality",
    replyTitle: "The itinerary", actions: "Holds placed (simulated)",
    lede: "Plan a trip two ways from one shared plan: specialist sub-agents routed by Clef, and the same work on the strongest model every time. Web search, guides and weather are real; flight fares, hotel rates and holds are simulated.",
  },
};
const agentInfo = () => app.info.agents[app.agent];
const deskOf = (run) => app.desks[run.agent];

function renderDeskHeader() {
  const v = app.saved, button = $("#prefs-summary"), copy = AGENT_COPY[app.agent];
  $("#desk-title").textContent = copy.title;
  $("#desk-lede").textContent = copy.lede;
  clearEl(button);
  button.append(
    h("span", {}, h("span", { class: "k", text: "Quality " }), LEVEL_NAME[v.quality]),
    h("span", {}, h("span", { class: "k", text: "Latency " }), LEVEL_NAME[v.latency]),
    h("span", {}, h("span", { class: "k", text: "Cost " }), LEVEL_NAME[v.cost]),
    h("span", { class: "edit", text: app.info.preferences.saved ? "Change" : "Review" }));
  $("#composer-support").hidden = app.agent !== "support";
  $("#composer-travel").hidden = app.agent !== "travel";
  $("#composer-title").textContent = app.agent === "travel" ? "Trip details" : "Customer message";
  buildCases();
  updateComposer();
  setRunning(!!(app.run && !app.run.done && !app.run.error));
  renderResults();
}

function buildCases() {
  const root = $("#cases");
  clearEl(root);
  for (const c of agentInfo().scenarios) {
    root.append(h("label", { class: "case" },
      h("input", { type: "radio", name: "case", value: c.id, checked: c.id === app.caseId, onchange: () => selectCase(c.id) }),
      h("span", { class: "body" }, h("span", { class: "title", text: c.title }), h("span", { class: "tag", text: c.tag }), h("span", { class: "sum", text: c.summary }))));
  }
}

function selectCase(id) {
  const c = agentInfo().scenarios.find((x) => x.id === id);
  if (!c) return;
  app.caseId = id;
  if (app.agent === "support") $("#message").value = c.message; else fillTravelForm(c.form);
  updateComposer();
}

const normForm = (f) => JSON.stringify({ o: f.origin.trim(), d: f.destination.trim(), s: f.start_date, e: f.end_date, t: Number(f.travelers), b: f.budget, i: [...f.interests].sort(), n: (f.notes || "").trim() });
const caseMatches = (c) => (app.agent === "support" ? c.message === $("#message").value : normForm(c.form) === normForm(readTravelForm()));

function updateComposer() {
  const c = agentInfo().scenarios.find((x) => x.id === app.caseId);
  if (app.agent === "support") {
    const area = $("#message");
    $("#message-count").textContent = `${area.value.length} / ${area.maxLength}`;
    $("#message-error").textContent = "";
  } else {
    $("#form-error").textContent = "";
  }
  const expect = $("#expect");
  clearEl(expect);
  if (c && caseMatches(c)) {
    expect.hidden = false;
    expect.append(h("b", { text: "A good outcome" }), h("ul", {}, c.expect.map((e) => h("li", { text: e.label }))));
  } else if (c) {
    expect.hidden = false;
    expect.append(h("span", { class: "hint", text: "Outcome checks only apply to the unedited example." }));
  } else {
    expect.hidden = true;
  }
}

/* ---- the travel form */
function isoPlus(days) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function buildTravelForm() {
  const root = $("#composer-travel");
  clearEl(root);
  const opts = app.info.travel_options;
  const field = (id, label, input, hint) => h("div", { class: "field" }, h("label", { for: id, text: label }), input, hint ? h("div", { class: "hint", text: hint }) : null);
  root.append(
    h("div", { class: "field-row" },
      field("tv-origin", "From", h("input", { id: "tv-origin", type: "text", maxLength: 60, value: "Dubai", oninput: updateComposer })),
      field("tv-destination", "To (optional)", h("input", { id: "tv-destination", type: "text", maxLength: 60, placeholder: "Leave empty for ideas", oninput: updateComposer }))),
    h("div", { class: "field-row" },
      field("tv-start", "Depart", h("input", { id: "tv-start", type: "date", min: isoPlus(0), value: isoPlus(45), onchange: updateComposer })),
      field("tv-end", "Return", h("input", { id: "tv-end", type: "date", min: isoPlus(0), value: isoPlus(50), onchange: updateComposer }))),
    h("div", { class: "field-row" },
      field("tv-travelers", "Travelers", h("input", { id: "tv-travelers", type: "number", min: 1, max: 9, value: 2, onchange: updateComposer })),
      h("div", { class: "field" }, h("label", { id: "tv-budget-label", text: "Budget level" }),
        h("div", { class: "seg", role: "radiogroup", "aria-labelledby": "tv-budget-label" },
          opts.budgets.map((b) => h("label", {}, h("input", { type: "radio", name: "tv-budget", value: b, checked: b === "mid-range", onchange: updateComposer }), h("span", { text: b === "mid-range" ? "Mid" : cap(b) })))))),
    h("fieldset", { class: "chips" }, h("legend", { text: "Interests" }),
      opts.interests.map((i) => h("label", { class: "chip-check" }, h("input", { type: "checkbox", class: "tv-int", value: i, onchange: updateComposer }), h("span", { text: cap(i) })))),
    field("tv-notes", "Anything else", h("textarea", { id: "tv-notes", rows: 2, maxLength: 400, placeholder: "Pace, kids, dietary needs…", oninput: updateComposer })),
    h("div", { class: "field-foot" }, h("span", { id: "form-error", class: "field-error", role: "alert" })));
}

function readTravelForm() {
  return {
    origin: $("#tv-origin").value, destination: $("#tv-destination").value, start_date: $("#tv-start").value, end_date: $("#tv-end").value,
    travelers: Number($("#tv-travelers").value) || 0, budget: (document.querySelector("input[name=tv-budget]:checked") || {}).value || "mid-range",
    interests: [...document.querySelectorAll(".tv-int:checked")].map((x) => x.value), notes: $("#tv-notes").value,
  };
}

function fillTravelForm(f) {
  $("#tv-origin").value = f.origin; $("#tv-destination").value = f.destination; $("#tv-start").value = f.start_date; $("#tv-end").value = f.end_date;
  $("#tv-travelers").value = f.travelers; $("#tv-notes").value = f.notes || "";
  for (const r of document.querySelectorAll("input[name=tv-budget]")) r.checked = r.value === f.budget;
  for (const c of document.querySelectorAll(".tv-int")) c.checked = f.interests.includes(c.value);
}

function travelProblem(f) {
  if (!f.origin.trim()) return "Say where the trip starts, so flights can be found.";
  if (!f.start_date || !f.end_date) return "Choose departure and return dates.";
  if (f.start_date < isoPlus(0)) return "The departure date is in the past.";
  if (f.end_date < f.start_date) return "The return date is before the departure date.";
  if ((new Date(f.end_date) - new Date(f.start_date)) / 864e5 > 21) return "Trips are limited to 21 nights.";
  if (!(f.travelers >= 1 && f.travelers <= 9)) return "Travelers must be from 1 to 9.";
  return "";
}

/* ---- run lifecycle */
function newRun(agent) {
  const lane = () => ({ status: "waiting", totals: null, steps: {}, checks: null, changes: null, error: null });
  return { agent, id: null, seq: -1, plan: null, planCost: 0, planNote: null, lanes: { clef: lane(), baseline: lane() }, compare: {}, summary: null, shadow: { decisions: {}, summary: null, on: false, replay: {} },
    error: null, open: new Set(), started: null, done: false, expected: false, source: null, pollTimer: null };
}

async function startRun() {
  const agent = app.agent, desk = app.desks[agent];
  if (desk.run && !desk.run.done && !desk.run.error) return;
  let body, matches = false;
  const c = agentInfo().scenarios.find((x) => x.id === desk.caseId);
  if (agent === "support") {
    const message = $("#message").value;
    if (!message.trim()) { $("#message-error").textContent = "Write a customer message first."; $("#message").focus(); return; }
    matches = !!(c && caseMatches(c));
    body = { agent, message, scenario_id: matches ? desk.caseId : undefined };
  } else {
    const form = readTravelForm(), problem = travelProblem(form);
    if (problem) { $("#form-error").textContent = problem; return; }
    matches = !!(c && caseMatches(c));
    body = { agent, form, scenario_id: matches ? desk.caseId : undefined };
  }
  const run = newRun(agent);
  run.expected = matches;
  desk.run = run;
  setRunning(true);
  renderResults();
  try {
    const { run_id: id } = await api("/api/run", body);
    run.id = id;
    follow(run);
  } catch (error) {
    run.error = error.message;
    run.done = true;
    setRunning(false);
    renderResults();
  }
}

function setRunning(on) {
  const button = $("#run"), copy = AGENT_COPY[app.agent];
  button.disabled = on;
  clearEl(button);
  if (on) button.append(h("span", { class: "spinner", "aria-hidden": "true" }), copy.running); else button.append(copy.run);
}

function follow(run) {
  const source = new EventSource(`/api/run/${run.id}/stream`);
  run.source = source;
  source.onmessage = (ev) => { if (deskOf(run).run === run) handleEvent(run, JSON.parse(ev.data)); };
  source.addEventListener("end", () => { source.close(); finishRun(run); });
  source.onerror = () => { source.close(); if (!run.done && deskOf(run).run === run) poll(run); };
}

function poll(run) {
  clearTimeout(run.pollTimer);
  api(`/api/run/${run.id}?since=${run.seq + 1}`).then((data) => {
    if (deskOf(run).run !== run) return;
    data.events.forEach((e) => handleEvent(run, e));
    if (data.done) finishRun(run); else run.pollTimer = setTimeout(() => poll(run), 800);
  }).catch((error) => { run.error = error.message; finishRun(run); });
}

function finishRun(run) {
  if (run.done) return;
  run.done = true;
  if (run.source) run.source.close();
  clearTimeout(run.pollTimer);
  if (!run.summary && !run.error) run.error = "The run ended before it finished. Try running it again.";
  if (app.agent === run.agent) { setRunning(false); if (app.view === "desk") renderResults(); }
  refreshState().catch(() => {});
}

function laneStep(run, lane, index) {
  const steps = run.lanes[lane].steps;
  steps[index] = steps[index] || { index, status: "running", execs: [], judges: [], notes: [], decisions: [], view: null };
  return steps[index];
}

function handleEvent(run, e) {
  run.seq = Math.max(run.seq, e.seq);
  const lane = e.lane && run.lanes[e.lane];
  switch (e.type) {
    case "run_started": run.started = e; break;
    case "plan": run.plan = e.steps; run.planCost = e.cost_usd; run.planNote = e.note; run.planEvent = e; break;
    case "lane_started": lane.status = "running"; break;
    case "step_started": laneStep(run, e.lane, e.index); break;
    case "decision": laneStep(run, e.lane, e.index).decisions.push(e); break;
    case "exec": laneStep(run, e.lane, e.index).execs.push(e); break;
    case "judge": laneStep(run, e.lane, e.index).judges.push(e); break;
    case "escalation": laneStep(run, e.lane, e.index).notes.push({ kind: "warn", text: `Escalated from ${(modelOf(e.from_model) || { display: e.from_model }).display}. ${e.reason}` }); break;
    case "note": laneStep(run, e.lane, e.index).notes.push({ kind: "info", text: e.text }); break;
    case "exec_error": laneStep(run, e.lane, e.index).notes.push({ kind: "bad", text: `A model call failed (${(modelOf(e.model) || { display: e.model }).display}).` }); break;
    case "step_failed": { const s = laneStep(run, e.lane, e.index); s.status = "failed"; s.notes.push({ kind: "bad", text: "This step could not be completed." }); break; }
    case "step_done": { const s = laneStep(run, e.lane, e.index); s.status = s.status === "failed" ? "failed" : "done"; s.view = e.step; lane.totals = e.totals; break; }
    case "step_compare": run.compare[e.index] = e; break;
    case "lane_totals": lane.totals = e.totals; break;
    case "lane_done": lane.status = lane.error ? "error" : "done"; lane.totals = e.totals; break;
    case "lane_error": lane.error = e.error; lane.status = "error"; break;
    case "outcome": lane.checks = e.checks; lane.changes = e.changes; break;
    case "shadow_started": run.shadow.on = true; run.shadow.started = e; break;
    case "shadow_decision": case "shadow_error": run.shadow.on = true; run.shadow.decisions[e.key] = e; break;
    case "replay_step": run.shadow.replay[e.key] = e; break;
    case "shadow_summary": run.shadow.on = true; run.shadow.summary = e; break;
    case "run_done": run.summary = e; break;
    case "run_error": run.error = "The run stopped unexpectedly. Nothing further will be charged."; break;
    default: break;
  }
  if (!app.frame) app.frame = requestAnimationFrame(() => { app.frame = 0; if (app.view === "desk" && app.agent === run.agent && app.run === run) renderResults(); });
}

/* ---- shadow selector: what the capability router and selector would do, applied to nothing */
const endpointName = (id) => (modelOf(id) || { display: id }).display;

function tierBars(d) {
  return h("div", { class: "bars tier-bars", title: "Clef's probability for how much capability the request needs" },
    ["low", "medium", "high"].map((tier) => h("div", { class: "bar-row" },
      h("span", { class: "name", text: tier }), h("div", { class: "track" }, h("div", { class: "fill", vars: { "--w": `${Math.round((d.tier_probabilities[tier] || 0) * 100)}%` } })),
      h("span", { class: "val", text: fraction(d.tier_probabilities[tier]) }))));
}

function replayCell(run, key, rowSummary) {
  const r = (rowSummary && rowSummary.replay) || run.shadow.replay[key];
  const base = rowSummary && rowSummary.actual ? rowSummary.actual.score : null;
  if (key === "plan") return h("span", { class: "hint", text: "not replayed" });
  if (!r) return run.shadow.summary ? h("span", { class: "hint", text: "not replayed" }) : h("span", { class: "pending" }, h("span", { class: "spinner" }), "Running…");
  if (r.status !== "ok") return h("div", {}, h("span", { class: "chip warn", text: r.status === "failed" ? "failed" : "skipped" }), h("div", { class: "hint", text: r.reason }));
  const worse = r.score !== null && base !== null && base !== undefined && r.score < base;
  return h("div", {}, h("div", { class: `tnum ${worse ? "bad" : ""}`, text: r.score === null ? "not scored" : `${r.score} / 5` }),
    h("div", { class: "hint", text: `${base !== null && base !== undefined ? `strongest ${base} / 5 · ` : ""}${usd(r.cost_usd)} · ${secs(r.latency_ms)}` }));
}

function shadowRow(run, key, rowSummary) {
  const d = run.shadow.decisions[key];
  const title = key === "plan" ? "Plan the case" : ((run.plan && run.plan[Number(key) - 1]) || { title: `Step ${key}` }).title;
  const head = h("td", {}, h("div", { class: "step-title", text: title }), h("div", { class: "hint", text: key === "plan" ? "planner" : (run.plan && run.plan[Number(key) - 1] && run.plan[Number(key) - 1].reply ? "final answer" : "tool step") }));
  if (!d) return h("tr", {}, head, h("td", { colSpan: 5 }, h("span", { class: "pending" }, h("span", { class: "spinner" }), "Asking Clef…")));
  if (d.type === "shadow_error") return h("tr", {}, head, h("td", { colSpan: 5 }, h("span", { class: "muted", text: "Clef did not answer, so the selector was skipped for this call. The run itself was not affected." })));
  const ran = rowSummary && rowSummary.actual;
  const ev = d.evidence, thin = !!ev && ev.pooled_share + ev.missing_share >= 0.5;
  const flags = [d.uncertain ? h("span", { class: "chip warn", title: d.uncertainty_reasons.join("; "), text: "unsure" }) : null,
    d.below_floor ? h("span", { class: "chip warn", title: "No endpoint reached the quality floor", text: "below floor" }) : null,
    thin ? h("span", { class: "chip warn", title: "Most of the expected success comes from results pooled across kinds of work, because the benchmark has few tasks of this exact kind and difficulty", text: "thin evidence" }) : null];
  const cost = rowSummary && rowSummary.est_cost_usd !== null && rowSummary.est_cost_usd !== undefined ? rowSummary.est_cost_usd + rowSummary.router_cost_usd : null;
  const more = h("details", { class: "more" }, h("summary", { text: "Why this endpoint" }),
    h("p", { class: "hint", text: d.reason }),
    h("table", {}, h("thead", {}, h("tr", {}, h("th", { text: "Endpoint" }), h("th", { class: "r", text: "Expected success" }), h("th", { class: "r", text: "Est. cost" }), h("th", { class: "r", text: "Est. time" }), h("th", { text: "Result" }))),
      h("tbody", {}, d.alternatives.map((c) => h("tr", {}, h("td", { text: endpointName(c.endpoint) + (c.price_verified ? "" : " *") }), h("td", { class: "r tnum", text: fraction(c.expected_success) }),
        h("td", { class: "r tnum", text: usd(c.est_cost_usd) }), h("td", { class: "r tnum", text: secs(c.est_latency_ms) }),
        h("td", { class: c.status === "selected" ? "good" : "muted", text: c.status === "selected" ? "Chosen" : c.status === "below_floor" ? "Below floor" : "Eligible" }))))),
    d.eliminated.length ? h("p", { class: "hint", text: "Ruled out: " + d.eliminated.map((c) => `${endpointName(c.endpoint)} (${c.reasons[0]})`).join(", ") + "." }) : null,
    ev ? h("p", { class: "hint", text: `Evidence for the choice: ${fraction(ev.exact_share)} from benchmark tasks of exactly this kind of work and difficulty, ${fraction(ev.pooled_share)} pooled across kinds of work${ev.missing_share ? `, ${fraction(ev.missing_share)} unmeasured` : ""}.` }) : null,
    h("p", { class: "hint", text: `Floor ${fraction(d.floor)}${d.effective_floor !== d.floor ? `, raised to ${fraction(d.effective_floor)} because Clef was unsure` : ""}. Clef answered in ${secs(d.router_ms)}. * price not yet verified.` }));
  return h("tr", {},
    head,
    h("td", {}, tierBars(d), h("div", { class: "hint", text: d.skill_used === false ? `Kind of work: taken from the step (${d.skill_hint ? Object.keys(d.skill_hint)[0] : "none"}); this router has no skill training` : `Kind of work: ${d.skill} (${fraction(d.skill_probabilities[d.skill])})` })),
    h("td", {}, h("div", {}, h("b", { text: endpointName(d.choice) }), " ", flags.filter(Boolean).flatMap((f) => [f, " "])), h("div", { class: "hint", text: `expected success ${fraction(d.expected_success)}${d.baseline.expected_success !== null ? ` · ${endpointName(d.baseline.endpoint)} ${fraction(d.baseline.expected_success)}` : ""}` }), more),
    h("td", { class: "r" }, ran ? h("div", {}, h("div", { class: "tnum", text: usd(ran.cost_usd) }), h("div", { class: "hint", text: endpointName(ran.model) })) : h("span", { class: "skel" })),
    h("td", {}, replayCell(run, key, rowSummary)),
    h("td", { class: "r" }, cost !== null ? h("div", {}, h("div", { class: "tnum", text: usd(cost) }), rowSummary.saved_vs_baseline && rowSummary.saved_vs_baseline.pct !== null ? h("div", { class: `delta ${rowSummary.saved_vs_baseline.value >= 0 ? "good" : "bad"}`, text: `${Math.abs(rowSummary.saved_vs_baseline.pct).toFixed(0)}% ${rowSummary.saved_vs_baseline.value >= 0 ? "less" : "more"}` }) : null) : h("span", { class: "skel" })));
}

function renderShadow(run) {
  const s = run.shadow.summary;
  const keys = ["plan", ...(run.plan || []).map((_, i) => String(i + 1))];
  const byKey = {};
  if (s) for (const row of s.rows) byKey[row.key] = row;
  const saved = s && s.saved_vs_baseline;
  const strip = s && s.decisions ? h("div", { class: "saved-strip" },
    h("span", {}, h("span", { class: "lbl", text: "Always strongest " }), usd(s.priced_baseline_cost_usd)),
    h("span", {}, h("span", { class: "lbl", text: "Selector would cost " }), usd(s.selector_est_cost_usd), saved && saved.pct !== null ? h("span", { class: `delta ${saved.value >= 0 ? "good" : "bad"}`, text: ` (${Math.abs(saved.pct).toFixed(0)}% ${saved.value >= 0 ? "less" : "more"})` }) : null),
    h("span", {}, h("span", { class: "lbl", text: "Clef lane actually cost " }), usd(s.clef_lane_cost_usd)),
    h("span", { class: "run", text: s.mean_expected_success !== null ? `Mean expected success ${fraction(s.mean_expected_success)} · same choice as the Clef lane on ${s.same_choice_as_clef_lane} of ${s.decisions}` : "" })) : null;
  const rp = s && s.replay;
  const checks = rp && rp.checks && rp.checks.replay ? `${rp.checks.replay[0]} / ${rp.checks.replay[1]}` : null;
  const baseChecks = rp && rp.checks && rp.checks.baseline ? `${rp.checks.baseline[0]} / ${rp.checks.baseline[1]}` : null;
  const worse = rp && (rp.below_baseline > 0 || rp.failed > 0 || (rp.checks && rp.checks.replay && rp.checks.baseline && rp.checks.replay[0] < rp.checks.baseline[0]));
  const replayStrip = rp ? h("div", { class: `alert ${worse ? "warn" : ""}`, style: "margin-bottom: 12px" }, h("div", { class: "grow" },
    h("b", { text: worse ? "The selector's choices did worse than the strongest model in this run." : "The selector's choices held up in this run." }),
    h("div", { class: "hint", text: `Replayed ${rp.steps_run} step${rp.steps_run === 1 ? "" : "s"} on the chosen endpoints (${usd(rp.cost_usd)} against ${usd(rp.baseline_cost_usd)} on the strongest model; the reviewer cost ${usd(rp.judge_cost_usd)} and is not counted). ${rp.mean_score !== null ? `Reviewer score ${rp.mean_score} / 5 against ${rp.baseline_mean_score} / 5; ${rp.below_baseline} step${rp.below_baseline === 1 ? "" : "s"} scored lower, ${rp.below_gate} under the ${rp.gate_min_score} / 5 your quality setting needs.` : "No step could be scored."}${checks ? ` Right actions ${checks} against ${baseChecks} for the strongest model.` : ""}${rp.failed ? ` ${rp.failed} call${rp.failed === 1 ? "" : "s"} failed.` : ""}${rp.skipped ? ` ${rp.skipped} skipped.` : ""}${rp.error ? ` The replay stopped early (${rp.error}).` : ""}` }))) : null;
  return h("section", { "aria-labelledby": "shadow-title" },
    h("div", { class: "section-title" }, h("h2", { id: "shadow-title", text: "Shadow selector" }),
      h("span", { class: "hint", text: s ? "Not applied: nothing above used these choices" : "Deciding beside the run…" })),
    h("p", { class: "hint", text: "Clef reads each request and says how much capability it needs. A selector then picks the best fit for your cost and speed settings among the endpoints, out of ten measured on a benchmark, that meet your quality bar. It runs beside the lanes and changes nothing." }),
    strip, replayStrip,
    h("div", { class: "table-wrap" }, h("table", { class: "shadow-table" },
      h("thead", {}, h("tr", {}, h("th", { text: "Call" }), h("th", { text: "Clef says" }), h("th", { text: "Selector would pick" }), h("th", { class: "r", text: "Always strongest ran" }), h("th", { text: "Replayed on its choice" }), h("th", { class: "r", text: "Selector est. cost" }))),
      h("tbody", {}, keys.map((key) => shadowRow(run, key, byKey[key]))))),
    s ? h("p", { class: "hint", text: `${s.note} Each selector cost includes ${usd(s.rows.length ? s.rows[0].router_cost_usd : 0)} for the Clef call.${s.errors ? ` Clef did not answer for ${s.errors} call${s.errors === 1 ? "" : "s"}.` : ""} Registry ${s.versions.registry}.` }) : null);
}

/* ---- results rendering */function renderResults() {
  const root = $("#results");
  clearEl(root);
  const run = app.run;
  if (!run) { root.append(emptyDesk()); return; }
  if (run.error) {
    root.append(h("div", { class: "alert bad", role: "alert" }, h("div", { class: "grow" },
      h("b", { text: "This run did not complete." }),
      h("div", { text: run.error }),
      h("div", { class: "hint", text: "Nothing in the store was changed outside this app. You can run it again." }))));
  }
  if (!run.plan && !run.error) {
    root.append(h("div", { class: "pending" }, h("span", { class: "spinner" }), "Planning the case…"));
    return;
  }
  if (!run.plan) return;
  for (const lane of ["clef", "baseline"]) {
    if (run.lanes[lane].error) {
      root.append(h("div", { class: "alert warn", role: "alert" }, h("div", { class: "grow" },
        h("b", { text: `The ${lane === "clef" ? "with Clef" : "without Clef"} lane stopped early.` }),
        h("div", { class: "hint", text: "The other lane's results are shown. Differences below only cover steps both lanes completed." }))));
    }
  }
  root.append(renderScorecard(run), renderReplies(run), renderPlanning(run), renderSteps(run));
  if (run.shadow.on && !(run.shadow.started && run.shadow.started.applied)) root.append(renderShadow(run));
  if (run.summary) {
    const use = run.summary.tool_usage || {};
    const search = use.web_searches ? ` Web search was shared by both lanes and is not compared: ${use.web_searches} searches, about ${usd(use.search_cost_usd)} in model tokens, before any per-search fee Azure may add.` : "";
    root.append(h("p", { class: "hint", text: `Finished in ${run.summary.elapsed_s} s. Planning cost ${usd(run.summary.planner_cost_usd)} and is shared by both lanes, so it is left out of the comparison.${search}` }));
  }
}

function renderPlanning(run) {
  const e = run.planEvent;
  if (!e || !e.routing) return h("div");
  const r = e.routing, fixed = (modelOf(e.fixed_model) || { display: e.fixed_model }).display;
  const used = (modelOf(e.planner_model) || { display: e.planner_model }).display;
  const top = r.probabilities ? Object.entries(r.probabilities).sort((a, b) => b[1] - a[1])[0] : null;
  const saved = e.fixed_cost_usd ? ` The same tokens on ${fixed} would have cost ${usd(e.fixed_cost_usd)}.` : "";
  const text = r.source === "clef"
    ? `Planned by ${used}. Clef read the planning task as ${ROUTE_LABEL[top[0]] || top[0]} (${fraction(top[1])}); ${r.reason} Cost ${usd(e.cost_usd)} including Clef.${saved}`
    : `Clef did not answer (${r.error || "timeout"}), so ${fixed} planned it. Cost ${usd(e.cost_usd)}.`;
  return h("p", { class: "hint", text: `${text} Both lanes run this one plan.${e.note ? ` ${e.note}` : ""}` });
}

function emptyDesk() {
  const travel = app.agent === "travel";
  return h("div", { class: "empty" },
    h("h2", { text: travel ? "Describe a trip and plan it" : "Pick a customer message and run it" }),
    h("p", { class: "muted", text: travel ? "Give dates, and a destination if you have one. Without one, the agent picks a place using the weather for your dates." : "Every case is handled twice from one shared plan, so you can see what Clef changes." }),
    h("ol", {},
      h("li", { text: travel ? "The agent plans the trip, with specialist sub-agents for flights, hotels, sights, food and weather that work in parallel." : "The agent plans the case into steps and tool calls." }),
      h("li", { text: "With Clef: each step is classified, then your preferences pick the model." }),
      h("li", { text: "Without Clef: every step goes to the strongest model with every tool attached." }),
      h("li", { text: travel ? "You compare cost, tokens, speed, itinerary quality and whether flights and a hotel were held." : "You compare cost, tokens, speed, reply quality and whether the right actions happened." })));
}

function cumulative(run) {
  const total = { cost: 0, tokens: 0, latency: 0, steps: 0 };
  for (const e of Object.values(run.compare)) {
    total.cost += e.saved.cost.value; total.tokens += e.saved.tokens.value; total.latency += e.saved.latency.value; total.steps += 1;
  }
  return total;
}

function deltaCell(value, formatter, unit, { lowerIsBetter = true, base = null } = {}) {
  if (value === null || value === undefined) return h("td", { class: "r" }, h("span", { class: "skel" }));
  const better = lowerIsBetter ? value > 0 : value < 0;
  const flat = Math.abs(value) < 1e-9;
  const share = base ? Math.abs(value) / base * 100 : null;
  return h("td", { class: "r" }, h("div", { class: `delta ${flat ? "" : better ? "good" : "bad"}` },
    flat ? "No difference" : `${formatter(Math.abs(value))} ${better ? unit[0] : unit[1]}`,
    share !== null && !flat ? h("span", { class: "sub", text: `${share < 1 ? "<1" : share.toFixed(0)}% ${better ? "lower" : "higher"}` }) : null));
}

function renderScorecard(run) {
  const done = !!run.summary;
  const clef = run.lanes.clef, base = run.lanes.baseline;
  const t = (lane) => lane.totals;
  const cum = done ? { cost: run.summary.saved.cost.value, tokens: run.summary.saved.tokens.value, latency: run.summary.saved.latency.value } : cumulative(run);
  const showDiff = done || Object.keys(run.compare).length > 0;
  const val = (lane, key, fmt) => (t(lane) && t(lane)[key] !== null && t(lane)[key] !== undefined) ? h("span", { class: "big" }, fmt(t(lane)[key])) : h("span", { class: "skel" });
  const laneHead = (name, sub, lane) => h("th", { class: "r" }, h("div", { class: "lane-name", style: "justify-content:flex-end" },
    lane.status === "running" ? h("span", { class: "spinner", "aria-label": "Running" }) : lane.status === "done" ? h("span", { class: "good", "aria-label": "Done", text: "✓" }) : lane.status === "error" ? h("span", { class: "bad", "aria-label": "Stopped", text: "!" }) : null, name),
    h("div", { class: "lane-sub", text: sub }));
  const cTot = t(clef) || {}, bTot = t(base) || {};
  const quality = (lane) => (t(lane) && t(lane).reply_score !== null && t(lane).reply_score !== undefined) ? h("span", { class: "big", text: `${t(lane).reply_score} / 5` }) : (lane.status === "done" ? h("span", { class: "big", text: "n/a" }) : h("span", { class: "skel" }));
  const checkTotals = (lane) => lane.checks && lane.checks.length ? `${lane.checks.filter((c) => c.passed).length} / ${lane.checks.length}` : null;
  const cChecks = checkTotals(clef), bChecks = checkTotals(base);
  const qDelta = (clef.totals && base.totals && clef.totals.reply_score != null && base.totals.reply_score != null) ? clef.totals.reply_score - base.totals.reply_score : null;
  const baselineName = (modelOf(run.started ? run.started.baseline_model : "") || { display: "the strongest model" }).display;
  const rows = [
    h("tr", {}, h("td", { class: "metric", text: "Cost" }),
      h("td", { class: "r" }, val(clef, "cost_usd", usd), cTot.router_cost_usd || cTot.check_cost_usd ? h("div", { class: "sub", text: `includes Clef ~${usd(cTot.router_cost_usd)} · checks ${usd(cTot.check_cost_usd)}` }) : null),
      h("td", { class: "r" }, val(base, "cost_usd", usd)),
      showDiff ? deltaCell(cum.cost, usd, ["saved", "more"], { base: bTot.cost_usd }) : h("td", { class: "r" }, h("span", { class: "skel" }))),
    h("tr", {}, h("td", { class: "metric", text: "Tokens" }),
      h("td", { class: "r" }, val(clef, "tokens_total", int), cTot.check_tokens ? h("div", { class: "sub", text: `${int(cTot.exec_tokens)} work + ${int(cTot.check_tokens)} checks` }) : null),
      h("td", { class: "r" }, val(base, "tokens_total", int)),
      showDiff ? deltaCell(cum.tokens, int, ["saved", "more"], { base: bTot.tokens_total }) : h("td", { class: "r" }, h("span", { class: "skel" }))),
    h("tr", {}, h("td", { class: "metric", text: "Time" }),
      h("td", { class: "r" }, val(clef, "latency_ms", secs), h("div", { class: "sub", text: "adds Clef and checks" })),
      h("td", { class: "r" }, val(base, "latency_ms", secs)),
      showDiff ? deltaCell(cum.latency, secs, ["faster", "slower"], { base: bTot.latency_ms }) : h("td", { class: "r" }, h("span", { class: "skel" }))),
    h("tr", {}, h("td", { class: "metric", text: AGENT_COPY[run.agent].quality }),
      h("td", { class: "r" }, quality(clef), h("div", { class: "sub", text: "scored by an AI reviewer" })),
      h("td", { class: "r" }, quality(base)),
      qDelta === null ? h("td", { class: "r" }, done ? "n/a" : h("span", { class: "skel" })) : deltaCell(qDelta, (x) => x.toFixed(0), ["higher", "lower"], { lowerIsBetter: false })),
    h("tr", {}, h("td", { class: "metric", text: "Right actions" }),
      h("td", { class: "r" }, cChecks ? h("span", { class: "big", text: cChecks }) : h("span", { class: "sub", text: done || !run.expected ? "n/a for custom requests" : "…" })),
      h("td", { class: "r" }, bChecks ? h("span", { class: "big", text: bChecks }) : h("span", { class: "sub", text: done || !run.expected ? "n/a" : "…" })),
      cChecks && bChecks ? deltaCell(clef.checks.filter((c) => c.passed).length - base.checks.filter((c) => c.passed).length, (x) => x.toFixed(0), ["more passed", "fewer passed"], { lowerIsBetter: false }) : h("td", { class: "r" }))];
  return h("section", { "aria-labelledby": "score-title" },
    h("div", { class: "section-title" }, h("h2", { id: "score-title", text: "Comparison" }),
      h("span", { class: "hint", text: done ? "Final totals" : "Updating as steps finish" })),
    h("div", { class: "table-wrap" }, h("table", { class: "scorecard" },
      h("thead", {}, h("tr", {}, h("th", {}),
        laneHead("With Clef", "Each step routed by your preferences", clef),
        laneHead("Without Clef", `${baselineName} on every step, all tools attached`, base),
        h("th", { class: "r" }, h("div", { class: "lane-name", style: "justify-content:flex-end", text: "Difference" }), h("div", { class: "lane-sub", text: done ? "Whole case" : "Completed steps so far" })))),
      h("tbody", {}, rows),
      h("tfoot", {}, h("tr", { class: "foot" }, h("td", { colSpan: 4, title: app.info.router_cost_note },
        `Clef's cost is an estimate of GPU time per decision (${usd(app.info.router_cost_usd)}); quality checks are counted for With Clef because they are part of keeping quality up. Clef's own tokens (${int(cTot.clef_tokens)}) run on your hosted model and are not billed per token. Tool time (web, weather) is left out of Time.`))))));
}

function finalStep(run, laneName) {
  if (!run.plan) return null;
  return run.lanes[laneName].steps[run.plan.length] || null;
}
const lastExec = (step) => (step && step.execs.length ? step.execs[step.execs.length - 1] : null);

function renderReplies(run) {
  const col = (laneName, title) => {
    const lane = run.lanes[laneName], step = finalStep(run, laneName), exec = lastExec(step);
    const score = step && step.view ? step.view.score : null;
    return h("div", { class: `reply ${laneName}` },
      h("div", { class: "reply-head" }, h("h3", { text: title }),
        h("span", { class: "hint" }, exec ? [modelChip(exec.model), " · "] : null, score !== null && score !== undefined ? h("span", { class: "score", text: `${score} / 5` }) : null)),
      exec ? (run.agent === "travel" ? renderItinerary(exec.output) : h("div", { class: "reply-text", text: exec.output || "(empty reply)" }))
        : step && step.status === "failed" ? h("div", { class: "reply-text bad", text: "No reply was produced for this lane." })
          : h("div", { class: "reply-text" }, lane.status === "error" ? "This lane stopped before replying." : h("span", { class: "pending" }, h("span", { class: "spinner" }), "Working on earlier steps…")),
      lane.checks && lane.checks.length ? h("ul", { class: "checks" }, lane.checks.map((c) => h("li", {},
        h("span", { class: `mark ${c.passed ? "good" : "bad"}`, "aria-label": c.passed ? "Passed" : "Missed", text: c.passed ? "✓" : "✕" }), c.label))) : null,
      lane.changes ? h("div", { class: "changes", text: lane.changes.length ? `${AGENT_COPY[run.agent].actions}: ${lane.changes.join("; ")}.` : `${AGENT_COPY[run.agent].actions}: none.` }) : null);
  };
  const travel = run.agent === "travel", mode = travel ? app.desks.travel.itin : "both";
  const cols = [mode !== "baseline" ? col("clef", "With Clef") : null, mode !== "clef" ? col("baseline", "Without Clef") : null].filter(Boolean);
  const toggle = h("div", { class: "seg compact", role: "radiogroup", "aria-label": "Which itinerary to show" },
    [["clef", "With Clef"], ["baseline", "Without"], ["both", "Side by side"]].map(([value, label]) => h("label", {},
      h("input", { type: "radio", name: "itin-view", value, checked: mode === value, onchange: () => { app.desks.travel.itin = value; renderResults(); } }), h("span", { text: label }))));
  return h("section", { "aria-labelledby": "reply-title" },
    h("div", { class: "section-title" }, h("h2", { id: "reply-title", text: AGENT_COPY[run.agent].replyTitle }), travel ? toggle : null),
    h("div", { class: cols.length > 1 ? "replies" : "single" }, cols));
}

const KIND_LABEL = { sight: "See", food: "Eat", transport: "Travel", rest: "Rest" };

function renderItinerary(text) {
  let data = null;
  try { data = JSON.parse(text); } catch (_) { /* shown as plain text below */ }
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return h("div", {}, h("div", { class: "alert warn" }, h("div", { class: "grow" }, h("b", { text: "This itinerary could not be displayed." }),
      h("div", { class: "hint", text: "The model did not return a complete, valid plan. The raw text is below." }))),
    h("pre", { class: "out", text: text || "(empty reply)" }));
  }
  const s = (v) => (v === null || v === undefined ? "" : String(v));
  const money = (v) => (typeof v === "number" ? `$${Math.round(v).toLocaleString("en-US")}` : s(v));
  const link = (url, label) => (typeof url === "string" && url.startsWith("https://") ? h("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: label }) : null);
  const host = (url) => { try { return new URL(url).hostname.replace(/^www\./, ""); } catch (_) { return url; } };
  const section = (title, ...kids) => h("section", { class: "itin-sec" }, h("h4", { text: title }), kids.flat(Infinity));
  const list = (items) => (Array.isArray(items) && items.length ? h("ul", {}, items.map((x) => h("li", { text: s(x) }))) : null);
  const kv = (pairs) => h("dl", { class: "kv" }, pairs.filter(([, v]) => s(v)).flatMap(([k, v]) => [h("dt", { text: k }), h("dd", { text: s(v) })]));
  const f = data.flights && typeof data.flights === "object" ? data.flights : null;
  const hotel = data.hotel && typeof data.hotel === "object" ? data.hotel : null;
  const days = Array.isArray(data.days) ? data.days : [];
  const budget = data.budget && typeof data.budget === "object" ? data.budget : null;
  const sources = (Array.isArray(data.sources) ? data.sources : []).filter((u) => typeof u === "string" && u.startsWith("https://")).slice(0, 12);
  return h("article", { class: "itin" },
    h("header", { class: "itin-head" }, h("h3", { text: s(data.title) || "Itinerary" }),
      h("p", { class: "sub", text: [data.destination, data.dates].map(s).filter(Boolean).join(" · ") }),
      data.summary ? h("p", { text: s(data.summary) }) : null,
      data.why_this_destination ? h("p", { class: "why" }, h("b", { text: "Why this destination: " }), s(data.why_this_destination)) : null),
    f ? section("Getting there", kv([["Carrier", f.carrier], ["Outbound", f.outbound], ["Return", f.return], ["Total", money(f.total_usd)], ["Offer", `${s(f.offer_id)}${f.held ? " · held (simulated)" : ""}`]])) : null,
    hotel ? section("Where you stay", kv([["Hotel", `${s(hotel.name)}${hotel.stars ? ` (${s(hotel.stars)}${/^\d/.test(s(hotel.stars)) ? " stars" : ""})` : ""}`], ["Stay", hotel.nights ? `${s(hotel.nights)} nights` : ""], ["Estimated total", money(hotel.total_usd)], ["Offer", `${s(hotel.offer_id)}${hotel.held ? " · held (simulated)" : ""}`]])) : null,
    data.weather ? section("Weather", h("p", { text: s(data.weather) })) : null,
    days.length ? section("Day by day", days.map((d) => h("div", { class: "itin-day" },
      h("h5", { text: `Day ${s(d.day)} · ${s(d.date)}${d.title ? ` · ${s(d.title)}` : ""}` }),
      h("ul", {}, (Array.isArray(d.items) ? d.items : []).map((it) => h("li", {},
        h("span", { class: "t", text: s(it.time) }), h("span", { class: `k k-${s(it.kind)}`, text: KIND_LABEL[s(it.kind)] || "" }),
        h("span", { class: "b" }, h("b", { text: s(it.title) }), " ", s(it.detail), " ", link(it.url, "source")))))))) : null,
    budget ? section("Budget", kv([["Flights", money(budget.flights_usd)], ["Hotel", money(budget.hotel_usd)], ["Food and activities", money(budget.food_and_activities_usd)], ["Total", money(budget.total_usd)], ["Note", budget.note]])) : null,
    Array.isArray(data.tips) && data.tips.length ? section("Tips", list(data.tips)) : null,
    sources.length ? section("Sources", h("ul", { class: "sources" }, sources.map((u) => h("li", {}, link(u, host(u)))))) : null,
    Array.isArray(data.caveats) && data.caveats.length ? section("Good to know", list(data.caveats)) : null);
}

function laneLine(run, laneName, index) {
  const step = run.lanes[laneName].steps[index];
  const who = laneName === "clef" ? "With Clef" : "Without Clef";
  if (!step) return h("div", { class: "lane-line" }, h("span", { class: "who", text: who }), h("span", { class: "pending", style: "grid-column: span 4" }, run.lanes[laneName].status === "error" ? "Not reached" : "Waiting for earlier steps"));
  const v = step.view;
  if (!v) {
    const e = lastExec(step), d = step.decisions[step.decisions.length - 1];
    return h("div", { class: "lane-line" }, h("span", { class: "who", text: who }),
      h("span", { class: "pending", style: "grid-column: span 4" }, step.status === "failed" ? "Failed" : h("span", { class: "spinner" }),
        step.status === "failed" ? null : (e ? "Checking the result…" : d ? `Running on ${(modelOf(d.decision.selected_model) || { display: "a model" }).display}…` : "Choosing a model…")));
  }
  return h("div", { class: "lane-line" },
    h("span", { class: "who" }, who, h("small", {}, v.failed ? "Failed" : modelChip(v.model), v.attempts > 1 ? ` · ${v.attempts} attempts` : "")),
    h("span", { class: "v" }, usd(v.cost_usd), h("small", { text: "cost" })),
    h("span", { class: "v" }, int(v.tokens), h("small", { text: "tokens" })),
    h("span", { class: "v" }, secs(v.latency_ms), h("small", { text: "time" })),
    h("span", { class: "v" }, v.score !== null && v.score !== undefined ? `${v.score} / 5` : "n/a", h("small", { text: "check" })));
}

function signed(value, format, words) {
  return Math.abs(value) < 1e-9 ? "no change" : `${format(Math.abs(value))} ${value > 0 ? words[0] : words[1]}`;
}

function strip(run, index, running) {
  const cmp = run.compare[index];
  if (!cmp) return h("div", { class: "saved-strip" }, h("span", { class: "pending" }, "Waiting for both lanes to finish this step"));
  const part = (label, saved, fmt, unit) => {
    const flat = Math.abs(saved.value) < 1e-9, better = saved.value > 0;
    return h("span", {}, h("span", { class: "lbl", text: `${label} ` }), h("span", { class: `delta ${flat ? "" : better ? "good" : "bad"}`, text: flat ? "same" : `${fmt(Math.abs(saved.value))} ${better ? unit[0] : unit[1]}${saved.pct !== null ? ` (${Math.abs(saved.pct).toFixed(0)}%)` : ""}` }));
  };
  return h("div", { class: "saved-strip" },
    part("Cost", cmp.saved.cost, usd, ["saved", "more"]), part("Tokens", cmp.saved.tokens, int, ["saved", "more"]), part("Time", cmp.saved.latency, secs, ["faster", "slower"]),
    h("span", { class: "run", text: `Running total: ${signed(running.cost, usd, ["saved", "more"])} · ${signed(running.tokens, (x) => `${int(x)} tokens`, ["fewer", "more"])} · ${signed(running.latency, secs, ["faster", "slower"])}` }));
}

function renderSteps(run) {
  const running = { cost: 0, tokens: 0, latency: 0 };
  const items = run.plan.map((planned, i) => {
    const index = i + 1;
    const cmp = run.compare[index];
    if (cmp) { running.cost += cmp.saved.cost.value; running.tokens += cmp.saved.tokens.value; running.latency += cmp.saved.latency.value; }
    const cs = run.lanes.clef.steps[index];
    const d = cs && cs.decisions.length ? cs.decisions[cs.decisions.length - 1].decision : null;
    const details = h("details", { class: "more", open: run.open.has(index), ontoggle: (ev) => { if (ev.target.open) run.open.add(index); else run.open.delete(index); } },
      h("summary", { text: "Show how each lane handled it" }), stepDetails(run, index));
    return h("li", { class: "step" },
      h("div", { class: "step-head" }, h("span", { class: "step-n", text: String(index) }), h("span", { class: "step-title", text: planned.title }),
        planned.specialist ? h("span", { class: "spec", title: planned.parallel ? "Runs at the same time as its neighbours" : "Runs on its own", text: `${planned.specialist}${planned.parallel ? " · parallel" : ""}` }) : null,
        planned.reply ? h("span", { class: "hint", text: "final answer" }) : null,
        d ? h("span", { class: "route" }, "Clef sees: ", h("b", { text: ROUTE_LABEL[d.clef.route] || d.clef.route }), d.clef.source === "fallback" ? " Clef did not answer in time, so a safe default was used." : ` ${fraction(d.clef.confidence)}`) : null),
      h("div", { class: "lanes" }, laneLine(run, "clef", index), laneLine(run, "baseline", index)),
      strip(run, index, { ...running }),
      cs ? cs.notes.map((n) => h("div", { class: `note ${n.kind === "bad" ? "bad" : ""}`, text: n.text })) : null,
      details);
  });
  return h("section", { "aria-labelledby": "steps-title" },
    h("div", { class: "section-title" }, h("h2", { id: "steps-title", text: "Step by step" }),
      h("span", { class: "hint", text: run.planNote || (run.agent === "travel" ? "Both lanes follow the same plan; parallel steps run as sub-agents." : "Both lanes follow the same plan.") })),
    h("ol", { class: "steps" }, items));
}

function toolCalls(step) {
  const calls = step ? step.execs.flatMap((e) => e.tool_calls || []) : [];
  if (!calls.length) return h("p", { class: "hint", text: "No tools were called." });
  return h("pre", { class: "out", text: calls.map((c) => `${c.name}(${Object.entries(c.arguments).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ")})\n  → ${JSON.stringify(c.result)}`).join("\n") });
}

function stepDetails(run, index) {
  const cs = run.lanes.clef.steps[index], bs = run.lanes.baseline.steps[index];
  const d = cs && cs.decisions.length ? cs.decisions[cs.decisions.length - 1].decision : null;
  const judgeLine = (step) => {
    const j = step && step.judges.length ? step.judges[step.judges.length - 1] : null;
    const by = j && j.routing ? ` Checked by ${(modelOf(j.model) || { display: j.model }).display}${j.fell_back ? " (Clef's pick did not give a score, so the fixed reviewer was used)" : ""}.` : "";
    return j ? h("p", { class: "hint", text: `Check: ${j.score === null ? "unavailable" : `${j.score}/5`}. ${j.reason}${by}` }) : null;
  };
  const clefCol = h("div", {}, h("h4", { text: "With Clef" }));
  if (d) {
    const top = Object.entries(d.clef.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3);
    clefCol.append(
      h("dl", { class: "kv" },
        h("dt", { text: "Why" }), h("dd", { text: d.reason }),
        h("dt", { text: "Clef" }), h("dd", { text: d.clef.source === "fallback" ? `No answer (${d.clef.error || "timeout"}). Safe default step type used.` : top.map(([r, p]) => `${ROUTE_LABEL[r] || r} ${fraction(p)}`).join(" · ") }),
        h("dt", { text: "Needs" }), h("dd", { text: d.requirement.parts.map(([n, v]) => `${n} ${v >= 0 ? "+" : ""}${(v * 100).toFixed(0)}`).join(" · ") + ` = ${fraction(d.requirement.total)}` })),
      h("table", {}, h("thead", {}, h("tr", {}, h("th", { text: "Model" }), h("th", { class: "r", text: "Expected quality" }), h("th", { class: "r", text: "Est. cost" }), h("th", { text: "Result" }))),
        h("tbody", {}, d.candidates.map((c) => h("tr", { class: c.status === "eliminated" ? "off" : "" },
          h("td", { text: c.display }), h("td", { class: "r tnum", text: fraction(c.quality) }), h("td", { class: "r tnum", text: usd(c.est_cost) }),
          h("td", { class: c.status === "selected" ? "good" : "muted", text: c.status === "selected" ? "Chosen" : c.status === "below_floor" ? "Below your quality bar" : c.reasons[0] || "Not chosen" }))))));
  }
  clefCol.append(h("h4", { text: "Tool calls" }), toolCalls(cs), h("h4", { text: "Output" }),
    h("pre", { class: "out", text: (lastExec(cs) || { output: "Waiting…" }).output }), judgeLine(cs));
  const baseCol = h("div", {}, h("h4", { text: "Without Clef" }), h("h4", { text: "Tool calls" }), toolCalls(bs), h("h4", { text: "Output" }),
    h("pre", { class: "out", text: (lastExec(bs) || { output: "Waiting…" }).output }), judgeLine(bs));
  return h("div", { class: "detail-grid" }, clefCol, baseCol);
}

/* ----------------------------------------------------------------- models */
function renderModels() {
  const root = $("#models-table");
  clearEl(root);
  const toggle = (model, field) => h("input", { type: "checkbox", class: "toggle", checked: model[field], "aria-label": `${field === "healthy" ? "Healthy" : "Allowed"}: ${model.display}`,
    onchange: async (ev) => {
      try { await api(`/api/registry/${model.id}`, { [field]: ev.target.checked }); await refreshState(); renderModels(); toast("Registry updated. It applies to the next run."); }
      catch (error) { toast(error.message); ev.target.checked = !ev.target.checked; }
    } });
  root.append(h("table", {},
    h("thead", {}, h("tr", {}, h("th", { text: "Model" }), h("th", { text: "State" }), h("th", { text: "Healthy" }), h("th", { text: "Allowed" }),
      h("th", { class: "r", text: "USD per 1M in / out" }), h("th", { class: "r", text: "Latency prior" }), h("th", { text: "Assumed skills" }))),
    h("tbody", {}, app.info.models.map((m) => h("tr", { class: m.live && m.healthy && m.allowed ? "" : "off" },
      h("td", {}, h("b", { text: m.display }), " ", h("span", { class: `tier-${m.tier}`, text: m.tier }), h("div", { class: "hint mono", text: m.id }), h("div", { class: "hint", text: m.notes })),
      h("td", { text: m.live ? "Deployed" : "Catalog only" }),
      h("td", {}, toggle(m, "healthy")), h("td", {}, toggle(m, "allowed")),
      h("td", { class: "r tnum" }, `${m.price_in} / ${m.price_out}`, h("div", { class: "hint", text: m.price_source })),
      h("td", { class: "r tnum", text: `${int(m.lat_base_ms)} ms + ${m.lat_per_token_ms} ms/token` }),
      h("td", { class: "tnum", text: Object.entries(m.skills).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(" · ") }))))));
  buildAddForm();
}

function buildAddForm() {
  const form = $("#add-model");
  if (form.dataset.built) return;
  form.dataset.built = "1";
  const input = (id, label, type = "text", extra = {}) => h("div", {}, h("label", { for: `am-${id}`, text: label }), h("input", { id: `am-${id}`, type, ...extra }));
  form.append(
    input("id", "Model id", "text", { required: true, placeholder: "my-model" }), input("display", "Display name", "text", { placeholder: "My Model" }),
    input("deployment", "Deployment name", "text", { required: true }),
    h("div", {}, h("label", { for: "am-tier", text: "Tier" }), h("select", { id: "am-tier" }, ["small", "mid", "frontier"].map((t) => h("option", { value: t, text: t })))),
    input("price_in", "USD per 1M input tokens", "number", { step: "0.01", min: "0", value: "1" }),
    input("price_out", "USD per 1M output tokens", "number", { step: "0.01", min: "0", value: "4" }),
    input("lat_base", "Latency base (ms)", "number", { min: "0", value: "2000" }),
    input("lat_tok", "Latency per token (ms)", "number", { min: "0", value: "15" }),
    ...["coding", "reasoning", "tools", "general"].map((s) => input(`skill-${s}`, `${s} skill (0 to 1)`, "number", { step: "0.01", min: "0", max: "1", value: "0.6" })),
    h("label", { class: "check wide" }, h("input", { type: "checkbox", id: "am-live" }), h("span", {}, h("b", { text: "Deployed and callable" }), h("div", { class: "hint", text: "Leave off to add it to the catalog without routing to it." }))),
    h("div", { class: "wide" }, h("button", { type: "submit", class: "btn primary", text: "Add to registry" }), " ", h("span", { id: "am-msg", class: "hint", role: "status" })));
  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const num = (id) => Number($(`#am-${id}`).value);
    try {
      await api("/api/registry", {
        id: $("#am-id").value.trim(), display: $("#am-display").value.trim() || $("#am-id").value.trim(), provider: "azure-openai",
        deployment: $("#am-deployment").value.trim(), tier: $("#am-tier").value, price_in: num("price_in"), price_out: num("price_out"),
        price_source: "entered by user", lat_base_ms: num("lat_base"), lat_per_token_ms: num("lat_tok"), live: $("#am-live").checked,
        skills: Object.fromEntries(["coding", "reasoning", "tools", "general"].map((s) => [s, num(`skill-${s}`)])),
      });
      $("#am-msg").textContent = "Added.";
      await refreshState();
      renderModels();
    } catch (error) {
      $("#am-msg").textContent = error.message;
    }
  });
}

/* --------------------------------------------------------------- insights */
async function loadInsights() {
  const root = $("#insights");
  clearEl(root);
  root.append(h("div", { class: "pending" }, h("span", { class: "spinner" }), "Loading…"));
  let data;
  try { data = await api(app.insightsAgent === "all" ? "/api/insights" : `/api/insights?agent=${app.insightsAgent}`); } catch (error) {
    clearEl(root);
    root.append(h("div", { class: "alert bad", role: "alert" }, h("div", { class: "grow" }, h("b", { text: "Insights could not load." }), h("div", { class: "hint", text: error.message })),
      h("button", { type: "button", class: "btn", text: "Try again", onclick: loadInsights })));
    return;
  }
  clearEl(root);
  const r = data.runs;
  root.append(h("div", { class: "seg compact", role: "radiogroup", "aria-label": "Agent", style: "margin-bottom:24px" },
    [["all", "All agents"], ["support", "Support desk"], ["travel", "Travel planner"]].map(([value, label]) => h("label", {},
      h("input", { type: "radio", name: "insights-agent", value, checked: app.insightsAgent === value, onchange: () => { app.insightsAgent = value; loadInsights(); } }), h("span", { text: label })))));
  if (!r.runs) {
    root.append(h("div", { class: "empty" }, h("h2", { text: "No runs yet" }),
      h("p", { class: "muted", text: "Insights add up every case you run, so they show measured savings rather than estimates." }),
      h("p", {}, h("button", { type: "button", class: "btn primary", text: "Go to the support desk", onclick: () => setView("desk", { agent: "support" }) }), " ",
        h("button", { type: "button", class: "btn", text: "Go to the travel planner", onclick: () => setView("desk", { agent: "travel" }) }))));
    return;
  }
  const stat = (label, big, sub, kind = "") => h("div", { class: "stat" }, h("div", { class: "label", text: label }), h("div", { class: `big ${kind}`, text: big }), h("div", { class: "sub", text: sub }));
  const saved = (p) => (p === null ? "n/a" : p >= 0 ? `${p}% lower` : `${Math.abs(p)}% higher`);
  const kind = (p) => (p === null ? "" : p > 0 ? "good" : p < 0 ? "bad" : "");
  root.append(h("div", { class: "stats" },
    stat("Cost", saved(r.cost_usd.saved_pct), `${usd(r.cost_usd.clef)} with Clef vs ${usd(r.cost_usd.baseline)} without, over ${r.runs} runs`, kind(r.cost_usd.saved_pct)),
    stat("Tokens", saved(r.tokens.saved_pct), `${int(r.tokens.clef)} vs ${int(r.tokens.baseline)}, checks included`, kind(r.tokens.saved_pct)),
    stat("Time per run", saved(r.latency_ms.saved_pct), `${secs(r.latency_ms.clef)} vs ${secs(r.latency_ms.baseline)} on average`, kind(r.latency_ms.saved_pct)),
    stat("Answer quality", `${r.reply_score.clef ?? "n/a"} vs ${r.reply_score.baseline ?? "n/a"}`, "average out of 5, with vs without Clef")));
  const mixTotal = Object.values(r.model_mix).reduce((a, m) => a + m.steps, 0) || 1;
  const routes = data.decisions.route_counts || {};
  const routeTotal = Object.values(routes).reduce((a, n) => a + n, 0) || 1;
  const barRow = (label, f, value) => h("div", { class: "bar-row" }, h("span", { class: "name", title: label, text: label }), h("div", { class: "track" }, h("div", { class: "fill", vars: { "--w": `${Math.min(1, f) * 100}%` } })), h("span", { class: "val", text: value }));
  root.append(h("div", { class: "two-col" },
    h("div", {}, h("h2", { text: "Which models handled the steps" }), h("p", { class: "hint", text: "With Clef lane only. Without Clef uses one model for everything." }),
      h("div", { class: "bars" }, Object.entries(r.model_mix).sort((a, b) => b[1].steps - a[1].steps).map(([id, m]) => barRow((modelOf(id) || { display: id }).display, m.steps / mixTotal, `${m.steps} steps`)))),
    h("div", {}, h("h2", { text: "What Clef saw" }), h("p", { class: "hint", text: `Router time p50 ${secs(data.decisions.router_ms_p50)}, p95 ${secs(data.decisions.router_ms_p95)}.` }),
      h("div", { class: "bars" }, Object.entries(routes).sort((a, b) => b[1] - a[1]).map(([id, n]) => barRow(ROUTE_LABEL[id] || id, n / routeTotal, String(n)))))));
  try {
    const sh = await api("/api/shadow");
    if (sh.enabled && sh.decisions) {
      const total = sh.decisions || 1;
      root.append(h("h2", { text: "Shadow selector", style: "margin-top:32px" }),
        h("p", { class: "hint", text: `${sh.decisions} decisions over ${sh.runs} runs, none applied. Mean expected success ${fraction(sh.mean_expected_success)}; ${sh.uncertain} were flagged unsure, ${sh.below_floor} had no endpoint above the floor and ${sh.thin_evidence} rest mostly on pooled evidence.${sh.errors ? ` Clef did not answer ${sh.errors} times.` : ""}` }),
        h("div", { class: "two-col" },
          h("div", {}, h("h3", { text: "Endpoints it would pick" }), h("div", { class: "bars" }, Object.entries(sh.by_endpoint).sort((a, b) => b[1] - a[1]).map(([id, n]) => barRow(endpointName(id), n / total, String(n))))),
          h("div", {}, h("h3", { text: "Capability Clef saw" }), h("div", { class: "bars" }, Object.entries(sh.by_tier).sort((a, b) => b[1] - a[1]).map(([id, n]) => barRow(id, n / total, String(n)))))));
    }
  } catch (_) { /* the shadow totals are optional */ }
  root.append(h("h2", { text: "Recent runs", style: "margin-top:32px" }),
    h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, h("th", { text: "When" }), h("th", { text: "Agent" }), h("th", { text: "Preferences" }), h("th", { class: "r", text: "Cost" }), h("th", { class: "r", text: "Tokens" }), h("th", { class: "r", text: "Time" }), h("th", { class: "r", text: "Quality (with vs without)" }))),
      h("tbody", {}, r.recent.map((x) => h("tr", {},
        h("td", { text: new Date(x.created_utc).toLocaleString() }),
        h("td", { text: x.agent === "travel" ? "Travel planner" : "Support desk" }),
        h("td", { text: `${LEVEL_NAME[x.preferences.quality]} · ${LEVEL_NAME[x.preferences.latency]} · ${LEVEL_NAME[x.preferences.cost]}` }),
        ...[x.cost_saved_pct, x.tokens_saved_pct, x.latency_saved_pct].map((p) => h("td", { class: `r tnum ${kind(p)}`, text: saved(p) })),
        h("td", { class: "r tnum", text: `${x.clef_score ?? "n/a"} vs ${x.baseline_score ?? "n/a"}` })))))));
}

/* ------------------------------------------------------------------ about */
function renderAbout() {
  const root = $("#about");
  clearEl(root);
  root.append(h("div", { class: "about" },
    h("h1", { id: "about-title", text: "How it works" }),
    h("p", { class: "lede", text: "This is a small, honest proof of concept for the Adaptive Intelligence Engine idea: send each step of an agent's work to the cheapest model that is good enough." }),
    h("h2", { text: "What happens to a customer message" }),
    h("ol", {},
      h("li", { text: "A planner turns the message into steps (Clef picks which model plans it, with GPT-4.1 as the fallback). The last step is always the reply to the customer." }),
      h("li", { text: "With Clef: the fine-tuned Clef-Flash model reads each step and the available tools, and says what kind of step it is, with a confidence." }),
      h("li", { text: "A transparent policy turns that step type, how risky the step looks, and your preferences into a capability the model must have, then picks the best-value model that meets it." }),
      h("li", { text: "A reviewer checks the result, on a model Clef also picks (held to a higher bar than the work it checks), and a weak result is retried on a stronger model. A step that already changed the store is never repeated." }),
      h("li", { text: "Without Clef: the same plan runs on the strongest model with every tool attached. Both lanes use separate copies of the store." })),
    h("h2", { text: "The travel planner" }),
    h("ul", {},
      h("li", { text: "Give dates and, if you have one, a destination. With no destination, the agent ranks candidate cities using the weather for your dates and your interests." }),
      h("li", { text: "The plan has specialist sub-agents (flights, hotels, sights, dining, weather) that run in parallel, then a booking step and the final itinerary." }),
      h("li", { text: "Real: web search with citations, destination guides (Wikivoyage), weather (Open-Meteo: a forecast when near, otherwise the same dates last year), and hotel names (OpenStreetMap)." }),
      h("li", { text: "Simulated: flight fares and times, hotel nightly rates, and holds. There is no free public booking service, so these are labelled illustrative and nothing can be booked or charged." }),
      h("li", { text: "Both lanes see the same tool results in a run, and web search is paid once and left out of the comparison." })),
    h("h2", { text: "What Clef does and does not do" }),
    h("p", { text: "Clef classifies steps into the tool-use categories it was trained on. It does not rate difficulty or compare models, and it is not asked to. Choosing the model is the policy's job, using your preferences." }),
    h("h2", { text: "How quality is measured" }),
    h("ul", {},
      h("li", { text: "A separate AI reviewer (GPT-4.1) scores every output from 1 to 5 against the store policy and the tool results. It is not a human." }),
      h("li", { text: "For the example cases, the app also checks that the right actions happened: for example, a refund of exactly $138.99." })),
    h("h2", { text: "Things to keep in mind" }),
    h("ul", {},
      h("li", { text: "Model skill scores in the registry are assumptions. Results depend on them. Latency priors come from a small sample of measured calls." }),
      h("li", { text: "Clef's cost is estimated from GPU time and assumes the GPU stays busy. On this setup a decision also takes about half a second, which counts against time." }),
      h("li", { text: "A handful of example cases per agent is a demonstration, not a benchmark. One run of each case is noisy." }),
      h("li", { text: "The fine-tuned Clef figures (about 95% agreement) come from SWE-bench and BFCL-style data labeled by AI, not from support conversations." }),
      h("li", { text: "Customer messages are sent to your Azure OpenAI resource to run. Tools are simulated and nothing outside this app changes." }))));
}

/* ------------------------------------------------------------------- init */
async function init() {
  document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => setView(tab.dataset.view, { agent: tab.dataset.agent || null })));
  $("#prefs-summary").addEventListener("click", () => setView("prefs"));
  $("#run").addEventListener("click", startRun);
  $("#refresh-insights").addEventListener("click", loadInsights);
  $("#message").addEventListener("input", updateComposer);
  $("#message").addEventListener("keydown", (ev) => { if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") startRun(); });
  try {
    await refreshState();
  } catch (error) {
    $("#results").append(h("div", { class: "alert bad", role: "alert" }, h("div", { class: "grow" }, h("b", { text: "The demo server did not respond." }), h("div", { class: "hint", text: error.message })),
      h("button", { type: "button", class: "btn", text: "Reload", onclick: () => location.reload() })));
    return;
  }
  app.saved = { ...app.info.preferences.value };
  app.draft = { ...app.saved };
  buildTravelForm();
  const wanted = location.hash.slice(1);
  setView(app.info.preferences.saved ? (VIEWS.includes(wanted) || wanted === "travel" ? wanted : "desk") : "prefs", { focus: false });
  setInterval(() => { if (document.visibilityState === "visible") refreshState().catch(() => {}); }, 15000);
}

init();
