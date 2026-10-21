"""The HTTP service: routes, the live stream, and the guards that protect the credentials it holds."""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
from pathlib import Path

import pytest
from agent_demo_fakes import FakeCapabilityRouter, FakeDecider, ScriptedLLM

from agent_demo.server import Engine, serve


@pytest.fixture
def service(tmp_path):
    llm = ScriptedLLM()
    engine = Engine(tmp_path, "https://x.eastus2.inference.ml.azure.com/score", "flash-ft-gpu", "a" * 64,
                    decider=FakeDecider({"code": "debugging", "tools": "tool_lookup"}), llm=llm)
    server = serve(engine, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield engine, server.server_address[1], llm
    server.shutdown()
    server.server_close()


def call(port, method, path, body=None, headers=None, raw=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    conn.putrequest(method, path, skip_host=host is not None)
    if host is not None:
        conn.putheader("Host", host)
    payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    send = {"Content-Type": "application/json", **(headers or {})} if method == "POST" else dict(headers or {})
    if payload is not None:
        send["Content-Length"] = str(len(payload))
    for key, value in send.items():
        conn.putheader(key, value)
    conn.endheaders(payload)
    response = conn.getresponse()
    data = response.read()
    conn.close()
    try:
        return response.status, json.loads(data), response
    except ValueError:
        return response.status, data, response


def wait_done(engine, run_id, timeout=15):
    run = engine.runs[run_id]
    deadline = time.time() + timeout
    while not run.done and time.time() < deadline:
        time.sleep(0.02)
    assert run.done
    return run


# -------------------------------------------------------------------------- pages
def test_the_page_is_served_with_a_fresh_nonce_and_a_strict_policy(service):
    _, port, _ = service
    status, html, response = call(port, "GET", "/")
    status2, html2, _ = call(port, "GET", "/")
    csp = response.getheader("Content-Security-Policy")
    nonce = csp.split("'nonce-")[1].split("'")[0]
    assert status == 200 and f'nonce="{nonce}"'.encode() in html and b"{{nonce}}" not in html
    assert "script-src 'self' 'nonce-" in csp and "style-src 'self';" in csp and "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp and html != html2
    assert response.getheader("X-Content-Type-Options") == "nosniff" and response.getheader("Cache-Control") == "no-store"


def test_theme_detection_script_comes_before_any_other_script_and_colors_are_all_theme_variables(service):
    root = Path(__file__).resolve().parents[1] / "agent_demo" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    assert html.index("clawpilotTheme") < html.index('src="/static/app.js"')
    css = (root / "app.css").read_text(encoding="utf-8")
    components = css.split("* { box-sizing", 1)[1]
    import re

    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", components.replace("rgba(0, 0, 0, 0.12)", "").replace("rgba(0, 0, 0, 0.14)", ""))
    assert "Segoe UI" in css and "--cp-accent" in css and 'html[data-theme="dark"]' in css


def test_switching_example_cases_clears_stale_results_and_cases_lock_during_a_run():
    script = (Path(__file__).resolve().parents[1] / "agent_demo" / "static" / "app.js").read_text(encoding="utf-8")
    select_case = script.split("function selectCase(id) {", 1)[1].split("const normForm", 1)[0]
    set_running = script.split("function setRunning(on) {", 1)[1].split("function follow(run)", 1)[0]

    assert "if (app.run && !app.run.done && !app.run.error)" in select_case
    assert "app.run = null;" in select_case and "renderResults();" in select_case
    assert 'document.querySelectorAll("#cases input")' in set_running and "input.disabled = on" in set_running


def test_prompt_capture_requires_explicit_opt_in(monkeypatch):
    from agent_demo.server import build_parser

    monkeypatch.delenv("AIE_CAPTURE_ROUTER_PROMPTS", raising=False)
    assert build_parser().parse_args([]).capture_router_prompts is False
    assert build_parser().parse_args(["--capture-router-prompts"]).capture_router_prompts is True
    monkeypatch.setenv("AIE_CAPTURE_ROUTER_PROMPTS", "1")
    assert build_parser().parse_args([]).capture_router_prompts is True


def test_decision_one_is_opt_in_and_does_not_reuse_clef_url_or_adapter(monkeypatch):
    from agent_demo.server import build_parser, build_shadow

    monkeypatch.delenv("AIE_ROUTER_PROVIDER", raising=False)
    assert build_parser().parse_args([]).router_provider == "clef"
    args = build_parser().parse_args(["--router-provider", "decision-1",
                                     "--llm-endpoint", "https://r.services.ai.azure.com/openai/v1/"])
    shadow = build_shadow(args, "https://old.eastus2.inference.ml.azure.com/score")
    assert shadow.versions["provider"] == "decision-1" and shadow.versions["adapter_sha256"] is None
    assert shadow.router.url == "https://r.services.ai.azure.com/providers/microsoft/v1/systemone"
    assert shadow.router.deployment == "decision-1"
    assert not shadow.use_skill  # keep the same skill hints for a controlled router comparison
    args.apply_selector = False
    with pytest.raises(ValueError, match="requires"):
        build_shadow(args, "unused")


@pytest.mark.parametrize("path", ["/static/app.css", "/static/app.js"])
def test_only_the_two_static_files_are_served(service, path):
    _, port, _ = service
    assert call(port, "GET", path)[0] == 200
    for bad in ("/static/../server.py", "/static/index.html", "/static/..%2fserver.py", "/static/", "/nope"):
        assert call(port, "GET", bad)[0] == 404


def test_state_describes_the_models_preferences_cases_and_the_clef_deployment(service):
    _, port, _ = service
    _, state, _ = call(port, "GET", "/api/state")
    assert state["clef"]["deployment"] == "flash-ft-gpu" and state["clef"]["adapter_sha256"] == "a" * 64
    assert {m["id"] for m in state["models"]} >= {"gpt-5.6-luna", "gpt-5.6-sol"} and "modes" not in state
    options = state["preferences"]["options"]
    assert [o["id"] for o in options["quality"]] == ["average", "good", "very_good"]
    assert [o["id"] for o in options["latency"]] == ["low", "medium", "high"] and state["preferences"]["saved"] is False
    assert set(state["preferences"]["presets"]) == {"lean", "balanced", "premium"}
    support, travel = state["agents"]["support"], state["agents"]["travel"]
    assert len(support["scenarios"]) == 6 and all(c["expect"] and c["message"] for c in support["scenarios"])
    assert {x["name"] for x in support["tools"]} >= {"issue_refund", "track_shipment"} and state["router_cost_usd"] > 0
    assert support["kind"] == "message" and travel["kind"] == "form" and travel["parallel"] is True
    assert {x["name"] for x in travel["tools"]} >= {"search_web", "search_flights", "hold_hotel"}
    assert {c["id"] for c in travel["scenarios"]} >= {"lisbon-weekend", "warm-escape"}
    assert state["travel_options"]["budgets"] == ["budget", "mid-range", "premium"]


def test_api_label_metadata_is_generic_including_empty_and_disabled_views(service):
    from agent_demo.labels import LANE_LABELS, UI_COPY

    _, port, _ = service
    for path in ("/api/state", "/api/insights", "/api/shadow"):
        status, data, _ = call(port, "GET", path)
        assert status == 200 and data["lane_labels"] == LANE_LABELS
    state = call(port, "GET", "/api/state")[1]
    assert state["ui_copy"] == UI_COPY
    state["lane_labels"]["clef"] = "changed by client"
    assert call(port, "GET", "/api/state")[1]["lane_labels"] == LANE_LABELS
    state["ui_copy"]["decision_prompt"] = "client change"
    assert call(port, "GET", "/api/state")[1]["ui_copy"] == UI_COPY


def test_run_api_emits_generic_labels_without_changing_lane_ids(service):
    from agent_demo.labels import LANE_LABELS

    engine, port, _ = service
    status, started, _ = call(port, "POST", "/api/run", {"message": "Where is order A-1042?"})
    assert status == 202
    wait_done(engine, started["run_id"])
    status, result, _ = call(port, "GET", f"/api/run/{started['run_id']}?since=0")
    assert status == 200 and result["done"]
    lane_events = [e for e in result["events"] if e.get("lane") in LANE_LABELS]
    assert {e["lane"] for e in lane_events} == {"clef", "baseline"}
    assert all(e["lane_label"] == LANE_LABELS[e["lane"]] for e in lane_events)
    summary = next(e for e in result["events"] if e["type"] == "run_done")
    assert summary["lane_labels"] == LANE_LABELS
    assert set(summary["lanes"]) == {"clef", "baseline"}


# ------------------------------------------------------------------------- guards
def test_requests_for_another_host_name_are_refused(service):
    _, port, _ = service
    assert call(port, "GET", "/api/state", host="evil.example.com")[0] == 403
    assert call(port, "POST", "/api/run", {"message": "x"}, host="127.0.0.1.evil.com")[0] == 403
    assert call(port, "GET", "/api/state", host="localhost:1234")[0] == 200


@pytest.mark.parametrize("content_type", ["text/plain", "application/x-www-form-urlencoded", "multipart/form-data"])
def test_state_changing_requests_must_be_json_so_browsers_cannot_send_them_cross_site(service, content_type):
    engine, port, _ = service
    status, body, _ = call(port, "POST", "/api/run", raw=b'{"task":"x"}', headers={"Content-Type": content_type})
    assert status == 415 and engine.runs == {}


def test_bodies_are_bounded_and_must_be_json_objects(service):
    _, port, _ = service
    assert call(port, "POST", "/v1/route", raw=b"x" * (256 * 1024 + 1))[0] == 413
    assert call(port, "POST", "/v1/route", raw=b"{broken")[0] == 400
    assert call(port, "POST", "/v1/route", raw=b"[1, 2]")[0] == 400
    assert call(port, "POST", "/v1/route", raw=b"")[0] == 400


@pytest.mark.parametrize("body", [
    {"domain": "code"}, {"step_text": "x", "domain": "pdf"}, {"step_text": "x", "domain": "code", "mode": "turbo"},
    {"step_text": 7, "domain": "code"}, {"step_text": "x", "domain": "code", "exclude": ["bad id!"]},
    {"step_text": "x", "domain": "tools", "tools": [1]}, {"step_text": "x", "domain": "tools", "tools": [{"name": "t"}] * 21},
    {"step_text": "x" * 6001, "domain": "code"}, {"step_text": "x", "domain": "code", "prior_failures": True},
])
def test_route_validates_every_field(service, body):
    _, port, _ = service
    assert call(port, "POST", "/v1/route", body)[0] == 400


def test_the_server_refuses_to_bind_beyond_loopback(tmp_path):
    engine = Engine(tmp_path, "https://x.eastus2.inference.ml.azure.com/score", "d", "a" * 64,
                    decider=FakeDecider(), llm=ScriptedLLM())
    for host in ("0.0.0.0", "192.168.1.5", ""):
        with pytest.raises(ValueError, match="loopback"):
            serve(engine, 0, host)


# ------------------------------------------------------------------- router service
def test_route_returns_a_traceable_decision_and_outcomes_can_be_reported(service):
    engine, port, _ = service
    status, decision, _ = call(port, "POST", "/v1/route", {"step_text": "Get the weather.", "domain": "tools", "mode": "fast"})
    assert status == 200 and decision["selected_model"] == "gpt-5.6-luna" and decision["registry_version"] == engine.registry.version
    status, trace, _ = call(port, "GET", f"/api/trace/{decision['trace_id']}")
    assert status == 200 and trace["selected_model"] == "gpt-5.6-luna" and trace["outcome"] is None
    assert call(port, "POST", "/v1/outcome", {"trace_id": decision["trace_id"], "outcome": "ok", "cost_usd": 0.001, "judge_score": 5})[0] == 200
    assert call(port, "GET", f"/api/trace/{decision['trace_id']}")[1]["outcome"] == "ok"


@pytest.mark.parametrize(("body", "status"), [
    ({"trace_id": "tr_000000000000", "outcome": "ok"}, 404), ({"trace_id": "nope", "outcome": "ok"}, 400),
    ({"trace_id": "tr_000000000000", "outcome": "weird"}, 400), ({"trace_id": "tr_000000000000"}, 400),
    ({"trace_id": "tr_000000000000", "cost_usd": "free"}, 400), ({"outcome": "ok"}, 400),
])
def test_outcome_reports_are_validated(service, body, status):
    _, port, _ = service
    assert call(port, "POST", "/v1/outcome", body)[0] == status


def test_unknown_traces_runs_and_models_are_404(service):
    _, port, _ = service
    assert call(port, "GET", "/api/trace/tr_000000000000")[0] == 404
    assert call(port, "GET", "/api/run/run_0000000000")[0] == 404
    assert call(port, "POST", "/api/registry/ghost", {"healthy": False})[0] == 404


def test_registry_changes_apply_to_the_next_decision_and_new_models_can_be_onboarded(service):
    engine, port, _ = service
    step = {"step_text": "Get the weather.", "domain": "tools", "mode": "fast"}
    before = engine.registry.version
    assert call(port, "POST", "/api/registry/gpt-5.6-luna", {"healthy": False})[0] == 200
    decision = call(port, "POST", "/v1/route", step)[1]
    assert decision["selected_model"] != "gpt-5.6-luna" and decision["registry_version"] != before
    assert call(port, "POST", "/api/registry/gpt-5.6-luna", {"price_in": 0})[0] == 400
    new = {"id": "tiny", "display": "Tiny", "provider": "p", "deployment": "tiny", "tier": "small", "price_in": 0.01,
           "price_out": 0.02, "price_source": "test", "lat_base_ms": 100, "lat_per_token_ms": 1, "live": True,
           "skills": {"coding": 0.9, "reasoning": 0.9, "tools": 0.9, "general": 0.9}}
    assert call(port, "POST", "/api/registry", new)[0] == 201
    assert call(port, "POST", "/api/registry", new)[0] == 400
    assert call(port, "POST", "/v1/route", step)[1]["selected_model"] == "tiny"


# ----------------------------------------------------------------------- preferences
def test_preferences_are_saved_previewed_and_survive_a_restart(service, tmp_path):
    engine, port, _ = service
    status, preview, _ = call(port, "GET", "/api/preferences/preview?quality=average&latency=low&cost=low")
    assert status == 200 and preview["preset"] == "lean" and preview["mode"]["quality_floor"] == 0.55
    rows = {(r["route"], r["customer_facing"]): r for r in preview["rows"]}
    assert rows[("tool_lookup", False)]["model"] == "gpt-5.6-luna" and ("no_matching_tool", True) in rows
    premium = call(port, "GET", "/api/preferences/preview?quality=very_good&latency=high&cost=high")[1]
    assert {r["route"]: r["tier"] for r in premium["rows"]}["tool_workflow"] == "frontier"
    status, saved, _ = call(port, "POST", "/api/preferences", {"quality": "very_good", "latency": "high", "cost": "high"})
    assert status == 200 and saved["preset"] == "premium"
    state = call(port, "GET", "/api/state")[1]["preferences"]
    assert state["saved"] is True and state["value"] == {"quality": "very_good", "latency": "high", "cost": "high"}
    reloaded = Engine(tmp_path, "https://x.eastus2.inference.ml.azure.com/score", "flash-ft-gpu", "a" * 64,
                      decider=FakeDecider(), llm=ScriptedLLM())
    assert reloaded.prefs.quality == "very_good" and reloaded.prefs_saved is True


@pytest.mark.parametrize("query", ["quality=great", "latency=fast&quality=good", "cost=3", "colour=red"])
def test_preview_rejects_unknown_levels(service, query):
    assert call(service[1], "GET", f"/api/preferences/preview?{query}")[0] == 400


@pytest.mark.parametrize("body", [{"quality": "great"}, {"latency": 3}, {"extra": "x"}])
def test_saving_invalid_preferences_changes_nothing(service, body):
    engine, port, _ = service
    assert call(port, "POST", "/api/preferences", body)[0] == 400 and engine.prefs_saved is False


# --------------------------------------------------------------------------- runs
def test_a_run_streams_its_events_and_ends(service):
    engine, port, _ = service
    status, started, _ = call(port, "POST", "/api/run", {"message": "Order A-1042 arrived broken."})
    assert status == 202
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    conn.request("GET", f"/api/run/{started['run_id']}/stream")
    response = conn.getresponse()
    text = response.read().decode()
    conn.close()
    events = [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:") and line != "data: {}"]
    types = [e["type"] for e in events]
    assert response.getheader("Content-Type") == "text/event-stream"
    assert types[:2] == ["run_started", "plan"] and types[-1] == "run_done" and types.count("step_compare") == 2
    assert {e["lane"] for e in events if "lane" in e} == {"clef", "baseline"} and "event: end" in text


def test_a_dropped_stream_resumes_without_repeating_events(service):
    engine, port, _ = service
    run_id = call(port, "POST", "/api/run", {"message": "Weather please"})[1]["run_id"]
    wait_done(engine, run_id)
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    conn.request("GET", f"/api/run/{run_id}/stream", headers={"Last-Event-ID": "3"})
    seqs = [int(line[3:]) for line in conn.getresponse().read().decode().splitlines() if line.startswith("id:")]
    conn.close()
    assert seqs == list(range(4, len(engine.runs[run_id].events)))
    polled = call(port, "GET", f"/api/run/{run_id}?since=6")[1]
    assert polled["done"] is True and [e["seq"] for e in polled["events"]] == list(range(6, len(engine.runs[run_id].events)))


@pytest.mark.parametrize("body", [
    {}, {"message": ""}, {"message": "   "}, {"message": 5}, {"message": "x" * 3001},
    {"message": "x", "preferences": {"quality": "great"}}, {"message": "x", "preferences": 5}, {"task": "legacy field"},
])
def test_run_requests_are_validated_before_anything_starts(service, body):
    engine, port, llm = service
    assert call(port, "POST", "/api/run", body)[0] == 400
    assert engine.runs == {} and llm.calls == []


def test_a_run_uses_the_saved_preferences_unless_given_others(service):
    engine, port, _ = service
    call(port, "POST", "/api/preferences", {"quality": "average", "latency": "low", "cost": "low"})
    saved = wait_done(engine, call(port, "POST", "/api/run", {"message": "Where is my order?"})[1]["run_id"])
    assert saved.events[0]["preferences"]["quality"] == "average"
    other = wait_done(engine, call(port, "POST", "/api/run", {
        "message": "Where is my order?", "preferences": {"quality": "very_good", "latency": "high", "cost": "high"}})[1]["run_id"])
    assert other.events[0]["preferences"]["quality"] == "very_good" and engine.prefs.quality == "average"


def test_expected_outcomes_apply_only_to_an_unedited_example(service):
    engine, port, _ = service
    case = call(port, "GET", "/api/state")[1]["agents"]["support"]["scenarios"][0]
    exact = wait_done(engine, call(port, "POST", "/api/run", {"message": case["message"], "scenario_id": case["id"]})[1]["run_id"])
    assert [len(e["checks"]) for e in exact.events if e["type"] == "outcome"] == [len(case["expect"])] * 2
    edited = wait_done(engine, call(port, "POST", "/api/run", {"message": case["message"] + " Thanks!", "scenario_id": case["id"]})[1]["run_id"])
    assert all(e["checks"] == [] for e in edited.events if e["type"] == "outcome")
    unknown = wait_done(engine, call(port, "POST", "/api/run", {"message": case["message"], "scenario_id": "nope"})[1]["run_id"])
    assert all(e["checks"] == [] for e in unknown.events if e["type"] == "outcome")


def test_too_many_simultaneous_runs_are_refused_not_queued(service, tmp_path):
    engine, port, llm = service
    gate = threading.Event()
    original = llm.complete

    def slow(*args, **kwargs):
        gate.wait(10)
        return original(*args, **kwargs)

    llm.complete = slow
    ids = [call(port, "POST", "/api/run", {"message": f"case {i}"})[1]["run_id"] for i in range(2)]
    status, body, _ = call(port, "POST", "/api/run", {"message": "one too many"})
    gate.set()
    assert status == 429 and "Too many" in body["error"]
    for run_id in ids:
        wait_done(engine, run_id)
    assert call(port, "POST", "/api/run", {"message": "now there is room"})[0] == 202


def test_a_crash_inside_a_run_is_reported_on_the_stream_not_lost(service):
    engine, port, llm = service

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    engine.agent.run = explode
    run_id = call(port, "POST", "/api/run", {"message": "x"})[1]["run_id"]
    run = wait_done(engine, run_id)
    assert run.events[-1]["type"] == "run_error" and "boom" in run.events[-1]["error"]


def test_insights_report_measured_totals_and_no_message_text(service):
    engine, port, _ = service
    assert call(port, "GET", "/api/insights")[1]["runs"] == {"runs": 0}
    wait_done(engine, call(port, "POST", "/api/run", {"message": "Where is my kettle, order A-1057? Ask Priya."})[1]["run_id"])
    insights = call(port, "GET", "/api/insights")[1]
    runs = insights["runs"]
    assert runs["runs"] == 1 and runs["cost_usd"]["saved_pct"] > 0 and len(runs["recent"]) == 1
    assert insights["recent"][0]["selected_model"] and "kettle" not in json.dumps(insights).lower()


# -------------------------------------------------------------------------- travel planner
TRIP_PLAN = json.dumps({"steps": [
    {"title": "Find flights", "instruction": "Search flights.", "specialist": "flights", "parallel": False, "reply": False},
    {"title": "Research sights", "instruction": "Research sights.", "specialist": "sights", "parallel": True, "reply": False},
    {"title": "Write the itinerary", "instruction": "Compose it.", "specialist": "", "parallel": False, "reply": True}]})
TRIP_REPLY = json.dumps({"title": "Lisbon", "days": [{"day": 1, "date": "2026-11-20", "title": "Arrive", "items": []}]})


def offline_travel(engine, llm):
    """Swap the travel world for one with no network, keeping every other part of the engine real."""
    import dataclasses

    from agent_demo.domains import DOMAINS
    from agent_demo.travel import TravelDesk
    from test_agent_demo_travel import TODAY, FakeWeb

    domain = dataclasses.replace(DOMAINS["travel"], make_desk=lambda llm_, shared: TravelDesk(llm_, "gpt-4.1", FakeWeb(), shared, today=lambda: TODAY))
    engine.domains = engine.agent.domains = {**DOMAINS, "travel": domain}
    llm.plan, llm.reply_text = TRIP_PLAN, TRIP_REPLY


def trip_form(service):
    _, port, _ = service
    case = call(port, "GET", "/api/state")[1]["agents"]["travel"]["scenarios"][0]
    return case, dict(case["form"])


def test_travel_cases_carry_real_future_dates(service):
    case, form = trip_form(service)
    start = time.strptime(form["start_date"], "%Y-%m-%d")
    assert time.mktime(start) > time.time() and form["end_date"] > form["start_date"] and "start_in_days" not in form
    assert case["expect"] and form["interests"]


@pytest.mark.parametrize("change", [
    {"start_date": "2020-01-01"}, {"end_date": "2020-01-01"}, {"travelers": 0}, {"budget": "lavish"}, {"interests": ["skydiving"]}, {"origin": ""},
])
def test_travel_runs_are_validated_before_anything_starts(service, change):
    engine, port, llm = service
    _, form = trip_form(service)
    assert call(port, "POST", "/api/run", {"agent": "travel", "form": {**form, **change}})[0] == 400
    assert call(port, "POST", "/api/run", {"agent": "travel", "message": "Plan me a trip"})[0] == 400
    assert call(port, "POST", "/api/run", {"agent": "spaceship", "message": "x"})[0] == 400
    assert engine.runs == {} and llm.calls == []


def test_a_travel_run_streams_both_lanes_and_checks_outcomes_only_for_an_unedited_form(service):
    engine, port, llm = service
    offline_travel(engine, llm)
    case, form = trip_form(service)
    exact = wait_done(engine, call(port, "POST", "/api/run", {"agent": "travel", "form": {**form, "interests": list(reversed(form["interests"])), "notes": " "},
                                                              "scenario_id": case["id"]})[1]["run_id"])
    assert exact.events[0]["agent"] == "travel" and "Lisbon" in exact.events[0]["request"]
    assert [len(e["checks"]) for e in exact.events if e["type"] == "outcome"] == [len(case["expect"])] * 2
    assert exact.events[-1]["type"] == "run_done" and exact.events[-1]["agent"] == "travel"
    plan = [e for e in exact.events if e["type"] == "plan"][0]["steps"]
    assert [s["parallel"] for s in plan] == [False, True, False] and plan[1]["specialist"] == "sights"
    edited = wait_done(engine, call(port, "POST", "/api/run", {"agent": "travel", "form": {**form, "travelers": 3}, "scenario_id": case["id"]})[1]["run_id"])
    assert all(e["checks"] == [] for e in edited.events if e["type"] == "outcome")


def test_insights_can_be_filtered_by_agent(service):
    engine, port, llm = service
    offline_travel(engine, llm)
    _, form = trip_form(service)
    wait_done(engine, call(port, "POST", "/api/run", {"agent": "travel", "form": form})[1]["run_id"])
    wait_done(engine, call(port, "POST", "/api/run", {"message": "Where is my kettle?"})[1]["run_id"])
    assert call(port, "GET", "/api/insights")[1]["runs"]["runs"] == 2
    travel = call(port, "GET", "/api/insights?agent=travel")[1]["runs"]
    support = call(port, "GET", "/api/insights?agent=support")[1]["runs"]
    assert travel["runs"] == 1 and travel["recent"][0]["agent"] == "travel" and support["runs"] == 1 and support["recent"][0]["agent"] == "support"
    assert call(port, "GET", "/api/insights?agent=nope")[0] == 400

def test_shadow_endpoint_reports_disabled_without_a_selector_and_totals_when_one_runs(service, tmp_path):
    from agent_demo.labels import LANE_LABELS

    engine, port, _ = service
    assert call(port, "GET", "/api/shadow")[1] == {"enabled": False, "lane_labels": LANE_LABELS}
    assert call(port, "GET", "/api/state")[1]["shadow"] == {"enabled": False, "versions": None, "replay": False, "applied": False, "route_planner": False, "route_judge": False, "last": None, "endpoints": []}

    from agent_demo.shadow import ShadowSelector
    from routing_lab.selector import MeasuredRegistry

    measured = Path(__file__).resolve().parents[1] / "data" / "registry" / "measured.json"
    if not measured.exists():
        pytest.skip("run python -m capability_lab registry")

    class Router:
        deployment = "fake"

    engine.shadow = ShadowSelector(Router(), MeasuredRegistry.load(measured), baseline="gpt-5.6-sol", router_cost_usd=0.0004,
                                   log_path=tmp_path / "shadow.jsonl")
    shown = call(port, "GET", "/api/shadow")[1]
    assert shown["enabled"] and shown["decisions"] == 0 and shown["versions"]["selector"] == "selector-v1"

def test_binding_beyond_loopback_needs_an_explicit_opt_in_and_still_checks_the_host_header(tmp_path):
    engine = Engine(tmp_path, "https://x.eastus2.inference.ml.azure.com/score", "d", "a" * 64,
                    decider=FakeDecider({"code": "debugging", "tools": "tool_lookup"}), llm=ScriptedLLM())
    with pytest.raises(ValueError, match="loopback"):
        serve(engine, 0, "0.0.0.0")  # noqa: S104
    server = serve(engine, 0, "0.0.0.0", allow_remote=True, extra_hosts=("aie-app", " Other.Svc "))  # noqa: S104
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        assert call(port, "GET", "/api/shadow", host="localhost")[0] == 200
        assert call(port, "GET", "/api/shadow", host="aie-app")[0] == 200
        assert call(port, "GET", "/api/shadow", host="other.svc")[0] == 200
        assert call(port, "GET", "/api/shadow", host="evil.example")[0] == 403
        assert call(port, "GET", "/api/shadow", host="10.0.0.5")[0] == 403  # reachable by address is not enough
    finally:
        server.shutdown()
        server.server_close()


def test_every_option_can_be_set_from_the_environment(monkeypatch, tmp_path):
    from agent_demo import server as srv

    for name, value in {"AIE_HOST": "0.0.0.0", "AIE_PORT": "9001", "AIE_ALLOW_NON_LOOPBACK": "true", "AIE_ALLOWED_HOSTS": "aie-app, ,b",
                        "AIE_RUNTIME_DIR": str(tmp_path), "AIE_SHADOW_URL": "http://clef-server:5001/score", "AIE_SHADOW_LOCAL": "1",
                        "AIE_SHADOW_DEPLOYMENT": "my-dep", "AIE_NO_REPLAY": "yes", "AIE_SHADOW_USE_SKILL": "0"}.items():
        monkeypatch.setenv(name, value)
    args = srv.build_parser().parse_args([])
    assert (args.host, args.port, args.allow_non_loopback, args.allowed_host) == ("0.0.0.0", 9001, True, ["aie-app", "b"])
    assert args.runtime_dir == tmp_path and args.shadow_url == "http://clef-server:5001/score" and args.shadow_local
    assert args.no_replay and not args.shadow_use_skill and not args.no_shadow
    explicit = srv.build_parser().parse_args(["--port", "1234", "--no-shadow"])
    assert explicit.port == 1234 and explicit.no_shadow


def test_defaults_stay_on_loopback_and_the_endpoint_can_come_from_the_environment(monkeypatch, tmp_path):
    from agent_demo import server as srv

    for name in [n for n in os.environ if n.startswith("AIE_")]:
        monkeypatch.delenv(name)
    args = srv.build_parser().parse_args([])
    assert args.host == "127.0.0.1" and not args.allow_non_loopback and args.port == 8765
    file = tmp_path / "c.json"
    file.write_text(json.dumps({"scoring_uri": "https://from-file.eastus2.inference.ml.azure.com/score"}), encoding="utf-8")
    args.connection = file
    monkeypatch.delenv("CLEF_ENDPOINT_URL", raising=False)
    assert srv.scoring_uri(args).startswith("https://from-file")
    monkeypatch.setenv("CLEF_ENDPOINT_URL", "https://from-env.eastus2.inference.ml.azure.com/score")
    assert srv.scoring_uri(args).startswith("https://from-env")


def test_the_shadow_router_can_be_a_local_clef_container_that_gets_no_azure_token(monkeypatch, tmp_path):
    from agent_demo import server as srv

    measured = Path(__file__).resolve().parents[1] / "data" / "registry" / "measured.json"
    if not measured.exists():
        pytest.skip("run python -m capability_lab registry")
    for name in [n for n in os.environ if n.startswith("AIE_")]:
        monkeypatch.delenv(name)
    args = srv.build_parser().parse_args(["--runtime-dir", str(tmp_path), "--shadow-local", "--shadow-url", "http://clef-server:5001/score"])
    shadow = srv.build_shadow(args, "https://unused.eastus2.inference.ml.azure.com/score")
    assert shadow.router.backend.local and shadow.router.backend.credential is None
    assert shadow.router.backend.url == "http://clef-server:5001/score" and shadow.versions["router"] == "clef-local"
    assert srv.build_shadow(srv.build_parser().parse_args(["--no-shadow"]), "x") is None

@pytest.fixture
def selecting(service, tmp_path):
    from agent_demo.shadow import ShadowSelector
    from routing_lab.selector import MeasuredRegistry

    measured = Path(__file__).resolve().parents[1] / "data" / "registry" / "measured.json"
    if not measured.exists():
        pytest.skip("run python -m capability_lab registry")
    engine, port, _ = service
    router = FakeCapabilityRouter()
    engine.shadow = ShadowSelector(router, MeasuredRegistry.load(measured), baseline="gpt-5.6-sol", router_cost_usd=0.0004,
                                   use_skill=False)
    return engine, port, router


def test_select_api_returns_the_routers_distribution_the_choice_and_why(selecting):
    _, port, router = selecting
    status, body, _ = call(port, "POST", "/v1/select", {"request": "What is the molar mass of NaCl?", "needs_tools": False,
                                                       "preferences": {"quality": "good", "latency": "medium", "cost": "low"}})
    assert status == 200 and router.calls == ["What is the molar mass of NaCl?"]
    assert body["tier"] == "low" and body["tier_probabilities"]["low"] == 0.9 and body["choice"] and "type" not in body
    assert body["preferences"] == {"quality": "good", "latency": "medium", "cost": "low"} and body["settings"]["floor"] == 0.7
    assert body["alternatives"] and body["versions"]["selector"] == "selector-v1" and body["expected_success"] >= 0.7


def test_select_api_applies_the_callers_constraints(selecting):
    _, port, _ = selecting
    status, body, _ = call(port, "POST", "/v1/select", {"request": "Call a tool for me", "needs_tools": True, "allow": ["gpt-5.6-luna", "phi-4"]})
    assert status == 200 and body["choice"] == "gpt-5.6-luna"
    assert {e["endpoint"] for e in body["eliminated"]} >= {"phi-4"}  # allowed, but it cannot call tools
    premium = call(port, "POST", "/v1/select", {"request": "Plan the migration", "preferences": {"quality": "very_good", "latency": "high", "cost": "high"}})[1]
    lean = call(port, "POST", "/v1/select", {"request": "Plan the migration", "preferences": {"quality": "average", "latency": "low", "cost": "low"}})[1]
    assert premium["floor"] > lean["floor"]


def test_select_api_can_require_json_and_reports_unverified_models(selecting):
    _, port, _ = selecting
    status, body, _ = call(port, "POST", "/v1/select", {
        "request": "Write an itinerary as JSON", "needs_tools": False, "needs_json": True,
        "allow": ["gpt-oss-120b", "llama-4-maverick"],
    })
    assert status == 200 and body["needs_json"] is True
    assert body["choice"] == "llama-4-maverick"
    assert body["json_api"] == "chat"
    assert "JSON" in next(c for c in body["eliminated"] if c["endpoint"] == "gpt-oss-120b")["reasons"][0]
    none = call(port, "POST", "/v1/select", {
        "request": "JSON required", "needs_tools": False, "needs_json": True, "allow": ["gpt-oss-120b"],
    })[1]
    assert none["choice"] is None and none["needs_json"] is True
    assert call(port, "POST", "/v1/select", {"request": "x", "needs_json": "yes"})[0] == 400


@pytest.mark.parametrize("body", [
    {}, {"request": ""}, {"request": "x" * 5000}, {"request": 5}, {"request": "ok", "preferences": {"quality": "best"}},
    {"request": "ok", "preferences": "balanced"}, {"request": "ok", "kind": "summary"}, {"request": "ok", "needs_tools": "yes"},
    {"request": "ok", "allow": "gpt-5.6-luna"}, {"request": "ok", "allow": ["bad id!"]}, {"request": "ok", "input_tokens": 0},
    {"request": "ok", "output_tokens": 10**7}, {"request": "ok", "input_tokens": True}])
def test_select_api_validates_every_field(selecting, body):
    assert call(selecting[1], "POST", "/v1/select", body)[0] == 400


def test_select_api_reports_an_unavailable_router_without_details(selecting):
    engine, port, router = selecting
    router.fail = True
    status, body, _ = call(port, "POST", "/v1/select", {"request": "hello"})
    assert status == 503 and "endpoint down" not in json.dumps(body)


def test_select_api_is_unavailable_when_no_router_is_configured(service):
    _, port, _ = service
    assert call(port, "POST", "/v1/select", {"request": "hello"})[0] == 503

def test_a_repeated_warning_is_logged_once():
    import logging

    from agent_demo.server import OncePerMessage

    once = OncePerMessage()
    make = lambda msg: logging.LogRecord("azure.identity", logging.WARNING, __file__, 1, msg, None, None)  # noqa: E731
    assert once.filter(make("DefaultAzureCredential failed")) is True
    assert once.filter(make("DefaultAzureCredential failed")) is False
    assert once.filter(make("something else")) is True

def test_the_selector_is_applied_by_default_and_can_be_turned_off(monkeypatch):
    from agent_demo import server as srv

    monkeypatch.delenv("AIE_APPLY_SELECTOR", raising=False)
    assert srv.build_parser().parse_args([]).apply_selector is True
    assert srv.build_parser().parse_args(["--no-apply-selector"]).apply_selector is False
    monkeypatch.setenv("AIE_APPLY_SELECTOR", "0")
    assert srv.build_parser().parse_args([]).apply_selector is False
    assert srv.build_parser().parse_args(["--apply-selector"]).apply_selector is True


def test_state_lists_the_endpoints_the_screen_needs_to_name_models(selecting):
    engine, port, _ = selecting
    engine.agent.apply_selector = True
    shadow = call(port, "GET", "/api/state")[1]["shadow"]
    assert shadow["applied"] is True
    by_id = {e["id"]: e for e in shadow["endpoints"]}
    assert by_id["gpt-5.6-sol"]["display"] == "GPT-5.6 Sol" and by_id["gpt-5.6-sol"]["tier"] == "frontier"
    assert by_id["gpt-5.6-luna"]["tier"] == "small" and by_id["gpt-4.1"]["tier"] == "mid" and len(by_id) == 10

def test_a_missing_connection_file_gives_a_clear_message_unless_the_url_is_set(tmp_path, monkeypatch):
    from agent_demo import server as srv

    monkeypatch.delenv("CLEF_ENDPOINT_URL", raising=False)
    args = srv.build_parser().parse_args(["--connection", str(tmp_path / "connection.json")])
    with pytest.raises(SystemExit, match="CLEF_ENDPOINT_URL"):
        srv.scoring_uri(args)
    monkeypatch.setenv("CLEF_ENDPOINT_URL", "https://x.eastus2.inference.ml.azure.com/score")
    assert srv.scoring_uri(args).startswith("https://x.")
