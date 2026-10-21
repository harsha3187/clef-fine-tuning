"""Local HTTP service: the router as an API, an agent runner with a live event stream, and the UI.

Bound to loopback only. It holds your Azure sign-in and can spend money, so it checks the Host header and
accepts state-changing requests only as application/json, which browsers will not send cross-site without a
preflight this server never answers.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import secrets
import sys
import threading
import time
import traceback
import uuid
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from agent_demo import ROUTER_VERSION
from agent_demo.agent import Agent
from agent_demo.clef_decider import ClefDecider
from agent_demo.domains import DOMAINS, travel_message
from agent_demo.llm import AzureResponsesClient
from agent_demo.policy import Complexity, Policy, compute_requirement, select_model
from agent_demo.preferences import LEVELS, PRESETS, Preferences
from agent_demo.labels import LANE_LABELS, UI_COPY
from agent_demo.registry import Registry
from agent_demo.shadow import ShadowItem, ShadowSelector, aggregate, display_name, endpoint_tier
from agent_demo.router import AdaptiveRouter, RoutingContext, RoutingError
from agent_demo.telemetry import Telemetry
from agent_demo.support import router_tools
from routing_lab.capability import CapabilityRouter
from routing_lab.selector import MeasuredRegistry, Settings

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
FINETUNED_ADAPTER_SHA256 = "bd6eb1be473a9b9f0f6c91bbc1f835d350351cd5695a5c3fd04ba21c67954e1f"
CAPABILITY_DEPLOYMENT = "flash-cap-ft3"
CAPABILITY_ADAPTER_SHA256 = "3651988bb592eda963ccf96afe38ec89068840c26464094b1f5f09e481ab6beb"
MEASURED_REGISTRY = ROOT / "data" / "registry" / "measured.json"
LLM_ENDPOINT = os.environ.get("AIE_LLM_ENDPOINT", "https://your-resource.services.ai.azure.com/openai/v1/")  # your Azure AI resource: set AIE_LLM_ENDPOINT (it is not stored in the repository)
MAX_BODY = 256 * 1024
MAX_RUNS_KEPT = 20
MAX_ACTIVE_RUNS = 2
STATIC = {"app.css": "text/css; charset=utf-8", "app.js": "text/javascript; charset=utf-8"}
_TRACE = re.compile(r"tr_[0-9a-f]{12}")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_OUTCOME_FIELDS = {"outcome": str, "exec_ms": (int, float), "input_tokens": int, "output_tokens": int,
                   "cost_usd": (int, float), "judge_score": int}
_OUTCOMES = {"ok", "weak", "escalated", "failed", "error_rerouted", "clarification"}


class Run:
    """One agent run: an append-only event log that any number of streams can follow."""

    def __init__(self, run_id: str) -> None:
        self.id = run_id
        self.events: list[dict[str, Any]] = []
        self.done = False
        self.cond = threading.Condition()

    def emit(self, event: dict[str, Any]) -> None:
        with self.cond:
            self.events.append({"seq": len(self.events), "t": round(time.time(), 3), **event})
            self.cond.notify_all()

    def finish(self) -> None:
        with self.cond:
            self.done = True
            self.cond.notify_all()


class Engine:
    def __init__(self, runtime_dir: Path, scoring_uri: str, deployment: str, adapter_sha256: str,
                 llm_endpoint: str = LLM_ENDPOINT, *, decider: Any = None, llm: Any = None,
                 shadow: ShadowSelector | None = None, shadow_replay: bool = False, apply_selector: bool = False,
                 route_planner: bool = False, route_judge: bool = False) -> None:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        self.registry = Registry.load(HERE / "data" / "registry.json", runtime_dir / "registry.json")
        self.policy = Policy.load(HERE / "data" / "policy.json")
        self.telemetry = Telemetry(runtime_dir / "telemetry.db")
        self._prefs_path = runtime_dir / "preferences.json"
        self.prefs, self.prefs_saved = self._load_prefs()
        self.decider = decider or ClefDecider(scoring_uri, deployment, adapter_sha256, self.policy.fallback_route)
        self.router = AdaptiveRouter(self.registry, self.policy, self.decider, self.telemetry)
        self.llm = llm or AzureResponsesClient(llm_endpoint)
        self.shadow = shadow
        self.agent = Agent(self.router, self.llm, self.registry, self.policy, self.telemetry, shadow=shadow, shadow_replay=shadow_replay,
                           apply_selector=apply_selector, route_planner=route_planner, route_judge=route_judge)
        self.domains = DOMAINS
        self.deployment, self.adapter_sha256 = deployment, adapter_sha256
        self.runs: dict[str, Run] = {}
        self._lock = threading.Lock()

    def cases(self, agent: str, today: date | None = None) -> list[dict[str, Any]]:
        """The example cases for an agent. Travel cases hold relative dates, resolved against today."""
        today = today or date.today()
        cases = self.domains[agent].scenarios()
        if agent != "travel":
            return cases
        for case in cases:
            form = dict(case["form"])
            start = today + timedelta(days=form.pop("start_in_days"))
            form["start_date"] = start.isoformat()
            form["end_date"] = (start + timedelta(days=form.pop("nights"))).isoformat()
            case["form"] = form
        return cases

    def warm_up(self) -> None:
        """Pay the first-call cost (token and connection setup) before a user is waiting."""
        try:
            self.llm._bearer()  # the first Azure sign-in token costs a few seconds
        except Exception:  # noqa: BLE001 - warming is best effort
            pass
        if not self.agent.apply_selector:  # the step-type router is only used when the selector is not applied
            self.decider.decide("tools", "Warm up the router.", tools=router_tools())
        if self.shadow is not None:
            self.shadow.warm_up()

    def _load_prefs(self) -> tuple[Preferences, bool]:
        try:
            return Preferences.from_dict(json.loads(self._prefs_path.read_text(encoding="utf-8-sig"))), True
        except (OSError, ValueError):
            return Preferences(), False

    def save_prefs(self, prefs: Preferences) -> None:
        self.prefs, self.prefs_saved = prefs, True
        self._prefs_path.write_text(json.dumps(prefs.to_dict()), encoding="utf-8")

    def preview(self, prefs: Preferences) -> dict[str, Any]:
        """Which model each kind of support step would get under these preferences, for a typical step."""
        derived = self.policy.derive(prefs)
        models = self.registry.snapshot()
        rows = []
        for route, profile in self.policy.profiles["tool-api-v1"].items():
            for facing in ((False, True) if route == "no_matching_tool" else (False,)):
                requirement = compute_requirement(self.policy, profile, route, Complexity(0.1, {}), 0.95,
                                                  derived.mode, 0, facing)
                tokens_in = 900 if profile.needs_tools else 300
                selection = select_model(self.policy, models, profile, requirement, derived.mode, tokens_in,
                                         int(profile.out_tokens * 1.1))
                chosen = next((c for c in selection.candidates if c.status == "selected"), None)
                rows.append({"route": route, "customer_facing": facing, "needs": round(requirement.total, 2),
                             "model": chosen.model_id if chosen else None,
                             "display": chosen.display if chosen else None, "tier": chosen.tier if chosen else None,
                             "quality": chosen.quality if chosen else None,
                             "est_cost_usd": chosen.est_cost if chosen else None})
        return {"preferences": prefs.to_dict(), "preset": prefs.preset(), "notes": derived.notes,
                "mode": {"quality_floor": derived.mode.quality_floor, "cost_weight": derived.mode.lambda_cost,
                         "latency_weight": derived.mode.mu_latency, "gate_min_score": derived.gate_min_score},
                "rows": rows}

    def start_run(self, params: dict[str, Any]) -> Run:
        with self._lock:
            if sum(not r.done for r in self.runs.values()) >= MAX_ACTIVE_RUNS:
                raise RuntimeError("Too many runs in progress; wait for one to finish.")
            run = Run("run_" + uuid.uuid4().hex[:10])
            self.runs[run.id] = run
            for stale in list(self.runs)[:-MAX_RUNS_KEPT]:
                if self.runs[stale].done:
                    del self.runs[stale]

        def work() -> None:
            try:
                self.agent.run(params["message"], prefs=params["prefs"], expectations=params["expectations"],
                               agent=params["agent"], emit=run.emit)
            except Exception as error:  # noqa: BLE001 - the stream must always end cleanly
                run.emit({"type": "run_error", "error": f"{type(error).__name__}: {str(error)[:300]}"})
            finally:
                run.finish()

        threading.Thread(target=work, daemon=True).start()
        return run

    def state(self) -> dict[str, Any]:
        table = self.policy.preference_table
        return {
            "router_version": ROUTER_VERSION, "registry_version": self.registry.version,
            "lane_labels": LANE_LABELS.copy(),
            "ui_copy": UI_COPY.copy(),
            "policy_version": self.policy.version,
            "preferences": {"value": self.prefs.to_dict(), "saved": self.prefs_saved, "presets": PRESETS,
                            "options": {field: [{"id": level, "label": table[field][level]["label"],
                                                 "hint": table[field][level].get("hint") or table[field][level]["note"]}
                                                for level in levels] for field, levels in LEVELS.items()}},
            "limits": self.policy.limits, "baseline_model": self.policy.baseline_model,
            "planner_model": self.policy.planner_model, "judge_model": self.policy.judge_model,
            "router_cost_usd": None if self.shadow and self.shadow.versions["provider"] == "decision-1" else self.policy.router_cost_usd,
            "router_cost_note": getattr(self.shadow.router, "cost_note", self.policy.data["router_cost_note"]) if self.shadow else self.policy.data["router_cost_note"],
            "clef": {"deployment": self.deployment, "adapter_sha256": self.adapter_sha256,
                     "last": getattr(self.decider, "last", None)},
            "agents": {d.id: {"title": d.title, "kind": "form" if d.id == "travel" else "message",
                              "scenarios": self.cases(d.id),
                              "tools": [{"name": x["name"], "description": x["description"]} for x in d.tools],
                              "parallel": d.parallel}
                       for d in self.domains.values()},
            "travel_options": {"budgets": ["budget", "mid-range", "premium"],
                               "interests": ["food", "culture", "nature", "beaches", "nightlife", "family", "shopping", "adventure"]},
            "models": [m.to_dict() for m in self.registry.snapshot()],
            "shadow": {"enabled": self.shadow is not None, "versions": self.shadow.versions if self.shadow else None,
                       "replay": self.agent.shadow_replay, "applied": self.agent.apply_selector,
                       "route_planner": self.agent.route_planner, "route_judge": self.agent.route_judge,
                       "last": self.shadow.last if self.shadow else None,
                       "endpoints": [{"id": e.id, "display": display_name(e.id), "tier": endpoint_tier(e), "price_in": e.price_in,
                                      "price_out": e.price_out, "verified_price": e.verified_price, "supports_tools": e.supports_tools}
                                     for e in (self.shadow.registry.endpoints if self.shadow else [])]},
        }

    def shadow_totals(self) -> dict[str, Any]:
        if self.shadow is None or self.shadow.log_path is None:
            return {"enabled": False, "lane_labels": LANE_LABELS.copy()}
        return {"enabled": True, "lane_labels": LANE_LABELS.copy(), "versions": self.shadow.versions, **aggregate(self.shadow.log_path)}


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


def _same_form(a: dict[str, Any], b: dict[str, Any]) -> bool:
    def norm(f: dict[str, Any]) -> dict[str, Any]:
        text = {k: str(f.get(k) or "").strip() for k in ("origin", "destination", "start_date", "end_date", "notes", "budget")}
        return {**text, "travelers": f.get("travelers"), "interests": sorted(f.get("interests") or [])}
    return norm(a) == norm(b)


def require(body: dict[str, Any], key: str, kind: type | tuple, *, default: Any = ...) -> Any:
    if key not in body:
        if default is ...:
            raise ApiError(400, f"Missing field: {key}")
        return default
    value = body[key]
    if isinstance(value, bool) and kind is not bool and (not isinstance(kind, tuple) or bool not in kind):
        raise ApiError(400, f"Field {key} has the wrong type")
    if not isinstance(value, kind):
        raise ApiError(400, f"Field {key} has the wrong type")
    return value


def make_handler(engine: Engine, allowed_hosts: set[str]) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AIE-PoC"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # request logs would echo task text
            return

        # -------------------------------------------------------------- plumbing
        def _guard(self) -> None:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            if host not in allowed_hosts:
                raise ApiError(403, "Unexpected Host header")

        def _json_body(self) -> dict[str, Any]:
            if (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
                raise ApiError(415, "Send application/json")
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError as error:
                raise ApiError(400, "Bad Content-Length") from error
            if length <= 0 or length > MAX_BODY:
                raise ApiError(413 if length > MAX_BODY else 400, "Body must be 1 byte to 256 KB")
            try:
                body = json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ApiError(400, "Body is not valid JSON") from error
            if not isinstance(body, dict):
                raise ApiError(400, "Body must be a JSON object")
            return body

        def _send(self, status: int, payload: Any, content_type: str = "application/json",
                  extra: dict[str, str] | None = None) -> None:
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def _dispatch(self, method: str) -> None:
            try:
                self._guard()
                url = urlsplit(self.path)
                getattr(self, f"_{method}")(url.path, parse_qs(url.query))
            except ApiError as error:
                self._send(error.status, {"error": error.message})
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception as error:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)  # type and stack only: request bodies are never logged
                try:
                    self._send(500, {"error": f"{type(error).__name__}: {str(error)[:200]}"})
                except OSError:
                    return

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch("get")

        def do_POST(self) -> None:  # noqa: N802
            self._dispatch("post")

        # ------------------------------------------------------------------ GET
        def _get(self, path: str, query: dict[str, list[str]]) -> None:
            if path in ("/", "/index.html"):
                nonce = secrets.token_urlsafe(16)
                html = (HERE / "static" / "index.html").read_text(encoding="utf-8").replace("{{nonce}}", nonce)
                csp = (f"default-src 'none'; script-src 'self' 'nonce-{nonce}'; style-src 'self'; "
                       "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
                       "frame-ancestors 'none'")
                return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8",
                                  {"Content-Security-Policy": csp})
            if path.startswith("/static/") and path[8:] in STATIC:
                return self._send(200, (HERE / "static" / path[8:]).read_bytes(), STATIC[path[8:]])
            if path == "/api/state":
                return self._send(200, engine.state())
            if path == "/api/insights":
                agent = (query.get("agent") or [None])[0]
                if agent is not None and agent not in engine.domains:
                    raise ApiError(400, "Unknown agent")
                return self._send(200, {"runs": engine.telemetry.run_aggregates(agent=agent),
                                        "lane_labels": LANE_LABELS.copy(),
                                        "decisions": engine.telemetry.aggregates(),
                                        "recent": engine.telemetry.recent(30),
                                        "registry_version": engine.registry.version})
            if path == "/api/shadow":
                return self._send(200, engine.shadow_totals())
            if path == "/api/preferences/preview":
                try:
                    prefs = Preferences.from_dict({k: v[0] for k, v in query.items()})
                except ValueError as error:
                    raise ApiError(400, str(error)) from error
                return self._send(200, engine.preview(prefs))
            match = re.fullmatch(r"/api/trace/(tr_[0-9a-f]{12})", path)
            if match:
                trace = engine.telemetry.trace(match.group(1))
                if trace is None:
                    raise ApiError(404, "Unknown trace")
                return self._send(200, trace)
            match = re.fullmatch(r"/api/run/(run_[0-9a-f]{10})(/stream)?", path)
            if match:
                run = engine.runs.get(match.group(1))
                if run is None:
                    raise ApiError(404, "Unknown run")
                resume = self.headers.get("Last-Event-ID")
                since = int((query.get("since") or [str(int(resume) + 1 if resume and resume.isdigit() else 0)])[0] or 0)
                if match.group(2):
                    return self._stream(run, max(0, since))
                return self._send(200, {"id": run.id, "done": run.done, "events": run.events[since:]})
            raise ApiError(404, "Not found")

        def _stream(self, run: Run, index: int) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            while True:
                with run.cond:
                    if index >= len(run.events) and not run.done:
                        run.cond.wait(timeout=10)
                    batch, done = run.events[index:], run.done
                if not batch and not done:
                    self.wfile.write(b": keep-alive\n\n")
                for event in batch:
                    self.wfile.write(f"id: {event['seq']}\ndata: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
                index += len(batch)
                if done and not batch:
                    self.wfile.write(b"event: end\ndata: {}\n\n")
                    self.wfile.flush()
                    return

        # ----------------------------------------------------------------- POST
        def _post(self, path: str, query: dict[str, list[str]]) -> None:
            body = self._json_body()
            if path == "/api/run":
                return self._post_run(body)
            if path == "/v1/route":
                return self._post_route(body)
            if path == "/v1/select":
                return self._post_select(body)
            if path == "/v1/outcome":
                return self._post_outcome(body)
            if path == "/api/registry":
                try:
                    return self._send(201, engine.registry.add(body).to_dict())
                except ValueError as error:
                    raise ApiError(400, str(error)) from error
            match = re.fullmatch(r"/api/registry/([A-Za-z0-9][A-Za-z0-9._-]{0,63})", path)
            if match:
                try:
                    return self._send(200, engine.registry.update(match.group(1), **body).to_dict())
                except KeyError as error:
                    raise ApiError(404, "Unknown model") from error
                except ValueError as error:
                    raise ApiError(400, str(error)) from error
            if path == "/api/preferences":
                try:
                    prefs = Preferences.from_dict(body)
                except ValueError as error:
                    raise ApiError(400, str(error)) from error
                engine.save_prefs(prefs)
                return self._send(200, engine.preview(prefs))
            raise ApiError(404, "Not found")

        def _post_run(self, body: dict[str, Any]) -> None:
            limits = engine.policy.limits
            agent = require(body, "agent", str, default="support")
            if agent not in engine.domains:
                raise ApiError(400, f"Agent must be one of {sorted(engine.domains)}")
            if agent == "travel":
                form = require(body, "form", dict)
                try:
                    message = travel_message(form)
                except ValueError as error:
                    raise ApiError(400, str(error)) from error
            else:
                form = None
                message = require(body, "message", str)
            if not message.strip() or len(message) > limits["max_task_chars"]:
                raise ApiError(400, f"The request must be 1 to {limits['max_task_chars']} characters")
            try:
                prefs = Preferences.from_dict(body["preferences"]) if "preferences" in body else engine.prefs
            except ValueError as error:
                raise ApiError(400, str(error)) from error
            # Expected outcomes only apply when the request is an unedited example case.
            expectations: list[dict[str, Any]] = []
            case_id = body.get("scenario_id")
            if case_id is not None:
                case = next((c for c in engine.cases(agent) if c["id"] == case_id), None)
                if case is not None and (_same_form(case["form"], form) if agent == "travel" else case["message"] == message):
                    expectations = case["expect"]
            try:
                run = engine.start_run({"message": message, "prefs": prefs, "expectations": expectations, "agent": agent})
            except RuntimeError as error:
                raise ApiError(429, str(error)) from error
            self._send(202, {"run_id": run.id})

        def _post_route(self, body: dict[str, Any]) -> None:
            text = require(body, "step_text", str)
            if not text.strip() or len(text) > engine.policy.limits["max_task_chars"]:
                raise ApiError(400, "step_text must be 1 to the task limit in characters")
            domain = require(body, "domain", str)
            if domain not in ("code", "tools"):
                raise ApiError(400, "domain must be code or tools")
            mode = require(body, "mode", str, default="balanced")
            if mode not in engine.policy.modes:
                raise ApiError(400, f"Mode must be one of {sorted(engine.policy.modes)}")
            budget = require(body, "remaining_budget_usd", (int, float), default=None) if "remaining_budget_usd" in body else None
            tools = body.get("tools")
            if tools is not None:
                if (not isinstance(tools, list) or len(tools) > 20
                        or any(not isinstance(t, dict) or not isinstance(t.get("name"), str) for t in tools)):
                    raise ApiError(400, "tools must be a list of at most 20 function descriptions")
            elif domain == "tools":
                tools = router_tools()
            exclude = body.get("exclude") or []
            if not isinstance(exclude, list) or any(not isinstance(x, str) or not _MODEL_ID.fullmatch(x) for x in exclude):
                raise ApiError(400, "exclude must be a list of model ids")
            ctx = RoutingContext(
                step_text=text, domain=domain, mode=mode, task_id=str(body.get("task_id", ""))[:40],
                step_id=str(body.get("step_id", ""))[:40], tools=tools if domain == "tools" else None,
                remaining_budget_usd=float(budget) if budget is not None else None,
                exclude=frozenset(exclude), prior_failures=int(require(body, "prior_failures", int, default=0)))
            try:
                self._send(200, engine.router.route(ctx).to_dict())
            except RoutingError as error:
                raise ApiError(503, str(error)) from error

        def _post_select(self, body: dict[str, Any]) -> None:
            """The capability router plus selector as an API for other applications: which endpoint would serve this request?"""
            if engine.shadow is None:
                raise ApiError(503, "The capability router is not configured on this service")
            text = require(body, "request", str)
            if not text.strip() or len(text) > engine.policy.limits["max_task_chars"]:
                raise ApiError(400, "request must be 1 to the task limit in characters")
            try:
                prefs = Preferences.from_dict(body["preferences"]) if "preferences" in body else Preferences()
            except ValueError as error:
                raise ApiError(400, str(error)) from error
            kind = require(body, "kind", str, default="step")
            if kind not in ("plan", "step", "reply"):
                raise ApiError(400, "kind must be plan, step or reply")
            needs_tools = require(body, "needs_tools", bool, default=kind == "step")
            needs_json = require(body, "needs_json", bool, default=False)
            allow = body.get("allow")
            if allow is not None and (not isinstance(allow, list) or len(allow) > 50
                                      or any(not isinstance(x, str) or not _MODEL_ID.fullmatch(x) for x in allow)):
                raise ApiError(400, "allow must be a list of endpoint ids")
            sizes = {}
            for field, default in (("input_tokens", 600), ("output_tokens", 400)):
                value = require(body, field, int, default=default)
                if not 1 <= value <= 200_000:
                    raise ApiError(400, f"{field} must be between 1 and 200000")
                sizes[field] = value
            mode = engine.policy.derive(prefs).mode
            settings = Settings(mode.quality_floor, mode.lambda_cost, mode.mu_latency)
            event = engine.shadow.decide(ShadowItem("api", kind, text, needs_tools, sizes["input_tokens"], sizes["output_tokens"], needs_json),
                                         settings, frozenset(allow) if allow is not None else None)
            if event["type"] == "shadow_error":
                raise ApiError(503, "The capability router did not answer; try again")
            event.pop("type")
            self._send(200, {**event, "preferences": prefs.to_dict(),
                             "settings": {"floor": settings.floor, "cost_weight": settings.cost_weight,
                                          "latency_weight": settings.latency_weight}})

        def _post_outcome(self, body: dict[str, Any]) -> None:
            trace_id = require(body, "trace_id", str)
            if not _TRACE.fullmatch(trace_id):
                raise ApiError(400, "Bad trace id")
            fields: dict[str, Any] = {}
            for key, kind in _OUTCOME_FIELDS.items():
                if key in body:
                    fields[key] = require(body, key, kind)
            if fields.get("outcome") is not None and fields["outcome"] not in _OUTCOMES:
                raise ApiError(400, f"outcome must be one of {sorted(_OUTCOMES)}")
            if not fields:
                raise ApiError(400, "Send at least one outcome field")
            try:
                engine.router.record_outcome(trace_id, **fields)
            except KeyError as error:
                raise ApiError(404, "Unknown trace") from error
            self._send(200, {"ok": True})

    return Handler


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address) -> None:
        if isinstance(sys.exc_info()[1], ConnectionError):
            return  # a browser tab closed mid-stream is routine
        super().handle_error(request, client_address)


LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def serve(engine: Engine, port: int, host: str = "127.0.0.1", *, allow_remote: bool = False,
          extra_hosts: tuple[str, ...] = ()) -> ThreadingHTTPServer:
    """Bind the service. Anything but loopback is refused unless `allow_remote` is set on purpose (a container must bind
    its own interface, and the host then publishes the port on loopback only). The Host header is still checked: only
    loopback names and the hosts listed in `extra_hosts` are accepted."""
    if host not in LOOPBACK_HOSTS and not allow_remote:
        raise ValueError("The demo service holds cloud credentials and must stay on loopback")
    allowed = set(LOOPBACK_HOSTS) | {h.strip().lower() for h in extra_hosts if h.strip()}
    server = QuietServer((host, port), make_handler(engine, allowed))
    server.daemon_threads = True
    return server


class OncePerMessage(logging.Filter):
    """Show each distinct warning once. A signed-out service would otherwise print the same ten-line credential
    failure for every model call; the first copy says why, and the UI reports each failed call."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        key = record.getMessage()[:300]
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.addFilter(OncePerMessage())
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(handler)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUE_VALUES


def build_parser() -> argparse.ArgumentParser:
    """Every option can also be set by an AIE_* environment variable, which is how the container is configured."""
    env = os.environ.get
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=env("AIE_HOST", "127.0.0.1"))
    parser.add_argument("--allow-non-loopback", action="store_true", default=env_flag("AIE_ALLOW_NON_LOOPBACK"),
                        help="Allow binding beyond loopback, for a container whose port is published on loopback only")
    parser.add_argument("--allowed-host", action="append", default=[h for h in env("AIE_ALLOWED_HOSTS", "").split(",") if h.strip()],
                        help="Extra Host header value to accept, for other containers calling this one (repeatable)")
    parser.add_argument("--port", type=int, default=int(env("AIE_PORT", "8765")))
    parser.add_argument("--runtime-dir", type=Path, default=Path(env("AIE_RUNTIME_DIR", str(ROOT / "outputs" / "agent_demo"))))
    parser.add_argument("--connection", type=Path, default=Path(env("AIE_CONNECTION", str(ROOT / "azure" / "online" / "connection.json"))))
    parser.add_argument("--deployment", default=env("AIE_DEPLOYMENT", "flash-ft-gpu"))
    parser.add_argument("--adapter-sha256", default=env("AIE_ADAPTER_SHA256", FINETUNED_ADAPTER_SHA256))
    parser.add_argument("--llm-endpoint", default=env("AIE_LLM_ENDPOINT", LLM_ENDPOINT))
    parser.add_argument("--router-provider", choices=("clef", "decision-1"), default=env("AIE_ROUTER_PROVIDER", "clef"),
                        help="Choose the capability decision provider without modifying existing endpoints")
    parser.add_argument("--decision-endpoint", default=env("AIE_DECISION_ENDPOINT"),
                        help="Resource URL for the native decision API; defaults to the existing model resource")
    parser.add_argument("--decision-deployment", default=env("AIE_DECISION_DEPLOYMENT", "decision-1"))
    parser.add_argument("--decision-token-price", type=float, default=float(env("AIE_DECISION_TOKEN_PRICE", "0.042")),
                        help="Estimated USD per million decision-model input/output tokens; verify the deployment's billing meter")
    parser.add_argument("--shadow-deployment", default=env("AIE_SHADOW_DEPLOYMENT", CAPABILITY_DEPLOYMENT))
    parser.add_argument("--shadow-adapter-sha256", default=env("AIE_SHADOW_ADAPTER_SHA256", CAPABILITY_ADAPTER_SHA256))
    parser.add_argument("--shadow-url", default=env("AIE_SHADOW_URL"),
                        help="Scoring URL for the shadow router, when it is not the endpoint in the connection file")
    parser.add_argument("--shadow-local", action="store_true", default=env_flag("AIE_SHADOW_LOCAL"),
                        help="Use a locally hosted router at http://host:port/score, without sending a cloud access token")
    parser.add_argument("--shadow-use-skill", action="store_true", default=env_flag("AIE_SHADOW_USE_SKILL"),
                        help="Let the selector use the router's skill answer (the current model was trained without skill labels)")
    parser.add_argument("--capture-router-prompts", action="store_true", default=env_flag("AIE_CAPTURE_ROUTER_PROMPTS"),
                        help="Include exact decision-model request bodies and selection inputs in events and shadow.jsonl. "
                             "Opt-in: stores user request text locally, but never authentication headers")
    parser.add_argument("--apply-selector", action=argparse.BooleanOptionalAction, default=os.environ.get("AIE_APPLY_SELECTOR", "1").strip().lower() in TRUE_VALUES,
                        help=f"Run {LANE_LABELS['clef']} on the endpoints the capability router and selector choose (default). --no-apply-selector "
                             "keeps it on the old step-type router and policy, and the selector only watches")
    parser.add_argument("--route-planner", action=argparse.BooleanOptionalAction, default=os.environ.get("AIE_ROUTE_PLANNER", "1").strip().lower() in TRUE_VALUES,
                        help="Let the decision model and selector choose the planner model too (default; needs --apply-selector). The fixed planner in the policy is the fallback")
    parser.add_argument("--route-judge", action=argparse.BooleanOptionalAction, default=os.environ.get("AIE_ROUTE_JUDGE", "1").strip().lower() in TRUE_VALUES,
                        help=f"Let the decision model and selector choose the reviewer for {LANE_LABELS['clef']} (default; needs --apply-selector). "
                             f"{LANE_LABELS['baseline']} is checked by the fixed reviewer in the policy")
    parser.add_argument("--no-replay", action="store_true", default=env_flag("AIE_NO_REPLAY"),
                        help="Decide in shadow mode but do not also run the chosen endpoints")
    parser.add_argument("--no-shadow", action="store_true", default=env_flag("AIE_NO_SHADOW"),
                        help="Do not run the capability router and selector beside the agent")
    return parser


def scoring_uri(args: argparse.Namespace) -> str:
    """CLEF_ENDPOINT_URL wins over the connection file, so a container needs no file edit."""
    override = os.environ.get("CLEF_ENDPOINT_URL", "").strip()
    if override:
        return override
    if not args.connection.exists():
        raise SystemExit(f"No legacy router endpoint configured: set CLEF_ENDPOINT_URL, or copy azure/online/connection.example.json to {args.connection.name} and fill it in.")
    return json.loads(args.connection.read_text(encoding="utf-8-sig"))["scoring_uri"]


def build_shadow(args: argparse.Namespace, uri: str) -> ShadowSelector | None:
    if args.router_provider == "decision-1" and (args.no_shadow or not args.apply_selector):
        raise ValueError("The selected native decision provider requires the applied capability selector; remove --no-shadow or --no-apply-selector")
    if args.no_shadow:
        return None
    policy = Policy.load(HERE / "data" / "policy.json")
    if args.router_provider == "decision-1":
        from routing_lab.decision_one import DecisionOneRouter

        router = DecisionOneRouter(args.decision_endpoint or args.llm_endpoint, args.decision_deployment,
                                   price_per_million_tokens=args.decision_token_price)
        adapter = None
    else:
        router = CapabilityRouter(args.shadow_url or uri, "clef-local" if args.shadow_local else args.shadow_deployment,
                                  adapter_sha256=args.shadow_adapter_sha256, local=args.shadow_local)
        adapter = args.shadow_adapter_sha256
    return ShadowSelector(router, MeasuredRegistry.load(MEASURED_REGISTRY), baseline=policy.baseline_model,
                          router_cost_usd=policy.router_cost_usd, log_path=args.runtime_dir / "shadow.jsonl",
                          adapter_sha256=adapter, use_skill=args.shadow_use_skill,
                          capture_prompts=args.capture_router_prompts)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging()
    if "your-resource" in args.llm_endpoint:
        raise SystemExit("Set AIE_LLM_ENDPOINT (or --llm-endpoint) to your Azure AI resource, for example https://<name>.services.ai.azure.com/openai/v1/")
    uri = scoring_uri(args)
    shadow = build_shadow(args, uri)
    engine = Engine(args.runtime_dir, uri, args.deployment, args.adapter_sha256, args.llm_endpoint, shadow=shadow,
                    shadow_replay=not args.no_replay, apply_selector=args.apply_selector, route_planner=args.route_planner, route_judge=args.route_judge)
    threading.Thread(target=engine.warm_up, daemon=True).start()
    server = serve(engine, args.port, args.host, allow_remote=args.allow_non_loopback, extra_hosts=tuple(args.allowed_host))
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host  # noqa: S104 - a display name, not a bind
    print(f"Adaptive Intelligence Engine PoC on http://{shown}:{args.port}  (Ctrl+C to stop)", flush=True)
    if args.host not in LOOPBACK_HOSTS:
        print("Bound beyond loopback on purpose: publish the port on 127.0.0.1 only, or put an authenticating proxy in front. "
              "The service holds cloud credentials and can spend money.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()