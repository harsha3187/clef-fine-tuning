"""System-1 step classification with the fine-tuned Clef model, plus a deterministic fallback (BR-008)."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import requests

from routing_lab.backends import state
from routing_lab.data import load_contracts
from routing_lab.endpoint import ClefEndpointBackend

CATALOGS = {
    "code": ("coding-v1", "contracts/agents.json"),
    "tools": ("tool-api-v1", "contracts/tool-api-v1.json"),
}
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ClefDecision:
    taxonomy: str
    route: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    margin: float | None = None
    entropy: float | None = None
    latency_ms: float = 0.0
    input_tokens: int | None = None
    source: str = "clef"  # "clef", or "fallback" when the model could not be reached
    error: str | None = None
    deployment: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def describe(probabilities: dict[str, float]) -> tuple[float, float, float]:
    """Top probability, its lead over the runner-up, and the normalized entropy of the distribution."""
    ordered = sorted(probabilities.values(), reverse=True)
    top, second = ordered[0], ordered[1] if len(ordered) > 1 else 0.0
    entropy = -sum(p * math.log(p) for p in ordered if p > 0) / math.log(len(ordered)) if len(ordered) > 1 else 0.0
    return round(top, 4), round(top - second, 4), round(entropy, 4)


def code_context(repository: str) -> str:
    return json.dumps({"repository": repository, "repository_content_supplied": False})


def tools_context(tools: list[dict[str, Any]], preceding: tuple[str, ...] = ()) -> str:
    """The exact shape the model was trained on: available functions plus preceding instructions."""
    return json.dumps({"available_functions": tools, "preceding_instructions": [p[:240] for p in preceding[-3:]]})


class ClefDecider:
    """One decision per call through the fine-tuned deployment; never raises."""

    def __init__(
        self, url: str, deployment: str, adapter_sha256: str, fallback_routes: dict[str, str],
        *, backends: dict[str, ClefEndpointBackend] | None = None, retries_on_429: int = 3,
    ) -> None:
        self.url, self.deployment, self.adapter_sha256 = url, deployment, adapter_sha256
        self.fallback_routes = fallback_routes
        self._backends = backends or {}
        self._lock = threading.Lock()  # the endpoint serves one request at a time
        self._retries = retries_on_429
        self.last: dict[str, Any] | None = None  # status of the most recent call, for the UI

    def _backend(self, taxonomy: str, contracts_path: str) -> ClefEndpointBackend:
        if taxonomy not in self._backends:
            contracts = load_contracts(ROOT / contracts_path, taxonomy)
            self._backends[taxonomy] = ClefEndpointBackend(
                contracts, url=self.url, taxonomy_id=taxonomy, deployment=self.deployment,
                adapter_sha256=self.adapter_sha256, max_gateway_retries=1, gateway_retry_backoff_seconds=0.5,
                timeout=(5, 12),  # a slow router must fall back quickly rather than hold the case up
            )
        return self._backends[taxonomy]

    def decide(self, domain: str, step_text: str, *, tools: list[dict[str, Any]] | None = None,
               repository: str = "demo/app", preceding: tuple[str, ...] = ()) -> ClefDecision:
        taxonomy, contracts_path = CATALOGS[domain]
        context = code_context(repository) if domain == "code" else tools_context(tools or [], preceding)
        text = state(SimpleNamespace(query=step_text, context=context), "context")
        started = time.perf_counter()
        try:
            backend = self._backend(taxonomy, contracts_path)
            with self._lock:
                for attempt in range(self._retries + 1):
                    try:
                        decision = backend.predict_state(text)
                        break
                    except requests.ConnectionError:
                        # The managed endpoint closes idle keep-alive connections; reconnect once instead of falling back.
                        if attempt == self._retries:
                            raise
                        backend.reset_connection()
                    except RuntimeError as error:
                        if "HTTP 429" not in str(error) or attempt == self._retries:
                            raise
                        time.sleep(0.3 * (attempt + 1))
            top, margin, entropy = describe(decision.probabilities)
            self.last = {"ok": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                         "at": time.time(), "error": None}
            return ClefDecision(
                taxonomy, decision.route, {k: round(v, 4) for k, v in decision.probabilities.items()},
                top, margin, entropy, round((time.perf_counter() - started) * 1000, 1),
                decision.input_tokens, "clef", None, self.deployment)
        except Exception as error:  # noqa: BLE001 - any failure must degrade to the safe path
            self.last = {"ok": False, "latency_ms": round((time.perf_counter() - started) * 1000, 1),
                         "at": time.time(), "error": f"{type(error).__name__}: {str(error)[:160]}"}
            return ClefDecision(
                taxonomy, self.fallback_routes[domain], latency_ms=round((time.perf_counter() - started) * 1000, 1),
                source="fallback", error=f"{type(error).__name__}: {str(error)[:160]}", deployment=self.deployment)
