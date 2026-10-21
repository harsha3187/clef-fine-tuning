"""Microsoft Decision-1's native typed-decision API, separate from the Clef endpoint."""

from __future__ import annotations

import math
import re
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import requests

from routing_lab.azure_auth import make_credential
from routing_lab.backends import questions
from routing_lab.capability import CONTRACTS, TIERS, CapabilityResult, capability_state
from routing_lab.data import load_contracts
from routing_lab.taxonomy import SKILLS

TOKEN_SCOPE = "https://ai.azure.com/.default"
API_PATH = "/providers/microsoft/v1/systemone"
HOST = re.compile(r"[a-z0-9-]+\.(?:services\.ai|openai|cognitiveservices)\.azure\.com")


class DecisionOneRouter:
    provider = "decision-1"
    display_name = "Microsoft Decision-1"

    def __init__(self, endpoint: str, deployment: str = "decision-1", *, credential: Any = None,
                 session: requests.Session | None = None, price_per_million_tokens: float = 0.042,
                 timeout: tuple[float, float] = (5, 30), retries: int = 2) -> None:
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "https" or not HOST.fullmatch(parsed.hostname or "") or parsed.port not in (None, 443)
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or parsed.path.rstrip("/") not in ("", "/openai/v1", API_PATH)):
            raise ValueError("The decision-model endpoint requires an HTTPS Azure Foundry resource URL without credentials or query parameters")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", deployment):
            raise ValueError("Invalid decision-model deployment name")
        if not math.isfinite(price_per_million_tokens) or price_per_million_tokens < 0:
            raise ValueError("Decision-model token price must be finite and non-negative")
        self.url = f"{parsed.scheme}://{parsed.netloc}{API_PATH}"
        self.deployment = deployment
        self.question = questions(load_contracts(CONTRACTS, "capability-v1"), taxonomy_id="capability-v1", skills=True)
        self.price_per_million_tokens = price_per_million_tokens
        self.cost_note = (
            f"Usage-based routing estimate at ${price_per_million_tokens:g} per million input/output tokens. "
            "Configured retail rate, not an invoice; DataZone billing should be verified. "
            "Retained hosted-router compute is billed separately."
        )
        self._credential = credential
        self._session = session or requests.Session()
        self._token: Any = None
        self._lock = threading.Lock()
        self._timeout, self._retries = timeout, retries

    def _bearer(self) -> str:
        with self._lock:
            if self._credential is None:
                self._credential = make_credential()
            if self._token is None or self._token.expires_on - time.time() < 300:
                self._token = self._credential.get_token(TOKEN_SCOPE)
            return self._token.token

    def request_payload(self, request: str, context: str = "") -> dict[str, Any]:
        return {"model": self.deployment, "state": capability_state(request, context), "questions": self.question}

    def classify(self, request: str, context: str = "") -> CapabilityResult:
        body = self.request_payload(request, context)
        started = time.perf_counter()
        for attempt in range(self._retries + 1):
            try:
                response = self._session.post(
                    self.url, json=body, timeout=self._timeout, allow_redirects=False,
                    headers={"Authorization": "Bearer " + self._bearer(), "Content-Type": "application/json"},
                )
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self._retries:
                    raise
                time.sleep(0.5 * (attempt + 1))
                continue
            if response.status_code == 200:
                return self._result(response.json(), (time.perf_counter() - started) * 1000)
            if response.status_code not in (408, 429, 500, 502, 503, 504) or attempt == self._retries:
                raise RuntimeError(f"Decision-model endpoint returned HTTP {response.status_code}")
            retry_after = response.headers.get("Retry-After", "")
            delay = float(retry_after) if re.fullmatch(r"\d+(?:\.\d+)?", retry_after) else 0.5 * (attempt + 1)
            time.sleep(min(delay, 30.0))
        raise AssertionError("Unreachable: each final attempt returns or raises")

    @staticmethod
    def _probabilities(answer: Any, options: tuple[str, ...], name: str) -> dict[str, float]:
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError(f"The decision model returned an invalid {name} answer")
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(options):
            raise ValueError(f"The decision model returned unexpected {name} options")
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1
               for v in probabilities.values()):
            raise ValueError(f"The decision model returned invalid {name} probabilities")
        if abs(sum(probabilities.values()) - 1) > 0.001:
            raise ValueError(f"The decision model's {name} probabilities do not sum to one")
        choice = answer.get("choice")
        if not isinstance(choice, str) or choice not in probabilities or probabilities[choice] < max(probabilities.values()) - 1e-6:
            raise ValueError(f"The decision model's {name} choice disagrees with its probabilities")
        return {k: round(float(v), 5) for k, v in probabilities.items()}

    def _result(self, payload: Any, latency_ms: float) -> CapabilityResult:
        if not isinstance(payload, dict) or str(payload.get("model", "")).lower() != "microsoft-decision-1":
            raise ValueError("The decision response did not identify the expected underlying model")
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("The decision response is missing answers")
        tiers = self._probabilities(answers.get("route"), TIERS, "route")
        skills = self._probabilities(answers.get("skill"), SKILLS, "skill")
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            raise ValueError("The decision response is missing token usage")
        for name in ("input_tokens", "output_tokens"):
            if isinstance(usage.get(name), bool) or not isinstance(usage.get(name), int) or usage[name] < 0:
                raise ValueError(f"The decision model returned invalid {name}")
        cost = (usage["input_tokens"] + usage["output_tokens"]) * self.price_per_million_tokens / 1e6
        return CapabilityResult(
            tiers, skills, round(latency_ms, 1), input_tokens=usage["input_tokens"], deployment=self.deployment,
            output_tokens=usage["output_tokens"], cost_usd=cost,
        )
