"""Capability routing with Clef: how much model capability a request needs, and what kind of work it is.

One call returns two answers from a single forward pass: a probability for each capability tier (low, medium,
high) and for each skill (coding, reasoning, tools, writing, extraction). The router never sees tools or picks a
model; selecting an endpoint from these probabilities is the selector's job, using measured model profiles.

Command line, for trying the deployed endpoint:

    python -m routing_lab.capability "Reformat this date as YYYY-MM-DD: 3rd of March 2026"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from routing_lab.backends import CapabilityDecision, Decision
from routing_lab.data import load_contracts
from routing_lab.endpoint import ClefEndpointBackend
from routing_lab.taxonomy import SKILLS, taxonomy_routes

TIERS = taxonomy_routes("capability-v1")
CONTRACTS = Path(__file__).resolve().parents[1] / "contracts" / "capability-v1.json"
MAX_REQUEST_CHARS = 20_000


@dataclass(frozen=True)
class CapabilityResult:
    """The router's answer: full distributions, never just a label."""

    tier_probabilities: dict[str, float]
    skill_probabilities: dict[str, float]
    latency_ms: float
    input_tokens: int | None = None
    deployment: str | None = None
    context_dropped_tokens: int = 0
    extras: dict[str, object] = field(default_factory=dict)
    output_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def tier(self) -> str:
        return max(self.tier_probabilities, key=self.tier_probabilities.get)

    @property
    def skill(self) -> str:
        return max(self.skill_probabilities, key=self.skill_probabilities.get)

    @property
    def confidence(self) -> float:
        return max(self.tier_probabilities.values())

    @property
    def expected_tier(self) -> float:
        """Probability-weighted tier on a 0 (low) to 2 (high) scale, a smoother signal than the argmax."""
        return sum(index * self.tier_probabilities[tier] for index, tier in enumerate(TIERS))

    def to_dict(self) -> dict[str, object]:
        return {"tier": self.tier, "tier_probabilities": self.tier_probabilities, "skill": self.skill,
                "skill_probabilities": self.skill_probabilities, "confidence": round(self.confidence, 4),
                "expected_tier": round(self.expected_tier, 4), "latency_ms": self.latency_ms,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cost_usd": self.cost_usd, "deployment": self.deployment}


def capability_state(request: str, context: str = "") -> str:
    """The decision-time text Clef sees. Requests are data; only the request and optional context are sent."""
    if not request.strip():
        raise ValueError("The request must not be empty")
    if len(request) > MAX_REQUEST_CHARS:
        raise ValueError(f"The request is longer than {MAX_REQUEST_CHARS} characters")
    text = f"User request:\n{request.strip()}"
    return f"{text}\nAvailable context:\n{context.strip()}" if context.strip() else text


class CapabilityRouter:
    """Client for the capability catalog on a Clef endpoint deployment."""

    RETRIES = 6
    provider = "clef"
    display_name = "Clef"

    def __init__(self, url: str, deployment: str | None = None, *, backend: ClefEndpointBackend | None = None,
                 adapter_sha256: str | None = None, timeout: tuple[float, float] = (5, 30), local: bool = False) -> None:
        self.backend = backend or ClefEndpointBackend(
            load_contracts(CONTRACTS, "capability-v1"), url=url, taxonomy_id="capability-v1", deployment=deployment,
            adapter_sha256=adapter_sha256, skills=True, timeout=timeout, local=local, max_gateway_retries=2,
            gateway_retry_backoff_seconds=0.5)
        self.deployment = deployment

    def request_payload(self, request: str, context: str = "") -> dict[str, object]:
        """Build the same native body used by classify without calling the endpoint."""
        return self.backend.request_payload(capability_state(request, context))

    def classify(self, request: str, context: str = "") -> CapabilityResult:
        state = capability_state(request, context)
        started = time.perf_counter()
        for attempt in range(self.RETRIES + 1):
            try:
                decision = self.backend.predict_state(state)
                break
            except requests.ConnectionError:
                # The managed endpoint closes idle connections; reconnect instead of failing the request.
                if attempt == self.RETRIES:
                    raise
                self.backend.reset_connection()
            except RuntimeError as error:
                # The endpoint serves one request at a time, so concurrent callers are throttled briefly.
                if "HTTP 429" not in str(error) or attempt == self.RETRIES:
                    raise
                time.sleep(0.3 * (attempt + 1))
        return self._result(decision, (time.perf_counter() - started) * 1000)

    def _result(self, decision: Decision, latency_ms: float) -> CapabilityResult:
        if not isinstance(decision, CapabilityDecision) or decision.probabilities is None:
            raise ValueError("The endpoint did not return both the tier and the skill distributions")
        if set(decision.skill_probabilities) != set(SKILLS):
            raise ValueError("The endpoint returned an unexpected skill list")
        return CapabilityResult(
            {k: round(v, 5) for k, v in decision.probabilities.items()},
            {k: round(v, 5) for k, v in decision.skill_probabilities.items()},
            round(latency_ms, 1), decision.input_tokens, self.deployment, decision.state_tokens_dropped or 0,
            output_tokens=decision.output_tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("request", nargs="+", help="One or more requests to classify")
    parser.add_argument("--connection", type=Path, default=Path(__file__).resolve().parents[1] / "azure" / "online" / "connection.json")
    parser.add_argument("--deployment", default="flash-gpu")
    args = parser.parse_args()
    url = json.loads(args.connection.read_text(encoding="utf-8-sig"))["scoring_uri"]
    router = CapabilityRouter(url, args.deployment)
    for request in args.request:
        result = router.classify(request)
        print(json.dumps({"request": request[:100], **result.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
