"""Call the pinned Clef deployment from a client, using the Azure identity configured in routing_lab.azure_auth.

By default that is the Azure CLI identity. A locally served Clef container (``local=True``) needs no identity at all.
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlsplit

from routing_lab import clef
from routing_lab.azure_auth import credential_name
from routing_lab.backends import Decision, decision_from_native, questions, state
from routing_lab.data import Contract, Record, records_taxonomy
from routing_lab.taxonomy import taxonomy_for_routes

TOKEN_SCOPE = "https://ml.azure.com/.default"


class ClefEndpointBackend:
    """Remote-only Clef adapter. No model dependencies or weights are loaded.

    Use ``az login`` first. The default credential is an explicit AzureCliCredential, which avoids silently selecting
    another developer or managed identity through a fallback chain; set ``AIE_AZURE_AUTH`` to choose another one.
    ``local=True`` allows a plain-HTTP ``/score`` URL on a private host (a container running the same scorer) and sends no
    token; the pinned model and adapter checks still apply to every answer.
    """

    def __init__(
        self, contracts: dict[str, Contract], *, url: str | None = None,
        context_mode: str = "context", variant: clef.ClefVariant = "flash",
        taxonomy_id: str | None = None,
        max_gateway_retries: int = 8, gateway_retry_backoff_seconds: float = 2.0,
        max_length: int | None = None, deployment: str | None = None,
        adapter_sha256: str | None = None, timeout: tuple[float, float] = (10, 190), skills: bool = False,
        local: bool = False,
    ) -> None:
        self.taxonomy_id = taxonomy_for_routes(contracts, taxonomy_id)
        if deployment is not None and not re.fullmatch(r"[a-z0-9-]{3,32}", deployment):
            raise ValueError("Deployment names are 3 to 32 lowercase letters, digits or hyphens")
        if adapter_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", adapter_sha256):
            raise ValueError("adapter_sha256 must be a lowercase SHA-256 hex digest")
        self.deployment = deployment
        self.adapter_sha256 = adapter_sha256
        self.local = local
        self.url = url if url is not None else os.environ["CLEF_ENDPOINT_URL"]
        parsed = urlsplit(self.url)
        if local:
            valid = (parsed.scheme == "http" and re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,62}", parsed.hostname or "") is not None
                     and parsed.hostname not in {"169.254.169.254", "metadata.google.internal"}
                     and parsed.path == "/score" and parsed.port is not None)
            message = "A local Clef URL is http://<host>:<port>/score, without credentials"
        else:
            valid = (parsed.scheme == "https"
                     and re.fullmatch(r"[a-z0-9-]+\.[a-z0-9-]+\.inference\.ml\.azure\.com", parsed.hostname or "") is not None
                     and parsed.path == "/score" and parsed.port in {None, 443})
            message = "Use the HTTPS Azure ML scoring_uri ending in /score, without credentials"
        if not valid or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
            raise ValueError(message)
        if context_mode not in {"query", "context"}:
            raise ValueError("Context mode must be query or context")
        self.context_mode = context_mode
        self.question = questions(contracts, taxonomy_id=self.taxonomy_id, skills=skills)
        self.expected_model = clef.model_info(variant)
        if local:
            self.credential = None
        else:
            from routing_lab.azure_auth import make_credential

            self.credential = make_credential(process_timeout=30)
        self._cached_token = None
        import requests

        # Reuse one connection (TCP+TLS) across every request instead of paying a fresh
        # handshake per call. Measured impact: one request took 834ms of actual server
        # processing time but 3.48s of client wall time; most of that gap is per-call
        # connection setup, not inference, and is avoidable via keep-alive reuse.
        self._session = requests.Session()
        self._max_gateway_retries = max_gateway_retries
        self._timeout = timeout
        self._gateway_retry_backoff_seconds = gateway_retry_backoff_seconds
        self.max_length = max_length
        import time

        self._sleep = time.sleep

    def _bearer_token(self):
        """Cache the Entra access token in-process across predict() calls.

        AzureCliCredential.get_token() shells out to a fresh `az` subprocess on
        every call (measured at ~2-2.8s each on this workstation), with no
        caching of its own. Refetching on every single prediction previously
        dominated per-request latency far more than network transit or model
        inference time. Refresh only when the cached token is missing or
        within five minutes of expiring.
        """
        import time

        now = time.time()
        if self._cached_token is None or self._cached_token.expires_on - now < 300:
            self._cached_token = self.credential.get_token(TOKEN_SCOPE)
        return self._cached_token

    def reset_connection(self) -> None:
        """Drop the pooled connection, for when the service has closed an idle one."""
        import requests
        self._session.close()
        self._session = requests.Session()

    def predict(self, record: Record) -> Decision:
        """Route one dataset record; only its decision-time text is transmitted, never its labels."""
        records_taxonomy([record], self.taxonomy_id)
        return self.predict_state(state(record, self.context_mode))

    def predict_state(self, state_text: str) -> Decision:
        """Route already-formatted decision-time text, with the retry policy below.

        A bounded retry absorbs the managed-endpoint gateway's observed intermittent
        HTTP 424 ("Failed Dependency"): the scoring container's own access log shows
        200 for every request during these incidents (confirmed via deployment logs
        across repeated runs), and a retry of the identical request always eventually
        succeeded, though some bursts lasted longer than a couple of seconds. Every
        other status code still fails the run immediately; this does not mask
        authentication, schema or genuine model/deployment failures.
        """
        for attempt in range(self._max_gateway_retries + 1):
            try:
                return self._send(state_text)
            except RuntimeError as exc:
                if "HTTP 424" not in str(exc) or attempt == self._max_gateway_retries:
                    raise
                self._sleep(min(self._gateway_retry_backoff_seconds * (attempt + 1), 20.0))
        raise AssertionError("Unreachable: loop always returns or raises")

    def _send(self, state_text: str) -> Decision:
        body: dict[str, object] = {"state": state_text, "questions": self.question}
        if self.max_length is not None:
            body["max_length"] = self.max_length
        headers = {"Content-Type": "application/json"}
        if not self.local:
            token = self._bearer_token()
            headers["Authorization"] = f"Bearer {token.token}"
        if self.deployment is not None:
            # Reach a specific deployment even while it holds no live traffic.
            headers["azureml-model-deployment"] = self.deployment
        with self._session.post(
            self.url,
            json=body,
            headers=headers,
            timeout=self._timeout, allow_redirects=False,
        ) as response:
            if response.status_code != 200:
                # Error bodies can echo source text; do not include them in local logs.
                raise RuntimeError(
                    f"Clef endpoint returned HTTP {response.status_code}; "
                    "check deployment health, scoring permissions and Azure logs"
                )
            payload = response.json()
        if not isinstance(payload, dict) or not self._provenance_matches(payload.get("experiment_model")):
            raise ValueError("Endpoint model provenance differs from the pinned Clef release")
        return decision_from_native(payload, self.taxonomy_id)

    def _provenance_matches(self, reported: object) -> bool:
        """Base model: exact match. Fine-tuned: same base plus the pinned adapter hash."""
        if not isinstance(reported, dict):
            return False
        if self.adapter_sha256 is None:
            return reported == self.expected_model
        base = {key: value for key, value in reported.items() if key != "fine_tuned"}
        tuned = reported.get("fine_tuned")
        return (base == self.expected_model and isinstance(tuned, dict)
                and tuned.get("adapter_sha256") == self.adapter_sha256)

    def close(self) -> None:
        """Release the pooled connection; harmless to skip for a short-lived process."""
        self._session.close()

    def metadata(self) -> dict[str, object]:
        return {
            "backend": "clef-endpoint", "url": self.url,
            "taxonomy_id": self.taxonomy_id,
            "expected_model": self.expected_model.copy(),
            "provenance_policy": "every response must match expected_model"
            + (" and the pinned adapter hash" if self.adapter_sha256 else ""),
            "deployment": self.deployment, "adapter_sha256": self.adapter_sha256,
            "auth": "none (local container)" if self.local else credential_name(), "context_mode": self.context_mode,
            "questions": self.question,
            "timing_scope": "client wall time including authentication, network, queue and inference",
            "max_length": self.max_length,
        }
