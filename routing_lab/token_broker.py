"""A token broker for containers: hands the signed-in Azure CLI user's tokens to a local container, the way a managed identity would.

Why: a container has no sign-in of its own, and device-code login is often blocked by tenant policy. The host usually is signed in
(`az login`). This small server runs on the host, speaks the App Service managed-identity protocol that every Azure SDK already
understands, and answers each request with `az account get-access-token` for one of a few allowed resources. The container then
needs no login and no secret file: it just points `IDENTITY_ENDPOINT` here.

    python -m routing_lab.token_broker --write-env .env      # on the host; leave it running
    docker compose up -d                                       # the container picks the settings up from .env

Safety: it listens on loopback only, answers only requests that carry a random per-run secret, and only hands out tokens for an
allow-list of resources (the Azure ML endpoint and the Azure OpenAI/Foundry API by default). The secret is written to `.env`,
which is not committed. Anything that can reach this port with the secret can act as you on those two resources until you stop
the broker, so run it only on a machine you control, and stop it when you are done.
"""

from __future__ import annotations

import argparse
import hmac
import json
import re
import secrets
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

DEFAULT_ALLOWED = ("https://ml.azure.com", "https://ai.azure.com")
API_VERSIONS = frozenset({"2019-08-01", "2017-09-01"})
HEADER = "X-IDENTITY-HEADER"
PATH = "/msi/token"
MARGIN_SECONDS = 300  # hand out a cached token only while it has at least this long left
LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})
RESOURCE = re.compile(r"https://[a-z0-9.-]+(/[A-Za-z0-9._-]*)?")


class BrokerError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status, self.message = status, message


def run_az(resource: str) -> dict[str, Any]:
    """The signed-in Azure CLI user's token for one resource."""
    try:
        done = subprocess.run(["az", "account", "get-access-token", "--resource", resource, "-o", "json"],
                              capture_output=True, text=True, timeout=60, check=False, shell=sys.platform == "win32")  # noqa: S602 - az is az.cmd on Windows
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BrokerError(502, "The Azure CLI did not answer") from error
    if done.returncode != 0:
        raise BrokerError(401, "The Azure CLI is not signed in on this machine: run `az login` here")
    try:
        return json.loads(done.stdout)
    except json.JSONDecodeError as error:
        raise BrokerError(502, "The Azure CLI returned something unreadable") from error


class Broker:
    """Validation, caching and the protocol, apart from the HTTP plumbing so it can be tested with a fake `az`."""

    def __init__(self, secret: str, allowed: tuple[str, ...] = DEFAULT_ALLOWED,
                 runner: Callable[[str], dict[str, Any]] = run_az, clock: Callable[[], float] = time.time) -> None:
        if len(secret) < 24:
            raise ValueError("The secret must be at least 24 characters")
        self.secret, self.allowed, self._run, self._clock = secret, frozenset(a.rstrip("/") for a in allowed), runner, clock
        self._cache: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def token(self, presented: str | None, resource: str | None, api_version: str | None) -> dict[str, Any]:
        if not presented or not hmac.compare_digest(presented.encode(), self.secret.encode()):
            raise BrokerError(401, "Missing or wrong identity header")
        if api_version not in API_VERSIONS:
            raise BrokerError(400, "Unsupported api-version")
        if not resource or not RESOURCE.fullmatch(resource) or resource.rstrip("/") not in self.allowed:
            raise BrokerError(403, "That resource is not on this broker's allow-list")
        key = resource.rstrip("/")
        with self._lock:  # one az call at a time; the SDK also caches, so calls are rare
            cached = self._cache.get(key)
            if cached is None or cached["expires_on"] - self._clock() < MARGIN_SECONDS:
                data = self._run(key)
                expires = data.get("expires_on")
                if not isinstance(data.get("accessToken"), str) or not isinstance(expires, (int, float)):
                    raise BrokerError(502, "The Azure CLI returned an unexpected token response")
                cached = {"access_token": data["accessToken"], "expires_on": int(expires)}
                self._cache[key] = cached
        return {"access_token": cached["access_token"], "expires_on": str(cached["expires_on"]), "resource": resource,
                "token_type": "Bearer"}


def make_handler(broker: Broker) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "AIE-TokenBroker"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # request lines would show resources; tokens never appear here either way
            return

        def _send(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            parts = urlsplit(self.path)
            try:
                if parts.path != PATH:
                    raise BrokerError(404, "Not found")
                query = parse_qs(parts.query)
                one = lambda name: (query.get(name) or [None])[0]  # noqa: E731
                self._send(200, broker.token(self.headers.get(HEADER), one("resource"), one("api-version")))
            except BrokerError as error:
                self._send(error.status, {"error": error.message})

    return Handler


def serve(broker: Broker, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    if host not in LOOPBACK:
        raise ValueError("The token broker hands out your identity and must stay on loopback")
    server = ThreadingHTTPServer((host, port), make_handler(broker))
    server.daemon_threads = True
    return server


def write_env(path: Path, values: dict[str, str]) -> None:
    """Add or replace these keys in an env file, leaving every other line alone."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    keep = [line for line in lines if line.split("=", 1)[0].strip() not in values]
    keep += [f"{key}={value}" for key, value in values.items()]
    path.write_text("\n".join(keep) + "\n", encoding="utf-8", newline="\n")  # LF, so a Linux shell or container can read it


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--allow", action="append", default=[], metavar="RESOURCE", help="Another resource to hand tokens out for (repeatable)")
    parser.add_argument("--write-env", type=Path, metavar="FILE", help="Write IDENTITY_ENDPOINT, IDENTITY_HEADER and AIE_AZURE_AUTH into this env file")
    parser.add_argument("--container-host", default="host.docker.internal", help="The name a container uses to reach this machine")
    args = parser.parse_args(argv)
    secret = secrets.token_urlsafe(32)
    allowed = (*DEFAULT_ALLOWED, *args.allow)
    probe = run_az(DEFAULT_ALLOWED[0])  # fail now, with the reason, if the host is not signed in
    del probe
    server = serve(Broker(secret, allowed), args.port)
    endpoint = f"http://{args.container_host}:{args.port}{PATH}"
    settings = {"AIE_AZURE_AUTH": "managed-identity", "IDENTITY_ENDPOINT": endpoint, "IDENTITY_HEADER": secret}
    if args.write_env:
        write_env(args.write_env, settings)
        print(f"wrote {', '.join(settings)} to {args.write_env}")
    else:
        print("\n".join(f"{key}={value}" for key, value in settings.items()))
    print(f"token broker on 127.0.0.1:{args.port} for {', '.join(sorted(a.rstrip('/') for a in allowed))}. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
