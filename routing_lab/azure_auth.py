"""Which Azure identity the clients use.

The default stays the Azure CLI identity: it never silently picks up another developer's or a managed identity. A container has no
developer sign-in, so it can opt in with ``AIE_AZURE_AUTH``:

- ``cli`` (default): the signed-in Azure CLI user. In a container, run ``az login --use-device-code`` once.
- ``default``: the standard credential chain: service principal from ``AZURE_TENANT_ID``, ``AZURE_CLIENT_ID`` and
  ``AZURE_CLIENT_SECRET``, then workload or managed identity, then the Azure CLI. Never an interactive browser prompt.
- ``managed-identity``: only the host's managed identity (``AZURE_CLIENT_ID`` picks a user-assigned one).
"""

from __future__ import annotations

import os
from typing import Any

AUTH_ENV = "AIE_AZURE_AUTH"
MODES = ("cli", "default", "managed-identity")


def auth_mode() -> str:
    mode = os.environ.get(AUTH_ENV, "cli").strip().lower() or "cli"
    if mode not in MODES:
        raise ValueError(f"{AUTH_ENV} must be one of {list(MODES)}, not {mode!r}")
    return mode


def make_credential(process_timeout: int = 30) -> Any:
    """Build the credential for the configured mode. The azure.identity import is lazy so tests can substitute it."""
    mode = auth_mode()
    import azure.identity as identity

    if mode == "default":
        return identity.DefaultAzureCredential(exclude_interactive_browser_credential=True, process_timeout=process_timeout)
    if mode == "managed-identity":
        return identity.ManagedIdentityCredential(client_id=os.environ.get("AZURE_CLIENT_ID") or None)
    return identity.AzureCliCredential(process_timeout=process_timeout)


def credential_name() -> str:
    return {"cli": "AzureCliCredential", "default": "DefaultAzureCredential",
            "managed-identity": "ManagedIdentityCredential"}[auth_mode()]
