"""Create the GPU training cluster in the Clef workspace and grant its identity the access it needs.

Creating a cluster with zero minimum nodes costs nothing; GPU billing starts only when a job runs.
Nothing changes unless ``--approve`` is passed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess

DEFAULTS = {
    "subscription": os.environ.get("AZURE_SUBSCRIPTION_ID", ""),  # not stored in the repository
    "resource_group": os.environ.get("AZURE_RESOURCE_GROUP", ""),  # not stored in the repository
    "workspace": os.environ.get("AZURE_ML_WORKSPACE", ""),  # not stored in the repository
    "vnet": "vnet-clef-exp",
    "subnet": "subnet-compute",
}
# Scoped to the experiment's own storage account and registry, nothing wider.
STORAGE_ROLES = ("Storage Blob Data Contributor", "Storage File Data Privileged Contributor")
REGISTRY_ROLES = ("AcrPull",)


def _grant(principal: str, role: str, scope: str) -> str:
    az = shutil.which("az")
    if az is None:
        raise RuntimeError("Azure CLI (az) is required to grant roles")
    result = subprocess.run(
        [az, "role", "assignment", "create", "--assignee-object-id", principal,
         "--assignee-principal-type", "ServicePrincipal", "--role", role, "--scope", scope,
         "--query", "roleDefinitionName", "-o", "tsv"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Could not grant {role}: {result.stderr.strip()[:300]}")
    return result.stdout.strip()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="clef-gpu-train")
    parser.add_argument("--size", default="Standard_NC40ads_H100_v5")
    parser.add_argument("--idle-seconds", type=int, default=600)
    parser.add_argument("--subscription", default=DEFAULTS["subscription"])
    parser.add_argument("--resource-group", default=DEFAULTS["resource_group"])
    parser.add_argument("--workspace", default=DEFAULTS["workspace"])
    parser.add_argument("--vnet", default=DEFAULTS["vnet"])
    parser.add_argument("--subnet", default=DEFAULTS["subnet"])
    parser.add_argument("--approve", action="store_true")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if not args.approve:
        raise ValueError("Pass --approve to create the cluster and grant its identity access")
    if not (args.subscription and args.resource_group and args.workspace):
        raise ValueError("Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP and AZURE_ML_WORKSPACE, "
                         "or pass --subscription, --resource-group and --workspace")
    from azure.ai.ml import MLClient
    from azure.ai.ml.entities import AmlCompute, IdentityConfiguration, NetworkSettings
    from azure.identity import AzureCliCredential

    client = MLClient(AzureCliCredential(), args.subscription, args.resource_group, args.workspace)
    subnet = (f"/subscriptions/{args.subscription}/resourceGroups/{args.resource_group}/providers/"
              f"Microsoft.Network/virtualNetworks/{args.vnet}/subnets/{args.subnet}")
    cluster = AmlCompute(
        name=args.name, size=args.size, min_instances=0, max_instances=1,
        idle_time_before_scale_down=args.idle_seconds, ssh_public_access_enabled=False,
        identity=IdentityConfiguration(type="system_assigned"),
        network_settings=NetworkSettings(vnet_name=args.vnet, subnet=subnet),
    )
    made = client.compute.begin_create_or_update(cluster).result()
    principal = made.identity.principal_id
    print(f"Cluster {made.name} ({made.size}) is {made.provisioning_state}; identity {principal}")
    workspace = client.workspaces.get(args.workspace)
    for scope, roles in ((workspace.storage_account, STORAGE_ROLES), (workspace.container_registry, REGISTRY_ROLES)):
        for role in roles:
            print(f"Granted {_grant(principal, role, scope)} on {scope.rsplit('/', 1)[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
