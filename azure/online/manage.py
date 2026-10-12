"""Manage the experiment endpoint using the SDK when the CLI ml extension is unavailable."""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from azure.ai.ml import MLClient, load_environment, load_online_deployment, load_online_endpoint
from azure.ai.ml.entities import Model
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import AzureCliCredential

SUBSCRIPTION = os.environ.get("AZURE_SUBSCRIPTION_ID", "")  # not stored in the repository: $env:AZURE_SUBSCRIPTION_ID = (az account show --query id -o tsv)
# Dedicated East US 2 workspace: the H100 quota is regional, and the UAE North Laya workspace has none.
GROUP = "rg-clef-exp-eus2"
WORKSPACE = "aml-clef-exp-eus2"
ENDPOINT = "clef-routing-exp"
DEPLOYMENT = "flash-gpu"
MODEL_NAME = "clef-flash-router-ft"


def package_scoring_code(destination: Path) -> Path:
    """Stage only scorer code and the canonical registry, never project data."""
    folder = Path(__file__).resolve().parent
    source = folder.parents[1] / "routing_lab"
    destination.mkdir(parents=True, exist_ok=False)
    package = destination / "routing_lab"
    package.mkdir()
    for name in ("__init__.py", "taxonomy.py", "clef.py"):
        shutil.copyfile(source / name, package / name)
    shutil.copyfile(folder / "code" / "score.py", destination / "score.py")
    return destination


def ensure_environment(client: MLClient, reference: str, spec_path: Path) -> None:
    """Reuse a registered environment; create it only if it is exactly the one the repo defines."""
    name, version = reference.removeprefix("azureml:").split(":")
    try:
        client.environments.get(name, version)
    except ResourceNotFoundError:
        spec = load_environment(spec_path)
        if (spec.name, spec.version) != (name, version):
            raise ValueError(f"Environment {name}:{version} is not registered and does not match environment.yml")
        client.environments.create_or_update(spec)


def register_model(client: MLClient, args: argparse.Namespace) -> int:
    """Register a fine-tuned export that already sits in the workspace datastore; nothing is downloaded."""
    if not args.model_path or not args.model_path.startswith("azureml://datastores/"):
        raise ValueError("--model-path must be an azureml://datastores/... folder URI")
    tags = dict(item.split("=", 1) for item in args.tag)
    model = client.models.create_or_update(Model(
        name=args.model_name, version=args.model_version, path=args.model_path, type="custom_model",
        description="Clef-Flash LoRA adapter, joint head and fitted temperature (routing_lab.finetune export)",
        tags=tags,
    ))
    print(json.dumps({"name": model.name, "version": model.version, "path": model.path, "tags": model.tags},
                     indent=2, default=str))
    return 0


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["status", "logs", "deploy", "register-model"])
    parser.add_argument("--approve-cost", action="store_true")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-version", default="1")
    parser.add_argument("--model-path", help="Datastore URI of a fine-tuned export folder (register-model)")
    parser.add_argument("--tag", action="append", default=[], metavar="KEY=VALUE",
                        help="Provenance tag for the registered model; repeatable")
    parser.add_argument("--deployment-name", default=DEPLOYMENT,
                        help="Deployment to target for status/logs, or to create for deploy")
    parser.add_argument("--deployment-config", type=Path, default=None,
                        help="Deployment YAML for 'deploy'; required for any non-default --deployment-name")
    parser.add_argument("--set-traffic-percent", type=int, default=None,
                        help="Explicitly route this percent of endpoint traffic to --deployment-name")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    if args.action == "deploy" and not args.approve_cost:
        raise ValueError("Deployment starts billed compute; explicitly pass --approve-cost")
    if not SUBSCRIPTION:
        raise SystemExit("Set AZURE_SUBSCRIPTION_ID to your Azure subscription id first.")
    client = MLClient(AzureCliCredential(), SUBSCRIPTION, GROUP, WORKSPACE)
    if args.action == "register-model":
        return register_model(client, args)
    if args.action == "logs":
        print(client.online_deployments.get_logs(
            name=args.deployment_name, endpoint_name=ENDPOINT, lines=100,
        ))
        return 0
    if args.action == "deploy":
        folder = Path(__file__).resolve().parent
        is_default_deployment = args.deployment_name == DEPLOYMENT
        deployment_config = args.deployment_config or (folder / "deployment.yml" if is_default_deployment else None)
        if deployment_config is None:
            raise ValueError("Provide --deployment-config for a non-default --deployment-name")
        try:
            endpoint = client.online_endpoints.get(ENDPOINT)
        except ResourceNotFoundError:
            endpoint = client.online_endpoints.begin_create_or_update(
                load_online_endpoint(folder / "endpoint.yml")
            ).result()
        if endpoint.auth_mode != "aad_token":
            raise ValueError("Refusing to change an existing endpoint with different authentication")
        deployment = load_online_deployment(deployment_config)
        if deployment.name != args.deployment_name:
            raise ValueError("Deployment YAML name does not match --deployment-name")
        ensure_environment(client, str(deployment.environment), folder / "environment.yml")
        if deployment.model:
            name, version = str(deployment.model).removeprefix("azureml:").split(":")
            client.models.get(name, version)  # fail now, not after provisioning a GPU
        with tempfile.TemporaryDirectory(prefix="clef-scoring-") as staging:
            code = package_scoring_code(Path(staging) / "code")
            try:
                existing = client.online_deployments.get(args.deployment_name, ENDPOINT)
            except ResourceNotFoundError:
                existing = None
            if existing is not None and str(existing.provisioning_state).lower().endswith("failed"):
                # A deployment that failed its initial provisioning cannot be updated in place.
                client.online_deployments.begin_delete(args.deployment_name, ENDPOINT).result()
            deployment.code_configuration.code = str(code)
            client.online_deployments.begin_create_or_update(deployment).result()
        if args.set_traffic_percent is not None:
            # Explicit opt-in traffic change; merges with (does not replace) existing splits.
            endpoint.traffic = {**(endpoint.traffic or {}), args.deployment_name: args.set_traffic_percent}
            client.online_endpoints.begin_create_or_update(endpoint).result()
        elif is_default_deployment:
            # Unchanged original behavior for the default deployment name only.
            endpoint.traffic = {DEPLOYMENT: 100}
            client.online_endpoints.begin_create_or_update(endpoint).result()
        # A non-default deployment with no --set-traffic-percent is created at 0% traffic,
        # reachable only via an explicit per-request deployment header, until validated.
    endpoint = client.online_endpoints.get(ENDPOINT)
    deployment = client.online_deployments.get(args.deployment_name, ENDPOINT)
    connection_path = Path(__file__).resolve().parent / "connection.json"
    connection = json.loads(connection_path.read_text(encoding="utf-8-sig"))
    if endpoint.scoring_uri and connection["scoring_uri"] != endpoint.scoring_uri:
        connection["scoring_uri"] = endpoint.scoring_uri
        connection_path.write_text(json.dumps(connection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "endpoint": endpoint.name,
        "scoring_uri": endpoint.scoring_uri,
        "authentication": endpoint.auth_mode,
        "traffic": endpoint.traffic,
        "deployment": deployment.name,
        "provisioning_state": deployment.provisioning_state,
        "instance_type": deployment.instance_type,
        "instance_count": deployment.instance_count,
        "environment": deployment.environment,
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
