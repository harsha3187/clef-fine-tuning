"""Deployment helper guards: registration needs a datastore path, environments are never invented."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest

pytest.importorskip("azure.ai.ml")
from azure.core.exceptions import ResourceNotFoundError  # noqa: E402

MANAGE = Path(__file__).resolve().parents[1] / "azure" / "online" / "manage.py"


@pytest.fixture(scope="module")
def manage():
    spec = importlib.util.spec_from_file_location("online_manage", MANAGE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def args(manage, *extra):
    return manage.create_parser().parse_args(["register-model", *extra])


@pytest.mark.parametrize("path", [None, "outputs/export", "https://example.com/export", "azureml://jobs/x/outputs/y"])
def test_registration_requires_a_datastore_folder_uri(manage, path):
    extra = ["--model-path", path] if path else []
    with pytest.raises(ValueError, match="azureml://datastores"):
        manage.register_model(Mock(), args(manage, *extra))


def test_registration_passes_provenance_tags_and_never_downloads(manage):
    client = Mock()
    client.models.create_or_update.side_effect = lambda model: model
    uri = "azureml://datastores/workspaceblobstore/paths/azureml/job-1/results/run/export/"
    assert manage.register_model(client, args(
        manage, "--model-path", uri, "--tag", "training_job=job-1", "--tag", "note=a=b")) == 0
    model = client.models.create_or_update.call_args.args[0]
    assert (model.name, model.version, model.path, model.type) == ("clef-flash-router-ft", "1", uri, "custom_model")
    assert model.tags == {"training_job": "job-1", "note": "a=b"}


def test_a_registered_environment_is_reused_not_recreated(manage):
    client = Mock()
    manage.ensure_environment(client, "azureml:clef-routing-gpu:3", MANAGE.parent / "environment.yml")
    client.environments.get.assert_called_once_with("clef-routing-gpu", "3")
    client.environments.create_or_update.assert_not_called()


def test_missing_environment_is_created_only_if_it_matches_the_repo_definition(manage):
    client = Mock()
    client.environments.get.side_effect = ResourceNotFoundError("missing")
    manage.ensure_environment(client, "azureml:clef-routing-gpu:3", MANAGE.parent / "environment.yml")
    created = client.environments.create_or_update.call_args.args[0]
    assert (created.name, created.version) == ("clef-routing-gpu", "3")
    client.environments.create_or_update.reset_mock()
    with pytest.raises(ValueError, match="does not match environment.yml"):
        manage.ensure_environment(client, "azureml:clef-routing-gpu:9", MANAGE.parent / "environment.yml")
    with pytest.raises(ValueError, match="does not match environment.yml"):
        manage.ensure_environment(client, "azureml:someone-elses-env:3", MANAGE.parent / "environment.yml")
    client.environments.create_or_update.assert_not_called()
