# Dev container

A VS Code dev container for working on this repository: Python 3.11, the service and test dependencies, ruff, Git LFS and the Azure CLI. It has no GPU and no model weights.
Clef itself runs on the Azure endpoint (or the optional `clef-server` container, see [running in containers](docker.md)).

## Open it

1. Install Docker and the VS Code Dev Containers extension.
2. Open the repository folder and choose **Reopen in Container** (or run `devcontainer up --workspace-folder .` with the `@devcontainers/cli` package).
3. The first build takes a few minutes. It installs `requirements-app.txt`, `requirements-test.txt` and `.devcontainer/requirements-dev.txt` into the image.

## Use it

```text
python -m pytest -q                  the whole suite (about 1,190 tests, under two minutes)
python -m ruff check agent_demo routing_lab capability_lab routing_data azure tests     lint the code (notebooks are left out)
python -m agent_demo.server          the app on http://localhost:8765 (VS Code forwards the port)
```

If another process already uses port 8765 on your machine (for example the compose service), VS Code forwards the dev container's port to a different local port and tells you which.

## Signing in to Azure

Device-code login can be blocked by a tenant, and a Windows `az` session cannot be shared with a Linux container. Use the host token broker, the same way as for the app container:

```text
python -m routing_lab.token_broker --write-env .env      on the host, leave it running
```

The container reaches it at `host.docker.internal` (the container is started with `--add-host=host.docker.internal:host-gateway`) and every new terminal reads `.env`. Open a new terminal after
creating `.env`. `az login` inside the container also works where your tenant allows it.

## Notes

- The repository is bind-mounted, so edits are live and `.env` (gitignored) is readable inside. Python packages live in the image, not in a `.venv`; the host's `.venv` is ignored.
- Docker Desktop's file sharing can fail with "Cannot allocate memory" when the VM is short of memory, most often in the minutes after a rebuild. Three clean runs in a row followed; if a test run fails with that
  error, give Docker more memory or stop other containers and rerun.
- The Azure CLI comes from the dev container feature and can be older than the latest release.