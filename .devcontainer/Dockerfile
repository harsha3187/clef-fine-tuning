# Development container: Python 3.11 with the service's and the tests' dependencies, ruff, and the Azure CLI (added as a feature).
# It has no GPU and no model weights; Clef runs on the Azure endpoint or the optional clef-server container.
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY pyproject.toml /tmp/deps/
RUN pip install --upgrade "pip>=25.1" \
    && cd /tmp/deps && pip install --group app --group test --group devcontainer \
    && rm -rf /tmp/deps