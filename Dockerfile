#
# Adaptive Intelligence Engine: the agent app, the routing API and the shadow selector.
#
#   docker build -t aie .                       the service (default target)
#   docker build --target test -t aie-test .    the same code plus the tests that need no GPU or cloud
#   docker build --build-arg WITH_AZ_CLI=0 .    without the Azure CLI (use a service principal or managed identity)
#
# The service calls Clef on an Azure ML endpoint (or a local Clef container, see docker/clef-server.Dockerfile) and the
# models on Azure OpenAI. No model weights are in this image.
#
# Layers are ordered slowest and most stable first, so an edit to the code rebuilds in seconds.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS deps
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
# Dependency groups in pyproject.toml (PEP 735). Installing them needs pip 25.1 or newer, and the project file stays out of the service image.
COPY pyproject.toml ./
RUN pip install --upgrade "pip>=25.1" \
    && pip install --group app \
    && rm pyproject.toml

# ------------------------------------------------------------------------------------------------ tests
FROM deps AS test
COPY pyproject.toml ./
RUN pip install --group test
COPY agent_demo/ agent_demo/
COPY routing_lab/ routing_lab/
COPY capability_lab/ capability_lab/
COPY contracts/ contracts/
COPY data/registry/measured.json data/registry/measured.json
COPY azure/online/connection.js* azure/online/
COPY tests/ tests/
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", \
     "tests/test_agent_demo_core.py", "tests/test_agent_demo_router_agent.py", "tests/test_agent_demo_server.py", \
     "tests/test_agent_demo_travel.py", "tests/test_selector.py", \
     "tests/test_endpoint.py", "tests/test_capability.py", "tests/test_verify_model.py", "tests/test_token_broker.py", "tests/test_agent_demo_apply.py"]

# ------------------------------------------------------------------------------------------------ service
FROM deps AS app

# The Azure CLI lives in its own virtual environment so its pinned dependencies cannot disturb the service's.
ARG WITH_AZ_CLI=1
RUN if [ "$WITH_AZ_CLI" = "1" ]; then \
      python -m venv /opt/azcli \
      && /opt/azcli/bin/pip install --no-cache-dir azure-cli \
      && ln -s /opt/azcli/bin/az /usr/local/bin/az ; \
    fi

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin aie \
    && mkdir -p /data/runtime /home/aie/.azure \
    && chown -R aie:aie /data /home/aie
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/aie-entrypoint

# Only what the service reads at run time. Code last: it changes most often.
COPY contracts/ contracts/
COPY data/registry/measured.json data/registry/measured.json
COPY azure/online/connection.js* azure/online/
COPY routing_lab/ routing_lab/
COPY capability_lab/ capability_lab/
COPY agent_demo/ agent_demo/

ENV AIE_HOST=0.0.0.0 \
    AIE_PORT=8765 \
    AIE_ALLOW_NON_LOOPBACK=1 \
    AIE_RUNTIME_DIR=/data/runtime \
    AZURE_CONFIG_DIR=/home/aie/.azure
USER aie
VOLUME ["/data", "/home/aie/.azure"]
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/api/shadow' % os.environ.get('AIE_PORT','8765'), timeout=4)"
ENTRYPOINT ["aie-entrypoint"]