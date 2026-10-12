# Clef served from a container: the same scorer Azure ML runs (azure/online/code/score.py) on the same HTTP server, so the
# request and response contract is identical to the deployed endpoint and any client that works against it works here.
#
#   docker build -f docker/clef-server.Dockerfile --target server -t clef-server:local .          model mounted at /models
#   docker build -f docker/clef-server.Dockerfile --target server-baked -t clef-server:baked .    model copied into the image
#
# Needs an NVIDIA GPU with the NVIDIA Container Toolkit (CUDA 12.6 capable driver). Set CLEF_DEVICE=cpu to try it without one: it
# needs about 20 GB of memory and takes tens of seconds per request. The 18 GB base model (Cloudflare/clef-flash, Apache-2.0,
# pinned revision) is downloaded from Hugging Face into /cache/huggingface on first start; mount a volume there.
#
# Run `git lfs pull` before building the baked image, or the weights will be pointer files (the container checks and says so).

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    CLEF_VARIANT=flash \
    CLEF_FINETUNED=true \
    CLEF_DEVICE=cuda \
    WORKER_COUNT=1 \
    WORKER_TIMEOUT=1800 \
    SCORING_TIMEOUT_MS=180000 \
    AZUREML_MODEL_DIR=/models \
    CLEF_MODEL_DIR=/models
WORKDIR /app

# The same packages as azure/online/conda.yml. The CUDA runtime comes with the torch wheel, so no CUDA base image is needed.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
COPY docker/requirements-clef-server.txt ./
RUN pip install --extra-index-url ${TORCH_INDEX} -r requirements-clef-server.txt

# Laid out exactly as azure/online/manage.py packages it for Azure, plus the model checker
COPY azure/online/code/score.py code/score.py
COPY routing_lab/__init__.py routing_lab/taxonomy.py routing_lab/clef.py routing_lab/verify_model.py code/routing_lab/
COPY --chmod=0755 docker/clef-server-entrypoint.sh /usr/local/bin/clef-entrypoint

RUN useradd --uid 10001 --create-home --shell /usr/sbin/nologin clef \
    && mkdir -p /cache/huggingface /models && chown -R clef:clef /cache /models
USER clef
VOLUME ["/cache/huggingface"]
EXPOSE 5001
# Loading the model, and downloading it the first time, takes many minutes.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=5 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:5001/', timeout=4)"
ENTRYPOINT ["clef-entrypoint"]

# ------------------------------------------------------------------------------------------------ model mounted
FROM runtime AS server

# ------------------------------------------------------------------------------------------------ model inside the image
FROM runtime AS server-baked
USER root
COPY models/clef-flash-capability-ft-v3/ /models/
RUN chown -R clef:clef /models
USER clef
