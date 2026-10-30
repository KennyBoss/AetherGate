FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    JAX_PLATFORM_NAME=cpu \
    JAX_PLATFORMS=cpu \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    PYTHONPATH=/app

WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock \
    && pip check \
    && mkdir /input /output \
    && chown 10001:10001 /output
COPY agent_ssm_core.py run_evolution.py ./
COPY research/evolution/*.py ./research/evolution/
COPY configs/evolution/ ./configs/evolution/

USER 10001:10001
ENTRYPOINT ["python", "/app/run_evolution.py"]
