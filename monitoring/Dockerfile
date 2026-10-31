FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.lock .
RUN pip install --no-cache-dir --require-hashes -r requirements.lock && pip check
COPY experiment_metrics.py .
COPY research/evolution/telemetry.py ./research/evolution/
USER 10001:10001
ENTRYPOINT ["python", "/app/experiment_metrics.py", "--root", "/artifacts", "--listen", "0.0.0.0"]
