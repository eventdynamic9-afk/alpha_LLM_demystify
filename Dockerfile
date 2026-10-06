# Reproducible environment for the rationale-fidelity pipeline (§16).
#   docker build -t rationale-fidelity .
#   docker run --rm -v "$PWD/runs:/app/runs" rationale-fidelity make smoke
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 MPLBACKEND=Agg
RUN apt-get update && apt-get install -y --no-install-recommends make git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.lock requirements.txt pyproject.toml ./
RUN pip install -r requirements.lock
COPY . .
RUN pip install --no-deps -e . && make test PY=python

CMD ["make", "smoke", "PY=python"]
