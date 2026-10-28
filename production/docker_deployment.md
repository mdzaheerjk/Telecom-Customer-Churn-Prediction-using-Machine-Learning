# 🐳 Production Deployment Guide: Docker & Containerization

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![Base Image](https://img.shields.io/badge/Base-python:3.11--slim--bookworm-3776AB?style=flat-square&logo=python&logoColor=white)](https://hub.docker.com/_/python)
[![User](https://img.shields.io/badge/Runs_as-non--root_(10001)-2EA043?style=flat-square)](#1-architecture--security-overview)
[![Last verified](https://img.shields.io/badge/Verified-28_Oct_2026-blue?style=flat-square)](#)

End-to-end instructions for building, running, hardening and troubleshooting the Telecom Customer Churn Prediction app as a containerized workload.

---

## 📑 Table of Contents
1. [Architecture & Security Overview](#1-architecture--security-overview)
2. [Prerequisites](#2-prerequisites)
3. [Build the Image](#3-build-the-image)
4. [Run the Container](#4-run-the-container)
5. [Configuration Reference](#5-configuration-reference)
6. [Optional: Hardened Dockerfile](#6-optional-hardened-dockerfile)
7. [Docker Compose](#7-docker-compose)
8. [Health Checks & Inspection](#8-health-checks--inspection)
9. [Runtime Hardening & Resource Limits](#9-runtime-hardening--resource-limits)
10. [Logging, Updates & CI](#10-logging-updates--ci)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. Architecture & Security Overview

What the current [`Dockerfile`](../Dockerfile) already does, and what this guide adds:

| Control | Current `Dockerfile` | Notes |
| :--- | :---: | :--- |
| Slim base (`python:3.11-slim-bookworm`) | ✅ | Pin by digest in CI for reproducible builds |
| Non-root user `appuser` (UID/GID 10001) | ✅ | Switch login shell to `/usr/sbin/nologin` (§6) |
| Dependency layer cached (`requirements.txt` copied first) | ✅ | Rebuilds in seconds when only code changes |
| `PYTHONDONTWRITEBYTECODE`, `PYTHONUNBUFFERED`, no pip cache | ✅ | Unbuffered output makes `docker logs` real-time |
| `HEALTHCHECK` on `/_stcore/health` | ✅ | Uses `curl` (installed in the image) |
| Multi-stage build | ➖ single stage | Only worthwhile if you compile wheels; slim single-stage is fine here |
| Copies only runtime files | ⚠️ `COPY . .` | Also ships `presentation/`, `production/`, `Interview Questions/` and two unused model files — see §6 |
| Image scanning, digest pinning | ⚠️ not yet | Add in CI (§10) |

> **Model artifact:** `Models/Ada_boost_churn_model.pkl` was trained with scikit-learn **1.5.0**. Pin `scikit-learn==1.5.0` in `requirements.txt` so the container loads it without version warnings (§11).

---

## 2. Prerequisites

- **Docker Engine** 20.10+ (24+ recommended) and **Docker Compose v2** (`docker compose`, not `docker-compose`).
- Verify:
  ```bash
  docker --version
  docker compose version
  ```

---

## 3. Build the Image

From the repository root:

```bash
# Tag with a version and the short git SHA so every image is traceable
GIT_SHA=$(git rev-parse --short HEAD)
docker build -t telecom-churn-app:latest -t telecom-churn-app:${GIT_SHA} .
```

### Inspect size and layers
```bash
docker images telecom-churn-app
docker history telecom-churn-app:latest
```
The scientific stack (NumPy, pandas, SciPy, scikit-learn) and Streamlit dominate the size. Record the number you measure rather than quoting an estimate; [`.dockerignore`](../.dockerignore) already keeps `Notebooks/` and virtual environments out of the build context.

### Scan for vulnerabilities (recommended)
```bash
docker scout cves telecom-churn-app:latest      # Docker Scout
# or
trivy image --severity HIGH,CRITICAL telecom-churn-app:latest
```

---

## 4. Run the Container

### Basic
```bash
docker run -d \
  --name telecom-churn \
  -p 8501:8501 \
  --restart unless-stopped \
  telecom-churn-app:latest
```
Open **http://localhost:8501**.

### With Streamlit settings overridden
```bash
docker run -d \
  --name telecom-churn \
  -p 8501:8501 \
  -e STREAMLIT_SERVER_MAX_UPLOAD_SIZE=5 \
  -e STREAMLIT_THEME_BASE=dark \
  telecom-churn-app:latest
```

> ℹ️ Streamlit reads any environment variable named `STREAMLIT_<SECTION>_<OPTION>` (for example `STREAMLIT_SERVER_PORT`).
> Variables such as `LOG_LEVEL`, `DATA_PATH` and `MODEL_PATH` from [`.env.example`](../.env.example) are **only used if `app.py` reads them** — see [§5](#5-configuration-reference).

---

## 5. Configuration Reference

| Variable | Read by | Set in image? | Purpose |
| :--- | :--- | :---: | :--- |
| `STREAMLIT_SERVER_PORT` | Streamlit | ✅ `8501` | Listening port |
| `STREAMLIT_SERVER_ADDRESS` | Streamlit | ✅ `0.0.0.0` | Bind address |
| `STREAMLIT_SERVER_HEADLESS` | Streamlit | ✅ `true` | No browser/email prompt |
| `STREAMLIT_BROWSER_GATHER_USAGE_STATS` | Streamlit | ✅ `false` | Disable telemetry |
| `LOG_LEVEL`, `DATA_PATH`, `MODEL_PATH`, `APP_ENV` | `app.py` **after the config patch** | ❌ | Application-level settings |

`SERVER_HOST` / `SERVER_PORT` from `.env.example` duplicate the Streamlit variables above; prefer the `STREAMLIT_*` names.

To make the application-level variables effective, apply the small config & logging patch documented in [Streamlit Cloud guide §4](streamlit_cloud_deployment.md#4-configuration-secrets--logging). The same patch works unchanged inside Docker.

---

## 6. Optional: Hardened Dockerfile

A drop-in variant that ships only what the app needs, uses a no-login shell and drops `curl` in favour of Python for the health probe:

```dockerfile
# syntax=docker/dockerfile:1
# In CI, pin by digest:  FROM python:3.11-slim-bookworm@sha256:<digest>
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN groupadd -g 10001 appgroup && \
    useradd -u 10001 -g appgroup -s /usr/sbin/nologin -m appuser

WORKDIR /app

# Dependency layer first -> cached until requirements.txt changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only the files the app actually loads
COPY --chown=appuser:appgroup app.py ./
COPY --chown=appuser:appgroup Data/Customer-Churn.csv ./Data/Customer-Churn.csv
COPY --chown=appuser:appgroup Models/Ada_boost_churn_model.pkl ./Models/Ada_boost_churn_model.pkl

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)" || exit 1

CMD ["streamlit", "run", "app.py"]
```

Why each change matters: fewer files → smaller image and attack surface; `nologin` blocks interactive shells for the service account; no `curl`/`apt` layer → fewer packages to patch.

---

## 7. Docker Compose

Create `docker-compose.yml` (Compose v2 — the legacy `version:` key is obsolete and can be omitted):

```yaml
name: telecom-churn

services:
  churn-service:
    build:
      context: .
      dockerfile: Dockerfile
    image: telecom-churn-app:latest
    container_name: telecom-churn-service
    restart: unless-stopped
    ports:
      # Use "8501:8501" for direct access.
      # Bind to loopback when Nginx/ALB terminates TLS in front of the container.
      - "127.0.0.1:8501:8501"
    environment:
      STREAMLIT_SERVER_PORT: "8501"
      STREAMLIT_BROWSER_GATHER_USAGE_STATS: "false"
      # Effective only after the config patch (see section 5):
      # LOG_LEVEL: INFO
    read_only: true            # test it: add tmpfs for /home/appuser/.streamlit if Streamlit needs to write
    tmpfs:
      - /tmp
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    pids_limit: 100
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        limits:
          cpus: "1.5"
          memory: 1024M
        reservations:
          cpus: "0.5"
          memory: 512M
```

> Using the hardened Dockerfile from §6 (no `curl`)? Replace the healthcheck `test` with
> `["CMD", "python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)"]`.

```bash
docker compose up -d --build        # build and start
docker compose ps                   # shows health status
docker compose logs -f churn-service
docker compose down
```

---

## 8. Health Checks & Inspection

```bash
# Current health: "starting" -> "healthy" | "unhealthy"
docker inspect --format='{{json .State.Health.Status}}' telecom-churn

# Last probe outputs
docker inspect --format='{{range .State.Health.Log}}{{.ExitCode}} {{.Output}}{{end}}' telecom-churn

# Live CPU / memory
docker stats telecom-churn

# Was the container killed for memory?
docker inspect --format='{{.State.OOMKilled}} exit={{.State.ExitCode}}' telecom-churn
```

Orchestrators (ECS, Kubernetes) use the **same endpoint** for load-balancer and liveness checks, so a failing `/_stcore/health` removes the task from rotation automatically.

---

## 9. Runtime Hardening & Resource Limits

```bash
docker run -d \
  --name telecom-churn-prod \
  -p 127.0.0.1:8501:8501 \
  --memory=1g --memory-swap=1g \
  --cpus=1.5 \
  --pids-limit=100 \
  --read-only --tmpfs /tmp \
  --cap-drop=ALL \
  --security-opt=no-new-privileges:true \
  --restart unless-stopped \
  telecom-churn-app:latest
```

| Flag | Protects against |
| :--- | :--- |
| `--memory`, `--memory-swap` | Memory leaks / one session exhausting the host |
| `--cpus` | CPU starvation of neighbours |
| `--pids-limit` | Fork bombs |
| `--read-only` + `--tmpfs /tmp` | Persistence of malware / tampering with app files |
| `--cap-drop=ALL` | Unneeded Linux capabilities |
| `no-new-privileges` | `setuid` privilege escalation |
| `127.0.0.1:` port bind | Accidentally exposing 8501 to the internet when a reverse proxy is in front |

**Reverse proxy:** Streamlit uses WebSockets. Nginx must forward `Upgrade`/`Connection` headers — a ready-made config is in the [AWS guide](aws_deployment.md#configure-nginx-reverse-proxy--https).

**Secrets:** the app currently needs none. If you add API keys, inject them at runtime (`--env-file`, Docker/Compose secrets, AWS Secrets Manager); never bake them into the image or commit `.env`.

---

## 10. Logging, Updates & CI

### Logs
```bash
docker logs -f --tail 100 telecom-churn
```
Containers should log to **stdout/stderr** (the image already sets `PYTHONUNBUFFERED=1`); ship them with the Docker logging driver, CloudWatch (`awslogs`) or Loki. Cap local log size:
```bash
docker run ... --log-opt max-size=10m --log-opt max-file=3 ...
```

### Updates
```bash
git pull
docker compose up -d --build         # rebuild; unchanged layers come from cache
docker image prune -f                # remove dangling layers
```

### Minimal CI (GitHub Actions)
```yaml
name: docker
on: [push]
jobs:
  build-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t telecom-churn-app:${{ github.sha }} .
      - uses: aquasecurity/trivy-action@0.28.0
        with:
          image-ref: telecom-churn-app:${{ github.sha }}
          severity: HIGH,CRITICAL
          exit-code: "1"
```
(Check the current action versions before copying.)

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
| :--- | :--- | :--- |
| Status stays `unhealthy` | Start-up slower than `start-period`, wrong port, or app crashed | `docker logs telecom-churn`; raise `--start-period` to 30–60 s |
| `port is already allocated` | Host process on 8501 | Map another host port: `-p 8080:8501` |
| `FileNotFoundError: Models/Ada_boost_churn_model.pkl` | File excluded from the build context, or `WORKDIR`/relative path mismatch | Confirm it is not matched by `.dockerignore`; run `docker run --rm telecom-churn-app ls -R /app` |
| `PermissionError` when writing | Running read-only or files owned by root | Keep `COPY --chown=appuser:appgroup`; add a `tmpfs` for any writable path |
| Dozens of "Trying to unpickle estimator … from version 1.5.0 when using version X" lines | `requirements.txt` allows a different scikit-learn patch/minor version | Pin `scikit-learn==1.5.0` and rebuild |
| Container exits with code **137** | Out of memory (OOM kill) | Raise the memory limit; confirm with `.State.OOMKilled` |
| Page loads but stays on "Please wait…" behind a proxy | WebSocket upgrade headers missing | Add `Upgrade`/`Connection` headers (see AWS guide Nginx block) |
| Build re-installs everything every time | `requirements.txt` changed or was copied after the code | Keep `COPY requirements.txt` + `pip install` before `COPY app.py …` |

---

*Authored for the Telecom Customer Churn Prediction Machine Learning Project.*
