# 🐳 Production Deployment Guide: Docker & Containerization

This guide provides end-to-end instructions for building, running, and managing the Telecom Customer Churn Prediction application as a secure, containerized workload using Docker.

---

## 📑 Table of Contents
1. [Architecture & Security Overview](#1-architecture--security-overview)
2. [Prerequisites](#2-prerequisites)
3. [Building the Optimized Docker Image](#3-building-the-optimized-docker-image)
4. [Running the Container Locally](#4-running-the-container-locally)
5. [Docker Compose Multi-Container Orchestration](#5-docker-compose-multi-container-orchestration)
6. [Container Health Checks & Inspection](#6-container-health-checks--inspection)
7. [Production Hardening & Resource Limits](#7-production-hardening--resource-limits)
8. [Troubleshooting Common Container Issues](#8-troubleshooting-common-container-issues)

---

## 1. Architecture & Security Overview

The project’s [`Dockerfile`](../Dockerfile) adheres to cloud-native, enterprise-grade container standards:
- **Base Image:** Official Debian Bookworm slim runtime (`python:3.11-slim-bookworm`).
- **Least Privilege (Non-Root User):** Runs as unprivileged system user `appuser` (UID 10001, GID 10001) to mitigate container breakout vulnerabilities.
- **Layer Caching:** Copies `requirements.txt` before code files to cache dependency installation.
- **Minimal Attack Surface:** Eliminates package manager caches (`rm -rf /var/lib/apt/lists/*`) and Python bytecode (`PYTHONDONTWRITEBYTECODE=1`).
- **Automated Health Probe:** Periodically checks Streamlit's internal `/_stcore/health` endpoint.

---

## 2. Prerequisites

- **Docker Engine:** Version 20.10 or later
- **Docker Compose:** Version 2.0 or later (optional, for compose workflow)
- Verify installation:
  ```bash
  docker --version
  docker compose version
  ```

---

## 3. Building the Optimized Docker Image

Run the build command from the repository root:

```bash
docker build -t telecom-churn-app:v1.0.0 -t telecom-churn-app:latest .
```

### Build Arguments & Inspection
- Verify image creation and size:
  ```bash
  docker images | grep telecom-churn-app
  ```
- **Expected Image Footprint:** ~300 MB (thanks to slim base and excluding training notebooks via [`.dockerignore`](../.dockerignore)).

---

## 4. Running the Container Locally

### Basic Execution
```bash
docker run -d \
  --name telecom-churn \
  -p 8501:8501 \
  --restart unless-stopped \
  telecom-churn-app:latest
```

### Running with Custom Environment Variables
You can override configuration using `.env` or explicit `-e` flags:

```bash
docker run -d \
  --name telecom-churn \
  -p 8501:8501 \
  --env-file .env.example \
  -e LOG_LEVEL=DEBUG \
  telecom-churn-app:latest
```

### Access the Application
Open your browser and navigate to:
```text
http://localhost:8501
```

---

## 5. Docker Compose Multi-Container Orchestration

For reproducible local or server deployments, use Docker Compose. Create a `docker-compose.yml` file:

```yaml
version: "3.8"

services:
  churn-service:
    build:
      context: .
      dockerfile: Dockerfile
    image: telecom-churn-app:latest
    container_name: telecom-churn-service
    restart: unless-stopped
    ports:
      - "8501:8501"
    environment:
      - APP_ENV=production
      - LOG_LEVEL=INFO
      - SERVER_HOST=0.0.0.0
      - SERVER_PORT=8501
      - DATA_PATH=Data/Customer-Churn.csv
      - MODEL_PATH=Models/Ada_boost_churn_model.pkl
    healthcheck:
      test: ["CMD", "curl", "--fail", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 15s
    deploy:
      resources:
        limits:
          cpus: "1.5"
          memory: 1024M
        reservations:
          cpus: "0.5"
          memory: 512M
```

### Compose Commands:
```bash
# Start service in detached background mode
docker compose up -d

# View live streaming container logs
docker compose logs -f churn-service

# Stop and remove containers
docker compose down
```

---

## 6. Container Health Checks & Inspection

The container includes a native health check polling `/_stcore/health`:

### Check Real-Time Health Status:
```bash
docker inspect --format='{{json .State.Health.Status}}' telecom-churn
# Output: "healthy"
```

### View Detailed Health Logs:
```bash
docker inspect --format='{{range .State.Health.Log}}{{.Output}}{{end}}' telecom-churn
```

### View Live Container Resource Utilization:
```bash
docker stats telecom-churn
```

---

## 7. Production Hardening & Resource Limits

When running in production outside Kubernetes, apply hard limits to prevent runaway memory leaks:

```bash
docker run -d \
  --name telecom-churn-prod \
  -p 8501:8501 \
  --memory="1g" \
  --memory-swap="1g" \
  --cpus="1.5" \
  --pids-limit 100 \
  --security-opt="no-new-privileges:true" \
  --cap-drop=ALL \
  --restart always \
  telecom-churn-app:latest
```

### Security Flags Explained:
- `--security-opt="no-new-privileges:true"`: Disallows child processes from gaining elevated privileges (prevents `setuid` exploits).
- `--cap-drop=ALL`: Drops all default Linux root capabilities.
- `--pids-limit 100`: Prevents fork-bomb denial-of-service attacks.

---

## 8. Troubleshooting Common Container Issues

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **Container status `unhealthy`** | App startup exceeded 15s or port mismatch | Increase `--start-period=30s` in Dockerfile or check logs with `docker logs telecom-churn`. |
| **Port 8501 already in use** | Host machine has a local process on 8501 | Map to a different host port: `-p 8080:8501`. Access at `http://localhost:8080`. |
| **Model file missing error** | `Models/` folder was excluded or empty | Ensure `Models/Ada_boost_churn_model.pkl` is committed and not ignored in `.dockerignore`. |
| **Permission Denied on files** | Files copied with root ownership | Ensure Dockerfile has `COPY --chown=appuser:appgroup . .` before `USER appuser`. |

---

*Authored for the Telecom Customer Churn Prediction Machine Learning Project.*
