# 🟧 Production Deployment Guide: Amazon Web Services (AWS)

[![AWS](https://img.shields.io/badge/AWS-EC2_%7C_ECS_Fargate_%7C_ECS_Express-FF9900?style=flat-square&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![Docker](https://img.shields.io/badge/Image-ECR-2496ED?style=flat-square&logo=docker&logoColor=white)](docker_deployment.md)
[![Last verified](https://img.shields.io/badge/Verified-28_Oct_2026-blue?style=flat-square)](#)

How to run the Telecom Customer Churn Prediction container on AWS using three patterns, from cheapest to most managed:

1. **Option A — EC2 + Docker + Nginx** (direct control, lowest cost)
2. **Option B — ECS on Fargate + Application Load Balancer** (production-grade, you design the network)
3. **Option C — Amazon ECS Express Mode** (managed path: image + two IAM roles → HTTPS service)

> ⚠️ **AWS App Runner is no longer an option for new accounts.** AWS closed it to new customers on **30 April 2026** and recommends ECS Express Mode as the replacement; existing customers can keep using it ([AWS notice](https://docs.aws.amazon.com/apprunner/latest/dg/apprunner-availability-change.html)). It is covered only as a [legacy note](#7-legacy-aws-app-runner-existing-accounts-only).

---

## 📑 Table of Contents
1. [Choosing an Architecture](#1-choosing-an-architecture)
2. [Prerequisites & Account Hygiene](#2-prerequisites--account-hygiene)
3. [Build & Push the Image to ECR](#3-build--push-the-image-to-ecr)
4. [Option A: EC2 + Docker + Nginx](#4-option-a-ec2--docker--nginx)
5. [Option B: ECS on Fargate + ALB](#5-option-b-ecs-on-fargate--alb)
6. [Option C: ECS Express Mode](#6-option-c-ecs-express-mode)
7. [Legacy: AWS App Runner](#7-legacy-aws-app-runner-existing-accounts-only)
8. [Security Checklist](#8-security-checklist)
9. [Monitoring & Alarms](#9-monitoring--alarms)
10. [Cost Control & Teardown](#10-cost-control--teardown)

---

## 1. Choosing an Architecture

| | **A. EC2 + Nginx** | **B. ECS Fargate + ALB** | **C. ECS Express Mode** |
| :--- | :--- | :--- | :--- |
| **Best for** | Personal demo, tight budget | Customer-facing, HA, custom networking | Fastest managed deploy |
| **You manage** | OS patches, Docker, TLS renewal | Task definition, ALB, VPC, scaling policy | Image + two IAM roles |
| **HTTPS** | Let's Encrypt via Certbot | ACM certificate on the ALB | Provisioned automatically |
| **Scaling** | Manual / Auto Scaling Group | Target-tracking on the ECS service | Built-in auto scaling |
| **Streamlit sessions** | Single instance ✅ | Enable **ALB stickiness** with > 1 task | Keep 1 task unless stickiness is confirmed |
| **Indicative cost\*** | ≈ $15–30 / month (`t3.small`–`t3.medium`) | ≈ $35 for 1 task + ALB, ≈ $55 for 2 tasks | Same resources as B, no Express surcharge |

\* Rough on-demand figures for `us-east-1`, always-on, excluding data transfer, NAT and storage. Fargate example: 0.5 vCPU / 1 GB ≈ $18 per task, ALB ≈ $16 + usage. Prices change and vary by region — confirm with the [AWS Pricing Calculator](https://calculator.aws/) before committing. Pick the region closest to your users (for example `ap-south-1` for India).

> **Why sticky sessions?** Streamlit keeps each browser session's state in the server process and talks over a WebSocket. If a reconnect lands on a different task, the session restarts. One task avoids the problem; several tasks need stickiness.

---

## 2. Prerequisites & Account Hygiene

1. **AWS account** with MFA on the root user (use root only for account-level tasks).
2. **Daily access via IAM Identity Center (SSO)** instead of long-lived access keys:
   ```bash
   aws configure sso            # follow the prompts
   aws sso login --profile <profile>
   ```
   (`aws configure` with static keys works but is a leak risk — avoid it for anything beyond a throwaway lab.)
3. **A budget alarm first:** AWS Budgets → monthly budget with an email alert at 80% of your limit.
4. **Tools:** AWS CLI v2, Docker, `git`.
5. **Image ready locally:** see the [Docker guide](docker_deployment.md). Pin `scikit-learn==1.5.0` so the pickled model loads cleanly.

---

## 3. Build & Push the Image to ECR

```bash
AWS_REGION="us-east-1"                       # choose your region
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="telecom-churn-prediction"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_TAG=$(git rev-parse --short HEAD)      # immutable, traceable tag
ECR_URI="${REGISTRY}/${REPO_NAME}:${IMAGE_TAG}"
```

### Step 1 — Create the repository (immutable tags + scan on push)
```bash
aws ecr create-repository \
  --repository-name $REPO_NAME \
  --image-tag-mutability IMMUTABLE \
  --image-scanning-configuration scanOnPush=true \
  --region $AWS_REGION
```

### Step 2 — Keep only the last 10 images
```bash
aws ecr put-lifecycle-policy --repository-name $REPO_NAME --region $AWS_REGION \
  --lifecycle-policy-text '{"rules":[{"rulePriority":1,"description":"Keep last 10 images",
    "selection":{"tagStatus":"any","countType":"imageCountMoreThan","countNumber":10},
    "action":{"type":"expire"}}]}'
```

### Step 3 — Authenticate Docker to the **registry host**
```bash
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $REGISTRY
```

### Step 4 — Build, tag, push
```bash
# Fargate/EC2 x86_64 targets. On Apple Silicon, cross-build explicitly:
docker build --platform linux/amd64 -t $ECR_URI .
docker push $ECR_URI
```
> Want a floating `latest` tag too? Create the repository with `--image-tag-mutability MUTABLE` instead. For production, deploy by SHA so every release is reproducible and roll-backs are one tag away.

---

## 4. Option A: EC2 + Docker + Nginx

### Launch the instance
| Setting | Value |
| :--- | :--- |
| AMI | Amazon Linux 2023 (or Ubuntu 24.04 LTS) |
| Type | `t3.small` (2 vCPU, 2 GB) — `t3.medium` for headroom |
| IAM instance profile | `AmazonEC2ContainerRegistryReadOnly` (pull from ECR) + `AmazonSSMManagedInstanceCore` (shell via Session Manager) |
| Metadata | **IMDSv2 required** |
| Storage | 20 GB gp3, encrypted |
| Elastic IP | Allocate and attach (stable DNS target) |

**Security group (inbound):** `80` and `443` from `0.0.0.0/0` only. With Session Manager you do **not** need SSH (`22`). Do **not** open `8501` publicly — Nginx is the only front door.

### Install Docker and run the container
Connect with *Session Manager* (or SSH if you opened 22 to your IP only), then:

```bash
sudo dnf update -y
sudo dnf install -y docker
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user && newgrp docker

# Variables as in section 3
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $REGISTRY

docker run -d \
  --name telecom-churn \
  --restart unless-stopped \
  -p 127.0.0.1:8501:8501 \
  --memory=1g --pids-limit=100 \
  --cap-drop=ALL --security-opt no-new-privileges:true \
  $ECR_URI
```
Binding to `127.0.0.1` means only Nginx on the same host can reach the app. Because `docker.service` is enabled and the restart policy is `unless-stopped`, the container comes back after a reboot — a separate systemd unit is unnecessary.

### Configure Nginx reverse proxy & HTTPS
```bash
sudo dnf install -y nginx
```

`/etc/nginx/conf.d/churn.conf`:
```nginx
map $http_upgrade $connection_upgrade {
    default upgrade;
    ''      close;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade           $http_upgrade;
        proxy_set_header   Connection        $connection_upgrade;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;   # keep idle WebSockets open
        proxy_buffering    off;
    }
}
```

```bash
sudo nginx -t && sudo systemctl enable --now nginx
```

Point your DNS **A record** at the Elastic IP, then get a free certificate. Amazon Linux 2023 does not ship Certbot in its default repositories, so install it in a virtualenv (on Ubuntu use `sudo apt install certbot python3-certbot-nginx` instead):

```bash
sudo dnf install -y python3-pip augeas-libs cronie
sudo python3 -m venv /opt/certbot
sudo /opt/certbot/bin/pip install certbot certbot-nginx
sudo ln -sf /opt/certbot/bin/certbot /usr/bin/certbot
sudo certbot --nginx -d your-domain.com

# Auto-renew twice a day; reload Nginx when a cert changes
sudo systemctl enable --now crond
echo '17 3,15 * * * root /usr/bin/certbot renew --quiet --deploy-hook "systemctl reload nginx"' | sudo tee /etc/cron.d/certbot-renew
```

### Deploy a new version
```bash
NEW_TAG=<git-sha>
docker pull ${REGISTRY}/${REPO_NAME}:${NEW_TAG}
docker stop telecom-churn && docker rm telecom-churn
docker run -d --name telecom-churn --restart unless-stopped -p 127.0.0.1:8501:8501 \
  --memory=1g --pids-limit=100 --cap-drop=ALL --security-opt no-new-privileges:true \
  ${REGISTRY}/${REPO_NAME}:${NEW_TAG}
```
Keep the previous tag handy: rolling back is the same command with the old SHA.

---

## 5. Option B: ECS on Fargate + ALB

### Network layout
```text
Internet ─► ALB (public subnets, 80/443) ─► Fargate tasks (private subnets, 8501 from ALB SG only)
                                                    └─► NAT Gateway *or* VPC endpoints ─► ECR / CloudWatch Logs
```
- **Security groups:** `alb-sg` allows 80/443 from the internet; `task-sg` allows **8501 only from `alb-sg`**.
- **Cheaper portfolio variant:** run tasks in public subnets with `assignPublicIp=ENABLED`, keeping `task-sg` locked to the ALB. This avoids NAT Gateway cost (see §10) at the price of public task IPs.
- **Private-subnet variant:** use a NAT Gateway, or VPC endpoints for `ecr.api`, `ecr.dkr`, `logs` and an S3 gateway endpoint.

### Cluster, log group and execution role
```bash
aws ecs create-cluster --cluster-name churn-production-cluster

aws logs create-log-group --log-group-name /ecs/telecom-churn
aws logs put-retention-policy --log-group-name /ecs/telecom-churn --retention-in-days 14
```
Create the **task execution role** `ecsTaskExecutionRole` with the managed policy `AmazonECSTaskExecutionRolePolicy` (ECR pull + log writes). The app calls no AWS APIs, so it needs **no task role**.

### Task definition (`task-definition.json`)
```json
{
  "family": "telecom-churn-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "runtimePlatform": { "cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX" },
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "telecom-churn-container",
      "image": "<ECR_URI>",
      "essential": true,
      "portMappings": [{ "containerPort": 8501, "protocol": "tcp" }],
      "environment": [
        { "name": "STREAMLIT_BROWSER_GATHER_USAGE_STATS", "value": "false" }
      ],
      "linuxParameters": { "initProcessEnabled": true },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 30
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/telecom-churn",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```
> Using the hardened image without `curl`? Change the command to `["CMD-SHELL", "python -c \"import urllib.request as u; u.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=5)\" || exit 1"]`.

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Application Load Balancer
1. **Target group:** type `IP`, protocol `HTTP`, port `8501`.
2. **Health check:** path `/_stcore/health`, healthy threshold `2`, unhealthy `3`, timeout `5s`, interval `30s`.
3. **Stickiness:** enable *load-balancer generated cookie* (e.g. 1 day) — required once you run more than one task.
4. **Deregistration delay:** `30s` for quicker deployments.
5. **Listeners:** `80` → redirect to `443`; `443` → forward to the target group with an **ACM certificate** for your domain.
6. **Idle timeout:** the default is 60 s. If users see "Connecting…" after leaving a tab idle, raise it (for example to 300 s).

### Create the service
```bash
aws ecs create-service \
  --cluster churn-production-cluster \
  --service-name churn-service \
  --task-definition telecom-churn-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --health-check-grace-period-seconds 60 \
  --deployment-configuration "deploymentCircuitBreaker={enable=true,rollback=true}" \
  --network-configuration "awsvpcConfiguration={subnets=[<PRIVATE_SUBNET_A>,<PRIVATE_SUBNET_B>],securityGroups=[<TASK_SG_ID>],assignPublicIp=DISABLED}" \
  --load-balancers "targetGroupArn=<TARGET_GROUP_ARN>,containerName=telecom-churn-container,containerPort=8501"
```
Use `desired-count 2` across two AZs for high availability (stickiness on). The deployment circuit breaker automatically rolls back a release whose tasks never become healthy.

### Auto scaling (target tracking on CPU)
```bash
aws application-autoscaling register-scalable-target \
  --service-namespace ecs --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/churn-production-cluster/churn-service \
  --min-capacity 1 --max-capacity 4

aws application-autoscaling put-scaling-policy \
  --policy-name cpu-60 --service-namespace ecs \
  --scalable-dimension ecs:service:DesiredCount \
  --resource-id service/churn-production-cluster/churn-service \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 60.0,
    "PredefinedMetricSpecification": {"PredefinedMetricType": "ECSServiceAverageCPUUtilization"},
    "ScaleOutCooldown": 60, "ScaleInCooldown": 120 }'
```

### Release a new version
Register a task-definition revision with the new image tag, then:
```bash
aws ecs update-service --cluster churn-production-cluster --service churn-service \
  --task-definition telecom-churn-task
```
Roll back by pointing the service at the previous revision.

---

## 6. Option C: ECS Express Mode

Express Mode turns *a container image plus two IAM roles* into a running HTTPS service on Fargate — it provisions the load balancer, certificate, logging, networking and scaling in your own account, with no extra charge beyond the resources it creates. It is AWS's recommended landing spot for former App Runner workloads.

### Console path (recommended)
1. **Amazon ECS → Express mode → Create.**
2. **Image:** `<ECR_URI>` from section 3.
3. **Roles:** let the console create/select the **task execution role** and the **infrastructure role**.
4. **Container port:** `8501`.
5. **Health check path:** `/_stcore/health` (override the default if the console offers it).
6. **Size:** 0.5 vCPU / 1 GB is enough for this app.
7. **Create.** AWS returns an HTTPS URL for the service.

### CLI shape
```bash
aws ecs create-express-gateway-service \
  --region $AWS_REGION \
  --primary-container image=$ECR_URI \
  --execution-role-arn arn:aws:iam::${AWS_ACCOUNT_ID}:role/ecsTaskExecutionRole
  # + infrastructure role, container port 8501, health-check path, cpu/memory:
  #   run `aws ecs create-express-gateway-service help` for the current flag names
```

### Notes
- **Sessions:** Streamlit needs session affinity once you scale beyond one task. Keep the service at **one task** unless you have confirmed stickiness; if sessions reset after scale-out, switch to Option B where you control target-group stickiness.
- **Release strategy:** if you need rolling or blue/green deployments with custom controls, create a standard ECS service (Option B).
- Express Mode is new: re-check the health-check path and port in the console after creation.

---

## 7. Legacy: AWS App Runner (existing accounts only)

App Runner stopped accepting **new customers on 30 April 2026**; existing services continue to run and can be updated, but no new features are planned. If you already use it:

- Source: ECR image `<ECR_URI>`, port `8501`.
- Set the **health check to HTTP** `/_stcore/health` (the default TCP check can mark a half-started app healthy).
- Plan a migration to **ECS Express Mode** (Option C); it takes the same image unchanged.

---

## 8. Security Checklist

| Area | Control |
| :--- | :--- |
| **Identity** | Root MFA; SSO/Identity Center for humans; roles (not keys) for workloads; GitHub Actions → AWS via **OIDC** |
| **Least privilege** | Execution role = `AmazonECSTaskExecutionRolePolicy` only; EC2 profile = ECR read + SSM |
| **Network** | Only the load balancer/Nginx is public; `8501` reachable only from the ALB SG or `127.0.0.1`; tasks in private subnets where budget allows |
| **TLS** | HTTPS everywhere (ACM or Let's Encrypt), HTTP → HTTPS redirect, modern TLS policy on the ALB |
| **Image** | Non-root user, scan-on-push, immutable SHA tags, ECR lifecycle policy, base image pinned by digest |
| **Secrets** | The app uses none today. If you add API keys, store them in **Secrets Manager** / SSM Parameter Store and inject via the ECS `secrets` field — never in the image or plain environment variables |
| **Data** | Dataset is a public sample; do not mount real customer data without encryption, access logging and a retention policy |
| **Edge protection** | Optional AWS WAF on the ALB (rate-based rule) for public deployments |
| **Hygiene** | Budget alarm, CloudTrail enabled, untag/clean unused resources |

---

## 9. Monitoring & Alarms

- **Logs:** `awslogs` → `/ecs/telecom-churn` (14-day retention). On EC2: `docker logs`, or install the CloudWatch agent.
- **Enable Container Insights** on the ECS cluster for CPU/memory/network per task.
- **Alarms (SNS email):**

| Metric | Threshold | Why |
| :--- | :--- | :--- |
| ALB `UnHealthyHostCount` | ≥ 1 for 5 min | Task failing health checks |
| ALB `HTTPCode_Target_5XX_Count` | > 5 in 5 min | App errors |
| ECS `MemoryUtilization` | > 80% for 10 min | Approaching the 1 GB task limit |
| ECS `CPUUtilization` | > 80% for 10 min | Scale out |

- **Model monitoring** (drift, score distributions) is separate from infrastructure monitoring — see the interview guide, Q47.

---

## 10. Cost Control & Teardown

### Savings levers
- **Stop what you don't use:** set ECS `desired-count` to `0` or stop the EC2 instance outside demo hours.
- **Fargate Spot** for non-critical environments: up to ~70% cheaper, but tasks can be interrupted with short notice.
- **Graviton/ARM64** tasks cost less than x86 — rebuild the image with `--platform linux/arm64` and set `cpuArchitecture: ARM64`.
- **Savings Plans / Reserved capacity** only for steady, always-on production (discount depends on term and commitment — check the current rates).
- **Beware the NAT Gateway:** its hourly charge plus data processing is often the largest line item on a small project. Prefer VPC endpoints, or public subnets with locked-down security groups for demos.
- Public IPv4 addresses (Elastic IPs, ALB nodes) are billed hourly too.

### Teardown (to stop all charges)
```bash
# ECS
aws ecs update-service --cluster churn-production-cluster --service churn-service --desired-count 0
aws ecs delete-service --cluster churn-production-cluster --service churn-service --force
aws ecs delete-cluster --cluster churn-production-cluster

# Load balancer, target group, ACM certificate, security groups  (console or CLI)
# NAT Gateway + Elastic IPs, EC2 instance + volumes
# Logs and registry
aws logs delete-log-group --log-group-name /ecs/telecom-churn
aws ecr delete-repository --repository-name telecom-churn-prediction --force
```
Check **Cost Explorer** the next day to confirm spend has dropped to zero.

---

*Authored for the Telecom Customer Churn Prediction Machine Learning Project.*
