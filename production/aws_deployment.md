# 🟧 Production Deployment Guide: Amazon Web Services (AWS)

This guide provides complete, production-grade instructions for deploying the Telecom Customer Churn Prediction container to **Amazon Web Services (AWS)** using three industry-standard architectural patterns:

1. **Option A: AWS EC2 with Docker & Nginx Reverse Proxy** (Direct Control / Cost-Effective)
2. **Option B: AWS ECS on AWS Fargate** (Enterprise-Grade Serverless Container Orchestration)
3. **Option C: AWS App Runner** (Zero-Ops Fully Managed Container Service)

---

## 📑 Table of Contents
1. [Architecture Comparison](#1-architecture-comparison)
2. [Prerequisites & AWS CLI Setup](#2-prerequisites--aws-cli-setup)
3. [Building & Pushing Image to Amazon ECR](#3-building--pushing-image-to-amazon-ecr)
4. [Option A: Deploying on Amazon EC2](#4-option-a-deploying-on-amazon-ec2)
   - [Launch EC2 Instance & Security Groups](#launch-ec2-instance--security-groups)
   - [Install Docker & Run App](#install-docker--run-app)
   - [Configure Nginx Reverse Proxy & SSL (HTTPS)](#configure-nginx-reverse-proxy--ssl-https)
   - [Systemd Service Daemon Setup](#systemd-service-daemon-setup)
5. [Option B: Deploying on AWS ECS Fargate](#5-option-b-deploying-on-aws-ecs-fargate)
   - [Create ECS Cluster](#create-ecs-cluster)
   - [Define ECS Task Definition](#define-ecs-task-definition)
   - [Configure Application Load Balancer (ALB)](#configure-application-load-balancer-alb)
   - [Create ECS Service & Auto-Scaling](#create-ecs-service--auto-scaling)
6. [Option C: Deploying on AWS App Runner](#6-option-c-deploying-on-aws-app-runner)
7. [IAM Least Privilege & Production Checklist](#7-iam-least-privilege--production-checklist)
8. [Cost Optimization Strategies](#8-cost-optimization-strategies)

---

## 1. Architecture Comparison

| Feature | AWS EC2 (Virtual Machine) | AWS ECS Fargate (Serverless) | AWS App Runner (PaaS) |
| :--- | :--- | :--- | :--- |
| **Best For** | Single VM, developer control, budget-conscious | Enterprise scale, zero server maintenance, high availability | Rapid deployment, zero infrastructure config |
| **Server Management** | Customer manages OS & patches | AWS manages underlying EC2 instances | Completely serverless |
| **Auto-Scaling** | Auto Scaling Groups (ASG) | Native ECS Task Auto-Scaling | Automated concurrency scaling |
| **SSL/TLS Setup** | Let's Encrypt Certbot via Nginx | AWS Certificate Manager (ACM) via ALB | Automatic free managed SSL |
| **Estimated Monthly Cost** | ~$15 - $30 (t3.small / t3.medium) | ~$25 - $50 (based on task vCPU/RAM) | Pay-per-request / memory (~$15 - $35) |

---

## 2. Prerequisites & AWS CLI Setup

1. **AWS Account:** With administrative permissions.
2. **AWS CLI v2 Installed:** [Installation Guide](https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html).
3. **Docker Installed & Running locally.**
4. Authenticate your CLI session:
   ```bash
   aws configure
   # Enter AWS Access Key ID, Secret Access Key, and Default Region (e.g., us-east-1)
   ```

---

## 3. Building & Pushing Image to Amazon ECR

Amazon Elastic Container Registry (ECR) stores our secure container image for AWS services.

### Step 1: Create an ECR Private Repository
```bash
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO_NAME="telecom-churn-prediction"

aws ecr create-repository \
    --repository-name $REPO_NAME \
    --image-scanning-configuration scanOnPush=true \
    --region $AWS_REGION
```

### Step 2: Authenticate Local Docker to ECR
```bash
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
```

### Step 3: Build, Tag, and Push
```bash
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}:latest"

# Build locally
docker build -t telecom-churn-app:latest .

# Tag for ECR
docker tag telecom-churn-app:latest $ECR_URI

# Push to Amazon ECR
docker push $ECR_URI
```

---

## 4. Option A: Deploying on Amazon EC2

### Launch EC2 Instance & Security Groups
1. **AMI:** Amazon Linux 2023 or Ubuntu 22.04 LTS.
2. **Instance Type:** `t3.medium` (2 vCPU, 4 GB RAM) or `t3.small` (2 vCPU, 2 GB RAM).
3. **Security Group Inbound Rules:**
   - `SSH` (Port 22) $\to$ Your IP only.
   - `HTTP` (Port 80) $\to$ `0.0.0.0/0`.
   - `HTTPS` (Port 443) $\to$ `0.0.0.0/0`.
   - `Custom TCP` (Port 8501) $\to$ `0.0.0.0/0` (for testing before Nginx).

### Install Docker & Run App
SSH into your EC2 instance:
```bash
ssh -i your-key.pem ec2-user@<EC2-PUBLIC-IP>
```

Install Docker on Amazon Linux 2023:
```bash
sudo dnf update -y
sudo dnf install -y docker git
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
newgrp docker
```

Login to ECR from EC2 and run the container:
```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ECR_URI>

docker run -d \
  --name telecom-churn \
  -p 8501:8501 \
  --restart always \
  <ECR_URI>
```

### Configure Nginx Reverse Proxy & SSL (HTTPS)
Install Nginx and Certbot:
```bash
sudo dnf install -y nginx
```

Create `/etc/nginx/conf.d/churn.conf`:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Start Nginx and provision free SSL:
```bash
sudo systemctl enable --now nginx
sudo dnf install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Systemd Service Daemon Setup
Ensure the container runs automatically across system reboots. Create `/etc/systemd/system/telecom-churn.service`:
```ini
[Unit]
Description=Telecom Churn Prediction Container
After=docker.service
Requires=docker.service

[Service]
TimeoutStartSec=0
Restart=always
ExecStartPre=-/usr/bin/docker stop telecom-churn
ExecStartPre=-/usr/bin/docker rm telecom-churn
ExecStart=/usr/bin/docker run --name telecom-churn -p 8501:8501 <ECR_URI>
ExecStop=/usr/bin/docker stop telecom-churn

[Install]
WantedBy=multi-user.target
```

Enable the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now telecom-churn.service
```

---

## 5. Option B: Deploying on AWS ECS Fargate

AWS Elastic Container Service (ECS) with Fargate provides a serverless, highly available production architecture.

### Create ECS Cluster
```bash
aws ecs create-cluster --cluster-name churn-production-cluster
```

### Define ECS Task Definition
Save as `task-definition.json`:
```json
{
  "family": "telecom-churn-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::<ACCOUNT_ID>:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "telecom-churn-container",
      "image": "<ECR_URI>",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8501,
          "hostPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "APP_ENV", "value": "production"},
        {"name": "LOG_LEVEL", "value": "INFO"}
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8501/_stcore/health || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 15
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

Register task definition:
```bash
# Create CloudWatch log group first
aws logs create-log-group --log-group-name /ecs/telecom-churn

# Register task
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Configure Application Load Balancer (ALB)
1. **Target Group:** Target type `IP`, Protocol `HTTP`, Port `8501`.
2. **Health Check Path:** `/_stcore/health`.
   - Healthy threshold: `2`
   - Unhealthy threshold: `3`
   - Timeout: `5s`
   - Interval: `30s`
3. **ALB Listener:** Port `80` (redirect to `443` HTTPS using ACM SSL certificate).

### Create ECS Service & Auto-Scaling
Deploy 2 container replicas across multiple Availability Zones:
```bash
aws ecs create-service \
    --cluster churn-production-cluster \
    --service-name churn-service \
    --task-definition telecom-churn-task \
    --desired-count 2 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-12345,subnet-67890],securityGroups=[sg-churn-ecs],assignPublicIp=ENABLED}" \
    --load-balancers targetGroupArn=arn:aws:elasticloadbalancing:<ARN>,containerName=telecom-churn-container,containerPort=8501
```

---

## 6. Option C: Deploying on AWS App Runner

AWS App Runner is the fastest serverless container deployment mechanism for web applications.

### Quick Deployment via AWS Console
1. Open **AWS App Runner Console** $\to$ **"Create service"**.
2. **Source:** Container registry $\to$ Select **Amazon ECR**.
3. **Image URI:** Select `<ECR_URI>`.
4. **Deployment Trigger:** Select **Automatic** (auto-deploys on every `docker push`).
5. **Configuration:**
   - **Port:** `8501`.
   - **vCPU:** `1 vCPU`.
   - **Memory:** `2 GB`.
   - **Environment Variables:**
     - `APP_ENV = production`
     - `LOG_LEVEL = INFO`
6. Click **"Create & Deploy"**.
7. AWS App Runner automatically provisions an endpoint with free HTTPS (e.g., `https://random-id.us-east-1.awsapprunner.com`).

---

## 7. IAM Least Privilege & Production Checklist

1. **ECS Task Execution Role:** Grant only `AmazonECSTaskExecutionRolePolicy` (grants ECR pull and CloudWatch logging permissions).
2. **Secret Management:** Store sensitive credentials in **AWS Secrets Manager** and reference them via ECS `secrets` attributes rather than plain-text environment variables.
3. **VPC Isolation:** Place ECS tasks inside **Private Subnets** with NAT Gateway for outbound traffic, accessible externally only through the Application Load Balancer.

---

## 8. Cost Optimization Strategies

- **Use AWS Fargate Spot:** For non-critical environments, Fargate Spot reduces container compute costs by **up to 70%**.
- **EC2 Savings Plans:** Committing to 1-year EC2 compute reduces costs by 40%.
- **EC2 Auto-Shutdown for Dev/Test:** Schedule an AWS Lambda cron job to shut down development instances outside business hours.

---

*Authored for the Telecom Customer Churn Prediction Machine Learning Project.*
