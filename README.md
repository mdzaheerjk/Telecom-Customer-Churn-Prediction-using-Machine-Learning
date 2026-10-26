# 📡 Telecom Customer Churn Prediction & Retention Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.0-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20ECS%20Fargate-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, industry-grade machine learning system designed to predict customer churn in the telecommunications sector. The platform identifies at-risk subscribers in real time, diagnoses root-cause churn drivers, and prescribes proactive retention interventions through a secure, containerized web application.

---

## 📌 Project Preview

<p align="center">
  <img src="https://d14omfvx1qlabb.cloudfront.net/krishnaik.in/media/project_banners/-gemini_generated_image_gwurc4gwurc4gwur-cab256ad19681fc06718f98c6974220_r8f1Qgp.jpg"
       alt="Project Preview"
       width="700" />
</p>

---

## 🏗️ System Architecture

<p align="center">
  <img src="https://career-platform-may-2026.s3.ap-south-1.amazonaws.com/krishnaik.in/media/project_architecture_diagrams/Gemini_Generated_Image_4hi37h4hi37h4hi3.jpg"
       alt="System Architecture"
       width="700" />
</p>

---

## 🚀 Key Highlights & Capabilities

- **Real-Time Machine Learning Scoring:** Sub-second churn probability evaluation powered by an AdaBoost classifier optimized with cost-sensitive sample weighting.
- **3-Tier Risk Segmentation:** Automatically categorizes customers into actionable tiers:
  - 🟢 **Low Risk (< 35%):** Healthy accounts receiving standard lifecycle engagements.
  - 🟡 **Moderate Risk (35% – 60%):** Emerging friction points triggering automated check-ins.
  - 🔴 **High Risk (≥ 60%):** Critical hazard requiring proactive account-manager intervention.
- **Explainable Driver Attribution:** Surfaces per-customer vulnerability signals (e.g., month-to-month contracts, electronic check payments, unbundled fiber optic service).
- **Proactive Retention Action Items:** Recommends targeted, high-impact business retention strategies based on identified risk drivers.
- **Hardened Cloud-Native Container:** Dockerfile featuring a dedicated non-root user (`appuser`, UID 10001), layer caching, and native healthcheck probing (`/_stcore/health`).
- **Production Observability:** Built-in Python logging with timestamps, severity levels, and defensive error boundaries preventing unhandled application crashes.

---

## 📂 Project Structure

```text
Telecom-Customer-Churn-Prediction-using-Machine-Learning/
├── Data/
│   └── Customer-Churn.csv          # Telecommunications customer dataset (7,043 rows)
├── Interview Questions/
│   └── Interview_Questions.md      # Top 50 Production-Grade AI/ML Interview Questions
├── Models/
│   ├── Ada_boost_churn_model.pkl   # Production-deployed AdaBoost classifier
│   ├── Best_optuna_churn_model.pkl # Optuna-tuned XGBoost model checkpoint
│   └── best_xgboost_churn_model.pkl# Baseline XGBoost model checkpoint
├── Notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis & statistical visualizations
│   ├── MODEL.ipynb                 # Model training, resampling bake-offs & Optuna sweeps
│   └── catboost_info/              # CatBoost experiment telemetry
├── presentation/
│   └── Telecom_Customer_Churn_Prediction.pptx # 12-slide executive presentation deck
├── production/
│   ├── aws_deployment.md           # AWS deployment (EC2 Nginx SSL, ECS Fargate, App Runner)
│   ├── docker_deployment.md        # Docker containerization, security & compose orchestration
│   └── streamlit_cloud_deployment.md# Streamlit Community Cloud GitOps deployment guide
├── .dockerignore                   # Docker build context exclusions
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git tracking rules for Python & ML artifacts
├── Dockerfile                      # Production-ready, secure, non-root Docker build
├── LICENSE                         # MIT License
├── README.md                       # Comprehensive project documentation
├── app.py                          # Streamlit web application & serving engine
└── requirements.txt                # Production dependencies with pinned versions
```

---

## 🔬 Machine Learning Pipeline & Benchmark

### 1. Data Cleaning & Pipeline Hygiene
- **Handling Blank Strings:** `TotalCharges` contained 11 blank records (`' '`) for brand-new customers with `tenure = 0`. Coerced to numeric via `pd.to_numeric(..., errors='coerce')` and cleaned without data corruption.
- **Discretization:** `tenure` transformed into 6 ordinal lifecycle cohorts: `1-12`, `13-24`, `25-36`, `37-48`, `49-60`, and `61-72` months.
- **Full-Rank Encoding:** Categoricals encoded using `pd.get_dummies(drop_first=True)` to prevent the Dummy Variable Trap, yielding exactly 34 standardized features.
- **Feature Normalization:** Standardized across all continuous and dummy dimensions using `StandardScaler`.

### 2. Comparative Model Benchmark

| Model | Imbalance Strategy | Minority F1 | Recall | Precision | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Decision Tree** | Default Unweighted | 0.51 | 0.50 | 0.52 | Baseline |
| **Random Forest** | `class_weight='balanced'` | 0.59 | 0.65 | 0.54 | Benchmark |
| **XGBoost (Baseline)** | Default Unweighted | 0.58 | 0.52 | 0.66 | Benchmark |
| **XGBoost** | SMOTEENN Resampling | 0.61 | 0.79 | 0.50 | Benchmark |
| **XGBoost (Optuna Tuned)**| 100 Trials, Stratified CV | 0.63 | 0.79 | 0.53 | Candidate |
| **AdaBoost Classifier** | **Cost-Sensitive Weighting** | **0.64** | **0.82** | **0.53** | **Production Deployed 🏆** |

> **Why AdaBoost was chosen for deployment:**  
> Weighted AdaBoost achieved the highest minority recall (0.82) and F1-score (0.64) while running on a single lightweight scikit-learn dependency, eliminating the heavy C++ runtimes and GPU requirements of gradient boosting engines while delivering sub-millisecond inference latencies.

---

## ⚙️ Configuration & Environment Variables

Configure application parameters using environment variables or a `.env` file at the project root:

```bash
# Copy example configuration template
cp .env.example .env
```

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | Deployment environment (`production` or `development`) |
| `SERVER_HOST` | `0.0.0.0` | Network binding host address |
| `SERVER_PORT` | `8501` | Application port |
| `LOG_LEVEL` | `INFO` | Logging severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DATA_PATH` | `Data/Customer-Churn.csv` | Relative or absolute path to reference dataset |
| `MODEL_PATH` | `Models/Ada_boost_churn_model.pkl` | Path to serialized production model artifact |

---

## 🛠️ Local Development Setup

### Prerequisites
- Python 3.10 or 3.11
- `git`
- Virtual environment tool (`venv` or `conda`)

```bash
# 1. Clone repository
git clone https://github.com/mdzaheerjk/Telecom-Customer-Churn-Prediction-using-Machine-Learning.git
cd Telecom-Customer-Churn-Prediction-using-Machine-Learning

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 3. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Launch web application
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

---

## 🐳 Docker Deployment

The repository includes a hardened, multi-stage `Dockerfile` adhering to the **CIS Docker Benchmark**:
- Runs under unprivileged non-root user `appuser` (UID 10001).
- Eliminates Python bytecode (`PYTHONDONTWRITEBYTECODE=1`).
- Implements continuous container healthcheck endpoint probing (`/_stcore/health`).
- Leverages layer caching for 2-second rebuilds.

```bash
# Build Docker image
docker build -t telecom-churn-app:latest .

# Run container
docker run -d \
  --name telecom-churn \
  -p 8501:8501 \
  --restart unless-stopped \
  telecom-churn-app:latest

# Verify container health status
docker inspect --format='{{json .State.Health.Status}}' telecom-churn
```

---

## 🚀 Multi-Cloud Production Runbooks

Comprehensive, step-by-step production deployment guides are available in the [`production/`](production/) directory:

| Hosting Environment | Runbook Guide | Architecture & Features |
| :--- | :--- | :--- |
| **Docker & Compose** | [**Docker Deployment Guide**](production/docker_deployment.md) | Multi-stage caching, security hardening, resource caps (`--cpus`, `--memory`), and Docker Compose orchestration |
| **Streamlit Cloud** | [**Streamlit Cloud Guide**](production/streamlit_cloud_deployment.md) | Zero-ops GitOps CI/CD from GitHub, encrypted secrets (`secrets.toml`), memory management, and live logs |
| **Amazon Web Services** | [**AWS Deployment Guide**](production/aws_deployment.md) | Complete setup for **EC2 (with Nginx & Let's Encrypt SSL)**, **ECS Fargate (serverless containers)**, and **AWS App Runner** |

---

## 📊 Executive Presentation Deck

An executive-ready, 12-slide PowerPoint presentation is located in the [`presentation/`](presentation/) folder:

- 📄 **File:** [`presentation/Telecom_Customer_Churn_Prediction.pptx`](presentation/Telecom_Customer_Churn_Prediction.pptx)
- **Deck Contents:**
  - Executive Problem Framing & Churn Economics (CAC vs. CLV)
  - Key Exploratory Data Analysis & Behavioral Drivers
  - Preprocessing & Class Imbalance Resampling Bake-off
  - Algorithmic Benchmark & Optuna Hyperparameter Optimization
  - Production Web Application & Risk Stratification UX
  - Enterprise Docker Containerization & Cloud Deployment Architecture
  - Business ROI & 6-Month Strategic Engineering Roadmap

---

## 📚 Interview Preparation & Self-Study Guide

A dedicated, comprehensive technical interview guide featuring **50 production-grade interview questions and answers** is located in [`Interview Questions/Interview_Questions.md`](Interview%20Questions/Interview_Questions.md):

- 📈 **5-Part Progression Roadmap:**
  - **Part 1 (Q1–Q12):** Data Pipeline, EDA & Hygiene *(Beginner to Intermediate)*
  - **Part 2 (Q13–Q22):** Feature Engineering & Class Imbalance Strategies *(Intermediate to Advanced)*
  - **Part 3 (Q23–Q34):** Algorithms, Mathematical Intuition & Hyperparameter Tuning *(Advanced)*
  - **Part 4 (Q35–Q42):** Metrics, Asymmetric Cost Thresholds & Explainability *(Advanced to Critical Thinking)*
  - **Part 5 (Q43–Q50):** Production Engineering, Deployment & MLOps *(Critical Thinking / Senior)*
- 💡 **Active Recall Format:** Interactive collapsible answers (`<details><summary>`), mathematical derivations, failure-mode scenarios, and real-world AI/ML interview insights.

👉 [**Open the 50-Question Interactive Interview Guide**](Interview%20Questions/Interview_Questions.md)

---

## 📄 License

This project is licensed under the terms described in the [LICENSE](LICENSE) file.
