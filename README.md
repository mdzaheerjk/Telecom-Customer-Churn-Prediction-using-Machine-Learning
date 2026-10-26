# Telecom Customer Churn Prediction

Predict telecommunications customer churn in real-time using production-grade machine learning. This system identifies high-risk customers, highlights key churn drivers, and provides proactive retention recommendations through an interactive web application.

---

## 📌 Project Preview

<p align="center">
  <img src="https://d14omfvx1qlabb.cloudfront.net/krishnaik.in/media/project_banners/-gemini_generated_image_gwurc4gwurc4gwur-cab256ad19681fc06718f98c6974220_r8f1Qgp.jpg"
       alt="Project Preview"
       width="650" />
</p>

---

## 🏗️ System Architecture

<p align="center">
  <img src="https://career-platform-may-2026.s3.ap-south-1.amazonaws.com/krishnaik.in/media/project_architecture_diagrams/Gemini_Generated_Image_4hi37h4hi37h4hi3.jpg"
       alt="System Architecture"
       width="650" />
</p>

---

## 🚀 Key Features

- **Real-Time Machine Learning Inference**: Sub-second churn prediction powered by an AdaBoost model trained with class-weight balancing.
- **Risk Tier Classification**: Customers are segmented into **Low Risk** (<35%), **Moderate Risk** (35%-60%), and **High Risk** (≥60%) categories with visual gauges.
- **Explainable Churn Drivers**: Dynamically highlights account vulnerability factors (e.g., month-to-month contracts, electronic check payments, unbundled fiber optic service).
- **Proactive Retention Action Items**: Recommends high-impact business retention strategies tailored to each individual customer profile.
- **Input Validation & Error Boundaries**: Validates incoming customer metrics with defensive error handling to prevent application crashes.
- **Structured Logging**: Built-in Python logging with timestamps and configurable severity levels for operational observability.
- **Enterprise Containerization**: Dockerfile featuring a non-root system user, layer-cached dependency installs, and container health checks.

---

## 📂 Project Structure

```text
Telecom-Customer-Churn-Prediction-using-Machine-Learning/
├── Data/
│   └── Customer-Churn.csv          # Telecommunications customer dataset
├── Interview Questions/
│   └── Interview_Questions.md      # Top 50 interactive self-study interview questions
├── Models/
│   ├── Ada_boost_churn_model.pkl   # Production AdaBoost classifier
│   ├── Best_optuna_churn_model.pkl # Optuna-tuned XGBoost model checkpoint
│   └── best_xgboost_churn_model.pkl# Baseline XGBoost model checkpoint
├── Notebooks/
│   ├── EDA.ipynb                   # Exploratory Data Analysis & visualizations
│   ├── MODEL.ipynb                 # Model experimentation, training & tuning
│   └── catboost_info/              # CatBoost experiment telemetry
├── .dockerignore                   # Docker build context exclusions
├── .env.example                    # Environment variable configuration template
├── .gitignore                      # Git tracking rules for Python & ML artifacts
├── Dockerfile                      # Production-ready, secure, non-root Docker build
├── LICENSE                         # Project license
├── README.md                       # Comprehensive documentation
├── app.py                          # Streamlit web application & serving engine
└── requirements.txt                # Production dependencies with pinned versions
```

---

## ⚙️ Configuration & Environment Variables

Configure application parameters using environment variables or a `.env` file at the project root:

```bash
# Copy example configuration template
cp .env.example .env
```

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `APP_ENV` | `production` | Application deployment environment (`production` or `development`) |
| `SERVER_HOST` | `0.0.0.0` | Network binding host address |
| `SERVER_PORT` | `8501` | Application port |
| `LOG_LEVEL` | `INFO` | Logging severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `DATA_PATH` | `Data/Customer-Churn.csv` | Relative or absolute path to reference dataset |
| `MODEL_PATH` | `Models/Ada_boost_churn_model.pkl` | Path to serialized production model artifact |

---

## 🛠️ Local Development & Setup

### Prerequisites

- Python 3.10 or 3.11
- `git`
- Virtual environment tool (`venv` or `conda`)

### Step 1: Clone Repository & Create Virtual Environment

```bash
git clone https://github.com/mdzaheerjk/Telecom-Customer-Churn-Prediction-using-Machine-Learning.git
cd Telecom-Customer-Churn-Prediction-using-Machine-Learning

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run Application

```bash
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 🐳 Docker Deployment

The project includes an optimized `Dockerfile` adhering to cloud-native security best practices:
- Runs as an unprivileged, non-root user (`appuser`, UID 10001).
- Eliminates Python bytecode overhead (`PYTHONDONTWRITEBYTECODE=1`).
- Implements continuous container healthcheck endpoint probe (`/_stcore/health`).
- Leverages Docker layer caching for rapid rebuilds.

### Build Docker Image

```bash
docker build -t telecom-churn-app:latest .
```

### Run Docker Container

```bash
docker run -d \
  --name telecom-churn \
  -p 8501:8501 \
  --restart unless-stopped \
  telecom-churn-app:latest
```

Access the service at `http://localhost:8501`.

### Check Container Health

```bash
docker inspect --format='{{json .State.Health.Status}}' telecom-churn
```

---

## 🔬 Machine Learning Pipeline

1. **Data Ingestion**: Raw customer data with demographic, service subscription, and billing information.
2. **Data Cleaning**:
   - TotalCharges converted to numeric; empty/whitespace values dropped.
3. **Feature Engineering**:
   - `tenure` transformed into 6 ordinal demographic buckets: `1-12`, `13-24`, `25-36`, `37-48`, `49-60`, `61-72`.
   - Categorical attributes encoded using `pd.get_dummies(drop_first=True)` producing 34 standardized features.
4. **Feature Scaling**: Normalized across all feature dimensions using `StandardScaler`.
5. **Model Architecture**:
   - **AdaBoost Classifier** trained with sample weighting on minority churn cases to address target class imbalance.
   - Evaluated against Decision Trees, Random Forests, CatBoost, and Optuna-tuned XGBoost models.

---

## 📚 Interview Preparation & Self-Study Guide

A dedicated, production-grade self-study guide containing **50 high-frequency technical interview questions and answers** is available in [`Interview Questions/Interview_Questions.md`](Interview%20Questions/Interview_Questions.md):

- 📊 **Self-Study Progress Tracker** for active recall and assessment.
- 💡 **Interactive Collapsible Answers (`<details><summary>`)** with step-by-step mathematical derivations, code demonstrations, and senior engineering insights.
- 🎯 Covers the complete lifecycle: Business context, EDA, tenure discretization, class imbalance (SMOTE vs. Cost-sensitive weighting), boosting algorithms (AdaBoost, XGBoost, CatBoost, LightGBM), Bayesian optimization, threshold tuning, and containerized MLOps.

👉 [**Open the 50-Question Interactive Self-Study Guide**](Interview%20Questions/Interview_Questions.md)

---

## 📄 License

This project is licensed under the terms described in the [LICENSE](LICENSE) file.

