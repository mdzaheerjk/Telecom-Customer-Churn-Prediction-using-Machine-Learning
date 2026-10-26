# ☁️ Production Deployment Guide: Streamlit Community Cloud

> 🌐 **Live Deployed Application:** [**https://telecom-customer-churn-prediction-using-machine-learning.streamlit.app/**](https://telecom-customer-churn-prediction-using-machine-learning.streamlit.app/)

This guide provides a step-by-step walkthrough for deploying the Telecom Customer Churn Prediction application to **Streamlit Community Cloud** with continuous deployment from GitHub.

---

## 📑 Table of Contents
1. [Overview & Architecture](#1-overview--architecture)
2. [Prerequisites](#2-prerequisites)
3. [Step-by-Step Deployment Instructions](#3-step-by-step-deployment-instructions)
4. [Configuring Environment Variables & Secrets](#4-configuring-environment-variables--secrets)
5. [Dependency & Performance Optimization](#5-dependency--performance-optimization)
6. [Managing Large Files & Artifacts](#6-managing-large-files--artifacts)
7. [App Monitoring, Logs & Auto-Reboots](#7-app-monitoring-logs--auto-reboots)
8. [Troubleshooting Common Streamlit Cloud Pitfalls](#8-troubleshooting-common-streamlit-cloud-pitfalls)

---

## 1. Overview & Architecture

**Streamlit Community Cloud** provides zero-ops managed hosting for Streamlit applications:
- **Continuous Deployment (GitOps):** Automatically pulls changes and rebuilds whenever new commits are pushed to the target GitHub branch (`main`).
- **Resource Allocation:** Free tier grants 1 GB RAM and shared CPU per app.
- **TLS/SSL Encryption:** Automatic free HTTPS certificate management.

---

## 2. Prerequisites

1. A **GitHub** account containing this repository:
   `https://github.com/mdzaheerjk/Telecom-Customer-Churn-Prediction-using-Machine-Learning`
2. A **Streamlit Community Cloud** account (sign up free using GitHub at [share.streamlit.io](https://share.streamlit.io)).
3. Verified repository structure:
   - `app.py` at repository root.
   - `requirements.txt` at repository root.
   - Models present in `Models/Ada_boost_churn_model.pkl`.
   - Data present in `Data/Customer-Churn.csv`.

---

## 3. Step-by-Step Deployment Instructions

### Step 1: Sign in to Streamlit Cloud
1. Navigate to [share.streamlit.io](https://share.streamlit.io).
2. Click **"Continue with GitHub"** to authenticate and authorize repository access.

### Step 2: Create a New App
1. From the Streamlit dashboard, click the **"New app"** button in the top right.
2. Select your repository:
   - **Repository:** `mdzaheerjk/Telecom-Customer-Churn-Prediction-using-Machine-Learning`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** `telecom-customer-churn-prediction-using-machine-learning.streamlit.app`

### Step 3: Configure Advanced Settings
1. Click **"Advanced settings..."** before deploying.
2. **Python Version:** Select **Python 3.11** (matches model pickling version).
3. **Secrets:** (See Section 4 below).

### Step 4: Deploy
Click **"Deploy!"**
Streamlit Cloud will provision the container, install packages from `requirements.txt`, and launch your app in ~45 seconds.

---

## 4. Configuring Environment Variables & Secrets

Streamlit Community Cloud manages configuration via a TOML secrets editor under **App Settings $\to$ Secrets**.

Paste the following into the **Secrets** text area:

```toml
# ==============================================================================
# Streamlit Cloud Environment Secrets
# ==============================================================================

APP_ENV = "production"
LOG_LEVEL = "INFO"
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8501

# File paths relative to repo root
DATA_PATH = "Data/Customer-Churn.csv"
MODEL_PATH = "Models/Ada_boost_churn_model.pkl"
```

Our updated `app.py` automatically checks `st.secrets` as well as standard environment variables:
```python
try:
    DATA_PATH = st.secrets.get("DATA_PATH", os.getenv("DATA_PATH", "Data/Customer-Churn.csv"))
    MODEL_PATH = st.secrets.get("MODEL_PATH", os.getenv("MODEL_PATH", "Models/Ada_boost_churn_model.pkl"))
except Exception:
    DATA_PATH = os.getenv("DATA_PATH", "Data/Customer-Churn.csv")
    MODEL_PATH = os.getenv("MODEL_PATH", "Models/Ada_boost_churn_model.pkl")
```

---

## 5. Dependency & Performance Optimization

To avoid deployment timeouts and excessive memory usage on Streamlit Cloud's 1 GB RAM limit:
- **Keep `requirements.txt` Lean:** Only runtime serving packages (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `joblib`, `python-dotenv`) are enabled. Heavy training libraries (`xgboost`, `catboost`, `optuna`, `matplotlib`, `seaborn`) are commented out.
- **Startup Caching:** `@st.cache_data` and `@st.cache_resource` in `app.py` prevent re-reading the CSV and re-fitting the preprocessor on every browser session, ensuring sub-second inference.

---

## 6. Managing Large Files & Artifacts

- **GitHub File Limits:** GitHub enforces a 100 MB hard limit per file.
- **Project Sizes:**
  - `Data/Customer-Churn.csv`: ~977 KB
  - `Models/Ada_boost_churn_model.pkl`: ~33 KB
  - `Models/Best_optuna_churn_model.pkl`: ~719 KB
- Because all files are well under 1 MB, standard git tracking works seamlessly without requiring Git LFS.

---

## 7. App Monitoring, Logs & Auto-Reboots

### Accessing Real-Time Logs
1. Click the **"Manage app"** panel in the bottom-right corner of your live Streamlit Cloud web page.
2. View real-time stdout/stderr logs. Look for:
   ```text
   2026-10-26 22:15:00 | INFO     | churn_app - Loading reference dataset from: Data/Customer-Churn.csv
   2026-10-26 22:15:01 | INFO     | churn_app - Preprocessor initialized: 34 template features
   2026-10-26 22:15:01 | INFO     | churn_app - Model loaded successfully: AdaBoostClassifier
   ```

### Clearing Cache or Rebooting:
- Click the three dots `...` in the top-right corner $\to$ **"Clear cache"** or **"Reboot app"**.

---

## 8. Troubleshooting Common Streamlit Cloud Pitfalls

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`ModuleNotFoundError: No module named '...'`** | Package missing from `requirements.txt` | Add the required package to `requirements.txt` and push to GitHub. Streamlit Cloud auto-rebuilds. |
| **App enters sleep mode** | Inactivity for several consecutive days | Streamlit Cloud puts inactive free apps to sleep. Clicking **"Yes, get this app back up!"** awakens it within 10 seconds. |
| **`InconsistentVersionWarning` on scikit-learn** | Pickle version mismatch | Ensure `requirements.txt` pins `scikit-learn>=1.4.0,<1.6.0` to match the model’s 1.5.0 training version. |
| **Memory Limit Exceeded (App Crashed)** | Out of memory (> 1 GB) | Ensure notebooks and unneeded model checkpoints are not imported or loaded into RAM simultaneously. |

---

*Authored for the Telecom Customer Churn Prediction Machine Learning Project.*
