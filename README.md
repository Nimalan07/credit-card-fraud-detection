# FinGuard AI: Real-Time Credit Card Fraud Detection with Automated MLOps Pipeline

FinGuard AI is a production-style, end-to-end MLOps pipeline designed to ingest, validate, preprocess, train, track, serve, containerize, and monitor credit card transaction fraud models.

---

## 🏗️ Folder Structure

```text
FinGuard-AI/
│
├── data/                          # Module 1 - Data Ingestion
│   ├── raw/
│   │   └── creditcard.csv         # Raw dataset (moved from data/)
│   ├── processed/
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   └── y_test.csv
│   └── validation/
│       └── validation_report.json
│
├── src/
│   ├── ingestion/                 # Module 1
│   │   ├── load_data.py
│   │   ├── validate_data.py
│   │   └── dvc_pipeline.py
│   │
│   ├── preprocessing/             # Module 2
│   │   ├── cleaning.py
│   │   ├── scaling.py
│   │   ├── smote.py
│   │   └── feature_engineering.py
│   │
│   ├── training/                  # Module 3
│   │   ├── train_lr.py
│   │   ├── train_rf.py
│   │   ├── train_xgb.py
│   │   ├── evaluate.py
│   │   └── train_pipeline.py
│   │
│   ├── mlflow_tracking/           # Module 4
│   │   ├── log_metrics.py
│   │   ├── log_model.py
│   │   └── register_model.py
│   │
│   ├── deployment/                # Module 5
│   │   ├── app.py
│   │   ├── routes.py
│   │   ├── predict.py
│   │   └── schemas.py
│   │
│   ├── monitoring/                # Module 8
│   │   ├── prometheus_metrics.py
│   │   ├── drift_detection.py
│   │   └── alerts.py
│   │
│   ├── retraining/                # Module 9
│   │   ├── retrain.py
│   │   ├── evaluate_new_model.py
│   │   └── deploy_best_model.py
│   │
│   ├── dashboard/                 # Module 10 (Served by FastAPI)
│   │   ├── templates/
│   │   │   └── index.html         # Rich Glassmorphic Dashboard
│   │   ├── static/
│   │   │   ├── css/
│   │   │   │   └── styles.css
│   │   │   └── js/
│   │   │       └── dashboard.js
│   │
│   └── utils/
│       ├── config.py
│       ├── logger.py
│       └── helper.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── best_model.pkl
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/
│   ├── test_api.py
│   ├── test_model.py
│   └── test_preprocessing.py
│
├── requirements.txt
├── params.yaml
└── .gitignore
```

---

## ⚡ Quick Start

### 1. Installation
Install all dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Unit Tests
Verify model serving and preprocessing setups:
```bash
python -m pytest tests/
```

### 3. Run Pipeline Orchestration
To execute ingestion, validate schema, split, scale, apply SMOTE, train models (Logistic Regression, Random Forest, XGBoost), log metrics to MLflow, and register the best model:
```bash
python -m src.training.train_pipeline
```

### 4. Serve the API & Glassmorphic Dashboard
Run the FastAPI web application serving both the inference endpoints and the interactive analytics dashboard:
```bash
python -m src.deployment.app
```
Access the dashboard at `http://localhost:8000/` and Swagger API docs at `http://localhost:8000/docs`.

### 5. Check MLflow Experiments
View the registered runs, parameters, F1 evaluation curves, confusion matrix plots, and register versions:
```bash
mlflow ui
```
Access the tracking interface at `http://localhost:5000`.

---

## 🐳 Containerization (Docker)

To build and launch the complete stack containerized:
```bash
# Build the container
docker build -t finguard-api -f docker/Dockerfile .

# Start the service using Docker Compose
docker compose -f docker/docker-compose.yml up --build
```
This runs the web server mapping container port 8000 to the host port 8000, retaining persistent model storage.

---

## 📊 Pipeline Validation Results

### 1. Model Comparisons (Sample Split)
The orchestrator evaluated Logistic Regression, Random Forest, and XGBoost models:

| Model Name | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 97.99% | 7.66% | 100.00% | 14.23% |
| **Random Forest** | 99.96% | 85.71% | 94.74% | **90.00%** |
| **XGBoost** | 99.82% | 48.72% | 100.00% | 65.52% |

* **Champion Model**: **Random Forest** (F1-score = 0.9000). Registered as Version 1 in the MLflow Registry.

### 2. Telemetry and Drift
- **Drift score**: `0.2189` average Kolmogorov-Smirnov score across key features (Amount, Time, V1-V3). Report saved in `data/drift_report.json`.
- **Retraining verification**: Retrained candidate model F1 is `0.8182`. Retrained candidate did not outperform production model (`0.9000`), so the champion was retained.
