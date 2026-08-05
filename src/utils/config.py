import os
from pathlib import Path

# Paths
BASE_DIR = Path("c:/Users/HP/mlops").resolve()

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VALIDATION_DIR = DATA_DIR / "validation"
MODELS_DIR = BASE_DIR / "models"

RAW_TRANSACTION_PATH = RAW_DATA_DIR / "train_transaction.csv"
RAW_IDENTITY_PATH = RAW_DATA_DIR / "train_identity.csv"
RAW_DATA_PATH = RAW_DATA_DIR / "merged_train.csv"
PROCESSED_TRAIN_X = PROCESSED_DATA_DIR / "X_train.csv"
PROCESSED_TEST_X = PROCESSED_DATA_DIR / "X_test.csv"
PROCESSED_TRAIN_Y = PROCESSED_DATA_DIR / "y_train.csv"
PROCESSED_TEST_Y = PROCESSED_DATA_DIR / "y_test.csv"

VALIDATION_REPORT_PATH = VALIDATION_DIR / "validation_report.json"
DRIFT_REPORT_PATH = DATA_DIR / "drift_report.json"

# Models
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
LR_MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
RF_MODEL_PATH = MODELS_DIR / "random_forest.pkl"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.pkl"

# Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
SAMPLE_FRACTION = 0.2  # Use 20% of dataset for faster CPU training in development (set to 1.0 for full dataset)

# Ensure directories exist
for folder in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, VALIDATION_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
