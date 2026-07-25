import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow

from src.utils.config import (
    RAW_DATA_PATH, PROCESSED_DATA_DIR, BEST_MODEL_PATH, 
    PROCESSED_TRAIN_X, PROCESSED_TRAIN_Y, PROCESSED_TEST_X, PROCESSED_TEST_Y,
    RANDOM_STATE, TEST_SIZE, SAMPLE_FRACTION
)
from src.utils.logger import get_logger

from src.ingestion.load_data import load_dataset
from src.ingestion.validate_data import validate_dataset
from src.preprocessing.cleaning import clean_data
from src.preprocessing.feature_engineering import engineer_features
from src.preprocessing.scaling import FeatureScaler
from src.preprocessing.smote import apply_smote

from src.training.train_lr import train_logistic_regression
from src.training.train_rf import train_random_forest
from src.training.train_xgb import train_xgboost
from src.mlflow_tracking.register_model import register_model_in_registry

logger = get_logger("training_pipeline")

def run_pipeline():
    logger.info("================== STARTING FINGUARD MLOPS PIPELINE ==================")
    
    # Step 1: Ingestion
    logger.info("[Step 1/7] Ingesting/checking raw dataset...")
    load_dataset()
    
    # Step 2: Validation
    logger.info("[Step 2/7] Running data validation checks...")
    if not validate_dataset():
        logger.error("Dataset validation failed! Stopping pipeline.")
        return False
        
    # Step 3: Preprocessing
    logger.info("[Step 3/7] Beginning data preprocessing...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Optional sampling for local speed
    if SAMPLE_FRACTION < 1.0:
        logger.info(f"Sampling dataset: keeping {SAMPLE_FRACTION*100}% of data for faster training...")
        # Stratify sample by class to preserve fraud ratio
        df, _ = train_test_split(df, train_size=SAMPLE_FRACTION, stratify=df["Class"], random_state=RANDOM_STATE)
        logger.info(f"Sampled dataset shape: {df.shape}")
        
    df_clean = clean_data(df)
    df_feat = engineer_features(df_clean)
    
    # Split into features and target
    X = df_feat.drop(columns=["Class"])
    y = df_feat["Class"]
    
    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    # Scale features
    scaler = FeatureScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE to training set only
    X_train_res, y_train_res = apply_smote(X_train_scaled, y_train)
    
    # Save processed data for logging/telemetry
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    X_train_res.to_csv(PROCESSED_TRAIN_X, index=False)
    y_train_res.to_csv(PROCESSED_TRAIN_Y, index=False)
    X_test_scaled.to_csv(PROCESSED_TEST_X, index=False)
    y_test.to_csv(PROCESSED_TEST_Y, index=False)
    
    # Save the scaler so the API/inference script can use it
    scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    logger.info("Preprocessed data and fit scaler saved successfully.")
    
    # Step 4: Setup MLflow Experiment
    logger.info("[Step 4/7] Setting up MLflow experiment...")
    mlflow.set_experiment("FinGuard_Fraud_Detection")
    
    # Step 5: Train Models
    logger.info("[Step 5/7] Training candidate models...")
    
    models_metrics = {}
    
    # Logistic Regression
    lr_model, lr_metrics, lr_run_id = train_logistic_regression(
        X_train_res, y_train_res, X_test_scaled, y_test
    )
    models_metrics["Logistic Regression"] = {
        "model": lr_model, "metrics": lr_metrics, "run_id": lr_run_id, "path": "logistic_regression_model"
    }
    
    # Random Forest
    rf_model, rf_metrics, rf_run_id = train_random_forest(
        X_train_res, y_train_res, X_test_scaled, y_test
    )
    models_metrics["Random Forest"] = {
        "model": rf_model, "metrics": rf_metrics, "run_id": rf_run_id, "path": "random_forest_model"
    }
    
    # XGBoost
    xgb_model, xgb_metrics, xgb_run_id = train_xgboost(
        X_train_res, y_train_res, X_test_scaled, y_test
    )
    models_metrics["XGBoost"] = {
        "model": xgb_model, "metrics": xgb_metrics, "run_id": xgb_run_id, "path": "xgboost_model"
    }
    
    # Step 6: Select Best Model
    logger.info("[Step 6/7] Comparing models and selecting the champion...")
    
    best_model_name = None
    best_f1 = -1.0
    
    # Print comparison
    logger.info("-" * 60)
    logger.info(f"{'Model Name':<25} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    logger.info("-" * 60)
    for name, info in models_metrics.items():
        m = info["metrics"]
        logger.info(f"{name:<25} | {m['accuracy']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f} | {m['f1_score']:<10.4f}")
        
        # Compare based on F1-score
        if m["f1_score"] > best_f1:
            best_f1 = m["f1_score"]
            best_model_name = name
    logger.info("-" * 60)
    
    best_info = models_metrics[best_model_name]
    logger.info(f"Champion Selected: {best_model_name} with F1-Score: {best_f1:.4f}")
    
    # Save best model locally
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_info["model"], BEST_MODEL_PATH)
    logger.info(f"Saved champion model to: {BEST_MODEL_PATH}")
    
    # Save a run summary file
    summary = {
        "best_model_name": best_model_name,
        "best_f1_score": best_f1,
        "run_id": best_info["run_id"],
        "metrics": best_info["metrics"],
        "all_models": {
            "Logistic Regression": {
                "metrics": lr_metrics,
                "params": {"C": 0.01, "max_iter": 1000, "solver": "lbfgs"}
            },
            "Random Forest": {
                "metrics": rf_metrics,
                "params": {"n_estimators": 50, "max_depth": 10}
            },
            "XGBoost": {
                "metrics": xgb_metrics,
                "params": {"n_estimators": 50, "max_depth": 6, "learning_rate": 0.1}
            }
        }
    }
    with open(PROCESSED_DATA_DIR / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    # Step 7: Register Best Model
    logger.info("[Step 7/7] Registering champion in the MLflow Model Registry...")
    register_model_in_registry(
        run_id=best_info["run_id"],
        artifact_path=best_info["path"],
        model_name="FinGuard_Fraud_Model"
    )
    
    logger.info("================== FINGUARD MLOPS PIPELINE COMPLETED ==================")
    return True

if __name__ == "__main__":
    run_pipeline()
