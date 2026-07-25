import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier
from src.utils.config import RF_MODEL_PATH, RANDOM_STATE
from src.utils.logger import get_logger
from src.training.evaluate import evaluate_predictions
from src.mlflow_tracking.log_metrics import log_experiment_metrics
from src.mlflow_tracking.log_model import log_trained_model

logger = get_logger("train_random_forest")

def train_random_forest(X_train, y_train, X_test, y_test):
    logger.info("Initializing Random Forest model...")
    
    # Hyperparameters optimized for performance and speed
    params = {
        "n_estimators": 50,
        "max_depth": 10,
        "random_state": RANDOM_STATE,
        "n_jobs": -1
    }
    
    model = RandomForestClassifier(**params)
    
    # MLflow tracking
    with mlflow.start_run(run_name="Random_Forest") as run:
        logger.info("Fitting Random Forest model...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        # Evaluate
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        
        # Log to MLflow
        log_experiment_metrics(metrics, params, "Random Forest", y_test, y_pred)
        log_trained_model(model, "random_forest_model")
        
        # Save locally
        RF_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, RF_MODEL_PATH)
        logger.info(f"Model saved locally at {RF_MODEL_PATH}")
        
        return model, metrics, run.info.run_id

if __name__ == "__main__":
    import pandas as pd
    from src.utils.config import PROCESSED_TRAIN_X, PROCESSED_TRAIN_Y, PROCESSED_TEST_X, PROCESSED_TEST_Y
    if all(p.exists() for p in [PROCESSED_TRAIN_X, PROCESSED_TRAIN_Y, PROCESSED_TEST_X, PROCESSED_TEST_Y]):
        X_train = pd.read_csv(PROCESSED_TRAIN_X)
        y_train = pd.read_csv(PROCESSED_TRAIN_Y).squeeze()
        X_test = pd.read_csv(PROCESSED_TEST_X)
        y_test = pd.read_csv(PROCESSED_TEST_Y).squeeze()
        train_random_forest(X_train, y_train, X_test, y_test)
    else:
        logger.warning("Processed data files not found. Train script cannot run standalone.")
