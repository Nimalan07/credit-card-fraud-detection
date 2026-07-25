import mlflow.sklearn
from src.utils.logger import get_logger

logger = get_logger("mlflow_model_logger")

def log_trained_model(model, artifact_path: str):
    logger.info(f"Logging model artifact under path: {artifact_path}")
    try:
        # Use mlflow.sklearn.log_model as it supports standard sklearn estimators and XGBClassifier
        mlflow.sklearn.log_model(model, artifact_path)
        logger.info("Model successfully logged to MLflow.")
    except Exception as e:
        logger.error(f"Failed to log model to MLflow: {str(e)}")
