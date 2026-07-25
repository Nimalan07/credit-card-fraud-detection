import mlflow
from src.utils.logger import get_logger

logger = get_logger("mlflow_model_registry")

def register_model_in_registry(run_id: str, artifact_path: str, model_name: str):
    logger.info(f"Registering model from run {run_id} under registry name: {model_name}")
    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered successfully. Version: {result.version}")
        return result
    except Exception as e:
        logger.error(f"Failed to register model in registry: {str(e)}")
        return None
