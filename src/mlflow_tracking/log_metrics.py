import os
import mlflow
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.utils.logger import get_logger

logger = get_logger("mlflow_metrics")

def log_experiment_metrics(metrics: dict, params: dict, model_name: str, y_true=None, y_pred=None):
    logger.info(f"Logging parameters and metrics to MLflow for model: {model_name}")
    try:
        # Log params
        mlflow.log_params(params)
        mlflow.log_param("model_name", model_name)
        
        # Log metrics
        for metric_name, val in metrics.items():
            mlflow.log_metric(metric_name, val)
            
        # Log confusion matrix if predictions are provided
        if y_true is not None and y_pred is not None:
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Genuine", "Fraud"])
            
            fig, ax = plt.subplots(figsize=(6, 6))
            disp.plot(cmap="Blues", values_format="d", ax=ax)
            plt.title(f"Confusion Matrix - {model_name}")
            
            # Save confusion matrix to a temp file and log as artifact
            plot_path = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            
            mlflow.log_artifact(plot_path)
            if os.path.exists(plot_path):
                os.remove(plot_path)
                
            logger.info("Successfully logged metrics and confusion matrix plot to MLflow.")
    except Exception as e:
        logger.error(f"Failed to log metrics to MLflow: {str(e)}")
