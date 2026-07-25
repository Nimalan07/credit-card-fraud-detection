from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.utils.logger import get_logger

logger = get_logger("model_evaluation")

def evaluate_predictions(y_true, y_pred, y_prob=None) -> dict:
    logger.info("Computing classification evaluation metrics...")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            metrics["roc_auc"] = float(auc)
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {str(e)}")
            
    logger.info(f"Computed metrics: {metrics}")
    return metrics
