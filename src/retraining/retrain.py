import json
import joblib
import pandas as pd
from src.utils.config import (
    PROCESSED_TRAIN_X, PROCESSED_TRAIN_Y, PROCESSED_TEST_X, PROCESSED_TEST_Y,
    BEST_MODEL_PATH, RANDOM_STATE
)
from src.utils.logger import get_logger
from xgboost import XGBClassifier
from src.training.evaluate import evaluate_predictions

logger = get_logger("retraining")

def retrain_model_pipeline():
    logger.info("Starting automated retraining job...")
    
    if not all(p.exists() for p in [PROCESSED_TRAIN_X, PROCESSED_TRAIN_Y, PROCESSED_TEST_X, PROCESSED_TEST_Y]):
        logger.error("Processed training datasets not found. Unable to run retraining.")
        return False
        
    try:
        # Load training and test splits
        X_train = pd.read_csv(PROCESSED_TRAIN_X)
        y_train = pd.read_csv(PROCESSED_TRAIN_Y).squeeze()
        X_test = pd.read_csv(PROCESSED_TEST_X)
        y_test = pd.read_csv(PROCESSED_TEST_Y).squeeze()
        
        # Load existing production model
        if not BEST_MODEL_PATH.exists():
            logger.error("Production model (best_model.pkl) does not exist. Run main training pipeline first.")
            return False
            
        prod_model = joblib.load(BEST_MODEL_PATH)
        y_pred_prod = prod_model.predict(X_test)
        prod_metrics = evaluate_predictions(y_test, y_pred_prod)
        logger.info(f"Existing Production Model F1-Score: {prod_metrics['f1_score']:.4f}")
        
        # Retrain with a new model configuration / new random state representing new session tuning
        logger.info("Fitting new candidate model for retraining...")
        retrained_model = XGBClassifier(
            n_estimators=60, # slightly more trees
            max_depth=5,     # slightly shallower
            learning_rate=0.15,
            eval_metric="logloss",
            random_state=RANDOM_STATE + 1, # different random seed
            n_jobs=-1
        )
        
        retrained_model.fit(X_train, y_train)
        y_pred_new = retrained_model.predict(X_test)
        new_metrics = evaluate_predictions(y_test, y_pred_new)
        logger.info(f"Retrained Candidate Model F1-Score: {new_metrics['f1_score']:.4f}")
        
        # Compare
        if new_metrics["f1_score"] > prod_metrics["f1_score"]:
            logger.info("Retrained model outperforms production model. Updating production model binary...")
            joblib.dump(retrained_model, BEST_MODEL_PATH)
            logger.info(f"Production model successfully updated at: {BEST_MODEL_PATH}")
            return True
        else:
            logger.info("Production model outperforms retrained candidate. Retaining existing model.")
            return False
            
    except Exception as e:
        logger.exception(f"Error during model retraining: {str(e)}")
        return False

if __name__ == "__main__":
    retrain_model_pipeline()
