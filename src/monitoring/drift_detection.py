import json
import pandas as pd
from scipy.stats import ks_2samp
from src.utils.config import PROCESSED_TRAIN_X, PROCESSED_TEST_X, DRIFT_REPORT_PATH
from src.utils.logger import get_logger
from src.monitoring.prometheus_metrics import DRIFT_SCORE

logger = get_logger("drift_detection")

def detect_drift() -> float:
    logger.info("Starting data drift detection...")
    
    if not PROCESSED_TRAIN_X.exists() or not PROCESSED_TEST_X.exists():
        logger.warning("Baseline train or test data not found. Run training pipeline first.")
        return 0.0
        
    try:
        # Load baseline (training) and current (test acting as new production data)
        df_baseline = pd.read_csv(PROCESSED_TRAIN_X)
        df_current = pd.read_csv(PROCESSED_TEST_X)
        
        drift_features = ["Amount", "Time", "V1", "V2", "V3"]
        drift_scores = {}
        drift_detected = False
        
        # Calculate Kolmogorov-Smirnov test for each feature
        for col in drift_features:
            if col in df_baseline.columns and col in df_current.columns:
                stat, p_val = ks_2samp(df_baseline[col], df_current[col])
                # standard threshold: p-value < 0.05 indicates different distributions (drift)
                drifted = p_val < 0.05
                drift_scores[col] = {
                    "ks_statistic": float(stat),
                    "p_value": float(p_val),
                    "drift_detected": bool(drifted)
                }
                if drifted:
                    drift_detected = True
                    
        # Average KS statistic as overall drift score
        overall_score = float(sum(d["ks_statistic"] for d in drift_scores.values()) / len(drift_scores))
        
        report = {
            "drift_detected": drift_detected,
            "overall_drift_score": overall_score,
            "feature_drift": drift_scores
        }
        
        # Save report
        with open(DRIFT_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Drift detection complete. Overall Drift Score: {overall_score:.4f}. Report saved to {DRIFT_REPORT_PATH}")
        
        # Log to Prometheus
        DRIFT_SCORE.set(overall_score)
        
        return overall_score
        
    except Exception as e:
        logger.exception(f"Error during drift detection: {str(e)}")
        return 0.0

if __name__ == "__main__":
    detect_drift()
