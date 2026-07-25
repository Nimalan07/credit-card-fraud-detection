import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering new features...")
    df_feat = df.copy()
    
    # Deriving hour of day (since Time is in seconds, 3600 seconds/hour, 24 hours/day)
    if "Time" in df_feat.columns:
        df_feat["HourOfDay"] = (df_feat["Time"] // 3600) % 24
        
        # Log-transform transaction Amount (adding a tiny epsilon to handle 0 amounts)
        # Note: Amount scaling is still handled by RobustScaler. This log transform creates a more normal distribution.
        if "Amount" in df_feat.columns:
            df_feat["LogAmount"] = np.log1p(df_feat["Amount"])
            
    logger.info(f"Feature engineering completed. Columns: {list(df_feat.columns)}")
    return df_feat
