import numpy as np
import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("feature_engineering")

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering new features...")
    df_feat = df.copy()
    
    # 1. Hour of Day extraction (since TransactionDT is in seconds)
    if "TransactionDT" in df_feat.columns:
        df_feat["HourOfDay"] = (df_feat["TransactionDT"] // 3600) % 24
        
    # 2. Log-transform transaction Amount
    if "TransactionAmt" in df_feat.columns:
        # Add a small epsilon to handle 0 or negative values if they exist
        df_feat["LogAmount"] = np.log1p(np.maximum(df_feat["TransactionAmt"], 0.0))
        
    logger.info(f"Feature engineering completed. Columns: {list(df_feat.columns)}")
    return df_feat

class FeatureEncoder:
    def __init__(self):
        self.mappings = {}
        self.categorical_cols = [
            "ProductCD", "card4", "card6", "P_emaildomain", 
            "R_emaildomain", "DeviceType", "DeviceInfo"
        ]
        
    def fit(self, df: pd.DataFrame):
        logger.info("Fitting feature encoder on categorical columns...")
        for col in self.categorical_cols:
            if col in df.columns:
                # Find all unique categories
                unique_vals = list(df[col].unique())
                if "unknown" not in unique_vals:
                    unique_vals.append("unknown")
                # Map categories to integer indices
                mapping = {val: idx for idx, val in enumerate(unique_vals)}
                self.mappings[col] = mapping
        return self
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_encoded = df.copy()
        for col, mapping in self.mappings.items():
            if col in df_encoded.columns:
                unknown_idx = mapping.get("unknown", 0)
                # Map existing categories, default to unknown_idx for new categories
                df_encoded[col] = df_encoded[col].map(lambda x: mapping.get(str(x), unknown_idx))
        return df_encoded
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
