import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.utils.logger import get_logger

logger = get_logger("data_scaling")

class FeatureScaler:
    def __init__(self):
        self.time_scaler = StandardScaler()
        self.amount_scaler = RobustScaler()
        
    def fit(self, X: pd.DataFrame):
        logger.info("Fitting scalers on training features...")
        if "Time" in X.columns:
            self.time_scaler.fit(X[["Time"]])
        if "Amount" in X.columns:
            self.amount_scaler.fit(X[["Amount"]])
        return self
            
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = X.copy()
        if "Time" in X.columns:
            X_scaled["Time"] = self.time_scaler.transform(X[["Time"]])
        if "Amount" in X.columns:
            X_scaled["Amount"] = self.amount_scaler.transform(X[["Amount"]])
        return X_scaled
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
