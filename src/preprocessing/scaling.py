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
        if "TransactionDT" in X.columns:
            self.time_scaler.fit(X[["TransactionDT"]])
        if "TransactionAmt" in X.columns:
            self.amount_scaler.fit(X[["TransactionAmt"]])
        return self
            
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_scaled = X.copy()
        if "TransactionDT" in X.columns:
            X_scaled["TransactionDT"] = self.time_scaler.transform(X[["TransactionDT"]])
        if "TransactionAmt" in X.columns:
            X_scaled["TransactionAmt"] = self.amount_scaler.transform(X[["TransactionAmt"]])
        return X_scaled
        
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.fit(X)
        return self.transform(X)
