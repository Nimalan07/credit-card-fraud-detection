import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils.config import BEST_MODEL_PATH, PROCESSED_DATA_DIR
from src.utils.logger import get_logger
from src.preprocessing.feature_engineering import engineer_features

logger = get_logger("model_inference")

class FraudPredictor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(FraudPredictor, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.scaler = None
            cls._instance.model_version = "v1.0"
            cls._instance.load_model_and_scaler()
        return cls._instance

    def load_model_and_scaler(self):
        scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
        
        # Load Best Model
        if BEST_MODEL_PATH.exists():
            try:
                self.model = joblib.load(BEST_MODEL_PATH)
                logger.info(f"Loaded champion model from {BEST_MODEL_PATH}")
            except Exception as e:
                logger.error(f"Error loading champion model: {str(e)}")
        else:
            logger.warning(f"Champion model not found at {BEST_MODEL_PATH}!")

        # Load Scaler
        if scaler_path.exists():
            try:
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded feature scaler from {scaler_path}")
            except Exception as e:
                logger.error(f"Error loading scaler: {str(e)}")
        else:
            logger.warning(f"Scaler not found at {scaler_path}!")

    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None

    def predict_single(self, transaction: dict) -> tuple[int, float]:
        if not self.is_ready():
            raise RuntimeError("Model or Scaler not loaded. Pipeline must be run first.")
            
        df = pd.DataFrame([transaction])
        
        # 1. Feature Engineering
        df_feat = engineer_features(df)
        
        # 2. Scaling (Time and Amount)
        df_scaled = self.scaler.transform(df_feat)
        
        # 3. Align column order with training features
        # Training feature columns order: Time, V1-V28, Amount, HourOfDay, LogAmount
        expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "HourOfDay", "LogAmount"]
        df_final = df_scaled[expected_cols]
        
        # 4. Predict
        is_fraud = int(self.model.predict(df_final)[0])
        prob = float(self.model.predict_proba(df_final)[0][1])
        
        return is_fraud, prob

    def predict_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        if not self.is_ready():
            raise RuntimeError("Model or Scaler not loaded. Pipeline must be run first.")
            
        # Keep copy
        df = df_batch.copy()
        
        # Ensure all columns exist
        expected_raw_cols = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
        missing = [col for col in expected_raw_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Batch dataset missing columns: {missing}")
            
        # 1. Feature Engineering
        df_feat = engineer_features(df)
        
        # 2. Scaling
        df_scaled = self.scaler.transform(df_feat)
        
        # 3. Align columns
        expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "HourOfDay", "LogAmount"]
        df_final = df_scaled[expected_cols]
        
        # 4. Predict
        df["is_fraud"] = self.model.predict(df_final).astype(int)
        df["probability"] = self.model.predict_proba(df_final)[:, 1].astype(float)
        
        return df
