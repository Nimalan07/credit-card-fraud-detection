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
            cls._instance.encoder = None
            cls._instance.model_version = "v2.0"
            cls._instance.load_model_scaler_encoder()
        return cls._instance

    def load_model_scaler_encoder(self):
        scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
        encoder_path = PROCESSED_DATA_DIR / "encoder.pkl"
        
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

        # Load Encoder
        if encoder_path.exists():
            try:
                self.encoder = joblib.load(encoder_path)
                logger.info(f"Loaded feature encoder from {encoder_path}")
            except Exception as e:
                logger.error(f"Error loading encoder: {str(e)}")
        else:
            logger.warning(f"Encoder not found at {encoder_path}!")

    def is_ready(self) -> bool:
        return self.model is not None and self.scaler is not None and self.encoder is not None

    def predict_single(self, transaction: dict) -> tuple[int, float]:
        if not self.is_ready():
            raise RuntimeError("Inference artifacts not fully loaded. Run the training pipeline first.")
            
        df = pd.DataFrame([transaction])
        
        # 1. Feature Engineering
        df_feat = engineer_features(df)
        
        # 2. Categorical Encoding
        df_encoded = self.encoder.transform(df_feat)
        
        # 3. Scaling (TransactionDT and TransactionAmt)
        df_scaled = self.scaler.transform(df_encoded)
        
        # 4. Align column order with training features
        expected_cols = [
            "TransactionAmt", "TransactionDT", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "HourOfDay", "LogAmount"
        ]
        df_final = df_scaled[expected_cols]
        
        # 5. Predict
        is_fraud = int(self.model.predict(df_final)[0])
        prob = float(self.model.predict_proba(df_final)[0][1])
        
        return is_fraud, prob

    def predict_batch(self, df_batch: pd.DataFrame) -> pd.DataFrame:
        if not self.is_ready():
            raise RuntimeError("Inference artifacts not fully loaded. Run the training pipeline first.")
            
        df = df_batch.copy()
        
        # Ensure all columns exist (except targets/keys like isFraud and TransactionID)
        expected_raw_cols = [
            "TransactionAmt", "TransactionDT", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"
        ]
        missing = [col for col in expected_raw_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Batch dataset missing columns: {missing}")
            
        # 1. Feature Engineering
        df_feat = engineer_features(df)
        
        # 2. Categorical Encoding
        df_encoded = self.encoder.transform(df_feat)
        
        # 3. Scaling
        df_scaled = self.scaler.transform(df_encoded)
        
        # 4. Align columns
        expected_cols = [
            "TransactionAmt", "TransactionDT", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo", "HourOfDay", "LogAmount"
        ]
        df_final = df_scaled[expected_cols]
        
        # 5. Predict
        df["is_fraud"] = self.model.predict(df_final).astype(int)
        df["probability"] = self.model.predict_proba(df_final)[:, 1].astype(float)
        
        return df
