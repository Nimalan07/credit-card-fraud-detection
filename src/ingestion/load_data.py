import os
import pandas as pd
from pathlib import Path
from src.utils.config import RAW_TRANSACTION_PATH, RAW_IDENTITY_PATH, RAW_DATA_PATH
from src.utils.logger import get_logger

logger = get_logger("data_ingestion")

def load_dataset():
    if RAW_DATA_PATH.exists():
        logger.info(f"Merged dataset already exists at: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    if not RAW_TRANSACTION_PATH.exists() or not RAW_IDENTITY_PATH.exists():
        msg = f"Raw competition files not found. Ensure {RAW_TRANSACTION_PATH.name} and {RAW_IDENTITY_PATH.name} are in data/raw/"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Loading train_transaction.csv and train_identity.csv...")
    tx_df = pd.read_csv(RAW_TRANSACTION_PATH)
    id_df = pd.read_csv(RAW_IDENTITY_PATH)

    logger.info(f"Transaction shape: {tx_df.shape}, Identity shape: {id_df.shape}")
    logger.info("Merging datasets on TransactionID (left join)...")
    merged = tx_df.merge(id_df, on="TransactionID", how="left")

    # Selected key features for the MLOps pipeline
    key_features = [
        "TransactionID", "isFraud", "TransactionAmt", "TransactionDT", "ProductCD",
        "card1", "card2", "card3", "card4", "card5", "card6",
        "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"
    ]
    
    # Keep only columns that exist
    available_features = [col for col in key_features if col in merged.columns]
    merged_subset = merged[available_features]

    logger.info(f"Saving merged subset with shape {merged_subset.shape} to {RAW_DATA_PATH}...")
    merged_subset.to_csv(RAW_DATA_PATH, index=False)
    logger.info("Ingestion and merging completed successfully.")
    return RAW_DATA_PATH

if __name__ == "__main__":
    load_dataset()
