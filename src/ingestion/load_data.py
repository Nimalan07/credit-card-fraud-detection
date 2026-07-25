import os
import shutil
from pathlib import Path
from src.utils.config import RAW_DATA_PATH, BASE_DIR
from src.utils.logger import get_logger

logger = get_logger("data_ingestion")

def load_dataset():
    # Expected initial location
    initial_path = BASE_DIR / "data" / "creditcard.csv"
    
    if RAW_DATA_PATH.exists():
        logger.info(f"Dataset already exists at raw path: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    if initial_path.exists():
        logger.info(f"Moving dataset from {initial_path} to {RAW_DATA_PATH}...")
        # Move dataset
        shutil.move(str(initial_path), str(RAW_DATA_PATH))
        logger.info("Dataset successfully moved.")
        return RAW_DATA_PATH
    else:
        logger.error(f"Dataset not found at {initial_path} or {RAW_DATA_PATH}!")
        raise FileNotFoundError("creditcard.csv not found in data folder.")

if __name__ == "__main__":
    load_dataset()
