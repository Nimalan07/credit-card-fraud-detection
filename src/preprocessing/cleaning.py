import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("data_cleaning")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data: checking for duplicates and missing values...")
    
    # Missing values
    initial_len = len(df)
    null_rows = df.isnull().any(axis=1).sum()
    if null_rows > 0:
        logger.info(f"Removing {null_rows} rows with missing values.")
        df = df.dropna()
        
    # Duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.info(f"Removing {duplicate_count} duplicate rows.")
        df = df.drop_duplicates()
        
    logger.info(f"Cleaned dataset: shape changed from {initial_len} to {len(df)}")
    return df
