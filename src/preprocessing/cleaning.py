import pandas as pd
from src.utils.logger import get_logger

logger = get_logger("data_cleaning")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Cleaning data: imputing missing values and removing duplicates...")
    initial_len = len(df)
    df = df.copy()
    
    # Define categorical vs numerical features in our merged dataset
    categorical_cols = [
        "ProductCD", "card4", "card6", "P_emaildomain", 
        "R_emaildomain", "DeviceType", "DeviceInfo"
    ]
    numerical_cols = [
        "TransactionAmt", "TransactionDT", 
        "card1", "card2", "card3", "card5", 
        "addr1", "addr2"
    ]
    
    # Impute categorical missing values with 'unknown'
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
            
    # Impute numerical missing values with -1.0
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(-1.0)
            
    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        logger.info(f"Removing {duplicate_count} duplicate rows.")
        df = df.drop_duplicates()
        
    logger.info(f"Cleaned dataset: shape changed from {initial_len} to {len(df)}")
    return df
