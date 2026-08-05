import pandas as pd
import numpy as np
from src.preprocessing.cleaning import clean_data
from src.preprocessing.feature_engineering import engineer_features

def test_clean_data():
    # Setup dummy data with duplicates and nulls
    data = {
        "TransactionDT": [0.0, 10.0, 0.0, 20.0],
        "TransactionAmt": [100.0, 200.0, 100.0, np.nan],
        "isFraud": [0, 1, 0, 0]
    }
    df = pd.DataFrame(data)
    
    # Run cleaning
    df_clean = clean_data(df)
    
    # Duplicate (row index 2) should be dropped, null amount (row index 3) imputed with -1.0
    assert len(df_clean) == 3
    assert df_clean.isnull().sum().sum() == 0
    assert df_clean.iloc[2]["TransactionAmt"] == -1.0

def test_engineer_features():
    # Setup dummy data
    data = {
        "TransactionDT": [0.0, 3600.0, 72000.0],
        "TransactionAmt": [10.0, 100.0, 1000.0]
    }
    df = pd.DataFrame(data)
    
    # Run feature engineering
    df_feat = engineer_features(df)
    
    # Check features exist
    assert "HourOfDay" in df_feat.columns
    assert "LogAmount" in df_feat.columns
    
    # Check HourOfDay calculations: 0, 1, 20
    assert list(df_feat["HourOfDay"].values) == [0.0, 1.0, 20.0]
    
    # Check LogAmount is log1p of TransactionAmt
    assert np.allclose(df_feat["LogAmount"], np.log1p(df["TransactionAmt"]))
