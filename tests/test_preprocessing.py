import pandas as pd
import numpy as np
from src.preprocessing.cleaning import clean_data
from src.preprocessing.feature_engineering import engineer_features

def test_clean_data():
    # Setup dummy data with duplicates and nulls
    data = {
        "Time": [0.0, 10.0, 0.0, 20.0],
        "Amount": [100.0, 200.0, 100.0, np.nan],
        "Class": [0, 1, 0, 0]
    }
    df = pd.DataFrame(data)
    
    # Run cleaning
    df_clean = clean_data(df)
    
    # Should drop duplicate (row 2) and null (row 3)
    assert len(df_clean) == 2
    assert df_clean.isnull().sum().sum() == 0

def test_engineer_features():
    # Setup dummy data
    data = {
        "Time": [0.0, 3600.0, 72000.0],
        "Amount": [10.0, 100.0, 1000.0]
    }
    df = pd.DataFrame(data)
    
    # Run feature engineering
    df_feat = engineer_features(df)
    
    # Check features exist
    assert "HourOfDay" in df_feat.columns
    assert "LogAmount" in df_feat.columns
    
    # Check HourOfDay calculations: 0, 1, 20
    assert list(df_feat["HourOfDay"].values) == [0.0, 1.0, 20.0]
    
    # Check LogAmount is log1p of Amount
    assert np.allclose(df_feat["LogAmount"], np.log1p(df["Amount"]))
