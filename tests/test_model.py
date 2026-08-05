import pytest
import pandas as pd
import numpy as np
from src.deployment.predict import FraudPredictor
from src.utils.config import BEST_MODEL_PATH

def test_predictor_uninitialized_exception():
    predictor = FraudPredictor()
    
    if not BEST_MODEL_PATH.exists():
        assert not predictor.is_ready()
        with pytest.raises(RuntimeError):
            predictor.predict_single({})
    else:
        assert predictor.is_ready()
        
        # Test structure of prediction payload using IEEE-CIS dataset variables
        dummy_input = {
            "TransactionAmt": 59.0,
            "TransactionDT": 86400.0,
            "ProductCD": "W",
            "card1": 13926.0,
            "card2": 327.0,
            "card3": 150.0,
            "card4": "discover",
            "card5": 142.0,
            "card6": "credit",
            "addr1": 315.0,
            "addr2": 87.0,
            "P_emaildomain": "gmail.com",
            "R_emaildomain": "gmail.com",
            "DeviceType": "desktop",
            "DeviceInfo": "Windows"
        }
        is_fraud, prob = predictor.predict_single(dummy_input)
        assert is_fraud in [0, 1]
        assert 0.0 <= prob <= 1.0
