import pytest
import pandas as pd
import numpy as np
from src.deployment.predict import FraudPredictor
from src.utils.config import BEST_MODEL_PATH

def test_predictor_uninitialized_exception():
    # If the model and scaler don't exist yet, it should return False or raise RuntimeError
    predictor = FraudPredictor()
    
    # Check that is_ready responds correctly based on file existence
    if not BEST_MODEL_PATH.exists():
        assert not predictor.is_ready()
        with pytest.raises(RuntimeError):
            predictor.predict_single({})
    else:
        assert predictor.is_ready()
        
        # Test structure of prediction payload
        dummy_input = {
            "Time": 0.0, "Amount": 10.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        }
        is_fraud, prob = predictor.predict_single(dummy_input)
        assert is_fraud in [0, 1]
        assert 0.0 <= prob <= 1.0
