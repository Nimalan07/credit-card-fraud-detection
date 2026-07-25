import pandas as pd
from imblearn.over_sampling import SMOTE
from src.utils.config import RANDOM_STATE
from src.utils.logger import get_logger

logger = get_logger("data_smote")

def apply_smote(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    logger.info(f"Applying SMOTE to balance classes. Original class distribution: {y_train.value_counts().to_dict()}")
    
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    logger.info(f"SMOTE applied. Balanced class distribution: {pd.Series(y_train_res).value_counts().to_dict()}")
    return X_train_res, y_train_res
