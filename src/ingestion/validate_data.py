import json
import pandas as pd
from src.utils.config import RAW_DATA_PATH, VALIDATION_REPORT_PATH
from src.utils.logger import get_logger

logger = get_logger("data_validation")

def validate_dataset() -> bool:
    logger.info("Starting dataset validation...")
    if not RAW_DATA_PATH.exists():
        logger.error(f"Merged dataset not found at {RAW_DATA_PATH}")
        return False
        
    try:
        # Load dataset
        df = pd.read_csv(RAW_DATA_PATH)
        num_rows, num_cols = df.shape
        logger.info(f"Loaded merged dataset with {num_rows} rows and {num_cols} columns.")
        
        # 1. Expected Column Check
        expected_cols = [
            "TransactionID", "isFraud", "TransactionAmt", "TransactionDT", "ProductCD",
            "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"
        ]
        
        # Keep track of columns present
        available_cols = list(df.columns)
        missing_cols = [col for col in expected_cols if col not in available_cols]
        
        # 2. Missing Value Check
        null_counts = df.isnull().sum().to_dict()
        total_nulls = sum(null_counts.values())
        
        # 3. Class Balance Check
        class_counts = df["isFraud"].value_counts().to_dict()
        num_genuine = class_counts.get(0, 0)
        num_fraud = class_counts.get(1, 0)
        fraud_ratio = float(num_fraud / num_rows) if num_rows > 0 else 0.0
        
        # Validation decision
        validation_passed = True
        error_messages = []
        
        if missing_cols:
            validation_passed = False
            error_messages.append(f"Missing expected columns: {missing_cols}")
            
        if num_fraud == 0:
            validation_passed = False
            error_messages.append("Dataset does not contain any fraudulent transaction records (isFraud=1).")
            
        report = {
            "validation_passed": validation_passed,
            "error_messages": error_messages,
            "dataset_info": {
                "num_rows": num_rows,
                "num_columns": num_cols,
                "columns_checked": available_cols
            },
            "missing_values": {
                "total_missing": total_nulls,
                "null_counts_per_column": null_counts
            },
            "class_distribution": {
                "genuine_count": num_genuine,
                "fraud_count": num_fraud,
                "fraud_percentage": fraud_ratio * 100
            }
        }
        
        # Ensure validation dir exists
        VALIDATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(VALIDATION_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Validation report saved to {VALIDATION_REPORT_PATH}")
        if validation_passed:
            logger.info("Dataset validation PASSED.")
        else:
            logger.error(f"Dataset validation FAILED: {error_messages}")
            
        return validation_passed
        
    except Exception as e:
        logger.exception(f"Error occurred during data validation: {str(e)}")
        return False

if __name__ == "__main__":
    validate_dataset()
