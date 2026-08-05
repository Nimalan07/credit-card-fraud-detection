from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TransactionInput(BaseModel):
    TransactionAmt: float = Field(..., description="Transaction amount in USD")
    TransactionDT: float = Field(..., description="Time delta in seconds from reference point")
    ProductCD: str = Field("W", description="Product code (e.g. W, H, C, S, R)")
    card1: float = Field(..., description="Card column 1")
    card2: float = Field(-1.0, description="Card column 2")
    card3: float = Field(-1.0, description="Card column 3")
    card4: str = Field("unknown", description="Card brand (e.g. visa, mastercard, discover)")
    card5: float = Field(-1.0, description="Card column 5")
    card6: str = Field("unknown", description="Card type (e.g. debit, credit)")
    addr1: float = Field(-1.0, description="Billing region")
    addr2: float = Field(-1.0, description="Billing country")
    P_emaildomain: str = Field("unknown", description="Purchaser email domain")
    R_emaildomain: str = Field("unknown", description="Recipient email domain")
    DeviceType: str = Field("unknown", description="Device type (e.g. desktop, mobile)")
    DeviceInfo: str = Field("unknown", description="Device details (e.g. Windows, iOS)")

    class Config:
        schema_extra = {
            "example": {
                "TransactionAmt": 59.00,
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
        }

class PredictionResponse(BaseModel):
    is_fraud: int = Field(..., description="0 = Genuine, 1 = Fraudulent")
    label: str = Field(..., description="Genuine or Fraudulent")
    probability: float = Field(..., description="Probability of transaction being fraudulent")
    model_version: str = Field(..., description="Version of the model that served this prediction")

class BatchPredictionItem(BaseModel):
    row_index: int
    is_fraud: int
    label: str
    probability: float

class BatchPredictionResponse(BaseModel):
    total_processed: int
    fraud_detected: int
    predictions: List[BatchPredictionItem]
    model_version: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
