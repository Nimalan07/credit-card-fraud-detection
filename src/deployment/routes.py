import time
import io
import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.deployment.schemas import TransactionInput, PredictionResponse, BatchPredictionResponse, BatchPredictionItem, HealthResponse
from src.deployment.predict import FraudPredictor
from src.monitoring.prometheus_metrics import REQUEST_COUNT, REQUEST_LATENCY, FRAUD_COUNT, PREDICTION_COUNT
from src.utils.logger import get_logger

logger = get_logger("api_routes")
router = APIRouter()
predictor = FraudPredictor()

@router.get("/health", response_model=HealthResponse)
def health():
    REQUEST_COUNT.labels(method="GET", endpoint="/health", status_code="200").inc()
    is_loaded = predictor.is_ready()
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=predictor.model_version
    )

@router.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionInput):
    start_time = time.time()
    logger.info("Received request for single transaction prediction.")
    
    if not predictor.is_ready():
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status_code="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded or pipeline not executed yet.")
        
    try:
        transaction_dict = payload.dict()
        is_fraud, prob = predictor.predict_single(transaction_dict)
        
        # Track Prometheus metrics
        status = "fraud" if is_fraud == 1 else "genuine"
        PREDICTION_COUNT.labels(status=status).inc()
        if is_fraud == 1:
            FRAUD_COUNT.inc()
            
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict").observe(latency)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status_code="200").inc()
        
        return PredictionResponse(
            is_fraud=is_fraud,
            label="Fraudulent" if is_fraud == 1 else "Genuine",
            probability=prob,
            model_version=predictor.model_version
        )
        
    except Exception as e:
        logger.exception(f"Error serving single prediction: {str(e)}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict", status_code="500").inc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/predict_csv", response_model=BatchPredictionResponse)
async def predict_csv(file: UploadFile = File(...)):
    start_time = time.time()
    logger.info(f"Received request for batch prediction via file: {file.filename}")
    
    if not predictor.is_ready():
        REQUEST_COUNT.labels(method="POST", endpoint="/predict_csv", status_code="503").inc()
        raise HTTPException(status_code=503, detail="Model not loaded or pipeline not executed yet.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Predict on batch
        df_pred = predictor.predict_batch(df)
        
        predictions_list = []
        fraud_count_batch = 0
        
        for idx, row in df_pred.iterrows():
            is_f = int(row["is_fraud"])
            prob = float(row["probability"])
            label = "Fraudulent" if is_f == 1 else "Genuine"
            
            predictions_list.append(
                BatchPredictionItem(
                    row_index=idx,
                    is_fraud=is_f,
                    label=label,
                    probability=prob
                )
            )
            
            # Prometheus status counts
            status = "fraud" if is_f == 1 else "genuine"
            PREDICTION_COUNT.labels(status=status).inc()
            if is_f == 1:
                FRAUD_COUNT.inc()
                fraud_count_batch += 1
                
        latency = time.time() - start_time
        REQUEST_LATENCY.labels(endpoint="/predict_csv").observe(latency)
        REQUEST_COUNT.labels(method="POST", endpoint="/predict_csv", status_code="200").inc()
        
        return BatchPredictionResponse(
            total_processed=len(df_pred),
            fraud_detected=fraud_count_batch,
            predictions=predictions_list,
            model_version=predictor.model_version
        )
        
    except Exception as e:
        logger.exception(f"Error serving batch prediction: {str(e)}")
        REQUEST_COUNT.labels(method="POST", endpoint="/predict_csv", status_code="500").inc()
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@router.get("/metrics")
def metrics():
    # Scraping endpoint for Prometheus
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
