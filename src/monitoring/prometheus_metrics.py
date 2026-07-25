from prometheus_client import Counter, Histogram, Gauge

# API Request Metrics
REQUEST_COUNT = Counter(
    "finguard_api_requests_total",
    "Total number of requests received",
    ["method", "endpoint", "status_code"]
)

REQUEST_LATENCY = Histogram(
    "finguard_api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"]
)

# Fraud Prediction Metrics
FRAUD_COUNT = Counter(
    "finguard_fraud_detections_total",
    "Total number of fraudulent transactions detected by the model"
)

PREDICTION_COUNT = Counter(
    "finguard_predictions_total",
    "Total number of transactions processed for prediction",
    ["status"] # e.g. genuine, fraud
)

# MLOps Telemetry Metrics
DRIFT_SCORE = Gauge(
    "finguard_data_drift_score",
    "Data drift score calculated by MLOps monitoring module"
)

MODEL_ACCURACY = Gauge(
    "finguard_model_accuracy_score",
    "Accuracy score of the active production model"
)
