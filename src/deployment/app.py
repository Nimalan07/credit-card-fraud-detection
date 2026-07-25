import json
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from src.utils.config import BASE_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger
from src.deployment.routes import router as api_router

logger = get_logger("fastapi_app")
app = FastAPI(
    title="FinGuard AI - Credit Card Fraud Detection Pipeline API",
    description="Automated MLOps Pipeline for Credit Card Fraud Detection",
    version="1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Path to static assets and templates
STATIC_PATH = BASE_DIR / "src" / "dashboard" / "static"
TEMPLATE_PATH = BASE_DIR / "src" / "dashboard" / "templates" / "index.html"

# Ensure static directories exist
STATIC_PATH.mkdir(parents=True, exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

# Expose API Router (endpoints under /)
app.include_router(api_router)

@app.get("/")
def serve_dashboard():
    logger.info("Serving dashboard template...")
    if not TEMPLATE_PATH.exists():
        logger.error(f"Dashboard template not found at: {TEMPLATE_PATH}")
        raise HTTPException(status_code=404, detail="Dashboard UI files not found.")
    return FileResponse(str(TEMPLATE_PATH))

@app.get("/api/telemetry")
def get_telemetry():
    summary_path = PROCESSED_DATA_DIR / "run_summary.json"
    if not summary_path.exists():
        logger.warning(f"Telemetry summary file not found at: {summary_path}")
        raise HTTPException(
            status_code=404, 
            detail="Pipeline summary telemetry not found. Please run the training pipeline first."
        )
    
    try:
        with open(summary_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.exception("Error loading telemetry summary.")
        raise HTTPException(status_code=500, detail=f"Failed to read telemetry: {str(e)}")

def main():
    logger.info("Starting FinGuard AI API Server...")
    uvicorn.run("src.deployment.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    main()
