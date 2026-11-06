"""
FastAPI HTTP server for Laya Anonymization Decision Engine.

Provides endpoints:
  - GET  /health
  - POST /v1/anonymization/decide-column
  - POST /v1/anonymization/evaluate-dataset
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import laya
import vegaml
from src.predictor import AnonymizationDecider

# Global decider instance
decider: Optional[AnonymizationDecider] = None


def get_decider() -> AnonymizationDecider:
    global decider
    if decider is None:
        calib_dir = os.getenv("CALIBRATION_DIR", "models/vega_anonymizer_calibration")
        print(f"[get_decider] Initializing AnonymizationDecider (calib_dir={calib_dir})...")
        decider = AnonymizationDecider(calibration_dir=calib_dir)
    return decider


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_decider()
    yield



app = FastAPI(
    title="Laya Column Anonymization Decision Engine",
    description="Multilingual (PT/EN) non-autoregressive decision engine for PII and column anonymization.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verify_api_key(x_api_key: Optional[str] = Header(None)) -> Optional[str]:
    """Validates X-API-Key if LAYA_API_KEY is configured in the environment."""
    required_key = os.getenv("LAYA_API_KEY")
    if not required_key:
        return x_api_key
    if not x_api_key or x_api_key != required_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header.",
        )
    return x_api_key


class ColumnDecideRequest(BaseModel):
    dataset: Dict[str, Any] = Field(..., description="Dataset metadata (dataset_name, domain, purpose, num_rows, num_cols)")
    column: Dict[str, Any] = Field(..., description="Column metadata, statistics, detectors")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Decision threshold for score_float")
    drop_semantic_type: Optional[bool] = Field(False, description="Whether to drop semantic_type from column state")


class DatasetEvaluateRequest(BaseModel):
    dataset_id: Optional[str] = Field(None, description="Dataset identifier")
    dataset_name: Optional[str] = Field(None, description="Dataset name")
    domain: Optional[str] = Field(None, description="Business domain")
    purpose: Optional[str] = Field(None, description="Dataset usage purpose")
    num_rows: Optional[int] = Field(None, description="Number of rows")
    num_cols: Optional[int] = Field(None, description="Number of columns")
    columns: List[Dict[str, Any]] = Field(..., description="List of columns metadata")
    threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Decision threshold")
    drop_semantic_type: Optional[bool] = Field(False, description="Whether to drop semantic_type")


@app.get("/health", tags=["Monitoring"])
def health_check() -> Dict[str, Any]:
    current_decider = get_decider()
    return {
        "status": "healthy",
        "service": "laya-anonymization-decider",
        "laya_version": getattr(laya, "__version__", "unknown"),
        "vegaml_version": getattr(vegaml, "__version__", "unknown"),
        "calibration_loaded": current_decider.calibration_loaded,
        "calibration_dir": current_decider.calibration_dir,
    }


@app.post("/v1/anonymization/decide-column", tags=["Anonymization"], dependencies=[Depends(verify_api_key)])
def decide_column(request: ColumnDecideRequest) -> Dict[str, Any]:
    current_decider = get_decider()
    try:
        return current_decider.predict_column(
            dataset_meta=request.dataset,
            column_data=request.column,
            threshold=request.threshold,
            drop_semantic_type=request.drop_semantic_type or False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/v1/anonymization/evaluate-dataset", tags=["Anonymization"], dependencies=[Depends(verify_api_key)])
def evaluate_dataset(request: DatasetEvaluateRequest) -> Dict[str, Any]:
    current_decider = get_decider()
    try:
        payload = request.model_dump()
        return current_decider.predict_dataset(
            dataset_payload=payload,
            threshold=request.threshold,
            drop_semantic_type=request.drop_semantic_type or False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting Laya Anonymization Server on {host}:{port}...")
    uvicorn.run("server:app", host=host, port=port, reload=False)
