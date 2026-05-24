import os
import sys
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Add parent directory to path to make sure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import EMGPipeline

# Define globally accessible components
pipeline = EMGPipeline(original_fs=1000.0, target_fs=100.0, window_size=20, overlap=0.0)
model = None

# Pydantic schemas for request and response validation
class EMGWindowRequest(BaseModel):
    emg_window: list[list[float]] = Field(
        ..., 
        description="A list of lists containing raw EMG channel values. Shape must be (N_samples, 11) where N_samples >= 200 (typically 200 samples representing 200ms at 1000Hz).",
        example=[[0.0] * 11] * 200
    )

class PredictionResponse(BaseModel):
    predicted_knee_angle: float = Field(..., description="The predicted right knee angle in degrees.")
    latency_ms: float = Field(..., description="The internal processing and inference latency in milliseconds.")
    features_extracted: dict = Field(..., description="A subset of the extracted feature values for verification.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events, loading the PyCaret model."""
    global model
    model_path = os.path.join(os.path.dirname(__file__), 'best_regressor_model.pkl')
    print(f"Loading regression model from '{model_path}'...")
    
    try:
        import joblib
        bundle = joblib.load(model_path)
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle
            print("Model loaded successfully (sklearn bundle)!")
        else:
            from pycaret.regression import load_model
            model = load_model(os.path.join(os.path.dirname(__file__), 'best_regressor_model'))
            print("Model loaded successfully (PyCaret)!")
    except Exception as e:
        print(f"CRITICAL: Failed to load model. Please run 'run_pipeline_viz.py' first. Error: {e}")
        # We don't raise here so the app can start and show a health check failure instead of crashing.
    yield
    print("Shutting down API server...")

# Initialize FastAPI App
app = FastAPI(
    title="Exoskeleton EMG Control API",
    description="Real-time joint angle prediction service utilizing preprocessed electromyography (EMG) signals and a tuned PyCaret regression model.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", tags=["System"])
async def health_check():
    """System status check."""
    status = "OK"
    details = "All systems operational."
    if model is None:
        status = "DEGRADED"
        details = "PyCaret model not loaded. Run the training script first."
    return {
        "status": status,
        "details": details,
        "loaded_model": "best_regressor_model" if model is not None else None
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_joint_angle(request: EMGWindowRequest):
    """
    Accepts raw 11-channel EMG signals, processes them through the DSP chain,
    extracts time-domain features, and runs the PyCaret regressor to predict the knee angle.
    """
    import time
    start_time = time.perf_counter()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Prediction service unavailable: Model not loaded.")
    
    # 1. Convert input to numpy array
    raw_data = np.array(request.emg_window)
    
    # 2. Validate shapes
    if len(raw_data.shape) != 2 or raw_data.shape[1] != 11:
        raise HTTPException(
            status_code=422, 
            detail=f"Invalid data shape. Expected (N_samples, 11). Got {raw_data.shape}"
        )
        
    if raw_data.shape[0] < 200:
        raise HTTPException(
            status_code=422,
            detail=f"EMG window too short. Must contain at least 200 samples to support decimation and feature extraction. Got {raw_data.shape[0]}"
        )
        
    try:
        # 3. DSP preprocessing and feature extraction
        # Since we are processing a single window, we enforce return_features=True
        dsp_results = pipeline.process_raw_emg(raw_data, return_features=True)
        features = dsp_results['features']
        
        if features is None or features.size == 0:
            raise ValueError("DSP pipeline failed to extract features. Ensure window size is appropriate.")
            
        # If multiple windows were returned, we take the average or the last window's features
        # In real-time single-window prediction, features has shape (1, 33)
        features_vector = features[-1, :]
        
        # 4. Predict using the PyCaret model
        # PyCaret expects a Pandas DataFrame matching the exact training feature names
        feature_names = pipeline.get_feature_names()
        features_df = pd.DataFrame([features_vector], columns=feature_names)
        
        if isinstance(model, dict) and "model" in model:
            X = model["scaler"].transform(features_df[model["feature_names"]].values)
            predicted_angle = float(model["model"].predict(X)[0])
        else:
            from pycaret.regression import predict_model
            predictions_df = predict_model(model, data=features_df)
            predicted_angle = float(predictions_df['prediction_label'].iloc[0])
        
        # Measure latency
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        
        # Package a few representative features for response logging
        sample_features = {
            "gastrocmed_mav": float(features_df["gastrocmed_mav"].iloc[0]),
            "vastusmedialis_rms": float(features_df["vastusmedialis_rms"].iloc[0]),
            "bicepsfemoris_wl": float(features_df["bicepsfemoris_wl"].iloc[0]),
        }
        
        return PredictionResponse(
            predicted_knee_angle=predicted_angle,
            latency_ms=latency_ms,
            features_extracted=sample_features
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference pipeline execution error: {e}")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
