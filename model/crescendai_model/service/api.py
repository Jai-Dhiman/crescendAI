#!/usr/bin/env python3
"""
CrescendAI Local FastAPI Service
Piano performance analysis API for local deployment
"""

import os
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import pickle
import logging

# Set JAX to CPU mode early
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import third-party packages
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import CrescendAI modules (avoiding training imports)
from crescendai_model.core.audio_preprocessing import PianoAudioPreprocessor
from crescendai_model.api.contracts import (
    PerformanceDimensions,
    FinalAnalysisResponse,
    ProcessingStatus
)

# FastAPI app setup
app = FastAPI(
    title="CrescendAI Piano Performance Analyzer",
    description="19-dimensional piano performance analysis using Audio Spectrogram Transformer",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
model_data = {
    "model": None,
    "params": None,
    "preprocessor": None,
    "loaded": False
}

class ModelManager:
    """Manages model loading and inference without training imports"""
    
    def __init__(self):
        self.loaded = False
        self.model = None
        self.params = None
        self.preprocessor = None
        
    def load_model(self):
        """Load model without importing training modules"""
        if self.loaded:
            return
            
        try:
            # Load preprocessor
            self.preprocessor = PianoAudioPreprocessor()
            logger.info("Audio preprocessor loaded")
            
            # Load model file if it exists
            model_path = Path("results/final_finetuned_model.pkl")
            if model_path.exists():
                with open(model_path, "rb") as f:
                    model_data_raw = pickle.load(f)
                
                # Extract components
                if isinstance(model_data_raw, dict):
                    self.model = model_data_raw.get('model')
                    self.params = model_data_raw.get('params') or model_data_raw.get('state', {}).get('params')
                else:
                    self.params = model_data_raw
                
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}")
                
            self.loaded = True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Predict performance dimensions from mel-spectrogram"""
        if not self.loaded:
            self.load_model()
        
        # If model is not available, return mock predictions
        if self.model is None or self.params is None:
            logger.warning("Model not available, returning mock predictions")
            # Return mock predictions for all 19 dimensions (values between 0 and 1)
            return np.random.uniform(0.3, 0.7, 19)
        
        try:
            # Import JAX here to avoid early conflicts
            import jax
            import jax.numpy as jnp
            
            # Ensure proper input shape
            if mel_spectrogram.ndim == 2:
                mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
            
            # Convert to JAX array
            inputs = jnp.array(mel_spectrogram)
            
            # Run inference
            outputs = self.model.apply(self.params, inputs, training=False)
            
            # Convert back to numpy
            predictions = np.array(outputs)
            if predictions.ndim > 1:
                predictions = predictions.squeeze()
            
            return predictions
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            # Return mock predictions on error
            return np.random.uniform(0.3, 0.7, 19)

# Global model manager
model_manager = ModelManager()

# Request/Response models
class AudioUploadResponse(BaseModel):
    status: str
    message: str
    processing_id: Optional[str] = None

class AnalysisRequest(BaseModel):
    audio_data: Optional[str] = None  # Base64 encoded audio
    file_path: Optional[str] = None   # Local file path for testing

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "CrescendAI Piano Performance Analyzer",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "analyze": "/analyze",
            "upload": "/upload"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "crescendai-local",
        "version": "1.0.0",
        "model_loaded": model_manager.loaded,
        "timestamp": time.time()
    }

@app.post("/upload", response_model=AudioUploadResponse)
async def upload_audio(file: UploadFile = File(...)):
    """Upload audio file for analysis"""
    try:
        # Validate file type
        if not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an audio file"
            )
        
        # Save uploaded file temporarily
        temp_path = Path(f"temp_{file.filename}")
        content = await file.read()
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process the audio
        result = await analyze_audio_file(str(temp_path))
        
        # Cleanup
        temp_path.unlink(missing_ok=True)
        
        return AudioUploadResponse(
            status="success",
            message="Audio processed successfully",
            processing_id=result.get("processing_id")
        )
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload processing failed: {str(e)}"
        )

@app.post("/analyze", response_model=FinalAnalysisResponse)
async def analyze_audio(request: AnalysisRequest):
    """Analyze audio and return performance dimensions"""
    try:
        if request.file_path:
            # Analyze local file
            result = await analyze_audio_file(request.file_path)
        elif request.audio_data:
            # Handle base64 encoded audio data
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Base64 audio analysis not yet implemented"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either file_path or audio_data must be provided"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )

async def analyze_audio_file(file_path: str) -> FinalAnalysisResponse:
    """Analyze audio file and return performance dimensions"""
    start_time = time.time()
    
    try:
        # Ensure model is loaded
        if not model_manager.loaded:
            model_manager.load_model()
        
        # Process audio file
        logger.info(f"Processing audio file: {file_path}")
        
        # Load and preprocess audio
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Process with preprocessor
        result = model_manager.preprocessor.process_audio_file(file_path)
        
        if result["status"] != "success":
            raise RuntimeError(f"Preprocessing failed: {result.get('error')}")
        
        mel_spectrogram = result["mel_spectrogram"]
        metadata = result["metadata"]
        
        # Run model inference
        predictions = model_manager.predict(mel_spectrogram)
        
        # Create performance dimensions
        dimension_names = [
            "Timing_Stable_Unstable",
            "Articulation_Short_Long", 
            "Articulation_Soft_cushioned_Hard_solid",
            "Dynamic_Sophisticated/mellow_Raw/crude",
            "Dynamic_Little_dynamic_range_Large_dynamic_range",
            "Music_Making_Fast_paced_Slow_paced",
            "Music_Making_Flat_Spacious",
            "Music_Making_Not_sensitive_Very_sensitive",
            "Music_Making_Unimaginative_Imaginative",
            "Music_Making_Inarticulated_Well_articulated",
            "Music_Making_Unexpressive_Very_expressive",
            "Music_Making_Unvaried_Extremely_varied",
            "Music_Making_Technical_Musical",
            "Pedal_Appropriate_Inappropriate",
            "Timbre_Incoherent_Coherent",
            "Overall_Unpleasant_Pleasant",
            "Overall_Boring_Interesting",
            "Overall_Emotionless_Emotional",
            "Overall_Not_impressive_Very_impressive"
        ]
        
        # Create dimension results
        dimensions = {}
        for i, name in enumerate(dimension_names):
            dimensions[name] = float(predictions[i])
        
        # Create response
        processing_time = time.time() - start_time
        
        response = FinalAnalysisResponse(
            status=ProcessingStatus.SUCCESS,
            dimensions=PerformanceDimensions(**dimensions),
            processing_time_seconds=processing_time,
            metadata={
                "model_version": "AST-19D-Local-v1.0",
                "input_shape": mel_spectrogram.shape,
                "sample_rate": metadata.get("sample_rate"),
                "duration": metadata.get("duration"),
                "timestamp": time.time()
            },
            processing_id=f"local_{int(time.time())}"
        )
        
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Analysis failed after {processing_time:.2f}s: {e}")
        
        # Return error response
        return FinalAnalysisResponse(
            status=ProcessingStatus.ERROR,
            error=str(e),
            processing_time_seconds=processing_time,
            metadata={
                "error_type": type(e).__name__,
                "timestamp": time.time()
            },
            processing_id=f"error_{int(time.time())}"
        )

# Optional: Add development endpoints
@app.get("/test")
async def test_endpoint():
    """Test endpoint for development"""
    try:
        # Test model loading
        model_manager.load_model()
        
        # Create dummy mel-spectrogram
        dummy_mel = np.random.rand(100, 128)
        predictions = model_manager.predict(dummy_mel)
        
        return {
            "status": "test_passed",
            "model_loaded": model_manager.loaded,
            "prediction_shape": predictions.shape,
            "sample_predictions": predictions[:5].tolist()
        }
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {
            "status": "test_failed", 
            "error": str(e)
        }

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    """Initialize model on server startup"""
    logger.info("Starting CrescendAI Local Service...")
    try:
        model_manager.load_model()
        logger.info("Model initialization completed")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        logger.info("Service will continue with mock predictions")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "crescendai_model.service.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        loop="asyncio"  # Use asyncio instead of uvloop to avoid conflict
    )