#!/usr/bin/env python3
"""
Mock Model Service for CrescendAI
Provides a simple FastAPI endpoint that returns sample analysis results
"""

import random
import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn


app = FastAPI(
    title="CrescendAI Mock Model Service",
    description="Mock ML model endpoint for piano analysis",
    version="1.0.0"
)

class SpectrogramRequest(BaseModel):
    """Request schema for spectrogram analysis"""
    file_id: str = Field(..., description="Unique identifier for the audio file")
    spectrogram_data: str = Field(..., description="Base64 encoded spectrogram data")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

class AnalysisResponse(BaseModel):
    """Response schema for analysis results"""
    file_id: str
    analysis: Dict[str, float]
    insights: list[str]
    processing_time: float
    model_version: str = "mock-v1.0.0"

# Sample insights pool for random selection
SAMPLE_INSIGHTS = [
    "Excellent rhythm consistency throughout the piece",
    "Consider more dynamic contrast in the middle section",
    "Great pedal technique, especially in the legato passages",
    "Tempo variations add nice musical expression",
    "Hand coordination is very well balanced",
    "Articulation could be clearer in the fast passages",
    "Beautiful phrasing in the melodic lines",
    "Technical execution shows solid practice foundation",
    "Musical interpretation demonstrates deep understanding",
    "Stage presence would benefit from more confident posture",
    "Creativity in ornamentation is noteworthy",
    "Listening skills show good ensemble awareness",
    "Stylistic choices align well with the period",
    "Voicing brings out the harmonic structure nicely",
    "Expression flows naturally with the musical line"
]

def generate_mock_analysis() -> Dict[str, float]:
    """Generate realistic-looking analysis scores"""
    # Base scores with some realistic variance
    base_scores = {
        "rhythm": random.uniform(75, 95),
        "pitch": random.uniform(80, 98),
        "dynamics": random.uniform(70, 90),
        "tempo": random.uniform(75, 92),
        "articulation": random.uniform(72, 88),
        "expression": random.uniform(78, 95),
        "technique": random.uniform(80, 94),
        "timing": random.uniform(74, 90),
        "phrasing": random.uniform(76, 93),
        "voicing": random.uniform(73, 89),
        "pedaling": random.uniform(71, 87),
        "hand_coordination": random.uniform(79, 95),
        "musical_understanding": random.uniform(82, 96),
        "stylistic_accuracy": random.uniform(75, 90),
        "creativity": random.uniform(70, 92),
        "listening": random.uniform(77, 91),
        "stage_presence": random.uniform(68, 85),
        "repertoire_difficulty": random.uniform(80, 95),
    }
    
    # Calculate overall as weighted average (not pure average)
    weights = {
        "technique": 0.15,
        "musical_understanding": 0.15,
        "expression": 0.12,
        "rhythm": 0.10,
        "pitch": 0.10,
        "timing": 0.08,
        "phrasing": 0.08,
        "dynamics": 0.07,
        "articulation": 0.05,
        "hand_coordination": 0.05,
        "pedaling": 0.05
    }
    
    overall = sum(base_scores[key] * weight for key, weight in weights.items())
    overall += sum(base_scores[key] * 0.01 for key in base_scores if key not in weights)
    
    base_scores["overall_performance"] = min(100, overall)
    
    return {key: round(value, 1) for key, value in base_scores.items()}

def select_random_insights(num_insights: int = 4) -> list[str]:
    """Select random insights for the analysis"""
    return random.sample(SAMPLE_INSIGHTS, min(num_insights, len(SAMPLE_INSIGHTS)))

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "CrescendAI Mock Model Service",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "mock-model",
        "ready": True
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_spectrogram(request: SpectrogramRequest):
    """
    Analyze a spectrogram and return mock piano performance results
    """
    start_time = time.time()
    
    # Simulate some processing time (1-3 seconds)
    processing_delay = random.uniform(1.0, 3.0)
    time.sleep(processing_delay)
    
    # Generate mock analysis
    analysis = generate_mock_analysis()
    insights = select_random_insights()
    
    processing_time = time.time() - start_time
    
    return AnalysisResponse(
        file_id=request.file_id,
        analysis=analysis,
        insights=insights,
        processing_time=round(processing_time, 2)
    )

@app.get("/models")
async def list_models():
    """List available models (mock)"""
    return {
        "models": [
            {
                "id": "mock-ast-v1",
                "name": "Mock Audio Spectrogram Transformer",
                "version": "1.0.0",
                "description": "Mock model for development and testing",
                "status": "active"
            }
        ]
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start CrescendAI Mock Model Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    print(f"🎹 Starting CrescendAI Mock Model Service...")
    print(f"📡 Listening on http://{args.host}:{args.port}")
    print(f"📚 API docs available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "mock_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )