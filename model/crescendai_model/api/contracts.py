#!/usr/bin/env python3
"""
API Contracts for CrescendAI Multi-Service Architecture
Defines data structures and interfaces for Cloudflare Workers <-> Modal integration
"""

from typing import Dict, List, Optional, Union, Any, Literal
from pydantic import BaseModel, Field, validator
from enum import Enum
import time

# ==================== ENUMS ====================

class ProcessingStatus(str, Enum):
    """Processing status enumeration"""
    SUCCESS = "success"
    ERROR = "error"
    PROCESSING = "processing"
    TIMEOUT = "timeout"

class AudioFormat(str, Enum):
    """Supported audio formats"""
    WAV = "wav"
    MP3 = "mp3"
    M4A = "m4a"
    FLAC = "flac"

class ErrorType(str, Enum):
    """Error type categorization"""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    TIMEOUT_ERROR = "timeout_error"
    SERVICE_ERROR = "service_error"
    MODEL_ERROR = "model_error"

# ==================== REQUEST MODELS ====================

class AudioUploadRequest(BaseModel):
    """Request model for audio upload to Cloudflare Workers"""
    
    audio_data: str = Field(..., description="Base64 encoded audio data")
    audio_format: AudioFormat = Field(..., description="Audio file format")
    client_metadata: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional metadata from client (iOS/web)"
    )
    
    @validator('audio_data')
    def validate_audio_data(cls, v):
        if not v or len(v) < 100:  # Minimum reasonable size
            raise ValueError("Audio data is too short or empty")
        return v

class PreprocessingRequest(BaseModel):
    """Request model for audio preprocessing in Cloudflare Workers"""
    
    audio_bytes: bytes = Field(..., description="Raw audio bytes")
    target_sample_rate: int = Field(default=22050, description="Target sample rate")
    max_duration: float = Field(default=180.0, description="Maximum audio duration in seconds")
    
class ModalInferenceRequest(BaseModel):
    """Request model for Modal inference service"""
    
    mel_spectrogram: List[List[float]] = Field(
        ..., 
        description="2D mel-spectrogram array [time_frames, mel_bands]"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional processing metadata"
    )
    
    @validator('mel_spectrogram')
    def validate_mel_spectrogram(cls, v):
        if not v or len(v) == 0:
            raise ValueError("Mel-spectrogram cannot be empty")
        
        # Check for consistent dimensions
        mel_bands = len(v[0]) if v else 0
        if mel_bands != 128:
            raise ValueError(f"Expected 128 mel bands, got {mel_bands}")
        
        for i, frame in enumerate(v):
            if len(frame) != mel_bands:
                raise ValueError(f"Inconsistent mel bands at frame {i}")
        
        return v

# ==================== RESPONSE MODELS ====================

class PerformanceDimensions(BaseModel):
    """19-dimensional piano performance analysis results"""
    
    # Timing dimension
    Timing_Stable_Unstable: float = Field(..., ge=0.0, le=1.0)
    
    # Articulation dimensions
    Articulation_Short_Long: float = Field(..., ge=0.0, le=1.0)
    Articulation_Soft_cushioned_Hard_solid: float = Field(..., ge=0.0, le=1.0)
    
    # Dynamic dimensions
    Dynamic_Sophisticated_mellow_Raw_crude: float = Field(..., ge=0.0, le=1.0, alias="Dynamic_Sophisticated/mellow_Raw/crude")
    Dynamic_Little_dynamic_range_Large_dynamic_range: float = Field(..., ge=0.0, le=1.0)
    
    # Musical expression dimensions
    Music_Making_Fast_paced_Slow_paced: float = Field(..., ge=0.0, le=1.0)
    Music_Making_Flat_Spacious: float = Field(..., ge=0.0, le=1.0)
    Music_Making_Not_sensitive_Very_sensitive: float = Field(..., ge=0.0, le=1.0)
    Music_Making_Unimaginative_Imaginative: float = Field(..., ge=0.0, le=1.0)
    Music_Making_Inarticulated_Well_articulated: float = Field(..., ge=0.0, le=1.0)
    Music_Making_Unexpressive_Very_expressive: float = Field(..., ge=0.0, le=1.0)
    Music_Making_Unvaried_Extremely_varied: float = Field(..., ge=0.0, le=1.0)
    Music_Making_Technical_Musical: float = Field(..., ge=0.0, le=1.0)
    
    # Pedal dimension
    Pedal_Appropriate_Inappropriate: float = Field(..., ge=0.0, le=1.0)
    
    # Timbre dimension
    Timbre_Incoherent_Coherent: float = Field(..., ge=0.0, le=1.0)
    
    # Overall assessment dimensions
    Overall_Unpleasant_Pleasant: float = Field(..., ge=0.0, le=1.0)
    Overall_Boring_Interesting: float = Field(..., ge=0.0, le=1.0)
    Overall_Emotionless_Emotional: float = Field(..., ge=0.0, le=1.0)
    Overall_Not_impressive_Very_impressive: float = Field(..., ge=0.0, le=1.0)
    
    class Config:
        allow_population_by_field_name = True

class ProcessingMetadata(BaseModel):
    """Metadata about processing pipeline"""
    
    model_version: str = Field(default="AST-19D-v1.0")
    processing_time_seconds: float = Field(..., ge=0.0)
    input_shape: Optional[List[int]] = Field(default=None)
    timestamp: float = Field(default_factory=time.time)
    service_chain: List[str] = Field(default_factory=lambda: ["cloudflare", "modal"])

class PreprocessingResponse(BaseModel):
    """Response from Cloudflare Workers preprocessing"""
    
    status: ProcessingStatus
    mel_spectrogram: Optional[List[List[float]]] = Field(default=None)
    metadata: Optional[Dict[str, Any]] = Field(default=None)
    error: Optional[str] = Field(default=None)
    error_type: Optional[ErrorType] = Field(default=None)

class ModalInferenceResponse(BaseModel):
    """Response from Modal inference service"""
    
    status: ProcessingStatus
    results: Optional[PerformanceDimensions] = Field(default=None)
    metadata: Optional[ProcessingMetadata] = Field(default=None)
    error: Optional[str] = Field(default=None)
    error_type: Optional[ErrorType] = Field(default=None)

class FinalAnalysisResponse(BaseModel):
    """Final response to client applications (iOS/Web)"""
    
    status: ProcessingStatus
    analysis_id: str = Field(..., description="Unique analysis identifier")
    performance_analysis: Optional[PerformanceDimensions] = Field(default=None)
    processing_metadata: Optional[ProcessingMetadata] = Field(default=None)
    error: Optional[str] = Field(default=None)
    error_type: Optional[ErrorType] = Field(default=None)
    
    # Client-specific metadata
    audio_duration_seconds: Optional[float] = Field(default=None)
    file_size_bytes: Optional[int] = Field(default=None)

# ==================== HEALTH CHECK MODELS ====================

class HealthCheckResponse(BaseModel):
    """Health check response model"""
    
    status: Literal["healthy", "unhealthy"]
    service: str
    version: str
    timestamp: float
    dependencies: Optional[Dict[str, str]] = Field(default=None)

class ServiceStatus(BaseModel):
    """Individual service status"""
    
    service_name: str
    status: Literal["healthy", "unhealthy", "unknown"]
    response_time_ms: Optional[float] = Field(default=None)
    last_check: float = Field(default_factory=time.time)

class SystemHealthResponse(BaseModel):
    """Overall system health response"""
    
    status: Literal["healthy", "degraded", "unhealthy"]
    services: List[ServiceStatus]
    timestamp: float = Field(default_factory=time.time)

# ==================== ERROR MODELS ====================

class APIError(BaseModel):
    """Standardized API error response"""
    
    status: Literal["error"] = "error"
    error_type: ErrorType
    error_message: str
    error_code: Optional[str] = Field(default=None)
    timestamp: float = Field(default_factory=time.time)
    request_id: Optional[str] = Field(default=None)

# ==================== CONFIGURATION MODELS ====================

class CloudflareWorkerConfig(BaseModel):
    """Configuration for Cloudflare Workers service"""
    
    modal_endpoint: str = Field(..., description="Modal service endpoint URL")
    modal_api_key: Optional[str] = Field(default=None, description="Modal API key")
    max_file_size_mb: float = Field(default=50.0, description="Maximum upload file size")
    max_processing_time_seconds: float = Field(default=30.0, description="Maximum processing timeout")
    
class ModalServiceConfig(BaseModel):
    """Configuration for Modal service"""
    
    gpu_type: str = Field(default="A10G", description="GPU type for inference")
    container_timeout_seconds: int = Field(default=300, description="Container timeout")
    container_idle_timeout_seconds: int = Field(default=60, description="Idle timeout for cold loading")
    memory_mb: int = Field(default=8192, description="Container memory allocation")

# ==================== UTILITY FUNCTIONS ====================

def create_error_response(
    error_type: ErrorType, 
    message: str, 
    request_id: Optional[str] = None
) -> APIError:
    """Create standardized error response"""
    return APIError(
        error_type=error_type,
        error_message=message,
        request_id=request_id
    )

def validate_audio_duration(duration: float, max_duration: float = 180.0) -> bool:
    """Validate audio duration against limits"""
    return 5.0 <= duration <= max_duration

def create_analysis_id() -> str:
    """Generate unique analysis identifier"""
    import uuid
    return f"analysis_{uuid.uuid4().hex[:12]}"

# ==================== CONSTANTS ====================

# Dimension name mapping for backward compatibility
DIMENSION_MAPPING = {
    "timing": "Timing_Stable_Unstable",
    "articulation_length": "Articulation_Short_Long", 
    "articulation_hardness": "Articulation_Soft_cushioned_Hard_solid",
    "dynamic_sophistication": "Dynamic_Sophisticated/mellow_Raw/crude",
    "dynamic_range": "Dynamic_Little_dynamic_range_Large_dynamic_range",
    "music_pace": "Music_Making_Fast_paced_Slow_paced",
    "music_space": "Music_Making_Flat_Spacious",
    "music_sensitivity": "Music_Making_Not_sensitive_Very_sensitive",
    "music_imagination": "Music_Making_Unimaginative_Imaginative",
    "music_articulation": "Music_Making_Inarticulated_Well_articulated",
    "music_expression": "Music_Making_Unexpressive_Very_expressive",
    "music_variation": "Music_Making_Unvaried_Extremely_varied",
    "music_technical": "Music_Making_Technical_Musical",
    "pedal": "Pedal_Appropriate_Inappropriate",
    "timbre": "Timbre_Incoherent_Coherent",
    "overall_pleasant": "Overall_Unpleasant_Pleasant",
    "overall_interesting": "Overall_Boring_Interesting",
    "overall_emotional": "Overall_Emotionless_Emotional",
    "overall_impressive": "Overall_Not_impressive_Very_impressive"
}

# Default configuration values
DEFAULT_CONFIG = {
    "target_sample_rate": 22050,
    "n_mels": 128,
    "max_audio_duration": 180.0,
    "min_audio_duration": 5.0,
    "max_file_size_mb": 50.0,
    "processing_timeout_seconds": 30.0
}

if __name__ == "__main__":
    # Test model creation
    print("🎹 Testing API contract models...")
    
    # Test performance dimensions
    sample_results = {name: 0.5 for name in DIMENSION_MAPPING.values()}
    dimensions = PerformanceDimensions(**sample_results)
    print(f"✅ Performance dimensions: {len(dimensions.dict())} fields")
    
    # Test final response
    response = FinalAnalysisResponse(
        status=ProcessingStatus.SUCCESS,
        analysis_id=create_analysis_id(),
        performance_analysis=dimensions,
        processing_metadata=ProcessingMetadata(processing_time_seconds=2.5)
    )
    print(f"✅ Final response: {response.status}")
    
    print("🚀 API contracts ready for integration!")