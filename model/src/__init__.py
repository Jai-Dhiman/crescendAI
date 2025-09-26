"""CrescendAI Piano Performance Analysis Model Package"""

__version__ = "1.0.0"
__author__ = "CrescendAI Team"
__description__ = "Audio Spectrogram Transformer for 19-dimensional piano performance analysis"

from src.core.audio_preprocessing import PianoAudioPreprocessor
from src.models.ast_transformer import AudioSpectrogramTransformer
from src.api.contracts import (
    PerformanceDimensions,
    FinalAnalysisResponse,
    ProcessingStatus
)

__all__ = [
    "PianoAudioPreprocessor",
    "AudioSpectrogramTransformer", 
    "PerformanceDimensions",
    "FinalAnalysisResponse",
    "ProcessingStatus",
]