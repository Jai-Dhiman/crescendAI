"""Core functionality for CrescendAI model"""

from src.core.audio_preprocessing import PianoAudioPreprocessor

# Conditional import of training to avoid uvloop/orbax conflicts in production
try:
    from src.core.training import ASTTrainingPipeline
    __all__ = ["PianoAudioPreprocessor", "ASTTrainingPipeline"]
except ImportError as e:
    # Training modules not available (e.g., in production deployment)
    __all__ = ["PianoAudioPreprocessor"]
    ASTTrainingPipeline = None