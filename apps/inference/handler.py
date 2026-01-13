"""HuggingFace Inference Endpoints handler for piano performance analysis.

D9c AsymmetricGatedFusion model using MERT+MuQ with per-dimension gating.
Returns 19-dimension performance evaluation scores.

Compatible with HuggingFace Inference Endpoints custom handler pattern.
"""

import base64
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np

from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
from models.loader import get_model_cache
from models.inference import (
    extract_mert_embeddings,
    extract_muq_embeddings,
    predict_with_fusion_ensemble,
)
from preprocessing.audio import (
    AudioDownloadError,
    AudioProcessingError,
    download_and_preprocess_audio,
    preprocess_audio_from_bytes,
)


class EndpointHandler:
    """HuggingFace Inference Endpoints handler for piano performance analysis."""

    def __init__(self, path: str = ""):
        """Initialize MERT, MuQ, and fusion models.

        Called once when the endpoint container starts.

        Args:
            path: Path to the model repository (provided by HF Inference Endpoints).
                  Contains the checkpoints/ directory with model weights.
        """
        print(f"Initializing D9c EndpointHandler with path: {path}")

        # Determine checkpoint directory
        # HF Inference Endpoints mount the repo at the provided path
        # Fall back to /repository (HF default) or current dir for local testing
        if path:
            model_path = Path(path)
        else:
            model_path = Path("/repository")
            if not model_path.exists():
                model_path = Path(".")

        checkpoint_dir = model_path / "checkpoints"
        if not checkpoint_dir.exists():
            # Try /app/checkpoints for backward compatibility
            checkpoint_dir = Path("/app/checkpoints")

        print(f"Using checkpoint directory: {checkpoint_dir}")

        # Initialize model cache (loads MERT, MuQ, and fusion heads)
        self._cache = get_model_cache()
        self._cache.initialize(device="cuda", checkpoint_dir=checkpoint_dir)

        print("D9c EndpointHandler initialization complete!")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference request.

        Args:
            data: Request payload. Supports two formats:

                HuggingFace format:
                {
                    "inputs": "<base64-audio>" or {"audio_url": "..."},
                    "parameters": {
                        "max_duration_seconds": 300
                    }
                }

                Legacy RunPod format (for backward compatibility):
                {
                    "input": {
                        "audio_url": "https://...",
                        "options": {...}
                    }
                }

        Returns:
            Prediction results:
            {
                "predictions": {"timing": 0.85, ...},
                "model_info": {"name": "D9c-AsymmetricGatedFusion", "r2": 0.531},
                "audio_duration_seconds": 180.5,
                "processing_time_ms": 1234
            }

            Or error:
            {
                "error": {"code": "...", "message": "..."}
            }
        """
        start_time = time.time()

        try:
            # Parse input - support both HF and legacy RunPod formats
            inputs, parameters = self._parse_request(data)

            # Extract parameters
            max_duration = parameters.get("max_duration_seconds", 300)

            # Load and preprocess audio
            audio, duration = self._load_audio(inputs, max_duration)
            print(f"Audio loaded: {duration:.1f}s")

            # Verify models are loaded
            if not self._cache.mert_model or not self._cache.muq_model:
                return {
                    "error": {
                        "code": "MODEL_NOT_LOADED",
                        "message": "Models not initialized",
                    }
                }

            # Extract MERT embeddings (concatenated layers 19-24)
            print("Extracting MERT embeddings...")
            mert_embeddings = extract_mert_embeddings(audio, self._cache)
            print(f"MERT embeddings shape: {mert_embeddings.shape}")

            # Extract MuQ embeddings
            print("Extracting MuQ embeddings...")
            muq_embeddings = extract_muq_embeddings(audio, self._cache)
            print(f"MuQ embeddings shape: {muq_embeddings.shape}")

            # Get fused predictions (4-fold ensemble)
            print("Running D9c fusion ensemble inference...")
            predictions = predict_with_fusion_ensemble(
                mert_embeddings, muq_embeddings, self._cache
            )

            # Build response
            processing_time_ms = int((time.time() - start_time) * 1000)

            result = {
                "predictions": self._predictions_to_dict(predictions),
                "model_info": {
                    "name": MODEL_INFO["name"],
                    "type": MODEL_INFO["type"],
                    "r2": MODEL_INFO["r2"],
                    "architecture": MODEL_INFO["architecture"],
                    "ensemble_folds": len(self._cache.fusion_heads),
                },
                "audio_duration_seconds": duration,
                "processing_time_ms": processing_time_ms,
            }

            print(f"Inference complete in {processing_time_ms}ms")
            return result

        except AudioDownloadError as e:
            return {
                "error": {
                    "code": "AUDIO_DOWNLOAD_FAILED",
                    "message": str(e),
                }
            }

        except AudioProcessingError as e:
            return {
                "error": {
                    "code": "AUDIO_PROCESSING_FAILED",
                    "message": str(e),
                }
            }

        except Exception as e:
            return {
                "error": {
                    "code": "INFERENCE_ERROR",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            }

    def _parse_request(self, data: Dict[str, Any]) -> tuple:
        """Parse request data supporting both HF and legacy formats.

        Returns:
            Tuple of (inputs, parameters)
        """
        # HF format: {"inputs": ..., "parameters": ...}
        if "inputs" in data:
            inputs = data["inputs"]
            parameters = data.get("parameters", {})
            return inputs, parameters

        # Legacy RunPod format: {"input": {"audio_url": ..., "options": ...}}
        if "input" in data:
            job_input = data["input"]
            inputs = {
                "audio_url": job_input.get("audio_url"),
                "performance_id": job_input.get("performance_id", "unknown"),
            }
            parameters = job_input.get("options", {})
            parameters["performance_id"] = inputs.get("performance_id", "unknown")
            return inputs, parameters

        # Fallback: treat entire data as inputs
        return data, {}

    def _load_audio(
        self, inputs: Union[str, bytes, Dict[str, Any]], max_duration: int
    ) -> tuple:
        """Load audio from various input formats.

        Args:
            inputs: One of:
                - str: Base64-encoded audio bytes
                - bytes: Raw audio bytes
                - dict: {"audio_url": "..."} for URL-based loading

        Returns:
            Tuple of (audio_array, duration_seconds)
        """
        if isinstance(inputs, str):
            # Base64-encoded audio
            try:
                audio_bytes = base64.b64decode(inputs)
                return preprocess_audio_from_bytes(audio_bytes, max_duration=max_duration)
            except Exception:
                # Maybe it's a URL string
                if inputs.startswith("http"):
                    return download_and_preprocess_audio(inputs, max_duration=max_duration)
                raise AudioProcessingError("Invalid input string: not base64 or URL")

        elif isinstance(inputs, bytes):
            # Raw bytes
            return preprocess_audio_from_bytes(inputs, max_duration=max_duration)

        elif isinstance(inputs, dict):
            # URL-based input
            audio_url = inputs.get("audio_url")
            if not audio_url:
                raise AudioProcessingError("No audio_url provided in inputs")
            return download_and_preprocess_audio(audio_url, max_duration=max_duration)

        else:
            raise AudioProcessingError(f"Unsupported input type: {type(inputs)}")

    def _predictions_to_dict(self, preds: np.ndarray) -> Dict[str, float]:
        """Convert prediction array to dimension dict."""
        return {dim: float(preds[i]) for i, dim in enumerate(PERCEPIANO_DIMENSIONS)}
