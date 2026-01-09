"""HuggingFace Inference Endpoints handler for piano performance analysis.

This handler loads three models and runs inference:
1. MERT-330M (audio) - Primary audio analysis
2. PercePiano (symbolic) - Uses pre-computed or synthetic predictions
3. Late Fusion - Combines audio and symbolic

Compatible with HuggingFace Inference Endpoints custom handler pattern.
"""

import base64
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

from constants import MODEL_INFO, PERCEPIANO_DIMENSIONS
from models.loader import get_model_cache
from models.mert_inference import extract_mert_embeddings, predict_with_mert_ensemble
from models.symbolic_inference import (
    SymbolicPredictions,
    predict_with_symbolic_model,
)
from models.fusion import late_fusion, load_fusion_weights
from preprocessing.audio import (
    AudioDownloadError,
    AudioProcessingError,
    download_and_preprocess_audio,
    preprocess_audio_from_bytes,
)


class EndpointHandler:
    """HuggingFace Inference Endpoints handler for piano performance analysis."""

    def __init__(self, path: str = ""):
        """Initialize models and resources.

        Called once when the endpoint container starts.

        Args:
            path: Path to the model repository (provided by HF Inference Endpoints).
                  Contains the checkpoints/ directory with model weights.
        """
        print(f"Initializing EndpointHandler with path: {path}")

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

        # Initialize model cache (loads MERT and MLP heads)
        self._cache = get_model_cache()
        self._cache.initialize(device="cuda", checkpoint_dir=checkpoint_dir)

        # Load pre-computed symbolic predictions if available
        predictions_path = checkpoint_dir / "symbolic_predictions.json"
        if predictions_path.exists():
            self._symbolic_predictions: Optional[SymbolicPredictions] = SymbolicPredictions(
                predictions_path
            )
            print(f"Loaded symbolic predictions from {predictions_path}")
        else:
            print("No pre-computed symbolic predictions found, using synthetic fallback")
            self._symbolic_predictions = SymbolicPredictions()

        # Load fusion weights
        weights_path = checkpoint_dir / "fusion" / "optimal_weights.json"
        self._fusion_weights = load_fusion_weights(weights_path)

        print("EndpointHandler initialization complete!")

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process inference request.

        Args:
            data: Request payload. Supports two formats:

                HuggingFace format:
                {
                    "inputs": "<base64-audio>" or {"audio_url": "...", "performance_id": "..."},
                    "parameters": {
                        "performance_id": "optional-id",
                        "return_intermediate": false,
                        "max_duration_seconds": 300
                    }
                }

                Legacy RunPod format (for backward compatibility):
                {
                    "input": {
                        "audio_url": "https://...",
                        "performance_id": "optional-id",
                        "options": {...}
                    }
                }

        Returns:
            Prediction results:
            {
                "predictions": {
                    "fusion": {"timing": 0.85, ...},
                    "audio": {...},      # if return_intermediate
                    "symbolic": {...}    # if return_intermediate
                },
                "model_info": {...},
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
            performance_id = parameters.get("performance_id", "unknown")
            return_intermediate = parameters.get("return_intermediate", False)
            max_duration = parameters.get("max_duration_seconds", 300)

            # Load and preprocess audio
            audio, duration = self._load_audio(inputs, max_duration)
            print(f"Audio loaded: {duration:.1f}s")

            # Verify models are loaded
            if not self._cache.mert_model:
                return {
                    "error": {
                        "code": "MODEL_NOT_LOADED",
                        "message": "Models not initialized",
                    }
                }

            # Extract MERT embeddings
            print("Extracting MERT embeddings...")
            embeddings = extract_mert_embeddings(audio, self._cache)
            print(f"Embeddings shape: {embeddings.shape}")

            # Get audio predictions (MERT ensemble)
            print("Running MERT ensemble inference...")
            audio_preds = predict_with_mert_ensemble(embeddings, self._cache)

            # Get symbolic predictions
            print("Getting symbolic predictions...")
            symbolic_preds, is_real_symbolic = predict_with_symbolic_model(
                sample_key=performance_id,
                audio_preds=audio_preds,
                precomputed=self._symbolic_predictions,
            )
            if is_real_symbolic:
                print("Using pre-computed symbolic predictions")
            else:
                print("Using synthetic symbolic predictions")

            # Apply late fusion
            print("Applying late fusion...")
            fusion_preds = late_fusion(audio_preds, symbolic_preds, self._fusion_weights)

            # Build response
            processing_time_ms = int((time.time() - start_time) * 1000)

            result = {
                "performance_id": performance_id,
                "predictions": {
                    "fusion": self._predictions_to_dict(fusion_preds),
                },
                "model_info": MODEL_INFO,
                "symbolic_is_real": is_real_symbolic,
                "audio_duration_seconds": duration,
                "processing_time_ms": processing_time_ms,
            }

            # Include intermediate predictions if requested
            if return_intermediate:
                result["predictions"]["audio"] = self._predictions_to_dict(audio_preds)
                result["predictions"]["symbolic"] = self._predictions_to_dict(symbolic_preds)

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
                raise AudioProcessingError(f"Invalid input string: not base64 or URL")

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
