"""RunPod serverless handler for piano performance analysis.

This handler loads three models and runs inference:
1. MERT-330M (audio) - Primary audio analysis
2. PercePiano (symbolic) - Uses pre-computed or synthetic predictions
3. Late Fusion - Combines audio and symbolic
"""

import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import runpod

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
)

# Global state
_symbolic_predictions: Optional[SymbolicPredictions] = None
_fusion_weights: Optional[Dict[str, float]] = None


def initialize():
    """Initialize models and load resources. Called once on container start."""
    global _symbolic_predictions, _fusion_weights

    print("Initializing inference handler...")
    checkpoint_dir = Path("/app/checkpoints")

    # Initialize model cache (loads MERT and MLP heads)
    cache = get_model_cache()
    cache.initialize(device="cuda", checkpoint_dir=checkpoint_dir)

    # Load pre-computed symbolic predictions if available
    predictions_path = checkpoint_dir / "symbolic_predictions.json"
    if predictions_path.exists():
        _symbolic_predictions = SymbolicPredictions(predictions_path)
    else:
        print("No pre-computed symbolic predictions found")
        _symbolic_predictions = SymbolicPredictions()

    # Load fusion weights
    weights_path = checkpoint_dir / "fusion" / "optimal_weights.json"
    _fusion_weights = load_fusion_weights(weights_path)

    print("Initialization complete!")


def predictions_to_dict(preds: np.ndarray) -> Dict[str, float]:
    """Convert prediction array to dimension dict."""
    return {dim: float(preds[i]) for i, dim in enumerate(PERCEPIANO_DIMENSIONS)}


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod serverless handler.

    Input schema:
    {
        "input": {
            "audio_url": "https://...",
            "performance_id": "optional-id",
            "options": {
                "return_intermediate": false,
                "max_duration_seconds": 300
            }
        }
    }

    Output schema:
    {
        "predictions": {
            "fusion": {"timing": 0.85, ...},
            "audio": {...},      # if return_intermediate
            "symbolic": {...}    # if return_intermediate
        },
        "model_info": {...},
        "processing_time_ms": 1234
    }
    """
    start_time = time.time()
    job_input = job.get("input", {})

    try:
        # Parse input
        audio_url = job_input.get("audio_url")
        performance_id = job_input.get("performance_id", "unknown")
        options = job_input.get("options", {})
        return_intermediate = options.get("return_intermediate", False)
        max_duration = options.get("max_duration_seconds", 300)

        if not audio_url:
            return {
                "error": {
                    "code": "MISSING_AUDIO_URL",
                    "message": "audio_url is required",
                }
            }

        # Step 1: Download and preprocess audio
        print(f"Downloading audio from {audio_url}...")
        audio, duration = download_and_preprocess_audio(
            audio_url,
            max_duration=max_duration,
        )
        print(f"Audio loaded: {duration:.1f}s")

        # Step 2: Get model cache
        cache = get_model_cache()
        if not cache.mert_model:
            return {
                "error": {
                    "code": "MODEL_NOT_LOADED",
                    "message": "Models not initialized",
                }
            }

        # Step 3: Extract MERT embeddings
        print("Extracting MERT embeddings...")
        embeddings = extract_mert_embeddings(audio, cache)
        print(f"Embeddings shape: {embeddings.shape}")

        # Step 4: Get audio predictions (MERT ensemble)
        print("Running MERT ensemble inference...")
        audio_preds = predict_with_mert_ensemble(embeddings, cache)

        # Step 5: Get symbolic predictions
        print("Getting symbolic predictions...")
        symbolic_preds, is_real_symbolic = predict_with_symbolic_model(
            sample_key=performance_id,
            audio_preds=audio_preds,
            precomputed=_symbolic_predictions,
        )
        if is_real_symbolic:
            print("Using pre-computed symbolic predictions")
        else:
            print("Using synthetic symbolic predictions")

        # Step 6: Apply late fusion
        print("Applying late fusion...")
        fusion_preds = late_fusion(audio_preds, symbolic_preds, _fusion_weights)

        # Build response
        processing_time_ms = int((time.time() - start_time) * 1000)

        result = {
            "performance_id": performance_id,
            "predictions": {
                "fusion": predictions_to_dict(fusion_preds),
            },
            "model_info": MODEL_INFO,
            "symbolic_is_real": is_real_symbolic,
            "audio_duration_seconds": duration,
            "processing_time_ms": processing_time_ms,
        }

        # Include intermediate predictions if requested
        if return_intermediate:
            result["predictions"]["audio"] = predictions_to_dict(audio_preds)
            result["predictions"]["symbolic"] = predictions_to_dict(symbolic_preds)

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


# Initialize on import (container start)
initialize()

# Start RunPod serverless
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
