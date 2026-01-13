"""Model loading and inference for D9c AsymmetricGatedFusion."""

from models.loader import (
    AsymmetricGatedFusionHead,
    ModelCache,
    get_model_cache,
)
from models.inference import (
    extract_mert_embeddings,
    extract_muq_embeddings,
    predict_with_fusion_ensemble,
)

__all__ = [
    "AsymmetricGatedFusionHead",
    "ModelCache",
    "get_model_cache",
    "extract_mert_embeddings",
    "extract_muq_embeddings",
    "predict_with_fusion_ensemble",
]
