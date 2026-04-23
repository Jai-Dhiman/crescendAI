"""Audio experiments models.

Only MuQ model classes are retained; masterclass_experiments.features imports
MuQStatsModel to load baseline MuQ checkpoints. Other Model v1 classes
(MERT, Mel CNN, Fusion, Contrastive, Phase3, Probes) are archived.
"""

from .muq_models import (
    MuQBaseModel,
    MuQStatsModel,
    MERTMuQEnsemble,
    MERTMuQConcatModel,
    AsymmetricGatedFusion,
)

__all__ = [
    "MuQBaseModel",
    "MuQStatsModel",
    "MERTMuQEnsemble",
    "MERTMuQConcatModel",
    "AsymmetricGatedFusion",
]
