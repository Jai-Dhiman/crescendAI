"""Feature extractors for audio experiments.

Only the MuQ extractor is retained as active infrastructure; MERT/Mel/
statistics extractors have been archived with the Model v1 experiments.
"""

from .muq import (
    MuQExtractor,
    extract_muq_embeddings,
)

__all__ = [
    "MuQExtractor",
    "extract_muq_embeddings",
]
