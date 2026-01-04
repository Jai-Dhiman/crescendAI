"""PercePiano data loading and processing."""

from .percepiano_dataset import (
    PercePianoDataset,
    create_dataloaders,
    DIMENSIONS,
)
from .percepiano_vnet_dataset import (
    PercePianoVNetDataset,
    PercePianoVNetDataModule,
    create_vnet_dataloaders,
    PercePianoKFoldDataset,
    PercePianoKFoldDataModule,
    PercePianoTestDataset,
)
from .percepiano_score_dataset import (
    PercePianoScoreDataset,
    create_score_dataloaders,
)
from .virtuosonet_feature_extractor import VirtuosoNetFeatureExtractor
from .kfold_split import (
    extract_piece_id,
    create_piece_based_folds,
    save_fold_assignments,
    load_fold_assignments,
    get_fold_samples,
    get_test_samples,
    print_fold_statistics,
)
from .audio_dataset import (
    AudioPercePianoDataset,
    AudioPercePianoDataModule,
    audio_collate_fn,
    create_audio_fold_assignments,
    PERCEPIANO_DIMENSIONS,
    DIMENSION_CATEGORIES,
)

__all__ = [
    "PercePianoDataset",
    "create_dataloaders",
    "DIMENSIONS",
    "PercePianoVNetDataset",
    "PercePianoVNetDataModule",
    "create_vnet_dataloaders",
    "PercePianoKFoldDataset",
    "PercePianoKFoldDataModule",
    "PercePianoTestDataset",
    "PercePianoScoreDataset",
    "create_score_dataloaders",
    "VirtuosoNetFeatureExtractor",
    "extract_piece_id",
    "create_piece_based_folds",
    "save_fold_assignments",
    "load_fold_assignments",
    "get_fold_samples",
    "get_test_samples",
    "print_fold_statistics",
    # Audio dataset
    "AudioPercePianoDataset",
    "AudioPercePianoDataModule",
    "audio_collate_fn",
    "create_audio_fold_assignments",
    "PERCEPIANO_DIMENSIONS",
    "DIMENSION_CATEGORIES",
]
