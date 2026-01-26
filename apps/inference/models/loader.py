"""Model loading and caching for M1c MuQ L9-12 inference."""

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from constants import MODEL_CONFIG, N_FOLDS


class MuQStatsHead(nn.Module):
    """Inference-only version of MuQStatsModel head.

    Loads trained weights and runs inference. Architecture must match training:
    - Input: pooled_dim (2048 for mean+std of 1024-dim embeddings)
    - Hidden: 512 -> 512 -> 19 with GELU and Dropout
    """

    def __init__(
        self,
        pooled_dim: int = 2048,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_labels = num_labels

        # MLP head matching MuQStatsModel architecture
        self.clf = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-pooled embeddings.

        Args:
            pooled: Stats-pooled embeddings [B, pooled_dim] or [pooled_dim]

        Returns:
            Predictions [B, num_labels] or [num_labels]
        """
        squeeze_output = False
        if pooled.dim() == 1:
            pooled = pooled.unsqueeze(0)
            squeeze_output = True

        result = self.clf(pooled)
        return result.squeeze(0) if squeeze_output else result


class ModelCache:
    """Singleton cache for loaded models."""

    _instance: Optional["ModelCache"] = None

    def __new__(cls) -> "ModelCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.muq_model = None
        self.muq_heads: List[MuQStatsHead] = []
        self.device = None
        self._initialized = True

    def initialize(self, device: str = "cuda", checkpoint_dir: Optional[Path] = None):
        """Load MuQ model and prediction heads. Called once on container start."""
        if self.muq_model is not None:
            return

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initializing M1c MuQ models on {self.device}...")

        # Load MuQ from HuggingFace
        print("Loading MuQ-large-msd-iter...")
        try:
            from muq import MuQ
            self.muq_model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
            self.muq_model = self.muq_model.to(self.device)
            self.muq_model.eval()
            print("MuQ loaded successfully")
        except ImportError as e:
            raise ImportError(
                "MuQ library not found. Install with: pip install muq"
            ) from e

        # Load prediction heads (4 folds)
        print("Loading MuQStatsHead prediction heads...")
        checkpoint_dir = checkpoint_dir or Path("/repository/checkpoints")
        if not checkpoint_dir.exists():
            checkpoint_dir = Path("/app/checkpoints")

        for fold in range(N_FOLDS):
            ckpt_path = checkpoint_dir / f"fold{fold}" / "best.ckpt"
            if ckpt_path.exists():
                head = self._load_muq_head(ckpt_path)
                self.muq_heads.append(head)
                print(f"  Loaded fold {fold} from {ckpt_path}")
            else:
                print(f"  Warning: {ckpt_path} not found")

        print(f"Initialization complete. {len(self.muq_heads)} heads loaded.")

    def _load_muq_head(self, ckpt_path: Path) -> MuQStatsHead:
        """Load a MuQStatsHead from PyTorch Lightning checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        hparams = checkpoint.get("hyper_parameters", {})

        # Determine pooled_dim from input_dim and pooling_stats
        input_dim = hparams.get("input_dim", MODEL_CONFIG["muq_dim"])
        pooling_stats = hparams.get("pooling_stats", MODEL_CONFIG["pooling_stats"])
        if pooling_stats == "mean_std":
            pooled_dim = input_dim * 2
        else:
            pooled_dim = input_dim * 4

        head = MuQStatsHead(
            pooled_dim=pooled_dim,
            hidden_dim=hparams.get("hidden_dim", MODEL_CONFIG["hidden_dim"]),
            num_labels=hparams.get("num_labels", MODEL_CONFIG["num_labels"]),
            dropout=hparams.get("dropout", MODEL_CONFIG["dropout"]),
        )

        # Load state dict - handle Lightning's "clf." prefix
        state_dict = checkpoint["state_dict"]
        # Filter to only clf.* keys and remove prefix
        clf_state = {k.replace("clf.", ""): v for k, v in state_dict.items() if k.startswith("clf.")}
        head.clf.load_state_dict(clf_state, strict=True)

        head.to(self.device)
        head.eval()
        return head


_cache: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Get the global model cache instance."""
    global _cache
    if _cache is None:
        _cache = ModelCache()
    return _cache
