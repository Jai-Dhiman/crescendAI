"""Model loading and caching for inference."""

import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

from ..constants import MERT_CONFIG, N_FOLDS, CHECKPOINT_PATHS


class MERTHead(nn.Module):
    """MLP head for MERT embeddings (must match training architecture)."""

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clf(x)


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
        self.mert_processor = None
        self.mert_model = None
        self.mert_heads: List[MERTHead] = []
        self.fusion_weights: dict = {}
        self.device = None
        self._initialized = True

    def initialize(self, device: str = "cuda", checkpoint_dir: Optional[Path] = None):
        """Load all models. Called once on container start."""
        if self.mert_model is not None:
            return

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initializing models on {self.device}...")

        # Load MERT-330M from HuggingFace
        print("Loading MERT-v1-330M...")
        self.mert_processor = AutoProcessor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True
        )
        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M",
            output_hidden_states=True,
            trust_remote_code=True,
        ).to(self.device)
        self.mert_model.eval()
        print(f"MERT loaded. Hidden size: {self.mert_model.config.hidden_size}")

        # Load MERT MLP heads (4 folds for ensemble)
        print("Loading MERT MLP heads...")
        checkpoint_dir = checkpoint_dir or Path("/app/checkpoints")
        for fold in range(N_FOLDS):
            ckpt_path = checkpoint_dir / "mert" / f"fold{fold}_best.ckpt"
            if ckpt_path.exists():
                head = self._load_mert_head(ckpt_path)
                self.mert_heads.append(head)
                print(f"  Loaded fold {fold} from {ckpt_path}")
            else:
                print(f"  Warning: {ckpt_path} not found")

        # Load fusion weights if available
        fusion_weights_path = checkpoint_dir / "fusion" / "optimal_weights.json"
        if fusion_weights_path.exists():
            with open(fusion_weights_path) as f:
                self.fusion_weights = json.load(f)
            print(f"Loaded fusion weights from {fusion_weights_path}")
        else:
            print("No fusion weights found, will use simple average")

        print(f"Initialization complete. {len(self.mert_heads)} MERT heads loaded.")

    def _load_mert_head(self, ckpt_path: Path) -> MERTHead:
        """Load a MERT MLP head from PyTorch Lightning checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # Extract hyperparameters
        hparams = checkpoint.get("hyper_parameters", {})
        input_dim = hparams.get("input_dim", MERT_CONFIG["input_dim"])
        hidden_dim = hparams.get("hidden_dim", MERT_CONFIG["hidden_dim"])
        num_labels = hparams.get("num_labels", MERT_CONFIG["num_labels"])
        dropout = hparams.get("dropout", MERT_CONFIG["dropout"])

        # Create model and load state dict
        head = MERTHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout,
        )

        # Extract clf weights from state_dict (they're prefixed with 'clf.')
        state_dict = checkpoint["state_dict"]
        clf_state_dict = {
            k.replace("clf.", ""): v for k, v in state_dict.items() if k.startswith("clf.")
        }
        head.clf.load_state_dict(clf_state_dict)
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
