"""Model loading and caching for A1-Max MuQ LoRA inference."""

import os
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn

from constants import MODEL_CONFIG, N_FOLDS


def _resolve_device(requested: str) -> torch.device:
    """Resolve device with env override and auto-detection.

    Supports: "cuda", "mps", "cpu", "auto".
    "auto" runs the full cascade: CUDA > MPS > CPU.
    CRESCEND_DEVICE env var overrides the requested device.
    """
    dev = os.environ.get("CRESCEND_DEVICE", requested)
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    elif dev == "cuda" and not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    elif dev == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        dev = "cpu"
    return torch.device(dev)


class A1MaxInferenceHead(nn.Module):
    """Inference-only version of MuQLoRAMaxModel's predict_scores path.

    Replicates the architecture needed for score prediction:
    - Attention pooling: [B, T, D] -> [B, D]
    - Encoder: 2-layer MLP [B, D] -> [B, hidden_dim]
    - Regression head: MLP + sigmoid [B, hidden_dim] -> [B, num_labels]

    Does NOT include ranking/contrastive/comparator modules (training-only).
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_labels: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_labels = num_labels

        # Attention pooling (matches MuQLoRAModel.attn)
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 256), nn.Tanh(), nn.Linear(256, 1)
        )

        # Shared encoder (matches MuQLoRAModel.encoder)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Regression head (matches MuQLoRAModel.regression_head)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels),
            nn.Sigmoid(),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict quality scores from frame embeddings.

        Args:
            embeddings: Frame embeddings [B, T, D] or [T, D].

        Returns:
            Scores [B, num_labels] or [num_labels] in [0, 1].
        """
        squeeze_output = False
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
            squeeze_output = True

        # Attention pool
        scores = self.attn(embeddings).squeeze(-1)  # [B, T]
        w = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        pooled = (embeddings * w).sum(1)  # [B, D]

        # Encode
        z = self.encoder(pooled)  # [B, hidden_dim]

        # Predict
        result = self.regression_head(z)  # [B, num_labels]

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
        self.muq_heads: List[A1MaxInferenceHead] = []
        self.device = None
        self._initialized = True

    def initialize(self, device: str = "cuda", checkpoint_dir: Optional[Path] = None):
        """Load MuQ model and A1-Max prediction heads. Called once on container start."""
        if self.muq_model is not None:
            return

        self.device = _resolve_device(device)
        print(f"Initializing A1-Max models on {self.device}...")

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

        # Load A1-Max prediction heads (4 folds)
        print("Loading A1-Max prediction heads...")
        checkpoint_dir = checkpoint_dir or Path("/repository/checkpoints")
        if not checkpoint_dir.exists():
            checkpoint_dir = Path("/app/checkpoints")

        for fold in range(N_FOLDS):
            ckpt_path = checkpoint_dir / f"fold_{fold}" / "best.ckpt"
            # Also try the epoch-based naming from sweep
            if not ckpt_path.exists():
                fold_dir = checkpoint_dir / f"fold_{fold}"
                if fold_dir.exists():
                    ckpts = sorted(fold_dir.glob("*.ckpt"))
                    if ckpts:
                        ckpt_path = ckpts[0]
            if ckpt_path.exists():
                head = self._load_a1max_head(ckpt_path)
                self.muq_heads.append(head)
                print(f"  Loaded fold {fold} from {ckpt_path}")
            else:
                print(f"  Warning: No checkpoint found for fold {fold}")

        print(f"Initialization complete. {len(self.muq_heads)} heads loaded.")

    def _load_a1max_head(self, ckpt_path: Path) -> A1MaxInferenceHead:
        """Load an A1MaxInferenceHead from PyTorch Lightning checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        hparams = checkpoint.get("hyper_parameters", {})

        head = A1MaxInferenceHead(
            input_dim=hparams.get("input_dim", MODEL_CONFIG["input_dim"]),
            hidden_dim=hparams.get("hidden_dim", MODEL_CONFIG["hidden_dim"]),
            num_labels=hparams.get("num_labels", MODEL_CONFIG["num_labels"]),
            dropout=hparams.get("dropout", MODEL_CONFIG["dropout"]),
        )

        # Load state dict from Lightning checkpoint
        state_dict = checkpoint["state_dict"]

        # Map Lightning keys to inference head keys
        # Lightning saves as: attn.0.weight, encoder.0.weight, regression_head.0.weight, etc.
        head_state = {}
        for key, value in state_dict.items():
            if key.startswith("attn.") or key.startswith("encoder.") or key.startswith("regression_head."):
                head_state[key] = value

        head.load_state_dict(head_state, strict=True)

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
