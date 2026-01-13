"""Model loading and caching for D9c AsymmetricGatedFusion inference."""

from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor

from constants import MODEL_CONFIG, N_FOLDS


class AsymmetricGatedFusionHead(nn.Module):
    """Inference-only version of AsymmetricGatedFusion.

    Loads trained weights and runs inference. Architecture must match training:
    - MERT: mert_dim -> mert_hidden -> shared_dim (2-stage projection)
    - MuQ: muq_dim -> shared_dim (single projection)
    - Attention pooling for both modalities
    - Per-dimension gating for each of 19 outputs
    """

    def __init__(
        self,
        mert_dim: int = 1024,
        muq_dim: int = 1024,
        mert_hidden: int = 512,
        shared_dim: int = 512,
        num_labels: int = 19,
        dropout: float = 0.2,
        pooling: str = "attention",
    ):
        super().__init__()
        self.num_labels = num_labels
        self.shared_dim = shared_dim
        self.pooling = pooling

        # Attention pooling modules (matching training)
        if pooling == "attention":
            self.mert_attn = nn.Sequential(
                nn.Linear(mert_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )
            self.muq_attn = nn.Sequential(
                nn.Linear(muq_dim, 256), nn.Tanh(), nn.Linear(256, 1)
            )

        # Asymmetric projections (must match training)
        self.mert_proj = nn.Sequential(
            nn.Linear(mert_dim, mert_hidden),
            nn.LayerNorm(mert_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mert_hidden, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.muq_proj = nn.Sequential(
            nn.Linear(muq_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-dimension gating network
        self.gate_net = nn.Sequential(
            nn.Linear(shared_dim * 2, shared_dim),
            nn.GELU(),
            nn.Linear(shared_dim, num_labels),
            nn.Sigmoid(),
        )

        # Per-dimension prediction heads
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(shared_dim, shared_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(shared_dim // 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(num_labels)
        ])

    def pool(self, x: torch.Tensor, attn_module: nn.Module, mask: torch.Tensor = None) -> torch.Tensor:
        """Pool sequence embeddings using attention or mean pooling."""
        if self.pooling == "attention" and attn_module is not None:
            scores = attn_module(x).squeeze(-1)
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))
            w = torch.softmax(scores, dim=-1).unsqueeze(-1)
            return (x * w).sum(1)
        else:
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                return (x * m).sum(1) / m.sum(1).clamp(min=1)
            return x.mean(dim=1)

    def forward(
        self,
        mert_emb: torch.Tensor,
        muq_emb: torch.Tensor,
        mert_mask: torch.Tensor = None,
        muq_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass with sequence embeddings."""
        # Handle unbatched input
        squeeze_output = False
        if mert_emb.dim() == 2:
            mert_emb = mert_emb.unsqueeze(0)
            muq_emb = muq_emb.unsqueeze(0)
            squeeze_output = True

        # Pool sequences to fixed-size vectors
        attn_mod = self.mert_attn if self.pooling == "attention" else None
        mert_pooled = self.pool(mert_emb, attn_mod, mert_mask)

        attn_mod = self.muq_attn if self.pooling == "attention" else None
        muq_pooled = self.pool(muq_emb, attn_mod, muq_mask)

        # Project each modality
        mert_proj = self.mert_proj(mert_pooled)
        muq_proj = self.muq_proj(muq_pooled)

        # Compute per-dimension gates
        combined = torch.cat([mert_proj, muq_proj], dim=-1)
        gates = self.gate_net(combined)

        # Apply per-dimension gated fusion and predict
        outputs = []
        for i, head in enumerate(self.heads):
            gate = gates[:, i:i+1]
            gated = gate * mert_proj + (1 - gate) * muq_proj
            out = head(gated)
            outputs.append(out)

        result = torch.cat(outputs, dim=1)
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
        self.mert_processor = None
        self.mert_model = None
        self.muq_model = None
        self.fusion_heads: List[AsymmetricGatedFusionHead] = []
        self.device = None
        self._initialized = True

    def initialize(self, device: str = "cuda", checkpoint_dir: Optional[Path] = None):
        """Load all models. Called once on container start."""
        if self.mert_model is not None:
            return

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Initializing D9c models on {self.device}...")

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

        # Load fusion heads (4 folds)
        print("Loading AsymmetricGatedFusion heads...")
        checkpoint_dir = checkpoint_dir or Path("/repository/checkpoints")
        if not checkpoint_dir.exists():
            checkpoint_dir = Path("/app/checkpoints")

        for fold in range(N_FOLDS):
            ckpt_path = checkpoint_dir / f"fold{fold}" / "best.ckpt"
            if ckpt_path.exists():
                head = self._load_fusion_head(ckpt_path)
                self.fusion_heads.append(head)
                print(f"  Loaded fold {fold} from {ckpt_path}")
            else:
                print(f"  Warning: {ckpt_path} not found")

        print(f"Initialization complete. {len(self.fusion_heads)} fusion heads loaded.")

    def _load_fusion_head(self, ckpt_path: Path) -> AsymmetricGatedFusionHead:
        """Load an AsymmetricGatedFusion head from PyTorch Lightning checkpoint."""
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        hparams = checkpoint.get("hyper_parameters", {})
        head = AsymmetricGatedFusionHead(
            mert_dim=hparams.get("mert_dim", MODEL_CONFIG["mert_dim"]),
            muq_dim=hparams.get("muq_dim", MODEL_CONFIG["muq_dim"]),
            mert_hidden=hparams.get("mert_hidden", MODEL_CONFIG["mert_hidden"]),
            shared_dim=hparams.get("shared_dim", MODEL_CONFIG["shared_dim"]),
            num_labels=hparams.get("num_labels", MODEL_CONFIG["num_labels"]),
            dropout=hparams.get("dropout", MODEL_CONFIG["dropout"]),
            pooling=hparams.get("pooling", MODEL_CONFIG["pooling"]),
        )

        state_dict = checkpoint["state_dict"]
        head.load_state_dict(state_dict, strict=True)
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
