import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SimpleAST(nn.Module):
    """Tiny CNN front-end producing a 256D embedding from [B,1,128,128] mel inputs."""

    def __init__(self, emb_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # [32,64,64]
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # [64,32,32]
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # [128,16,16]
        )
        self.proj = nn.Linear(128 * 16 * 16, emb_dim)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = rearrange(h, "b c h w -> b (c h w)")
        h = self.proj(h)
        h = self.norm(h)
        return self.dropout(h)


class Evaluator(nn.Module):
    """Audio evaluator with dataset embedding and regression head.

    Returns both predictions and the audio embedding (for distillation).
    """

    def __init__(self, num_dims: int, num_datasets: int = 8, ds_embed_dim: int = 16, emb_dim: int = 256):
        super().__init__()
        self.backbone = SimpleAST(emb_dim=emb_dim)
        self.ds_embed = nn.Embedding(num_datasets, ds_embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim + ds_embed_dim),
            nn.Linear(emb_dim + ds_embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_dims),
        )
        # Distillation projection to match external embedding space (256 by default)
        self.distill_proj = nn.Linear(emb_dim, 256)

    def forward(self, mel01: torch.Tensor, ds_id: torch.Tensor):
        emb = self.backbone(mel01)  # [B, E]
        de = self.ds_embed(ds_id)   # [B, D_e]
        z = torch.cat([emb, de], dim=-1)
        pred = self.head(z)         # [B, D]
        return pred, emb


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp_min(1e-6)
    return diff.sum() / denom


def masked_mae_weighted(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    # weight per-dimension per-sample; only applies where mask==1
    w = weight * mask
    diff = (pred - target).abs() * w
    denom = w.sum().clamp_min(1e-6)
    return diff.sum() / denom


def cosine_distillation_loss(audio_emb: torch.Tensor, distill_vec: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
    # Project audio embedding and compare to external (stop-grad) embedding via cosine similarity
    a = proj(audio_emb)                          # [B, 256]
    b = distill_vec.detach()                     # [B, 256]
    # Normalize
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    cos = (a * b).sum(dim=-1)                    # [B]
    return (1 - cos).mean()