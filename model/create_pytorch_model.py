# Based on your evaluation report, recreate the model:

import torch
import torch.nn as nn


class SimpleAST(nn.Module):
    """12-layer Audio Spectrogram Transformer with Regression Head"""

    def __init__(self, num_classes=19, emb_dim=768, num_heads=12, num_layers=12):
        super().__init__()

        # Patch embedding for 128x128 mel-spectrograms
        self.patch_embed = nn.Conv2d(
            1, emb_dim, kernel_size=16, stride=16
        )  # 16x16 patches

        # Positional encoding
        num_patches = (128 // 16) * (128 // 16)  # 8x8 = 64 patches
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Regression head for 19 perceptual dimensions
        self.head = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding: [B, 1, 128, 128] -> [B, 768, 8, 8]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, 64, 768]

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 65, 768]

        # Add positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Use class token for prediction
        cls_output = x[:, 0]  # [B, 768]

        # Regression head
        output = self.head(cls_output)  # [B, 19]

        return torch.sigmoid(output)  # [0, 1] range


# Create model matching your evaluation report specs:
model = SimpleAST(
    num_classes=19,  # 19 perceptual dimensions
    emb_dim=768,  # 768 embedding dimension
    num_heads=12,  # 12 attention heads
    num_layers=12,  # 12 transformer layers
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
# Should be close to 85,706,003 from your report

# Train this model on your data, then convert to ONNX
