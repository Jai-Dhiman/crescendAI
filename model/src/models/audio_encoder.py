import torch
import torch.nn as nn
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from typing import Optional, Tuple


class MERTEncoder(nn.Module):
    """
    MERT-95M audio encoder for music understanding.

    Loads pre-trained m-a-p/MERT-v1-95M from HuggingFace.
    - 95M parameters (Colab-friendly)
    - Pre-trained on 160K hours of music
    - Dual-teacher approach (acoustic + musical CQT)
    - 12 transformer layers, 768 hidden dim

    Input: CQT spectrogram [batch, 168, time_frames]
    Output: Frame-level embeddings [batch, time_frames', 768]
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = True,
    ):
        """
        Initialize MERT encoder.

        Args:
            model_name: HuggingFace model identifier
            freeze_backbone: If True, freeze MERT weights (feature extraction only)
            gradient_checkpointing: Enable gradient checkpointing to save memory
        """
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone

        # Load pre-trained MERT model
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MERT model '{model_name}'. "
                f"Error: {e}\n"
                f"Make sure you have internet connection and HuggingFace access."
            )

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        self.hidden_size = self.model.config.hidden_size  # 768 for MERT-95M

    def forward(
        self,
        cqt: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MERT encoder.

        Args:
            cqt: CQT spectrogram [batch, 168, time_frames]
            attention_mask: Optional attention mask [batch, time_frames]
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of:
                - embeddings: Frame-level features [batch, time_frames', hidden_size]
                - hidden_states: All layer outputs if requested (optional)
        """
        batch_size, n_bins, time_frames = cqt.shape

        # MERT expects input similar to wav2vec2 format
        # We need to convert CQT to the expected format
        # Note: MERT was trained on raw audio, but we're using CQT
        # This is a simplification for MVP - ideally we'd retrain the feature extractor

        # Reshape CQT for processing: [batch, features, time]
        # MERT expects [batch, sequence_length]
        # We'll flatten the frequency bins as features
        inputs = cqt.transpose(1, 2)  # [batch, time_frames, 168]
        inputs = inputs.reshape(batch_size, time_frames * n_bins)  # Flatten

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, time_frames, device=cqt.device)

        # Forward through MERT
        # Note: This is a simplified approach for MVP
        # Production version should properly handle CQT -> MERT input conversion
        outputs = self.model(
            inputs,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        # Extract last hidden state
        embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_size]

        hidden_states = None
        if output_hidden_states:
            hidden_states = outputs.hidden_states

        return embeddings, hidden_states

    def freeze(self):
        """Freeze all parameters in the backbone."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.freeze_backbone = True

    def unfreeze(self):
        """Unfreeze all parameters in the backbone."""
        for param in self.model.parameters():
            param.requires_grad = True
        self.freeze_backbone = False

    def get_output_dim(self) -> int:
        """Get output dimension (hidden size)."""
        return self.hidden_size


class SimplifiedAudioEncoder(nn.Module):
    """
    Simplified audio encoder for testing without MERT.

    Uses CNN layers to process CQT spectrograms when MERT is unavailable.
    This is a fallback for local testing.
    """

    def __init__(
        self,
        input_channels: int = 168,
        hidden_size: int = 768,
        num_layers: int = 4,
    ):
        """
        Initialize simplified encoder.

        Args:
            input_channels: Number of CQT bins (default: 168)
            hidden_size: Output embedding dimension
            num_layers: Number of CNN layers
        """
        super().__init__()

        self.hidden_size = hidden_size

        # CNN layers to process CQT
        layers = []
        in_channels = 1
        out_channels = 64

        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            in_channels = out_channels
            out_channels = min(out_channels * 2, 512)

        self.cnn = nn.Sequential(*layers)

        # Calculate flattened dimension after CNN
        # This is approximate - actual dimension depends on input size
        self.flatten_dim = 512 * (input_channels // (2 ** num_layers))

        # Project to hidden size
        self.projection = nn.Sequential(
            nn.Linear(self.flatten_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(
        self,
        cqt: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through simplified encoder.

        Args:
            cqt: CQT spectrogram [batch, 168, time_frames]
            attention_mask: Optional attention mask (not used in this version)
            output_hidden_states: Whether to return hidden states (not used)

        Returns:
            Tuple of (embeddings, None)
        """
        batch_size, n_bins, time_frames = cqt.shape

        # Add channel dimension: [batch, 1, n_bins, time_frames]
        x = cqt.unsqueeze(1)

        # CNN processing
        x = self.cnn(x)  # [batch, channels, reduced_bins, reduced_time]

        # Reshape for projection
        batch, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2)  # [batch, time, channels, height]
        x = x.reshape(batch, width, -1)  # [batch, time, channels*height]

        # Project to hidden size
        embeddings = self.projection(x)  # [batch, time, hidden_size]

        return embeddings, None

    def get_output_dim(self) -> int:
        """Get output dimension."""
        return self.hidden_size


def create_audio_encoder(
    use_mert: bool = True,
    model_name: str = "m-a-p/MERT-v1-95M",
    freeze_backbone: bool = False,
    gradient_checkpointing: bool = True,
) -> nn.Module:
    """
    Factory function to create audio encoder.

    Args:
        use_mert: If True, use MERT encoder; otherwise use simplified CNN
        model_name: HuggingFace model name for MERT
        freeze_backbone: Whether to freeze the backbone
        gradient_checkpointing: Enable gradient checkpointing

    Returns:
        Audio encoder module
    """
    if use_mert:
        try:
            return MERTEncoder(
                model_name=model_name,
                freeze_backbone=freeze_backbone,
                gradient_checkpointing=gradient_checkpointing,
            )
        except Exception as e:
            print(f"Warning: Failed to load MERT model: {e}")
            print("Falling back to simplified encoder")
            use_mert = False

    if not use_mert:
        return SimplifiedAudioEncoder(
            input_channels=168,
            hidden_size=768,
            num_layers=4,
        )


if __name__ == "__main__":
    print("Audio encoder module loaded successfully")
    print("- MERT-95M encoder (m-a-p/MERT-v1-95M)")
    print("- Simplified CNN encoder (fallback)")
    print("- Gradient checkpointing support")
    print("- Freeze/unfreeze capability")
