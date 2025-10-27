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
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            )
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
        audio_waveform: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through MERT encoder.

        IMPORTANT: MERT expects raw audio waveforms at 24kHz, NOT spectrograms.
        The Wav2Vec2FeatureExtractor handles all preprocessing internally.

        Args:
            audio_waveform: Raw audio waveform [batch, num_samples]
                          Should be sampled at 24kHz (MERT's expected rate)
            attention_mask: Optional attention mask [batch, sequence_length]
            output_hidden_states: Whether to return all hidden states

        Returns:
            Tuple of:
                - embeddings: Frame-level features [batch, sequence_length, hidden_size]
                - hidden_states: All layer outputs if requested (optional)
        """
        # Process raw audio through MERT's feature extractor
        # The processor handles:
        # - Resampling to 24kHz if needed
        # - Normalization
        # - Conversion to appropriate input format
        inputs = self.processor(
            audio_waveform.cpu().numpy() if audio_waveform.is_cuda else audio_waveform.numpy(),
            sampling_rate=24000,  # MERT expects 24kHz
            return_tensors="pt",
            padding=True,
        )

        # Move to same device as model
        inputs = {k: v.to(audio_waveform.device) for k, v in inputs.items()}

        # Forward through MERT
        outputs = self.model(
            inputs["input_values"],
            attention_mask=inputs.get("attention_mask", attention_mask),
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


if __name__ == "__main__":
    print("Audio encoder module loaded successfully")
    print("PRODUCTION: MERT-95M encoder only (m-a-p/MERT-v1-95M)")
    print("- Raw audio waveforms at 24kHz")
    print("- Gradient checkpointing support")
    print("- Freeze/unfreeze capability")
