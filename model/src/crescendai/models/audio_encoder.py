import torch
import torch.nn as nn
import gc
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

    Research shows middle layers (5-7) are optimal for music understanding tasks,
    as lower layers contain acoustic info and upper layers overfit pre-train objective.
    """

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = True,
        use_layer_selection: bool = True,
        selected_layers: Optional[Tuple[int, ...]] = (5, 6, 7),
        freeze_bottom_layers: bool = False,
        num_freeze_layers: int = 6,
    ):
        """
        Initialize MERT encoder.

        Args:
            model_name: HuggingFace model identifier
            freeze_backbone: If True, freeze MERT weights (feature extraction only)
            gradient_checkpointing: Enable gradient checkpointing to save memory
            use_layer_selection: If True, use specific layers instead of last_hidden_state
            selected_layers: Which layers to average (default: 5,6,7 for MERT-95M)
            freeze_bottom_layers: If True, freeze bottom N layers of MERT
            num_freeze_layers: Number of bottom layers to freeze (default: 6 for MERT-95M)
        """
        super().__init__()

        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.use_layer_selection = use_layer_selection
        self.selected_layers = selected_layers
        self.freeze_bottom_layers = freeze_bottom_layers
        self.num_freeze_layers = num_freeze_layers

        # Load pre-trained MERT model
        # Try to load from local snapshot directory first (for offline environments)
        import os

        # Check for local snapshot directory (used in Thunder Compute/Colab)
        snapshot_dir = os.path.join(
            os.path.expanduser("~/.cache/huggingface/hub"),
            f"models--{model_name.replace('/', '--')}/snapshots/main"
        )

        if os.path.exists(snapshot_dir) and os.path.isdir(snapshot_dir):
            # Load from snapshot directory directly (don't use local_files_only with direct paths)
            model_path = snapshot_dir
            use_local_only = False  # Direct path, no HF Hub lookup needed
            print(f"Loading MERT from local snapshot: {snapshot_dir}")
        else:
            # Fall back to model name (will use HF cache resolution)
            model_path = model_name
            use_local_only = True  # Use cache only, don't download

        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=use_local_only
            )
            self.model = AutoModel.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=use_local_only
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load MERT model '{model_name}'. "
                f"Error: {e}\n"
                f"Tried loading from: {model_path}\n"
                f"Make sure the model is cached locally at ~/.cache/huggingface/hub/"
            )

        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        elif freeze_bottom_layers:
            # Freeze bottom N layers (0 to num_freeze_layers-1)
            # For MERT-95M with 12 layers, freezing 0-5 leaves 6-11 trainable
            for i, layer in enumerate(self.model.encoder.layers[:num_freeze_layers]):
                for param in layer.parameters():
                    param.requires_grad = False
            print(f"Froze bottom {num_freeze_layers} layers of MERT encoder")

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
        Audio should be normalized (mean=0, std=1) before passing to the model.

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
        # Manual normalization to avoid HuggingFace processor memory leak
        # The processor just does: (audio - mean) / std, which we can do directly
        # This avoids CPU<->numpy conversions that accumulate memory
        # NOTE: We don't clone - the input tensor is from DataLoader and not reused
        mean = audio_waveform.mean(dim=-1, keepdim=True)
        std = audio_waveform.std(dim=-1, keepdim=True).clamp(min=1e-7)
        input_values = (audio_waveform - mean) / std

        # Build inputs dict matching HuggingFace format
        inputs = {"input_values": input_values}

        # Forward through MERT
        # Always get hidden states if using layer selection
        need_hidden_states = output_hidden_states or self.use_layer_selection

        outputs = self.model(
            inputs["input_values"],
            attention_mask=attention_mask,
            output_hidden_states=need_hidden_states,
        )

        # Extract embeddings
        if self.use_layer_selection and self.selected_layers:
            # Use average of selected middle layers (research-backed)
            # For MERT-95M: layers 5-7 contain optimal music understanding features
            selected_hidden_states = [
                outputs.hidden_states[layer_idx]
                for layer_idx in self.selected_layers
            ]
            embeddings = torch.stack(selected_hidden_states, dim=0).mean(dim=0)

            # Free unused hidden states immediately (12 layers, only 3 used)
            # This prevents holding all layer outputs in memory until function returns
            if not output_hidden_states:
                del outputs
        else:
            # Fallback to last hidden state
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
    print("- Layer selection: Use middle layers (5-7) for optimal features")
    print("- Partial freezing: Freeze bottom layers to reduce overfitting")
