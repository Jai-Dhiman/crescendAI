"""
InfoNCE (Noise Contrastive Estimation) loss for contrastive learning.

Used to align audio and MIDI representations in multimodal space.
Encourages matched audio-MIDI pairs to have high similarity while
pushing apart unmatched pairs.

Reference: "Representation Learning with Contrastive Predictive Coding"
(van den Oord et al., 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE loss for audio-MIDI alignment.

    Treats diagonal elements of the similarity matrix as positive pairs
    (same performance) and off-diagonal as negatives.
    """

    def __init__(self, temperature: float = 0.07):
        """
        Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for softmax (default: 0.07)
                        Lower values make the model more discriminative
        """
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        audio_embed: torch.Tensor,
        midi_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss for audio-MIDI alignment.

        Args:
            audio_embed: Audio embeddings [batch, dim]
            midi_embed: MIDI embeddings [batch, dim]

        Returns:
            InfoNCE loss (scalar)
        """
        batch_size = audio_embed.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=audio_embed.device, requires_grad=True)

        # Normalize embeddings to unit sphere
        audio_embed = F.normalize(audio_embed, dim=-1)
        midi_embed = F.normalize(midi_embed, dim=-1)

        # Compute similarity matrix [batch, batch]
        # similarity[i, j] = cosine_similarity(audio[i], midi[j])
        similarity = torch.matmul(audio_embed, midi_embed.t()) / self.temperature

        # Labels: diagonal elements are positive pairs (same performance)
        labels = torch.arange(batch_size, device=audio_embed.device)

        # Bidirectional loss (audio->MIDI and MIDI->audio)
        loss_audio_to_midi = F.cross_entropy(similarity, labels)
        loss_midi_to_audio = F.cross_entropy(similarity.t(), labels)

        # Average both directions
        loss = (loss_audio_to_midi + loss_midi_to_audio) / 2.0

        return loss


if __name__ == "__main__":
    print("Contrastive loss module loaded successfully")
    print("- InfoNCE for audio-MIDI alignment")
    print("- Bidirectional (audio->MIDI and MIDI->audio)")
