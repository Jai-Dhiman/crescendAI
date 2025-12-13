"""
Hierarchical Attention Network (HAN) Encoder for score representation.

Ported from PercePiano's virtuoso/encoder_score.py.
Implements the hierarchical structure: Note -> Beat -> Measure with attention aggregation.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from typing import Dict, Optional
from dataclasses import dataclass

from .context_attention import ContextAttention
from .hierarchy_utils import (
    make_higher_node,
    span_beat_to_note_num,
    run_hierarchy_lstm_with_pack,
    compute_actual_lengths,
)


@dataclass
class HanNetParams:
    """Configuration for HAN encoder."""
    input_size: int = 84  # Input feature dimension
    note_size: int = 128  # Note-level hidden dimension
    note_layers: int = 2
    voice_size: int = 64  # Voice-level hidden dimension
    voice_layers: int = 1
    beat_size: int = 64  # Beat-level hidden dimension
    beat_layers: int = 1
    measure_size: int = 64  # Measure-level hidden dimension
    measure_layers: int = 1
    num_attention_heads: int = 4
    dropout: float = 0.2


class HanEncoder(nn.Module):
    """
    Hierarchical Attention Network encoder.

    Architecture:
        Notes -> Note LSTM -> Voice LSTM (parallel per voice)
            -> Beat Attention + LSTM -> Measure Attention + LSTM
            -> Span back to notes -> Concatenate all levels

    The hierarchical structure captures musical relationships at multiple levels:
    - Note level: Individual note features enhanced with sequential context
    - Voice level: Parallel processing by voice (handles polyphony)
    - Beat level: Aggregates notes within each beat using attention
    - Measure level: Aggregates beats within each measure using attention

    Final output concatenates representations from all levels.
    """

    def __init__(
        self,
        input_size: int = 84,
        note_size: int = 128,
        note_layers: int = 2,
        voice_size: int = 64,
        voice_layers: int = 1,
        beat_size: int = 64,
        beat_layers: int = 1,
        measure_size: int = 64,
        measure_layers: int = 1,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_size: Dimension of input note features
            note_size: Hidden size for note-level LSTM
            note_layers: Number of note LSTM layers
            voice_size: Hidden size for voice-level LSTM
            voice_layers: Number of voice LSTM layers
            beat_size: Hidden size for beat-level LSTM
            beat_layers: Number of beat LSTM layers
            measure_size: Hidden size for measure-level LSTM
            measure_layers: Number of measure LSTM layers
            num_attention_heads: Number of attention heads for context attention
            dropout: Dropout rate
        """
        super().__init__()

        self.input_size = input_size
        self.note_size = note_size
        self.voice_size = voice_size
        self.beat_size = beat_size
        self.measure_size = measure_size

        # Input projection (if needed)
        self.note_fc = nn.Linear(input_size, note_size) if input_size != note_size else nn.Identity()

        # Note-level bidirectional LSTM
        self.lstm = nn.LSTM(
            note_size,
            note_size,
            note_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if note_layers > 1 else 0,
        )

        # Voice-level LSTM (processes notes by voice)
        self.voice_net = nn.LSTM(
            note_size,
            voice_size,
            voice_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if voice_layers > 1 else 0,
        )

        # Combined note+voice dimension (bidirectional -> *2)
        combined_note_voice_dim = (note_size + voice_size) * 2

        # Beat-level attention and LSTM
        self.beat_attention = ContextAttention(combined_note_voice_dim, num_attention_heads)
        self.beat_rnn = nn.LSTM(
            combined_note_voice_dim,
            beat_size,
            beat_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if beat_layers > 1 else 0,
        )

        # Measure-level attention and LSTM
        self.measure_attention = ContextAttention(beat_size * 2, num_attention_heads)
        self.measure_rnn = nn.LSTM(
            beat_size * 2,
            measure_size,
            measure_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Output dimension: note+voice (bidirectional) + beat (bidirectional) + measure (bidirectional)
        self.output_dim = combined_note_voice_dim + beat_size * 2 + measure_size * 2

    def forward(
        self,
        x: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        edges: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical encoder.

        Args:
            x: Input tensor of shape (N, T, input_size) - note features
            note_locations: Dict with keys:
                - 'beat': (N, T) beat index per note
                - 'measure': (N, T) measure index per note
                - 'voice': (N, T) voice index per note (1-indexed)
            edges: Optional graph edges (not used in base HAN)

        Returns:
            Dict with keys:
                - 'note': (N, T, note_size*2) note-level representations
                - 'voice': (N, T, (note_size+voice_size)*2) note+voice representations
                - 'beat': (N, T_beat, beat_size*2) beat-level representations
                - 'measure': (N, T_measure, measure_size*2) measure-level representations
                - 'beat_spanned': (N, T, beat_size*2) beat reps broadcast to notes
                - 'measure_spanned': (N, T, measure_size*2) measure reps broadcast to notes
                - 'total_note_cat': (N, T, output_dim) all levels concatenated
        """
        voice_numbers = note_locations['voice']
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']

        # Compute actual sequence lengths for proper handling throughout hierarchy
        actual_lengths = compute_actual_lengths(beat_numbers)

        # Project input if needed
        x = self.note_fc(x)

        # Pack sequence for efficient LSTM processing
        if not isinstance(x, nn.utils.rnn.PackedSequence):
            x_packed = pack_padded_sequence(x, actual_lengths.cpu(), batch_first=True, enforce_sorted=False)
        else:
            x_packed = x

        # Process through voice network (parallel per voice)
        max_voice = torch.max(voice_numbers).item()
        voice_out = self._run_voice_net(x, voice_numbers, max_voice)

        # Process through note LSTM
        note_out, _ = self.lstm(x_packed)
        note_out, _ = pad_packed_sequence(note_out, batch_first=True)

        # Concatenate note and voice outputs
        hidden_out = torch.cat((note_out, voice_out), dim=2)

        # Process through beat and measure levels - pass actual_lengths
        beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned = self._run_beat_and_measure(
            hidden_out, note_locations, actual_lengths=actual_lengths
        )

        # Concatenate all levels for final output
        total_note_cat = torch.cat([hidden_out, beat_out_spanned, measure_out_spanned], dim=-1)

        return {
            'note': note_out,
            'voice': hidden_out,
            'beat': beat_hidden_out,
            'measure': measure_hidden_out,
            'beat_spanned': beat_out_spanned,
            'measure_spanned': measure_out_spanned,
            'total_note_cat': total_note_cat,
        }

    def _run_voice_net(
        self,
        batch_x: torch.Tensor,
        voice_numbers: torch.Tensor,
        max_voice: int,
    ) -> torch.Tensor:
        """
        Process notes by voice in parallel.

        Each voice gets its own LSTM sequence, then results are mapped back
        to original note positions.

        Args:
            batch_x: Input tensor (N, T, C)
            voice_numbers: Voice index per note (N, T), 1-indexed
            max_voice: Maximum voice number

        Returns:
            Voice-level representations (N, T, voice_size*2)
        """
        if isinstance(batch_x, nn.utils.rnn.PackedSequence):
            batch_x, _ = pad_packed_sequence(batch_x, batch_first=True)

        num_notes = batch_x.size(1)
        output = torch.zeros(
            batch_x.shape[0], batch_x.shape[1], self.voice_net.hidden_size * 2
        ).to(batch_x.device)

        for i in range(1, max_voice + 1):
            # Find notes belonging to voice i
            voice_x_bool = voice_numbers == i
            num_voice_notes = torch.sum(voice_x_bool)
            num_batch_voice_notes = torch.sum(voice_x_bool, dim=1)

            if num_voice_notes > 0:
                # Extract notes for this voice from each batch
                voice_notes = [
                    batch_x[j, voice_x_bool[j]] if torch.sum(voice_x_bool[j]) > 0
                    else torch.zeros(1, batch_x.shape[-1]).to(batch_x.device)
                    for j in range(len(batch_x))
                ]
                voice_x = pad_sequence(voice_notes, batch_first=True)

                # Run LSTM on voice notes
                pack_voice_x = pack_padded_sequence(
                    voice_x,
                    [len(x) for x in voice_notes],
                    batch_first=True,
                    enforce_sorted=False,
                )
                ith_voice_out, _ = self.voice_net(pack_voice_x)
                ith_voice_out, _ = pad_packed_sequence(ith_voice_out, batch_first=True)

                # Create span matrix to map voice outputs back to original positions
                span_mat = torch.zeros(batch_x.shape[0], num_notes, voice_x.shape[1]).to(batch_x.device)
                voice_where = torch.nonzero(voice_x_bool)
                span_mat[
                    voice_where[:, 0],
                    voice_where[:, 1],
                    torch.cat([torch.arange(num_batch_voice_notes[j]) for j in range(len(batch_x))]),
                ] = 1

                # Add voice output to corresponding positions
                output += torch.bmm(span_mat, ith_voice_out)

        return output

    def _run_beat_and_measure(
        self,
        hidden_out: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        actual_lengths: Optional[torch.Tensor] = None,
    ):
        """
        Aggregate from note level to beat and measure levels.

        Args:
            hidden_out: Note+voice representations (N, T, combined_dim)
            note_locations: Dict with beat and measure indices
            actual_lengths: Optional tensor of shape (N,) with actual sequence lengths.
                           If not provided, will be computed from beat_numbers.

        Returns:
            Tuple of:
                - beat_hidden_out: (N, T_beat, beat_size*2)
                - measure_hidden_out: (N, T_measure, measure_size*2)
                - beat_out_spanned: (N, T, beat_size*2) - broadcast back to notes
                - measure_out_spanned: (N, T, measure_size*2) - broadcast back to notes
        """
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']

        # Compute actual_lengths if not provided
        if actual_lengths is None:
            actual_lengths = compute_actual_lengths(beat_numbers)

        # Aggregate notes to beats using attention
        beat_nodes = make_higher_node(
            hidden_out, self.beat_attention, beat_numbers, beat_numbers,
            lower_is_note=True, actual_lengths=actual_lengths
        )

        # Process beats through LSTM
        beat_hidden_out = run_hierarchy_lstm_with_pack(beat_nodes, self.beat_rnn)

        # Aggregate beats to measures using attention
        measure_nodes = make_higher_node(
            beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers,
            actual_lengths=actual_lengths
        )

        # Process measures through LSTM
        measure_hidden_out = run_hierarchy_lstm_with_pack(measure_nodes, self.measure_rnn)

        # Span back to note level
        beat_out_spanned = span_beat_to_note_num(beat_hidden_out, beat_numbers, actual_lengths=actual_lengths)
        measure_out_spanned = span_beat_to_note_num(measure_hidden_out, measure_numbers, actual_lengths=actual_lengths)

        return beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned

    def get_output_dim(self) -> int:
        """Get the output dimension of total_note_cat."""
        return self.output_dim


class SimplifiedHanEncoder(nn.Module):
    """
    Simplified HAN encoder without voice processing.

    Useful when voice information is not available or not needed.
    Still maintains the hierarchical note -> beat -> measure structure.
    """

    def __init__(
        self,
        input_size: int = 84,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden dimension for all LSTMs
            num_layers: Number of LSTM layers at each level
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.hidden_size = hidden_size

        # Note-level
        self.note_fc = nn.Linear(input_size, hidden_size)
        self.note_lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Beat-level
        self.beat_attention = ContextAttention(hidden_size * 2, num_attention_heads)
        self.beat_lstm = nn.LSTM(
            hidden_size * 2, hidden_size, 1,
            batch_first=True, bidirectional=True,
        )

        # Measure-level
        self.measure_attention = ContextAttention(hidden_size * 2, num_attention_heads)
        self.measure_lstm = nn.LSTM(
            hidden_size * 2, hidden_size, 1,
            batch_first=True, bidirectional=True,
        )

        # Output: note + beat_spanned + measure_spanned
        self.output_dim = hidden_size * 2 * 3

    def forward(
        self,
        x: torch.Tensor,
        note_locations: Dict[str, torch.Tensor],
        edges: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        batch_size, orig_seq_len, _ = x.shape
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']

        # Compute actual sequence lengths for proper handling throughout hierarchy
        actual_lengths = compute_actual_lengths(beat_numbers)

        # Note-level processing
        x = self.note_fc(x)

        x_packed = pack_padded_sequence(x, actual_lengths.cpu(), batch_first=True, enforce_sorted=False)
        note_out, _ = self.note_lstm(x_packed)
        note_out, _ = pad_packed_sequence(note_out, batch_first=True, total_length=orig_seq_len)

        # Beat aggregation - pass actual_lengths for proper boundary handling
        beat_nodes = make_higher_node(
            note_out, self.beat_attention, beat_numbers, beat_numbers,
            lower_is_note=True, actual_lengths=actual_lengths
        )
        beat_out = run_hierarchy_lstm_with_pack(beat_nodes, self.beat_lstm)
        beat_spanned = span_beat_to_note_num(beat_out, beat_numbers, actual_lengths=actual_lengths)

        # Ensure beat_spanned matches original sequence length
        if beat_spanned.shape[1] < orig_seq_len:
            padding = torch.zeros(
                batch_size, orig_seq_len - beat_spanned.shape[1], beat_spanned.shape[2],
                device=beat_spanned.device, dtype=beat_spanned.dtype
            )
            beat_spanned = torch.cat([beat_spanned, padding], dim=1)

        # Measure aggregation - pass actual_lengths for proper boundary handling
        measure_nodes = make_higher_node(
            beat_out, self.measure_attention, beat_numbers, measure_numbers,
            actual_lengths=actual_lengths
        )
        measure_out = run_hierarchy_lstm_with_pack(measure_nodes, self.measure_lstm)
        measure_spanned = span_beat_to_note_num(measure_out, measure_numbers, actual_lengths=actual_lengths)

        # Ensure measure_spanned matches original sequence length
        if measure_spanned.shape[1] < orig_seq_len:
            padding = torch.zeros(
                batch_size, orig_seq_len - measure_spanned.shape[1], measure_spanned.shape[2],
                device=measure_spanned.device, dtype=measure_spanned.dtype
            )
            measure_spanned = torch.cat([measure_spanned, padding], dim=1)

        # Concatenate all levels
        total = torch.cat([note_out, beat_spanned, measure_spanned], dim=-1)

        return {
            'note': note_out,
            'beat': beat_out,
            'measure': measure_out,
            'beat_spanned': beat_spanned,
            'measure_spanned': measure_spanned,
            'total_note_cat': total,
        }

    def get_output_dim(self) -> int:
        return self.output_dim
