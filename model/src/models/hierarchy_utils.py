"""
Utility functions for hierarchical attention networks.

Ported from PercePiano's virtuoso/utils.py and virtuoso/model_utils.py.
These functions handle hierarchical aggregation (note -> beat -> measure)
and broadcasting back to note level.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Tuple, Optional


def compute_actual_lengths(beat_numbers: torch.Tensor) -> torch.Tensor:
    """
    Compute actual sequence lengths from zero-padded beat numbers.

    This is the core helper that enables proper length tracking throughout
    the hierarchical processing pipeline.

    Args:
        beat_numbers: Tensor of shape (N, T) with beat indices per note.
                      Assumes padding positions have beat_number == 0.

    Returns:
        Tensor of shape (N,) with actual sequence lengths.
    """
    batch_size, seq_len = beat_numbers.shape

    # Find last non-zero position for each batch element
    non_zero_mask = beat_numbers != 0  # (N, T)

    lengths = torch.zeros(batch_size, dtype=torch.long, device=beat_numbers.device)
    for i in range(batch_size):
        non_zero_positions = torch.where(non_zero_mask[i])[0]
        if len(non_zero_positions) > 0:
            lengths[i] = non_zero_positions[-1].item() + 1
        else:
            lengths[i] = 1  # Minimum length

    return lengths


def find_boundaries(diff_boundary: torch.Tensor, beat_numbers: torch.Tensor, batch_idx: int) -> List[int]:
    """
    Find boundary indices for a single batch element.

    Args:
        diff_boundary: Tensor of shape (num_boundaries, 2) with [batch_idx, position]
        beat_numbers: Tensor of shape (N, T) with beat numbers
        batch_idx: Index of the batch element to process

    Returns:
        List of boundary indices including 0 and sequence length
    """
    mask = diff_boundary[:, 0] == batch_idx
    indices = diff_boundary[mask, 1].tolist()

    # Add 1 to indices (boundaries are after the position)
    boundaries = [0] + [i + 1 for i in indices]

    # Calculate actual sequence length (first position where beat number doesn't increase)
    seq_len = beat_numbers.shape[1]
    if len(beat_numbers[batch_idx]) > 1:
        diffs = beat_numbers[batch_idx, 1:] - beat_numbers[batch_idx, :-1]
        # Find where diff becomes negative (padding starts)
        neg_positions = torch.where(diffs < 0)[0]
        if len(neg_positions) > 0:
            seq_len = int(neg_positions[0].item()) + 1

    boundaries.append(seq_len)
    return boundaries


def find_boundaries_batch(
    beat_numbers: torch.Tensor,
    actual_lengths: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """
    Find beat/measure boundaries for a batch of sequences.

    A boundary is where beat_number[t] != beat_number[t-1].

    Args:
        beat_numbers: Tensor of shape (N, T) with beat/measure indices per note.
                      Zero-padded sequences.
        actual_lengths: Optional tensor of shape (N,) with actual sequence lengths.
                       If not provided, will be computed from beat_numbers.

    Returns:
        List of lists, each containing boundary indices for that batch element.
        Includes 0 at start and actual sequence length at end.
    """
    if actual_lengths is None:
        actual_lengths = compute_actual_lengths(beat_numbers)

    batch_size = beat_numbers.shape[0]
    all_boundaries = []

    for i in range(batch_size):
        length = int(actual_lengths[i].item())
        valid_beats = beat_numbers[i, :length]

        # Find where beat number changes (increases by 1)
        if length > 1:
            diffs = valid_beats[1:] - valid_beats[:-1]
            change_positions = torch.where(diffs == 1)[0].cpu().tolist()
            # Boundaries are after the change position
            boundaries = [0] + [p + 1 for p in change_positions] + [length]
        else:
            boundaries = [0, length]

        all_boundaries.append(boundaries)

    return all_boundaries


def get_softmax_by_boundary(
    similarity: torch.Tensor,
    boundaries: List[int],
) -> List[torch.Tensor]:
    """
    Apply softmax within each segment defined by boundaries.

    Args:
        similarity: Tensor of shape (T, C) - attention scores for single sequence
        boundaries: List of boundary indices defining segments

    Returns:
        List of tensors with softmax applied within each segment
    """
    result = []
    for i in range(1, len(boundaries)):
        start, end = boundaries[i - 1], boundaries[i]
        if start < end:
            segment = similarity[start:end, :]
            result.append(torch.softmax(segment, dim=0))
    return result


def cal_length_from_padded_beat_numbers(beat_numbers: torch.Tensor) -> torch.Tensor:
    """
    Calculate actual sequence lengths from zero-padded beat numbers.

    Args:
        beat_numbers: Tensor of shape (N, T) with beat indices per note.
                      Assumes zero-padding at the end.

    Returns:
        Tensor of shape (N,) with actual sequence lengths
    """
    batch_size = beat_numbers.shape[0]
    seq_len = beat_numbers.shape[1]

    # Handle edge cases
    if seq_len <= 1:
        return torch.ones(batch_size, dtype=torch.long, device=beat_numbers.device)

    # Find first position where diff becomes negative (padding starts)
    diffs = torch.diff(beat_numbers, dim=1)

    # Handle empty diffs
    if diffs.shape[1] == 0:
        return torch.ones(batch_size, dtype=torch.long, device=beat_numbers.device)

    # For each batch, find where padding starts (diff < 0) or use full length
    len_note = torch.zeros(batch_size, dtype=torch.long, device=beat_numbers.device)
    for i in range(batch_size):
        neg_pos = torch.where(diffs[i] < 0)[0]
        if len(neg_pos) > 0:
            len_note[i] = neg_pos[0].item() + 1
        else:
            # No negative diff found - either all valid or all padding
            # Check if sequence has any non-zero values
            non_zero = torch.where(beat_numbers[i] > 0)[0]
            if len(non_zero) > 0:
                len_note[i] = seq_len  # All valid
            else:
                len_note[i] = 1  # All zeros, use minimum length

    return len_note


def make_higher_node(
    lower_out: torch.Tensor,
    attention_weights: nn.Module,
    lower_indices: torch.Tensor,
    higher_indices: torch.Tensor,
    lower_is_note: bool = False,
    actual_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Aggregate lower-level nodes into higher-level nodes using attention.

    Used for:
    - Note -> Beat aggregation (lower_is_note=True)
    - Beat -> Measure aggregation (lower_is_note=False)

    Args:
        lower_out: Tensor of shape (N, T_lower, C) - lower level representations
        attention_weights: Module with get_attention() method (e.g., ContextAttention)
        lower_indices: Tensor of shape (N, T_lower) - beat/measure indices at lower level
        higher_indices: Tensor of shape (N, T_lower) - measure indices for each lower element
        lower_is_note: If True, lower_out is notes, else it's beats
        actual_lengths: Optional tensor of shape (N,) with actual sequence lengths.
                       If not provided, will be computed from indices.

    Returns:
        Tensor of shape (N, T_higher, C) - aggregated higher-level representations
    """
    # Get attention scores
    similarity = attention_weights.get_attention(lower_out)  # (N, T, num_heads)

    # Compute actual_lengths if not provided
    if actual_lengths is None:
        actual_lengths = compute_actual_lengths(higher_indices)

    if lower_is_note:
        # Notes -> Beats: use higher_indices directly
        boundaries = find_boundaries_batch(higher_indices, actual_lengths)
    else:
        # Beats -> Measures: need to map through lower_indices
        higher_boundaries = find_boundaries_batch(higher_indices, actual_lengths)
        zero_shifted_lower_indices = lower_indices - lower_indices[:, 0:1]

        # Calculate actual lengths for beat-level output
        num_zero_padded = ((lower_out != 0).sum(-1) == 0).sum(1)
        len_lower_out = (lower_out.shape[1] - num_zero_padded).tolist()

        boundaries = [
            zero_shifted_lower_indices[i, higher_boundaries[i][:-1]].tolist() + [len_lower_out[i]]
            for i in range(len(lower_out))
        ]

    # Apply softmax within each segment
    softmax_results = []
    for batch_idx in range(len(lower_out)):
        segments = get_softmax_by_boundary(similarity[batch_idx], boundaries[batch_idx])
        if len(segments) > 0:
            softmax_results.append(torch.cat(segments))
        else:
            # Handle edge case: no valid segments, create a single-element tensor
            softmax_results.append(similarity[batch_idx, :1, :])
    softmax_similarity = torch.nn.utils.rnn.pad_sequence(softmax_results, batch_first=True)

    # Ensure softmax_similarity matches lower_out sequence length
    if softmax_similarity.shape[1] < lower_out.shape[1]:
        # Pad if softmax is shorter
        padding = torch.zeros(
            softmax_similarity.shape[0],
            lower_out.shape[1] - softmax_similarity.shape[1],
            softmax_similarity.shape[2],
            device=softmax_similarity.device,
            dtype=softmax_similarity.dtype,
        )
        softmax_similarity = torch.cat([softmax_similarity, padding], dim=1)
    elif softmax_similarity.shape[1] > lower_out.shape[1]:
        # Truncate if softmax is longer (can happen in beat->measure aggregation)
        softmax_similarity = softmax_similarity[:, :lower_out.shape[1], :]

    # Compute weighted sum within each segment
    if hasattr(attention_weights, 'head_size'):
        x_split = torch.stack(lower_out.split(split_size=attention_weights.head_size, dim=2), dim=2)
        weighted_x = x_split * softmax_similarity.unsqueeze(-1).repeat(1, 1, 1, x_split.shape[-1])
        weighted_x = weighted_x.view(x_split.shape[0], x_split.shape[1], lower_out.shape[-1])

        higher_nodes_list = []
        for i in range(len(lower_out)):
            batch_segments = []
            for j in range(1, len(boundaries[i])):
                start, end = boundaries[i][j-1], boundaries[i][j]
                if start < end and end <= weighted_x.shape[1]:
                    segment_sum = torch.sum(weighted_x[i:i+1, start:end, :], dim=1)
                    batch_segments.append(segment_sum)
            if len(batch_segments) > 0:
                higher_nodes_list.append(torch.cat(batch_segments, dim=0))
            else:
                # Handle edge case: no valid segments, use first position
                higher_nodes_list.append(weighted_x[i, :1, :])

        higher_nodes = torch.nn.utils.rnn.pad_sequence(higher_nodes_list, batch_first=True)
    else:
        # Non-head-size path (for compatibility)
        weighted_sum = softmax_similarity * lower_out
        higher_nodes_list = []
        for i in range(len(lower_out)):
            batch_segments = []
            for j in range(1, len(boundaries[i])):
                start, end = boundaries[i][j-1], boundaries[i][j]
                if start < end and end <= weighted_sum.shape[1]:
                    segment_sum = torch.sum(weighted_sum[i:i+1, start:end, :], dim=1)
                    batch_segments.append(segment_sum)
            if len(batch_segments) > 0:
                higher_nodes_list.append(torch.cat(batch_segments, dim=0))
            else:
                higher_nodes_list.append(weighted_sum[i, :1, :])

        higher_nodes = torch.nn.utils.rnn.pad_sequence(higher_nodes_list, batch_first=True)

    return higher_nodes


def span_beat_to_note_num(
    beat_out: torch.Tensor,
    beat_number: torch.Tensor,
    actual_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Broadcast beat-level representations back to note level.

    Each note receives the representation of its corresponding beat.

    Args:
        beat_out: Tensor of shape (N, T_beat, C) - beat-level representations
        beat_number: Tensor of shape (N, T_note) - beat index for each note
        actual_lengths: Optional tensor of shape (N,) with actual sequence lengths.
                       If not provided, will be computed from beat_number.

    Returns:
        Tensor of shape (N, T_note, C) - beat representations repeated for each note
    """
    # Shift beat numbers to start from 0
    zero_shifted_beat_number = beat_number - beat_number[:, 0:1]

    # Get actual note lengths - use provided or compute
    if actual_lengths is not None:
        len_note = actual_lengths
    else:
        len_note = cal_length_from_padded_beat_numbers(beat_number)

    # Build index arrays for sparse matrix
    batch_indices = torch.cat([torch.ones(length) * i for i, length in enumerate(len_note)]).long()
    note_indices = torch.cat([torch.arange(length) for length in len_note])
    beat_indices = torch.cat([
        zero_shifted_beat_number[i, :length]
        for i, length in enumerate(len_note)
    ]).long()

    # Clamp beat_indices to valid range to prevent out-of-bounds errors
    max_beat_idx = beat_out.shape[1] - 1
    beat_indices = beat_indices.clamp(0, max_beat_idx)

    # Create span matrix (N, T_note, T_beat)
    span_mat = torch.zeros(beat_number.shape[0], beat_number.shape[1], beat_out.shape[1]).to(beat_out.device)
    span_mat[batch_indices.to(beat_out.device), note_indices.to(beat_out.device), beat_indices.to(beat_out.device)] = 1

    # Multiply to get note-level representations
    spanned_beat = torch.bmm(span_mat, beat_out)

    return spanned_beat


def run_hierarchy_lstm_with_pack(sequence: torch.Tensor, lstm: nn.LSTM) -> torch.Tensor:
    """
    Run LSTM on zero-padded sequences using pack/pad for efficiency.

    Args:
        sequence: Tensor of shape (N, T, C) - zero-padded sequences
        lstm: LSTM layer to apply

    Returns:
        Tensor of shape (N, T, H) - LSTM output (zero-padded)
    """
    # Calculate actual sequence lengths
    batch_note_length = sequence.shape[1] - (sequence == 0).all(dim=-1).sum(-1)
    batch_note_length = batch_note_length.clamp(min=1)  # Ensure at least 1 for pack

    # Pack, run LSTM, unpack
    packed_sequence = pack_padded_sequence(sequence, batch_note_length.cpu(), batch_first=True, enforce_sorted=False)
    hidden_out, _ = lstm(packed_sequence)
    hidden_out, _ = pad_packed_sequence(hidden_out, batch_first=True)

    return hidden_out
