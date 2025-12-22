"""
Utility functions for hierarchical attention networks.

Ported from PercePiano's virtuoso/utils.py and virtuoso/model_utils.py.
These functions handle hierarchical aggregation (note -> beat -> measure)
and broadcasting back to note level.
"""

import warnings
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Tuple, Optional

# Global debug flag - set to True to enable detailed logging
HIERARCHY_DEBUG = False

def set_hierarchy_debug(enabled: bool) -> None:
    """Enable or disable hierarchy debug logging."""
    global HIERARCHY_DEBUG
    HIERARCHY_DEBUG = enabled
    if enabled:
        print("[HIERARCHY_DEBUG] Debug mode ENABLED - will log detailed information")


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


def find_boundaries(diff_boundary: torch.Tensor, higher_indices: torch.Tensor, batch_idx: int) -> List[int]:
    """
    Find boundary indices for a single batch element.

    Matches original PercePiano logic exactly.

    Args:
        diff_boundary: Tensor of shape (num_boundaries, 2) with [batch_idx, position]
        higher_indices: Tensor of shape (N, T) with beat/measure numbers
        batch_idx: Index of the batch element to process

    Returns:
        List of boundary indices including 0 and sequence length
    """
    # Original: [0] + (positions where diff == 1) + [last non-zero position + 1]
    out = [0] + (diff_boundary[diff_boundary[:, 0] == batch_idx][:, 1] + 1).tolist()

    # Final boundary: last non-zero position + 1
    # Handle edge case where sequence might be all zeros
    nonzero_positions = torch.nonzero(higher_indices[batch_idx])
    if len(nonzero_positions) > 0:
        out.append(torch.max(nonzero_positions).item() + 1)
    else:
        out.append(1)  # Minimum sequence length

    # Original duplicate removal: if the first boundary occurs at 0, it will be duplicated
    if len(out) > 1 and out[1] == 0:
        out.pop(0)

    return out


def find_boundaries_batch(
    beat_numbers: torch.Tensor,
    actual_lengths: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """
    Find beat/measure boundaries for a batch of sequences.

    Matches original PercePiano logic exactly: uses batched torch.nonzero().

    Args:
        beat_numbers: Tensor of shape (N, T) with beat/measure indices per note.
                      Zero-padded sequences.
        actual_lengths: Optional tensor - ignored for compatibility, not used in original.

    Returns:
        List of lists, each containing boundary indices for that batch element.
        Includes 0 at start and actual sequence length at end.
    """
    global HIERARCHY_DEBUG

    # Original: batched boundary detection using nonzero
    diff_boundary = torch.nonzero(beat_numbers[:, 1:] - beat_numbers[:, :-1] == 1).cpu()
    boundaries = [find_boundaries(diff_boundary, beat_numbers, i) for i in range(len(beat_numbers))]

    # Debug logging
    if HIERARCHY_DEBUG:
        batch_size = len(beat_numbers)
        print(f"\n[HIERARCHY_DEBUG] find_boundaries_batch:")
        print(f"  Input shape: {beat_numbers.shape}")
        print(f"  Batch size: {batch_size}")

        # Check for potential issues
        for i in range(min(3, batch_size)):  # Log first 3 samples
            b = boundaries[i]
            bt = beat_numbers[i].cpu()
            non_zero = torch.nonzero(bt != 0).squeeze(-1)
            actual_len = len(non_zero) if len(non_zero.shape) > 0 else 0

            print(f"  Sample {i}:")
            print(f"    Beat range: [{bt[bt != 0].min().item() if actual_len > 0 else 0}, "
                  f"{bt[bt != 0].max().item() if actual_len > 0 else 0}]")
            print(f"    Boundaries: {b[:10]}{'...' if len(b) > 10 else ''} (total: {len(b)})")

            if len(b) < 2:
                print(f"    [WARNING] Only {len(b)} boundaries - hierarchy may fail!")

            # Check for gaps
            if actual_len > 1:
                valid = bt[:actual_len]
                diffs = valid[1:] - valid[:-1]
                gaps = ((diffs != 0) & (diffs != 1)).sum().item()
                if gaps > 0:
                    print(f"    [WARNING] {gaps} beat index gaps > 1 detected!")

    return boundaries


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

    Matches original PercePiano logic exactly: uses torch.min() to find the
    position of minimum diff value (where padding starts or boundary occurs).

    Args:
        beat_numbers: Tensor of shape (N, T) with beat indices per note.
                      Assumes zero-padding at the end.

    Returns:
        Tensor of shape (N,) with actual sequence lengths
    """
    try:
        # Original logic: find position of minimum diff (where padding/boundary occurs)
        # torch.min(...)[1] returns the indices of the minimum values
        len_note = torch.min(torch.diff(beat_numbers, dim=1), dim=1)[1] + 1
    except Exception:
        # Fallback for edge cases (empty sequences, etc.)
        len_note = torch.LongTensor([beat_numbers.shape[1]] * len(beat_numbers)).to(beat_numbers.device)

    # Original correction: if len_note==1, use full sequence length
    # This handles cases where the minimum diff occurs at position 0
    len_note[len_note == 1] = beat_numbers.shape[1]

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
    global HIERARCHY_DEBUG

    if HIERARCHY_DEBUG:
        level_name = "Note->Beat" if lower_is_note else "Beat->Measure"
        print(f"\n[HIERARCHY_DEBUG] make_higher_node ({level_name}):")
        print(f"  lower_out shape: {lower_out.shape}")
        print(f"  higher_indices shape: {higher_indices.shape}")
        print(f"  lower_indices shape: {lower_indices.shape}")

    # Get attention scores
    similarity = attention_weights.get_attention(lower_out)  # (N, T, num_heads)

    if HIERARCHY_DEBUG:
        print(f"  attention similarity shape: {similarity.shape}")
        print(f"  attention stats: mean={similarity.mean():.4f}, std={similarity.std():.4f}")

    # Note: actual_lengths parameter kept for backward compatibility but not used
    # Original find_boundaries_batch uses nonzero() to detect boundaries directly

    if lower_is_note:
        # Notes -> Beats: use higher_indices directly
        boundaries = find_boundaries_batch(higher_indices)
    else:
        # Beats -> Measures: need to map through lower_indices
        higher_boundaries = find_boundaries_batch(higher_indices)
        zero_shifted_lower_indices = lower_indices - lower_indices[:, 0:1]

        # Calculate actual lengths for beat-level output
        num_zero_padded = ((lower_out != 0).sum(-1) == 0).sum(1)
        len_lower_out = (lower_out.shape[1] - num_zero_padded).tolist()

        boundaries = [
            zero_shifted_lower_indices[i, higher_boundaries[i][:-1]].tolist() + [len_lower_out[i]]
            for i in range(len(lower_out))
        ]

    # Apply softmax within each segment - match original exactly (no edge case fallback)
    softmax_similarity = torch.nn.utils.rnn.pad_sequence(
        [torch.cat(get_softmax_by_boundary(similarity[batch_idx], boundaries[batch_idx]))
         for batch_idx in range(len(lower_out))],
        batch_first=True
    )

    # Pad softmax_similarity to match lower_out sequence length (for fixed-padding scenarios)
    if softmax_similarity.shape[1] < lower_out.shape[1]:
        padding = torch.zeros(
            softmax_similarity.shape[0],
            lower_out.shape[1] - softmax_similarity.shape[1],
            softmax_similarity.shape[2],
            device=softmax_similarity.device,
            dtype=softmax_similarity.dtype,
        )
        softmax_similarity = torch.cat([softmax_similarity, padding], dim=1)

    # Compute weighted sum within each segment
    if hasattr(attention_weights, 'head_size'):
        x_split = torch.stack(lower_out.split(split_size=attention_weights.head_size, dim=2), dim=2)
        weighted_x = x_split * softmax_similarity.unsqueeze(-1).repeat(1, 1, 1, x_split.shape[-1])
        weighted_x = weighted_x.view(x_split.shape[0], x_split.shape[1], lower_out.shape[-1])

        # Original: single list comprehension with torch.cat for each batch
        higher_nodes = torch.nn.utils.rnn.pad_sequence(
            [torch.cat([torch.sum(weighted_x[i:i+1, boundaries[i][j-1]:boundaries[i][j], :], dim=1)
                        for j in range(1, len(boundaries[i]))], dim=0)
             for i in range(len(lower_out))],
            batch_first=True
        )
    else:
        # Non-head-size path (for compatibility) - matches original single-batch logic
        weighted_sum = softmax_similarity * lower_out
        higher_nodes = torch.cat(
            [torch.sum(weighted_sum[:, boundaries[i-1]:boundaries[i], :], dim=1)
             for i in range(1, len(boundaries))]).unsqueeze(0)

    if HIERARCHY_DEBUG:
        print(f"  output higher_nodes shape: {higher_nodes.shape}")

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

    # Build index arrays for sparse matrix (matches original exactly)
    batch_indices = torch.cat([torch.ones(length) * i for i, length in enumerate(len_note)]).long()
    note_indices = torch.cat([torch.arange(length) for length in len_note])
    beat_indices = torch.cat([
        zero_shifted_beat_number[i, :length]
        for i, length in enumerate(len_note)
    ]).long()

    # Create span matrix (N, T_note, T_beat)
    # Clamp indices to valid range to prevent CUDA assertion errors
    span_mat = torch.zeros(beat_number.shape[0], beat_number.shape[1], beat_out.shape[1]).to(beat_out.device)
    beat_indices_clamped = beat_indices.clamp(0, beat_out.shape[1] - 1)
    note_indices_clamped = note_indices.clamp(0, beat_number.shape[1] - 1)
    span_mat[batch_indices.to(beat_out.device), note_indices_clamped.to(beat_out.device), beat_indices_clamped.to(beat_out.device)] = 1

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
