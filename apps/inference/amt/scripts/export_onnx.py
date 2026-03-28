"""Export Aria-AMT encoder and decoder to ONNX format.

Converts the Aria-AMT PyTorch encoder-decoder model (Whisper-class, 49M params)
to ONNX for deployment on Cloudflare Containers without PyTorch.

Produces up to 3 ONNX files:
  - encoder.onnx: log-mel spectrogram -> encoder features
  - decoder_prefill.onnx: initial token sequence + encoder features -> logits + KV cache
  - decoder_step.onnx: single token + KV cache -> next logits + updated KV cache

KV cache export strategy (tried in order):
  A) Externalized KV cache as explicit I/O tensors (preferred)
  B) Stateless decoder without KV cache reuse (slower, guaranteed to work)
  C) Encoder-only export (decoder stays in PyTorch)

Usage:
    uv run python scripts/export_onnx.py \
        --checkpoint /path/to/model.safetensors \
        --output-dir /path/to/output/ \
        --config medium-double \
        --validate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# -- Patch amt.config before any other amt imports (same as transcription.py) --
import amt.config as _amt_config

_AMT_CONFIG = {
    "tokenizer": {
        "velocity_quantization": {"step": 5, "default": 60},
        "time_quantization": {"num_steps": 3000, "step": 10},
    },
    "audio": {
        "sample_rate": 16000,
        "n_fft_large": 2048,
        "n_fft_med": 2048,
        "n_fft_small": 800,
        "hop_len": 160,
        "chunk_len": 30,
        "n_mels_large": 384,
        "n_mels_med": 256,
        "n_mels_small": 128,
    },
    "data": {"stride_factor": 15, "max_seq_len": 4096},
}

_original_load_config = _amt_config.load_config


def _patched_load_config():
    cfg = _original_load_config()
    if "audio" not in cfg or "time_quantization" not in cfg.get("tokenizer", {}):
        return _AMT_CONFIG
    return cfg


_amt_config.load_config = _patched_load_config

from amt.config import load_model_config
from amt.inference.model import AmtEncoderDecoder, ModelConfig, KVCache
from amt.tokenizer import AmtTokenizer

# Import weight loader from sibling transcription module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from transcription import _load_weight

# -- Constants --
MAX_BLOCK_LEN = 4096
MAX_AUDIO_LEN = 1500  # n_audio_ctx for medium-double config


# ---------------------------------------------------------------------------
# Strategy A: Wrapper modules that externalize KV cache as tensor I/O
# ---------------------------------------------------------------------------


class EncoderWrapper(nn.Module):
    """Thin wrapper around AudioEncoder for clean ONNX export.

    Input:  log_mels [1, n_mels, T] (T = 3000 for 30s at 16kHz with hop=160)
    Output: audio_features [1, n_audio_ctx, n_audio_state]
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, log_mels: Tensor) -> Tensor:
        return self.encoder(xa=log_mels)


class DecoderPrefillWrapper(nn.Module):
    """Decoder wrapper for prefill (first forward pass).

    Externalizes all KV caches: takes empty cache tensors as input,
    returns populated cache tensors as output alongside logits.

    For each decoder block i, the KV cache has 4 tensors:
      - self_attn_k_cache_i, self_attn_v_cache_i  (self-attention)
      - cross_attn_k_cache_i, cross_attn_v_cache_i (cross-attention)

    Inputs:
      - x: token ids [1, seq_len]
      - xa: encoder features [1, audio_len, d_model]
      - x_input_pos: [seq_len] position indices for tokens
      - xa_input_pos: [audio_len] position indices for encoder features
      - *kv_cache_inputs: flattened list of cache tensors (all zeros for prefill)

    Outputs:
      - logits: [1, seq_len, vocab_size]
      - *kv_cache_outputs: updated cache tensors after prefill
    """

    def __init__(self, decoder: nn.Module, n_layers: int):
        super().__init__()
        self.decoder = decoder
        self.n_layers = n_layers

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        x_input_pos: Tensor,
        xa_input_pos: Tensor,
        *kv_cache_flat: Tensor,
    ) -> tuple:
        # Install cache tensors into decoder blocks before forward pass
        self._install_caches(kv_cache_flat)

        # Run decoder forward
        logits = self.decoder(
            x=x, xa=xa, x_input_pos=x_input_pos, xa_input_pos=xa_input_pos
        )

        # Extract updated caches
        updated_caches = self._extract_caches()
        return (logits, *updated_caches)

    def _install_caches(self, kv_cache_flat: tuple[Tensor, ...]) -> None:
        """Write external tensors into the decoder's KVCache buffers."""
        idx = 0
        for block in self.decoder.blocks:
            # Self-attention cache
            block.attn.kv_cache.k_cache.copy_(kv_cache_flat[idx])
            block.attn.kv_cache.v_cache.copy_(kv_cache_flat[idx + 1])
            idx += 2
            # Cross-attention cache
            block.cross_attn.kv_cache.k_cache.copy_(kv_cache_flat[idx])
            block.cross_attn.kv_cache.v_cache.copy_(kv_cache_flat[idx + 1])
            idx += 2

    def _extract_caches(self) -> list[Tensor]:
        """Read updated KV cache tensors from decoder blocks."""
        caches = []
        for block in self.decoder.blocks:
            caches.append(block.attn.kv_cache.k_cache.clone())
            caches.append(block.attn.kv_cache.v_cache.clone())
            caches.append(block.cross_attn.kv_cache.k_cache.clone())
            caches.append(block.cross_attn.kv_cache.v_cache.clone())
        return caches


class DecoderStepWrapper(nn.Module):
    """Decoder wrapper for autoregressive step (single token).

    Same externalized KV cache pattern as prefill, but:
      - x is a single token [1, 1]
      - x_input_pos is a single position [1]
      - xa_input_pos is empty [] (cross-attention KV is already cached)

    Inputs:
      - x: single token [1, 1]
      - xa: encoder features [1, audio_len, d_model] (passed through but
             cross-attn uses cached values when xa_input_pos is empty)
      - x_input_pos: [1] current position
      - *kv_cache_inputs: cache tensors from previous step

    Outputs:
      - logits: [1, 1, vocab_size]
      - *kv_cache_outputs: updated cache tensors
    """

    def __init__(self, decoder: nn.Module, n_layers: int):
        super().__init__()
        self.decoder = decoder
        self.n_layers = n_layers

    def forward(
        self,
        x: Tensor,
        xa: Tensor,
        x_input_pos: Tensor,
        *kv_cache_flat: Tensor,
    ) -> tuple:
        # Install cache tensors
        self._install_caches(kv_cache_flat)

        # For step mode, xa_input_pos is empty (cross-attn uses cached KV)
        xa_input_pos = torch.tensor([], dtype=torch.long, device=x.device)

        logits = self.decoder(
            x=x, xa=xa, x_input_pos=x_input_pos, xa_input_pos=xa_input_pos
        )

        updated_caches = self._extract_caches()
        return (logits, *updated_caches)

    def _install_caches(self, kv_cache_flat: tuple[Tensor, ...]) -> None:
        idx = 0
        for block in self.decoder.blocks:
            block.attn.kv_cache.k_cache.copy_(kv_cache_flat[idx])
            block.attn.kv_cache.v_cache.copy_(kv_cache_flat[idx + 1])
            idx += 2
            block.cross_attn.kv_cache.k_cache.copy_(kv_cache_flat[idx])
            block.cross_attn.kv_cache.v_cache.copy_(kv_cache_flat[idx + 1])
            idx += 2

    def _extract_caches(self) -> list[Tensor]:
        caches = []
        for block in self.decoder.blocks:
            caches.append(block.attn.kv_cache.k_cache.clone())
            caches.append(block.attn.kv_cache.v_cache.clone())
            caches.append(block.cross_attn.kv_cache.k_cache.clone())
            caches.append(block.cross_attn.kv_cache.v_cache.clone())
        return caches


# ---------------------------------------------------------------------------
# Strategy B: Stateless decoder (no KV cache reuse)
# ---------------------------------------------------------------------------


class DecoderStatelessWrapper(nn.Module):
    """Stateless decoder that re-computes full attention each step.

    Takes all tokens generated so far + encoder features, returns logits.
    No KV cache -- slower but guaranteed to export cleanly to ONNX.

    This bypasses the KVCache entirely by creating a standalone forward pass
    that replicates the decoder logic without in-place cache mutations.
    """

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(self, x: Tensor, xa: Tensor) -> Tensor:
        """Run decoder without KV cache.

        Args:
            x: all token ids so far [1, seq_len]
            xa: encoder features [1, audio_len, d_model]

        Returns:
            logits [1, seq_len, vocab_size]
        """
        seq_len = x.shape[1]
        audio_len = xa.shape[1]
        x_input_pos = torch.arange(seq_len, device=x.device, dtype=torch.long)
        xa_input_pos = torch.arange(audio_len, device=x.device, dtype=torch.long)

        # Ensure causal mask covers this sequence length
        if self.decoder.causal_mask is None or self.decoder.causal_mask.shape[0] < seq_len:
            self.decoder.causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
            )

        logits = self.decoder(
            x=x, xa=xa, x_input_pos=x_input_pos, xa_input_pos=xa_input_pos
        )
        return logits


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    checkpoint_path: str,
    config_name: str,
    device: str = "cpu",
) -> AmtEncoderDecoder:
    """Load Aria-AMT model from safetensors checkpoint.

    Args:
        checkpoint_path: Path to .safetensors file.
        config_name: Model config name (e.g. "medium-double").
        device: Device to load onto.

    Returns:
        Loaded AmtEncoderDecoder in inference mode.
    """
    model_config = ModelConfig(**load_model_config(config_name))

    tokenizer = AmtTokenizer()
    model_config.set_vocab_size(tokenizer.vocab_size)

    model = AmtEncoderDecoder(model_config)
    state_dict = _load_weight(checkpoint_path, device=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def setup_kv_cache(
    model: AmtEncoderDecoder,
    max_seq_len: int = MAX_BLOCK_LEN,
    max_audio_len: int = MAX_AUDIO_LEN,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> None:
    """Initialize KV caches on the decoder blocks.

    Replicates the setup from transcription.py but targets the specified device
    and dtype (float32 for ONNX compatibility).
    """
    decoder = model.decoder
    decoder.causal_mask = torch.tril(
        torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=device)
    )
    for b in decoder.blocks:
        head_dim = decoder.n_state // decoder.n_head
        b.attn.kv_cache = KVCache(
            max_batch_size=1,
            max_seq_length=max_seq_len,
            n_heads=decoder.n_head,
            head_dim=head_dim,
            dtype=dtype,
        ).to(device)
        b.cross_attn.kv_cache = KVCache(
            max_batch_size=1,
            max_seq_length=max_audio_len,
            n_heads=decoder.n_head,
            head_dim=head_dim,
            dtype=dtype,
        ).to(device)


# ---------------------------------------------------------------------------
# ONNX export functions
# ---------------------------------------------------------------------------


def export_encoder(
    model: AmtEncoderDecoder,
    output_path: Path,
    n_mels: int,
    opset_version: int = 17,
) -> Path:
    """Export the audio encoder to ONNX.

    Args:
        model: Loaded AmtEncoderDecoder.
        output_path: Directory to write encoder.onnx.
        n_mels: Number of mel frequency bins (from model config).
        opset_version: ONNX opset version.

    Returns:
        Path to the exported encoder.onnx file.
    """
    encoder_path = output_path / "encoder.onnx"
    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()

    # Dummy input: [batch=1, n_mels, time_steps]
    # For 30s audio at 16kHz with hop=160: 480000/160 = 3000 frames
    time_steps = 3000
    dummy_mels = torch.randn(1, n_mels, time_steps)

    print(f"Exporting encoder to {encoder_path}")
    print(f"  Input shape: [1, {n_mels}, {time_steps}]")

    torch.onnx.export(
        wrapper,
        (dummy_mels,),
        str(encoder_path),
        opset_version=opset_version,
        input_names=["log_mels"],
        output_names=["audio_features"],
        dynamic_axes={
            "log_mels": {2: "time_steps"},
            "audio_features": {1: "audio_len"},
        },
    )

    print(f"  Encoder exported successfully ({encoder_path.stat().st_size / 1e6:.1f} MB)")
    return encoder_path


def _get_cache_names(n_layers: int, suffix: str = "") -> list[str]:
    """Generate ordered cache tensor names for decoder I/O.

    For each layer, produces 4 names:
      self_attn_k_cache_{i}, self_attn_v_cache_{i},
      cross_attn_k_cache_{i}, cross_attn_v_cache_{i}

    Args:
        n_layers: Number of decoder blocks.
        suffix: Optional suffix (e.g. "_out") for output names.

    Returns:
        List of cache tensor names.
    """
    names = []
    for i in range(n_layers):
        names.append(f"self_attn_k_cache_{i}{suffix}")
        names.append(f"self_attn_v_cache_{i}{suffix}")
        names.append(f"cross_attn_k_cache_{i}{suffix}")
        names.append(f"cross_attn_v_cache_{i}{suffix}")
    return names


def _make_empty_caches(
    n_layers: int,
    n_heads: int,
    head_dim: int,
    max_seq_len: int = MAX_BLOCK_LEN,
    max_audio_len: int = MAX_AUDIO_LEN,
) -> list[Tensor]:
    """Create zero-initialized cache tensors for all decoder layers.

    Returns list of tensors in the order:
      [self_k_0, self_v_0, cross_k_0, cross_v_0, self_k_1, ...]
    """
    caches = []
    for _ in range(n_layers):
        # Self-attention caches: [1, n_heads, max_seq_len, head_dim]
        caches.append(torch.zeros(1, n_heads, max_seq_len, head_dim))
        caches.append(torch.zeros(1, n_heads, max_seq_len, head_dim))
        # Cross-attention caches: [1, n_heads, max_audio_len, head_dim]
        caches.append(torch.zeros(1, n_heads, max_audio_len, head_dim))
        caches.append(torch.zeros(1, n_heads, max_audio_len, head_dim))
    return caches


def export_decoder_strategy_a(
    model: AmtEncoderDecoder,
    output_path: Path,
    opset_version: int = 17,
) -> tuple[Optional[Path], Optional[Path]]:
    """Export decoder with externalized KV cache (Strategy A).

    Produces two ONNX files:
      - decoder_prefill.onnx: initial sequence -> logits + populated KV cache
      - decoder_step.onnx: single token + KV cache -> logits + updated KV cache

    Args:
        model: Loaded AmtEncoderDecoder with KV caches initialized.
        output_path: Directory to write ONNX files.
        opset_version: ONNX opset version.

    Returns:
        Tuple of (prefill_path, step_path), or (None, None) if export fails.

    Raises:
        Exception: Re-raised with context if ONNX export fails.
    """
    decoder = model.decoder
    n_layers = len(list(decoder.blocks))
    n_heads = decoder.n_head
    head_dim = decoder.n_state // decoder.n_head
    n_audio_ctx = MAX_AUDIO_LEN

    print(f"Strategy A: externalizing KV cache for {n_layers} layers")
    print(f"  n_heads={n_heads}, head_dim={head_dim}")

    # -- Prefill export --
    prefill_path = output_path / "decoder_prefill.onnx"
    prefill_wrapper = DecoderPrefillWrapper(decoder, n_layers)
    prefill_wrapper.eval()

    # Dummy inputs for prefill
    prefill_seq_len = 1  # BOS token
    dummy_x = torch.zeros(1, prefill_seq_len, dtype=torch.long)
    dummy_xa = torch.randn(1, n_audio_ctx, decoder.n_state)
    dummy_x_pos = torch.arange(prefill_seq_len, dtype=torch.long)
    dummy_xa_pos = torch.arange(n_audio_ctx, dtype=torch.long)
    dummy_caches = _make_empty_caches(n_layers, n_heads, head_dim)

    # Build input/output name lists
    cache_in_names = _get_cache_names(n_layers, suffix="")
    cache_out_names = _get_cache_names(n_layers, suffix="_out")
    prefill_input_names = ["x", "xa", "x_input_pos", "xa_input_pos"] + cache_in_names
    prefill_output_names = ["logits"] + cache_out_names

    # Dynamic axes: token sequence length is variable
    prefill_dynamic_axes = {
        "x": {1: "seq_len"},
        "x_input_pos": {0: "seq_len"},
        "logits": {1: "seq_len"},
    }

    print(f"  Exporting prefill decoder to {prefill_path}")
    prefill_args = (dummy_x, dummy_xa, dummy_x_pos, dummy_xa_pos, *dummy_caches)

    torch.onnx.export(
        prefill_wrapper,
        prefill_args,
        str(prefill_path),
        opset_version=opset_version,
        input_names=prefill_input_names,
        output_names=prefill_output_names,
        dynamic_axes=prefill_dynamic_axes,
    )
    print(f"  Prefill exported ({prefill_path.stat().st_size / 1e6:.1f} MB)")

    # -- Step export --
    step_path = output_path / "decoder_step.onnx"
    step_wrapper = DecoderStepWrapper(decoder, n_layers)
    step_wrapper.eval()

    dummy_step_x = torch.zeros(1, 1, dtype=torch.long)
    dummy_step_xa = torch.randn(1, n_audio_ctx, decoder.n_state)
    dummy_step_pos = torch.tensor([prefill_seq_len], dtype=torch.long)
    # For step, reuse same cache shape (would be populated from prefill)
    dummy_step_caches = _make_empty_caches(n_layers, n_heads, head_dim)

    step_input_names = ["x", "xa", "x_input_pos"] + cache_in_names
    step_output_names = ["logits"] + cache_out_names

    print(f"  Exporting step decoder to {step_path}")
    step_args = (dummy_step_x, dummy_step_xa, dummy_step_pos, *dummy_step_caches)

    torch.onnx.export(
        step_wrapper,
        step_args,
        str(step_path),
        opset_version=opset_version,
        input_names=step_input_names,
        output_names=step_output_names,
    )
    print(f"  Step exported ({step_path.stat().st_size / 1e6:.1f} MB)")

    return prefill_path, step_path


def export_decoder_strategy_b(
    model: AmtEncoderDecoder,
    output_path: Path,
    opset_version: int = 17,
) -> Optional[Path]:
    """Export stateless decoder without KV cache (Strategy B).

    Single ONNX file that takes all tokens + encoder features, returns logits.
    Each step re-runs full attention over all tokens. Slower but no in-place
    mutation issues.

    Args:
        model: Loaded AmtEncoderDecoder.
        output_path: Directory to write ONNX file.
        opset_version: ONNX opset version.

    Returns:
        Path to decoder_stateless.onnx, or None if export fails.
    """
    decoder_path = output_path / "decoder_stateless.onnx"
    wrapper = DecoderStatelessWrapper(model.decoder)
    wrapper.eval()

    n_audio_ctx = MAX_AUDIO_LEN
    n_state = model.decoder.n_state

    # Need KV caches initialized for the forward pass even in stateless mode,
    # since the underlying decoder code still writes to them.
    setup_kv_cache(model, dtype=torch.float32)

    dummy_x = torch.zeros(1, 1, dtype=torch.long)
    dummy_xa = torch.randn(1, n_audio_ctx, n_state)

    print(f"Strategy B: exporting stateless decoder to {decoder_path}")

    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_xa),
        str(decoder_path),
        opset_version=opset_version,
        input_names=["x", "xa"],
        output_names=["logits"],
        dynamic_axes={
            "x": {1: "seq_len"},
            "logits": {1: "seq_len"},
        },
    )
    print(f"  Stateless decoder exported ({decoder_path.stat().st_size / 1e6:.1f} MB)")
    return decoder_path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_encoder(
    model: AmtEncoderDecoder,
    encoder_onnx_path: Path,
    n_mels: int,
) -> bool:
    """Validate ONNX encoder output matches PyTorch within tolerance.

    Runs both models on the same random input and checks max absolute
    difference is below 0.01.

    Args:
        model: Original PyTorch model.
        encoder_onnx_path: Path to exported encoder.onnx.
        n_mels: Number of mel bins.

    Returns:
        True if validation passes.

    Raises:
        ImportError: If onnxruntime is not installed.
        ValueError: If drift exceeds threshold.
    """
    import onnxruntime as ort

    print("Validating encoder ONNX export...")

    # Generate deterministic random input
    torch.manual_seed(42)
    dummy_mels = torch.randn(1, n_mels, 3000)

    # PyTorch reference
    with torch.no_grad():
        pt_output = model.encoder(xa=dummy_mels).numpy()

    # ONNX Runtime
    session = ort.InferenceSession(str(encoder_onnx_path))
    ort_output = session.run(
        ["audio_features"],
        {"log_mels": dummy_mels.numpy()},
    )[0]

    max_diff = np.max(np.abs(pt_output - ort_output))
    mean_diff = np.mean(np.abs(pt_output - ort_output))
    print(f"  Encoder validation: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    if max_diff >= 0.01:
        raise ValueError(
            f"Encoder ONNX drift too large: max_diff={max_diff:.6f} (threshold: 0.01)"
        )

    print("  Encoder validation PASSED")
    return True


def validate_decoder_strategy_a(
    model: AmtEncoderDecoder,
    prefill_path: Path,
    step_path: Path,
) -> bool:
    """Validate Strategy A decoder export against PyTorch.

    Runs a short prefill + 1 step sequence through both PyTorch and ONNX,
    comparing logits at each stage.

    Args:
        model: Original PyTorch model (with KV caches set up).
        prefill_path: Path to decoder_prefill.onnx.
        step_path: Path to decoder_step.onnx.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If drift exceeds threshold.
    """
    import onnxruntime as ort

    print("Validating Strategy A decoder ONNX export...")

    decoder = model.decoder
    n_layers = len(list(decoder.blocks))
    n_heads = decoder.n_head
    head_dim = decoder.n_state // decoder.n_head
    n_audio_ctx = MAX_AUDIO_LEN

    # Deterministic inputs
    torch.manual_seed(42)
    dummy_xa = torch.randn(1, n_audio_ctx, decoder.n_state)
    dummy_x = torch.zeros(1, 1, dtype=torch.long)  # BOS

    # -- PyTorch reference: prefill --
    setup_kv_cache(model, dtype=torch.float32)
    with torch.no_grad():
        pt_prefill_logits = decoder(
            x=dummy_x,
            xa=dummy_xa,
            x_input_pos=torch.arange(1, dtype=torch.long),
            xa_input_pos=torch.arange(n_audio_ctx, dtype=torch.long),
        ).numpy()

    # Extract PT caches after prefill for step comparison
    pt_caches_after_prefill = []
    for block in decoder.blocks:
        pt_caches_after_prefill.append(block.attn.kv_cache.k_cache.clone().numpy())
        pt_caches_after_prefill.append(block.attn.kv_cache.v_cache.clone().numpy())
        pt_caches_after_prefill.append(block.cross_attn.kv_cache.k_cache.clone().numpy())
        pt_caches_after_prefill.append(block.cross_attn.kv_cache.v_cache.clone().numpy())

    # -- ONNX prefill --
    prefill_session = ort.InferenceSession(str(prefill_path))
    empty_caches = _make_empty_caches(n_layers, n_heads, head_dim)
    cache_in_names = _get_cache_names(n_layers, suffix="")

    prefill_feeds = {
        "x": dummy_x.numpy(),
        "xa": dummy_xa.numpy(),
        "x_input_pos": np.arange(1, dtype=np.int64),
        "xa_input_pos": np.arange(n_audio_ctx, dtype=np.int64),
    }
    for name, tensor in zip(cache_in_names, empty_caches):
        prefill_feeds[name] = tensor.numpy()

    ort_prefill_outputs = prefill_session.run(None, prefill_feeds)
    ort_prefill_logits = ort_prefill_outputs[0]

    prefill_diff = np.max(np.abs(pt_prefill_logits - ort_prefill_logits))
    print(f"  Prefill logits max_diff: {prefill_diff:.6f}")

    if prefill_diff >= 0.01:
        raise ValueError(
            f"Prefill decoder drift too large: {prefill_diff:.6f} (threshold: 0.01)"
        )

    # -- PyTorch reference: step --
    dummy_step_x = torch.tensor([[1]], dtype=torch.long)  # arbitrary token
    with torch.no_grad():
        pt_step_logits = decoder(
            x=dummy_step_x,
            xa=dummy_xa,
            x_input_pos=torch.tensor([1], dtype=torch.long),
            xa_input_pos=torch.tensor([], dtype=torch.long),
        ).numpy()

    # -- ONNX step --
    step_session = ort.InferenceSession(str(step_path))
    cache_out_tensors = ort_prefill_outputs[1:]  # caches from prefill

    step_feeds = {
        "x": np.array([[1]], dtype=np.int64),
        "xa": dummy_xa.numpy(),
        "x_input_pos": np.array([1], dtype=np.int64),
    }
    for name, arr in zip(cache_in_names, cache_out_tensors):
        step_feeds[name] = arr

    ort_step_outputs = step_session.run(None, step_feeds)
    ort_step_logits = ort_step_outputs[0]

    step_diff = np.max(np.abs(pt_step_logits - ort_step_logits))
    print(f"  Step logits max_diff: {step_diff:.6f}")

    if step_diff >= 0.01:
        raise ValueError(
            f"Step decoder drift too large: {step_diff:.6f} (threshold: 0.01)"
        )

    print("  Strategy A decoder validation PASSED")
    return True


def validate_decoder_strategy_b(
    model: AmtEncoderDecoder,
    decoder_path: Path,
) -> bool:
    """Validate Strategy B decoder export against PyTorch.

    Args:
        model: Original PyTorch model.
        decoder_path: Path to decoder_stateless.onnx.

    Returns:
        True if validation passes.

    Raises:
        ValueError: If drift exceeds threshold.
    """
    import onnxruntime as ort

    print("Validating Strategy B decoder ONNX export...")

    decoder = model.decoder
    n_audio_ctx = MAX_AUDIO_LEN

    torch.manual_seed(42)
    dummy_xa = torch.randn(1, n_audio_ctx, decoder.n_state)
    dummy_x = torch.zeros(1, 1, dtype=torch.long)

    # PyTorch reference (re-init caches for clean comparison)
    setup_kv_cache(model, dtype=torch.float32)
    with torch.no_grad():
        x_pos = torch.arange(1, dtype=torch.long)
        xa_pos = torch.arange(n_audio_ctx, dtype=torch.long)
        pt_logits = decoder(
            x=dummy_x, xa=dummy_xa, x_input_pos=x_pos, xa_input_pos=xa_pos
        ).numpy()

    # ONNX
    session = ort.InferenceSession(str(decoder_path))
    ort_logits = session.run(
        ["logits"],
        {"x": dummy_x.numpy(), "xa": dummy_xa.numpy()},
    )[0]

    max_diff = np.max(np.abs(pt_logits - ort_logits))
    print(f"  Stateless decoder logits max_diff: {max_diff:.6f}")

    if max_diff >= 0.01:
        raise ValueError(
            f"Stateless decoder drift too large: {max_diff:.6f} (threshold: 0.01)"
        )

    print("  Strategy B decoder validation PASSED")
    return True


# ---------------------------------------------------------------------------
# Main export pipeline
# ---------------------------------------------------------------------------


def export_all(
    checkpoint_path: str,
    output_dir: str,
    config_name: str = "medium-double",
    validate: bool = False,
    opset_version: int = 17,
) -> dict[str, str]:
    """Run the full ONNX export pipeline.

    Tries Strategy A for the decoder first. If it fails, falls back to B,
    then C (encoder-only). Documents which strategy succeeded.

    Args:
        checkpoint_path: Path to .safetensors model checkpoint.
        output_dir: Directory to write ONNX files.
        config_name: Model config name.
        validate: Whether to run validation after export.
        opset_version: ONNX opset version.

    Returns:
        Dict with keys: "strategy", "encoder", "decoder_prefill", "decoder_step",
        "decoder_stateless" -- values are file paths or None.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = {
        "strategy": None,
        "encoder": None,
        "decoder_prefill": None,
        "decoder_step": None,
        "decoder_stateless": None,
    }

    # Load model
    print(f"Loading model from {checkpoint_path} (config: {config_name})")
    start = time.time()
    model = load_model(checkpoint_path, config_name, device="cpu")
    model_config = ModelConfig(**load_model_config(config_name))
    print(f"Model loaded in {time.time() - start:.1f}s")

    # Export encoder (same for all strategies)
    start = time.time()
    encoder_path = export_encoder(model, output_path, model_config.n_mels, opset_version)
    result["encoder"] = str(encoder_path)
    print(f"Encoder export took {time.time() - start:.1f}s")

    if validate:
        validate_encoder(model, encoder_path, model_config.n_mels)

    # Try decoder strategies in order
    # Strategy A: externalized KV cache
    print("\n--- Attempting Strategy A (externalized KV cache) ---")
    try:
        setup_kv_cache(model, dtype=torch.float32)
        prefill_path, step_path = export_decoder_strategy_a(
            model, output_path, opset_version
        )
        result["strategy"] = "A"
        result["decoder_prefill"] = str(prefill_path)
        result["decoder_step"] = str(step_path)

        if validate:
            validate_decoder_strategy_a(model, prefill_path, step_path)

        print("\nStrategy A succeeded.")

    except Exception as e:
        print(f"\nStrategy A failed: {e}")
        print("Falling back to Strategy B (stateless decoder)...")

        # Strategy B: stateless decoder
        try:
            decoder_path = export_decoder_strategy_b(model, output_path, opset_version)
            result["strategy"] = "B"
            result["decoder_stateless"] = str(decoder_path)

            if validate:
                validate_decoder_strategy_b(model, decoder_path)

            print("\nStrategy B succeeded.")

        except Exception as e2:
            print(f"\nStrategy B failed: {e2}")
            print("Falling back to Strategy C (encoder-only)...")

            result["strategy"] = "C"
            print(
                "\nStrategy C: encoder exported to ONNX. "
                "Decoder must remain in PyTorch."
            )
            print(
                "  The server.py should use ONNX encoder + PyTorch decoder hybrid."
            )

    # Write manifest
    _write_manifest(result, output_path)

    return result


def _write_manifest(result: dict, output_path: Path) -> None:
    """Write export_manifest.json documenting what was exported.

    Args:
        result: Export result dict from export_all().
        output_path: Directory containing ONNX files.
    """
    import json

    manifest = {
        "strategy": result["strategy"],
        "files": {},
    }
    for key in ["encoder", "decoder_prefill", "decoder_step", "decoder_stateless"]:
        if result[key] is not None:
            path = Path(result[key])
            manifest["files"][key] = {
                "filename": path.name,
                "size_bytes": path.stat().st_size,
            }

    manifest_path = output_path / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Export Aria-AMT model to ONNX format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python scripts/export_onnx.py \\
      --checkpoint model.safetensors \\
      --output-dir ./onnx_output/ \\
      --config medium-double \\
      --validate
        """,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to .safetensors model checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write ONNX files.",
    )
    parser.add_argument(
        "--config",
        default="medium-double",
        help="Model config name (default: medium-double).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation comparing ONNX output to PyTorch.",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )

    args = parser.parse_args()

    # Verify checkpoint exists
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        parser.error(f"Checkpoint not found: {ckpt}")
    if not ckpt.suffix == ".safetensors":
        parser.error(f"Only safetensors checkpoints supported. Got: {ckpt}")

    result = export_all(
        checkpoint_path=str(ckpt),
        output_dir=args.output_dir,
        config_name=args.config,
        validate=args.validate,
        opset_version=args.opset_version,
    )

    print("\n=== Export Summary ===")
    print(f"Strategy: {result['strategy']}")
    for key, path in result.items():
        if key != "strategy" and path is not None:
            print(f"  {key}: {path}")

    if result["strategy"] == "C":
        print(
            "\nWARNING: Only encoder was exported to ONNX. "
            "Decoder requires PyTorch at runtime."
        )
        sys.exit(1)
    elif result["strategy"] == "B":
        print(
            "\nNOTE: Stateless decoder exported (no KV cache reuse). "
            "Each decoding step will re-run full attention. "
            "Expect slower inference than Strategy A."
        )


if __name__ == "__main__":
    main()
