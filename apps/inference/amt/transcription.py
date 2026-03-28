"""HuggingFace Inference Endpoints handler for Aria-AMT piano transcription.

Whisper-based automatic music transcription using EleutherAI's aria-amt model.
Accepts two audio fields (context + chunk) for overlapping-window transcription
with deduplication. Returns MIDI notes and pedal events compatible with
PerfNote/PerfPedalEvent Rust structs.

Compatible with HuggingFace Inference Endpoints custom handler pattern.

    REQUEST FLOW:
    +------------------+     +------------------+     +------------------+
    | WebM/Opus bytes  | --> | ffmpeg decode    | --> | 16kHz mono PCM   |
    | (base64 encoded) |     | to PCM float32   |     | (context + chunk)|
    +------------------+     +------------------+     +------------------+
                                                             |
                                                             v
                                                      +------------------+
                                                      | Aria-AMT         |
                                                      | log-mel -> enc   |
                                                      | -> dec -> tokens |
                                                      +------------------+
                                                             |
                                                             v
                                                      +------------------+
                                                      | Deduplicate:     |
                                                      | onset >= context |
                                                      | Adjust times     |
                                                      +------------------+
                                                             |
                                                             v
                                                      [{pitch, onset,
                                                        offset, velocity}]
"""

from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

# aria-amt imports (EleutherAI/aria-amt package)
#
# When both aria-amt and aria are installed (local dev), ariautils overwrites
# aria-amt's config/config.json, dropping the "audio" section entirely and
# replacing "tokenizer" with an incompatible schema (no time_quantization).
# Patch load_config before any amt.* module-level code runs.
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
    # If ariautils overwrote aria-amt's config, restore the full aria-amt config
    if "audio" not in cfg or "time_quantization" not in cfg.get("tokenizer", {}):
        return _AMT_CONFIG
    return cfg


_amt_config.load_config = _patched_load_config

from amt.config import load_model_config
from amt.inference.model import AmtEncoderDecoder, ModelConfig
from amt.tokenizer import AmtTokenizer
from amt.audio import AudioTransform

SAMPLE_RATE = 16000
CHUNK_LEN_S = 30
MAX_BLOCK_LEN = 4096
FFMPEG_DECODE_TIMEOUT_S = 60


def _load_weight(ckpt_path: str, device: str = "cpu") -> dict:
    """Load model weights from a safetensors checkpoint file.

    Only safetensors format is supported (safe serialization, no
    arbitrary code execution risk). Strips torch.compile prefixes
    if present.

    Args:
        ckpt_path: Path to a .safetensors checkpoint file.
        device: Target device for tensor placement.

    Raises:
        ValueError: If the checkpoint is not in safetensors format.
    """
    if not ckpt_path.endswith(".safetensors"):
        raise ValueError(
            f"Only safetensors checkpoints are supported. Got: {ckpt_path}"
        )

    from safetensors.torch import load_file

    state_dict = load_file(filename=ckpt_path, device=device)

    # Strip torch.compile prefixes
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            cleaned[k[len("_orig_mod."):]] = v
        else:
            cleaned[k] = v

    return cleaned


def decode_webm_to_pcm(audio_bytes: bytes) -> np.ndarray:
    """Decode WebM/Opus encoded audio bytes to 16kHz mono PCM float32.

    Uses ffmpeg subprocess for robust decoding of WebM containers with
    independent EBML headers (each chunk from MediaRecorder has its own).

    Args:
        audio_bytes: Raw WebM/Opus encoded bytes.

    Returns:
        numpy float32 array of audio samples at 16kHz mono.

    Raises:
        RuntimeError: If ffmpeg decoding fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as tmp_in:
        tmp_in.write(audio_bytes)
        tmp_in.flush()
        tmp_in_path = tmp_in.name

        result = subprocess.run(
            [
                "ffmpeg",
                "-i", tmp_in_path,
                "-f", "f32le",
                "-acodec", "pcm_f32le",
                "-ar", str(SAMPLE_RATE),
                "-ac", "1",
                "-v", "error",
                "pipe:1",
            ],
            capture_output=True,
            timeout=FFMPEG_DECODE_TIMEOUT_S,
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg decoding failed (exit {result.returncode}): "
            f"{result.stderr.decode('utf-8', errors='replace')}"
        )

    pcm_data = np.frombuffer(result.stdout, dtype=np.float32)
    if len(pcm_data) == 0:
        raise RuntimeError("ffmpeg produced empty output")

    return pcm_data


def midi_dict_to_notes_and_pedals(
    midi_dict: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert ariautils MidiDict to note and pedal event lists.

    Extracts note_msgs and pedal_msgs from the MidiDict, converting
    tick-based timestamps to seconds using the MidiDict's tempo map.

    Args:
        midi_dict: An ariautils.midi.MidiDict instance from tokenizer.detokenize().

    Returns:
        Tuple of (notes, pedal_events) in PerfNote/PerfPedalEvent format:
        - notes: [{"pitch": int, "onset": float, "offset": float, "velocity": int}]
        - pedal_events: [{"time": float, "value": int}]
    """
    notes = []
    for msg in midi_dict.note_msgs:
        data = msg["data"]
        start_ms = midi_dict.tick_to_ms(data["start"])
        end_ms = midi_dict.tick_to_ms(data["end"])
        notes.append({
            "pitch": int(data["pitch"]),
            "onset": round(start_ms / 1000.0, 4),
            "offset": round(end_ms / 1000.0, 4),
            "velocity": int(data["velocity"]),
        })

    pedal_events = []
    for msg in midi_dict.pedal_msgs:
        tick_ms = midi_dict.tick_to_ms(msg["tick"])
        # MidiDict pedal data: 0 = off, 1 = on
        # PerfPedalEvent value: 0 = off, 127 = on (CC64 convention)
        value = 127 if msg["data"] == 1 else 0
        pedal_events.append({
            "time": round(tick_ms / 1000.0, 4),
            "value": value,
        })

    notes.sort(key=lambda n: (n["onset"], n["pitch"]))
    pedal_events.sort(key=lambda e: e["time"])

    return notes, pedal_events


def deduplicate_notes(
    notes: list[dict[str, Any]],
    pedal_events: list[dict[str, Any]],
    context_duration_s: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter notes/pedals to only those in the current chunk, adjust timestamps.

    Notes with onset >= context_duration are from the current chunk.
    Their timestamps are adjusted to be relative to the chunk start
    (i.e., onset -= context_duration).

    Args:
        notes: All transcribed notes from combined context+chunk audio.
        pedal_events: All pedal events from combined audio.
        context_duration_s: Duration of context audio in seconds.

    Returns:
        Tuple of (filtered_notes, filtered_pedals) with adjusted timestamps.
    """
    if context_duration_s <= 0:
        return notes, pedal_events

    filtered_notes = []
    for note in notes:
        if note["onset"] >= context_duration_s:
            filtered_notes.append({
                "pitch": note["pitch"],
                "onset": round(note["onset"] - context_duration_s, 4),
                "offset": round(note["offset"] - context_duration_s, 4),
                "velocity": note["velocity"],
            })

    filtered_pedals = []
    for pedal in pedal_events:
        if pedal["time"] >= context_duration_s:
            filtered_pedals.append({
                "time": round(pedal["time"] - context_duration_s, 4),
                "value": pedal["value"],
            })

    return filtered_notes, filtered_pedals


class EndpointHandler:
    """HuggingFace Inference Endpoints handler for Aria-AMT transcription."""

    def __init__(self, path: str = ""):
        """Initialize Aria-AMT model, tokenizer, and audio transform.

        Called once when the endpoint container starts.

        Args:
            path: Path to the model repository (provided by HF Inference Endpoints).
                  Must contain a .safetensors checkpoint file.
        """
        print(f"Initializing Aria-AMT EndpointHandler with path: {path}")

        # Determine model path
        if path:
            model_path = Path(path)
        else:
            model_path = Path("/repository")
            if not model_path.exists():
                model_path = Path(".")

        # Find checkpoint file
        checkpoint_path = self._find_checkpoint(model_path)
        print(f"Using checkpoint: {checkpoint_path}")

        # Load model config and instantiate
        config_name = os.environ.get("AMT_MODEL_CONFIG", "medium-double")
        print(f"Loading Aria-AMT model config: {config_name}")
        model_config = ModelConfig(**load_model_config(config_name))

        self._tokenizer = AmtTokenizer()
        model_config.set_vocab_size(self._tokenizer.vocab_size)

        self._model = AmtEncoderDecoder(model_config)

        # Load weights
        state_dict = _load_weight(str(checkpoint_path))
        self._model.load_state_dict(state_dict)

        # Move to device and set up for inference
        device = os.environ.get("CRESCEND_DEVICE", "cuda")
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = device
        self._model.to(device)
        self._model.eval()

        # Compute dtype once; reused for KV cache setup
        self._cache_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        )

        # Set up KV cache for decoder.
        # The upstream setup_cache() hardcodes .cuda() on KVCache objects.
        # Monkey-patch to use the resolved device instead.
        self._setup_kv_cache(MAX_BLOCK_LEN)

        # Look up EOS token id once; raise if missing to catch tokenizer mismatches
        eos_token_id = self._tokenizer.tok_to_id.get(self._tokenizer.eos_tok)
        if eos_token_id is None:
            raise RuntimeError(
                f"EOS token '{self._tokenizer.eos_tok}' not found in tokenizer vocabulary"
            )
        self._eos_token_id = eos_token_id

        # Audio transform for log-mel spectrogram (move to device for STFT buffers)
        self._audio_transform = AudioTransform().to(device)

        print("Aria-AMT EndpointHandler initialization complete!")

    def _setup_kv_cache(self, max_seq_len: int) -> None:
        """Initialize KV caches on the correct device.

        The upstream TextDecoder.setup_cache() hardcodes .cuda() on KVCache
        objects. This reimplements it to use self._device instead, enabling
        MPS and CPU inference for local dev.
        """
        from amt.inference.model import KVCache

        decoder = self._model.decoder
        decoder.causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool, device=self._device)
        )
        for b in decoder.blocks:
            head_dim = decoder.n_state // decoder.n_head
            b.attn.kv_cache = KVCache(
                max_batch_size=1,
                max_seq_length=max_seq_len,
                n_heads=decoder.n_head,
                head_dim=head_dim,
                dtype=self._cache_dtype,
            ).to(self._device)
            b.cross_attn.kv_cache = KVCache(
                max_batch_size=1,
                max_seq_length=1500,
                n_heads=decoder.n_head,
                head_dim=head_dim,
                dtype=self._cache_dtype,
            ).to(self._device)

    def _find_checkpoint(self, model_path: Path) -> Path:
        """Find the safetensors checkpoint file in the model directory.

        Args:
            model_path: Root directory to search.

        Returns:
            Path to the checkpoint file.

        Raises:
            FileNotFoundError: If no checkpoint is found.
        """
        for pattern in ["*.safetensors", "**/*.safetensors"]:
            candidates = list(model_path.glob(pattern))
            if candidates:
                print(f"Checkpoint candidates ({pattern}): {[str(c) for c in candidates]}")
                # Prefer files with 'amt' or 'piano' in name
                for c in candidates:
                    if "amt" in c.name.lower() or "piano" in c.name.lower():
                        return c
                return candidates[0]

        raise FileNotFoundError(
            f"No safetensors checkpoint found in {model_path}."
        )

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Process transcription request.

        Args:
            data: Request payload:
                {
                    "inputs": {
                        "chunk_audio": "<base64-encoded-webm>",  # required
                        "context_audio": "<base64-encoded-webm>"  # optional
                    }
                }

        Returns:
            {
                "midi_notes": [{"pitch": int, "onset": float,
                                "offset": float, "velocity": int}],
                "pedal_events": [{"time": float, "value": int}],
                "transcription_info": {
                    "note_count": int,
                    "pitch_range": [int, int],
                    "pedal_event_count": int,
                    "transcription_time_ms": int,
                    "context_duration_s": float,
                    "chunk_duration_s": float
                }
            }
        """
        start_time = time.time()

        try:
            inputs = data.get("inputs", data)
            if isinstance(inputs, str):
                # Single base64 audio string -- treat as chunk_audio
                inputs = {"chunk_audio": inputs}

            # Decode chunk audio (required)
            chunk_audio_b64 = inputs.get("chunk_audio")
            if not chunk_audio_b64:
                return {
                    "error": {
                        "code": "MISSING_CHUNK_AUDIO",
                        "message": "chunk_audio field is required",
                    }
                }

            chunk_audio_bytes = base64.b64decode(chunk_audio_b64)
            chunk_pcm = decode_webm_to_pcm(chunk_audio_bytes)
            chunk_duration_s = len(chunk_pcm) / SAMPLE_RATE

            # Decode context audio (optional)
            context_audio_b64 = inputs.get("context_audio")
            context_duration_s = 0.0

            if context_audio_b64:
                context_audio_bytes = base64.b64decode(context_audio_b64)
                context_pcm = decode_webm_to_pcm(context_audio_bytes)
                context_duration_s = len(context_pcm) / SAMPLE_RATE
                # Concatenate: context first, then chunk
                combined_pcm = np.concatenate([context_pcm, chunk_pcm])
            else:
                combined_pcm = chunk_pcm

            print(
                f"Audio decoded: context={context_duration_s:.1f}s, "
                f"chunk={chunk_duration_s:.1f}s, "
                f"combined={len(combined_pcm)/SAMPLE_RATE:.1f}s"
            )

            # Run Aria-AMT transcription
            midi_notes, pedal_events = self._transcribe(combined_pcm)

            # Deduplicate: only return notes from the current chunk
            midi_notes, pedal_events = deduplicate_notes(
                midi_notes, pedal_events, context_duration_s
            )

            # Build response
            processing_time_ms = int((time.time() - start_time) * 1000)
            pitches = [n["pitch"] for n in midi_notes]

            result = {
                "midi_notes": midi_notes,
                "pedal_events": pedal_events,
                "transcription_info": {
                    "note_count": len(midi_notes),
                    "pitch_range": (
                        [min(pitches), max(pitches)] if pitches else [0, 0]
                    ),
                    "pedal_event_count": len(pedal_events),
                    "transcription_time_ms": processing_time_ms,
                    "context_duration_s": round(context_duration_s, 2),
                    "chunk_duration_s": round(chunk_duration_s, 2),
                },
            }

            print(
                f"Transcription complete: {len(midi_notes)} notes, "
                f"{len(pedal_events)} pedal events in {processing_time_ms}ms"
            )
            return result

        except Exception as e:
            return {
                "error": {
                    "code": "TRANSCRIPTION_ERROR",
                    "message": str(e),
                    "traceback": traceback.format_exc(),
                }
            }

    @torch.inference_mode()
    def _transcribe(
        self, audio: np.ndarray
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run Aria-AMT inference on PCM audio.

        Converts audio to log-mel spectrogram, runs encoder-decoder inference,
        and converts output tokens to note/pedal event lists.

        Args:
            audio: 16kHz mono float32 PCM audio array.

        Returns:
            Tuple of (notes, pedal_events).
        """
        # Convert to torch tensor and ensure correct length
        audio_tensor = torch.from_numpy(audio).float()

        # Pad or truncate to chunk_len samples
        chunk_samples = CHUNK_LEN_S * SAMPLE_RATE
        if len(audio_tensor) < chunk_samples:
            # Pad with zeros
            padding = torch.zeros(chunk_samples - len(audio_tensor))
            audio_tensor = torch.cat([audio_tensor, padding])
        elif len(audio_tensor) > chunk_samples:
            # Truncate to model's maximum input length
            audio_tensor = audio_tensor[:chunk_samples]

        # Compute log-mel spectrogram
        audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dim
        log_mels = self._audio_transform.log_mel(
            audio_tensor.to(self._device)
        )

        # Encode audio
        audio_features = self._model.encoder(xa=log_mels)

        # Autoregressive decoding
        tokenizer = self._tokenizer
        prefix = [tokenizer.bos_tok]
        seq = torch.tensor(
            [tokenizer.encode(prefix)], dtype=torch.long, device=self._device
        )

        # Reset KV cache for new sequence
        self._setup_kv_cache(MAX_BLOCK_LEN)

        generated_ids = list(seq[0].tolist())
        idx = seq.shape[1]
        xa_len = audio_features.shape[1]

        # Prefill with initial tokens
        logits = self._model.decoder(
            x=seq,
            xa=audio_features,
            x_input_pos=torch.arange(0, idx, device=self._device),
            xa_input_pos=torch.arange(0, xa_len, device=self._device),
        )

        for _step in range(MAX_BLOCK_LEN - len(generated_ids)):
            next_token_logits = logits[:, -1, :]

            # Boost pedal-off probability slightly (from aria-amt source)
            pedal_off_id = tokenizer.tok_to_id.get(("pedal", 0))
            if pedal_off_id is not None:
                next_token_logits[:, pedal_off_id] *= 1.05

            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            generated_ids.append(next_token_id)
            idx += 1

            # Check for EOS
            if next_token_id == self._eos_token_id:
                break

            # Decode next token (empty xa_input_pos since encoder is cached)
            next_input = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=self._device
            )
            logits = self._model.decoder(
                x=next_input,
                xa=audio_features,
                x_input_pos=torch.tensor(
                    [idx - 1], device=self._device, dtype=torch.int
                ),
                xa_input_pos=torch.tensor(
                    [], device=self._device, dtype=torch.int
                ),
            )

        # Detokenize to MidiDict
        decoded_seq = tokenizer.decode(generated_ids)
        # Find last onset time for len_ms parameter
        last_onset_ms = 0
        for tok in decoded_seq:
            if isinstance(tok, tuple) and tok[0] == "onset":
                last_onset_ms = max(last_onset_ms, tok[1])

        total_duration_ms = int(len(audio) / SAMPLE_RATE * 1000)
        midi_dict = tokenizer.detokenize(
            tokenized_seq=decoded_seq,
            len_ms=max(last_onset_ms, total_duration_ms),
        )

        return midi_dict_to_notes_and_pedals(midi_dict)
