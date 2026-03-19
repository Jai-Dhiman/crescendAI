"""Aria embedding extraction from MIDI files.

Supports two variants:
- "embedding": 512-dim via TransformerEMB + get_global_embedding_from_midi
- "base": 1536-dim via Transformer + last-token pooling
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from safetensors.torch import load_file

from ariautils.midi import MidiDict
from ariautils.tokenizer import AbsTokenizer
from aria.embedding import get_global_embedding_from_midi
from aria.model import ModelConfig, Transformer, TransformerEMB
from paths import Weights

logger = logging.getLogger(__name__)

# Lazy-loaded caches to avoid reloading per file.
_model_cache: dict[str, object] = {}
_tokenizer_cache: dict[str, AbsTokenizer] = {}


def _get_tokenizer() -> AbsTokenizer:
    """Get or create a cached AbsTokenizer instance."""
    if "abs" not in _tokenizer_cache:
        _tokenizer_cache["abs"] = AbsTokenizer()
    return _tokenizer_cache["abs"]

EMBEDDING_WEIGHTS = Weights.root / "aria-medium-embedding"
BASE_WEIGHTS = Weights.root / "aria-medium-base"


def _load_embedding_model() -> TransformerEMB:
    """Load the embedding variant (512-dim) of Aria."""
    if "embedding" in _model_cache:
        return _model_cache["embedding"]

    config_path = EMBEDDING_WEIGHTS / "config.json"
    weights_path = EMBEDDING_WEIGHTS / "model.safetensors"

    with open(config_path) as f:
        config = json.load(f)

    mc = ModelConfig(
        d_model=config["hidden_size"],
        n_heads=config["num_attention_heads"],
        n_layers=config["num_hidden_layers"],
        max_seq_len=config["max_seq_len"],
        ff_mult=config["intermediate_size"] // config["hidden_size"],
        emb_size=config["embedding_size"],
        vocab_size=config["vocab_size"],
        drop_p=0.0,
        grad_checkpoint=False,
    )
    model = TransformerEMB(mc)
    state_dict = load_file(str(weights_path))
    model.load_state_dict(state_dict)
    model.eval()

    _model_cache["embedding"] = model
    logger.info("Loaded Aria embedding model from %s", EMBEDDING_WEIGHTS)
    return model


def _load_base_model() -> Transformer:
    """Load the base variant (1536-dim) of Aria."""
    if "base" in _model_cache:
        return _model_cache["base"]

    config_path = BASE_WEIGHTS / "config.json"
    weights_path = BASE_WEIGHTS / "model.safetensors"

    with open(config_path) as f:
        config = json.load(f)

    mc = ModelConfig(
        d_model=config["hidden_size"],
        n_heads=config["num_attention_heads"],
        n_layers=config["num_hidden_layers"],
        max_seq_len=config["max_seq_len"],
        ff_mult=config["intermediate_size"] // config["hidden_size"],
        vocab_size=config["vocab_size"],
        drop_p=0.0,
        grad_checkpoint=False,
    )
    model = Transformer(mc)
    state_dict = load_file(str(weights_path))
    # Strip 'model.' prefix and skip lm_head weights
    stripped = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }
    model.load_state_dict(stripped)
    model.eval()

    _model_cache["base"] = model
    logger.info("Loaded Aria base model from %s", BASE_WEIGHTS)
    return model


def tokenize_midi(midi_path: Path) -> list:
    """Tokenize a MIDI file using AbsTokenizer.

    Args:
        midi_path: Path to the MIDI file.

    Returns:
        List of tokens.

    Raises:
        FileNotFoundError: If the MIDI file does not exist.
    """
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    midi_dict = MidiDict.from_midi(str(midi_path))
    tokenizer = _get_tokenizer()
    tokens = tokenizer.tokenize(midi_dict)
    return tokens


def extract_embedding(midi_path: Path, variant: str = "embedding") -> torch.Tensor:
    """Extract an embedding from a single MIDI file.

    Args:
        midi_path: Path to the MIDI file.
        variant: "embedding" (512-dim) or "base" (1536-dim).

    Returns:
        1-D tensor of shape (512,) or (1536,) depending on variant.

    Raises:
        FileNotFoundError: If the MIDI file does not exist.
        ValueError: If variant is not "embedding" or "base".
    """
    midi_path = Path(midi_path)
    if not midi_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_path}")

    if variant == "embedding":
        model = _load_embedding_model()
        emb = get_global_embedding_from_midi(
            model, midi_path=str(midi_path), device="cpu"
        )
        return emb.float()

    elif variant == "base":
        model = _load_base_model()
        tokens = tokenize_midi(midi_path)
        tokenizer = _get_tokenizer()
        token_ids = tokenizer.encode(tokens)

        # Limit sequence length for consistency with embedding variant
        if len(token_ids) > 2048:
            token_ids = token_ids[:2048]

        input_ids = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            output = model(input_ids)  # shape [1, seq_len, 1536]
            embedding = output[0, -1, :]  # last-token pooling: shape [1536]

        return embedding.float()

    else:
        raise ValueError(
            f"Unknown variant '{variant}'. Must be 'embedding' or 'base'."
        )


def extract_all_embeddings(
    midi_dir: Path, variant: str = "embedding"
) -> dict[str, torch.Tensor]:
    """Extract embeddings from all MIDI files in a directory.

    Args:
        midi_dir: Directory containing .mid files.
        variant: "embedding" (512-dim) or "base" (1536-dim).

    Returns:
        Dict mapping filename stem to embedding tensor.
    """
    midi_dir = Path(midi_dir)
    midi_files = sorted(midi_dir.glob("*.mid"))

    if not midi_files:
        raise FileNotFoundError(f"No .mid files found in {midi_dir}")

    results: dict[str, torch.Tensor] = {}
    for i, midi_path in enumerate(midi_files):
        segment_id = midi_path.stem
        try:
            emb = extract_embedding(midi_path, variant=variant)
            results[segment_id] = emb
        except Exception:
            logger.exception("Failed to extract embedding for %s", midi_path)
            raise

        if (i + 1) % 50 == 0:
            logger.info("Processed %d / %d files", i + 1, len(midi_files))

    logger.info(
        "Extracted %d embeddings (variant=%s) from %s",
        len(results),
        variant,
        midi_dir,
    )
    return results


def main() -> None:
    """CLI entry point for batch embedding extraction."""
    parser = argparse.ArgumentParser(
        description="Extract Aria embeddings from MIDI files."
    )
    parser.add_argument(
        "--variant",
        choices=["embedding", "base"],
        default="embedding",
        help="Model variant: 'embedding' (512-dim) or 'base' (1536-dim).",
    )
    parser.add_argument(
        "--midi-dir",
        type=Path,
        required=True,
        help="Directory containing .mid files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for the .pt file containing the embedding dict.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    embeddings = extract_all_embeddings(args.midi_dir, variant=args.variant)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, args.output)
    logger.info("Saved %d embeddings to %s", len(embeddings), args.output)


if __name__ == "__main__":
    main()
