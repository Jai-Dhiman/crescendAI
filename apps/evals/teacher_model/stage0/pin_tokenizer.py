"""Pin the tokenizer of a Hugging Face model by hashing its source files.

Stage 0 commits the resulting tokenizer_pin.json so Stage 1 can verify the
exact same tokenizer + chat template is in use during finetuning.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from transformers import AutoTokenizer

# Files that participate in the pin. Anything else (model weights, optimizer state)
# is irrelevant to tokenization equivalence.
_PIN_FILE_PATTERNS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "added_tokens.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)


class MissingChatTemplateError(RuntimeError):
    """Raised when a model has no chat template; Stage 1 cannot proceed."""


@dataclass
class TokenizerPin:
    model_id: str
    sha256: str
    files: list[str]            # sorted relative paths included in the hash
    cache_dir: str              # absolute path to the local cache dir
    chat_template_present: bool


def _collect_pin_files(cache_dir: Path) -> list[Path]:
    """Find all files in cache_dir matching _PIN_FILE_PATTERNS, sorted by relpath."""
    matches: list[Path] = []
    for pattern in _PIN_FILE_PATTERNS:
        matches.extend(cache_dir.rglob(pattern))
    # Deduplicate (a single file matches at most one pattern, but rglob may revisit symlinks)
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in matches:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)
    unique.sort(key=lambda p: str(p.relative_to(cache_dir)))
    return unique


def _hash_files(files: Iterable[Path], cache_dir: Path) -> str:
    h = hashlib.sha256()
    for f in files:
        rel = str(f.relative_to(cache_dir)).encode("utf-8")
        h.update(b"PATH:")
        h.update(rel)
        h.update(b"\nDATA:")
        h.update(f.read_bytes())
        h.update(b"\n---\n")
    return h.hexdigest()


def pin_tokenizer(
    model_id: str,
    out_path: Path,
    cache_dir: Path | None = None,
    force_local: bool = False,
) -> TokenizerPin:
    """Download the tokenizer, hash its source files, write a pin record.

    Args:
        model_id: HuggingFace model id, e.g. 'Qwen/Qwen3.6-35B-A3B-Instruct'.
        out_path: where to write the JSON pin record.
        cache_dir: optional override of the HF cache directory (used in tests).
        force_local: if True, do not re-download; require files already in cache_dir.

    Raises:
        MissingChatTemplateError: if the loaded tokenizer has no chat_template.
    """
    kwargs: dict = {}
    if force_local and cache_dir is not None:
        # When cache_dir is a snapshot dir (already resolved), pass it directly as
        # the model path so transformers doesn't try to navigate HF hub cache structure.
        tokenizer = AutoTokenizer.from_pretrained(str(cache_dir), local_files_only=True)
    else:
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)

    chat_template = getattr(tokenizer, "chat_template", None)
    if not chat_template:
        raise MissingChatTemplateError(
            f"Model {model_id!r} has no chat_template. "
            "Stage 1 requires a chat-template-native tool-call slot."
        )

    # Resolve the pin_dir: the directory whose tokenizer files we will hash.
    # When force_local + cache_dir is a snapshot dir, we already loaded from it directly.
    if force_local and cache_dir is not None:
        pin_dir = Path(cache_dir)
    else:
        # For from_pretrained, the files land under cache_dir/models--{org}--{name}/snapshots/<sha>/
        if cache_dir is None:
            from huggingface_hub import constants as hf_const

            resolved_cache = Path(hf_const.HF_HUB_CACHE)
        else:
            resolved_cache = Path(cache_dir)

        snapshots = list(resolved_cache.rglob(f"models--{model_id.replace('/', '--')}/snapshots"))
        if not snapshots:
            raise RuntimeError(f"Could not locate cache snapshot for {model_id!r} under {resolved_cache}")
        snapshot_root = snapshots[0]
        snapshot_dirs = sorted(p for p in snapshot_root.iterdir() if p.is_dir())
        if not snapshot_dirs:
            raise RuntimeError(f"No snapshots present under {snapshot_root}")
        pin_dir = snapshot_dirs[-1]  # most recent

    files = _collect_pin_files(pin_dir)
    if not files:
        raise RuntimeError(
            f"No tokenizer source files found under {pin_dir}. "
            "Pinning requires at least one of: " + ", ".join(_PIN_FILE_PATTERNS)
        )

    sha = _hash_files(files, pin_dir)
    pin = TokenizerPin(
        model_id=model_id,
        sha256=sha,
        files=[str(f.relative_to(pin_dir)) for f in files],
        cache_dir=str(pin_dir),
        chat_template_present=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(pin), indent=2))
    return pin
