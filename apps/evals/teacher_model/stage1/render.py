import hashlib
import json
from pathlib import Path


class TokenizerPinMismatchError(Exception):
    pass


def _hash_files(directory: Path, filenames: list[str]) -> str:
    sha = hashlib.sha256()
    for filename in sorted(filenames):
        path = directory / filename
        if not path.exists():
            raise TokenizerPinMismatchError(
                f"Pinned file missing from tokenizer dir: {filename}"
            )
        sha.update(filename.encode())
        sha.update(b"\0")
        sha.update(path.read_bytes())
        sha.update(b"\0")
    return sha.hexdigest()


def verify_tokenizer_pin(tokenizer_dir: Path, pin_path: Path) -> None:
    pin = json.loads(pin_path.read_text())
    expected = pin["sha256"]
    actual = _hash_files(tokenizer_dir, pin["files"])
    if actual != expected:
        raise TokenizerPinMismatchError(
            f"sha256 mismatch: expected {expected}, got {actual}"
        )
