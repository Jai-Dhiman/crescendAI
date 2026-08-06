# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0", "scipy>=1.10.0", "torch>=2.0.0", "peft==0.11.1",
#     "scikit-learn",  # lazy in bakeoff_cv; unused on this path, declared anyway
#     "pretty_midi",  # lazily imported by write_notes_midi -- fatal at CALL time
#     "mido", "music21", "pandas", "tqdm", "regex", "requests",
#     "filelock", "pyyaml", "safetensors", "tokenizers==0.19.1",
#     "huggingface_hub",
# ]
# ///
"""#138 Phase 1 Stage 5(b): MoonBeam embeddings for the AUDIO-derived MIDIs.

Reads realaudio_check.py's transcription cache (one `{"notes": [...],
"pedals": [...]}` JSON per piece), and for each piece runs it through THAT
piece's OWN fold adapter -- the fold it is a test piece of under
bakeoff_cv.composer_disjoint_folds at the same (n_folds, seed) the whole phase
uses. Scoring a piece through an adapter that was trained on it would be
train-on-test, which is exactly the contamination #135's 0.824 anchor died of.

Output is the standard per-piece bakeoff_npz contract with pooling key
"mean_pool", which is what the runbook's Stage 5 gate snippet reads.

    cd model/src/claim_measurement/difficulty
    uv run --no-project --script audio_emb_extract.py \\
        --cache-dir .../audio_midi_cache --out-dir .../audio_emb \\
        --adapter-root .../fold_embeddings --data-root .../model/data \\
        --checkpoint .../moonbeam_839M.pt --repo-root .../moonbeam/repo \\
        --model-config .../repo/src/llama_recipes/configs/model_config.json

Run under the ISOLATED uv env this file's `# /// script` header defines, NOT
the shared model/.venv -- exactly like moonbeam_extract_script.py, and for the
same reason: model/.venv carries transformers 5.5.4 and tokenizers 0.22.1,
while the fork's vendored transformers hard-requires `tokenizers>=0.19,<0.20`
and dies on import there. peft is pinned to 0.11.1 to match both that ~4.41-era
fork and the version that WROTE these adapters inside the job container.
`python -m claim_measurement.difficulty.audio_emb_extract` still works for the
tests (which inject a fake loader_factory and never touch the fork).

Resumable: a piece whose .npz already exists is skipped and its adapter is
never loaded, so this long local run survives interruption. All GPU/model work
is behind `loader_factory` (train_fold.py's `_real_loader`, the SAME strict
checkpoint check -- a half-loaded backbone aborts rather than being measured),
so the tests exercise the real wiring on CPU with no network.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Under `uv run --script` this file runs as a loose script, so the package it
# imports from is not on sys.path. __file__-anchored, never CWD-relative: the
# documented invocation cd's into this directory. A no-op under `python -m`.
_SRC_ROOT = str(Path(__file__).resolve().parents[2])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from claim_measurement.difficulty.bakeoff_cv import (  # noqa: E402
    composer_disjoint_folds,
)
from claim_measurement.difficulty.bakeoff_npz import (  # noqa: E402
    read_embedding_npz,
    write_embedding_npz,
)
from claim_measurement.difficulty.train_fold import (  # noqa: E402
    _extract_full_piece,
    _real_loader,
    _stub_absent_transformers_models,
)

N_FOLDS, SEED = 5, 2026
POOLING = "mean_pool"


def write_notes_midi(notes: list, path: Path) -> None:
    """Materialise cached note dicts back into a one-instrument piano MIDI,
    because MoonBeam's MusicTokenizer.midi_to_compound consumes a FILE, not a
    note list. Pedals are deliberately dropped: MoonBeam's 6-dim compound
    tokens (onset, dur, octave, pitch_class, instrument, velocity) carry no
    pedal channel, so a pedal event could not reach the encoder anyway."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for note in notes:
        instrument.notes.append(pretty_midi.Note(
            velocity=int(note["velocity"]), pitch=int(note["pitch"]),
            start=float(note["onset"]), end=float(note["offset"])))
    pm.instruments.append(instrument)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(path))


def fold_of_seg_ids(seg_ids: list, composers: np.ndarray, n_folds: int,
                     seed: int) -> dict:
    """seg_id -> the fold it is a TEST piece of. Delegates to
    bakeoff_cv.composer_disjoint_folds rather than reimplementing the greedy
    bin-packing: "the same folds" only holds if it is literally the same
    function at the same (n_folds, seed)."""
    folds = composer_disjoint_folds(composers, n_folds, seed)
    return {seg_ids[i]: f for f, idx in enumerate(folds) for i in idx}


def build_fold_embedder(adapter_dir: Path, checkpoint: Path, repo_root: Path,
                         model_config: Path, loader_factory=_real_loader):
    """A `embed(midi_path) -> np.ndarray` closure over the base MoonBeam
    backbone with one fold's LoRA adapter applied.

    The `.eval()` below is load-bearing and must not be removed: LoRA is
    configured with lora_dropout=0.05, and torch.no_grad() does NOT disable
    dropout -- without eval mode every extraction would be a different random
    draw, and the gate would be measuring noise."""
    # Order is a contract, not a style choice, and mirrors train_fold.main():
    # loader_factory puts the fork's PARTIAL vendored transformers on sys.path
    # FIRST, the stub then supplies the models.bloom that peft/utils/constants
    # feature-probes for, and only then may peft be imported. Importing peft
    # first binds whatever transformers is already installed and skips the stub.
    base_model, tokenize, max_len = loader_factory(
        checkpoint, repo_root=repo_root, model_config=model_config)
    _stub_absent_transformers_models()
    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    peft_model.eval()
    transformer = peft_model.model.model  # inner transformer, LoRA-injected

    def embed(midi_path: Path) -> np.ndarray:
        return _extract_full_piece(transformer, tokenize(Path(midi_path)), max_len)

    return embed


@dataclass
class AudioExtractionReport:
    ok: int = 0
    skipped: int = 0
    failed: list = field(default_factory=list)


def extract_audio_embeddings(cache_dir: Path, out_dir: Path, fold_of: dict,
                              grades: dict, composer_ids: dict,
                              embedder_for_fold, skip_existing: bool = True
                              ) -> AudioExtractionReport:
    """One mean-pooled .npz per cached transcription, extracted through that
    piece's own fold adapter. Pieces are grouped by fold so each adapter (a
    full 839M backbone load) is built at most once, and a fold whose pieces
    are all already extracted is never loaded at all."""
    cache_dir, out_dir = Path(cache_dir), Path(out_dir)
    report = AudioExtractionReport()

    todo: dict = {}
    for cache_path in sorted(cache_dir.glob("*.json")):
        seg_id = cache_path.stem
        if skip_existing and (out_dir / f"{seg_id}.npz").exists():
            report.skipped += 1
            continue
        if seg_id not in fold_of:
            report.failed.append(f"{seg_id}: not an eval piece (no fold assignment)")
            continue
        todo.setdefault(fold_of[seg_id], []).append(cache_path)

    for fold in sorted(todo):
        embed = embedder_for_fold(fold)
        for cache_path in todo[fold]:
            seg_id = cache_path.stem
            try:
                notes = json.loads(cache_path.read_text())["notes"]
                with tempfile.TemporaryDirectory() as tmp:
                    midi_path = Path(tmp) / f"{seg_id}.mid"
                    write_notes_midi(notes, midi_path)
                    vector = embed(midi_path)
                write_embedding_npz(out_dir / f"{seg_id}.npz", {POOLING: vector},
                                     grade=grades[seg_id],
                                     composer_id=composer_ids[seg_id])
                report.ok += 1
            except Exception as exc:  # noqa: BLE001 -- record and continue; the report is the source of truth
                report.failed.append(f"{seg_id}: {exc!r}")
    return report


def _load_eval_index(features37_dir: Path):
    """seg_ids (canonical order), grades and composer_ids straight out of the
    features37 .npz files -- the same source ft_eval.py's _load_features37
    reads, so the fold assignment here cannot drift from the one the gate
    scores through."""
    from claim_measurement.difficulty.bakeoff_paths import features37_seg_ids

    features37_dir = Path(features37_dir)
    seg_ids = features37_seg_ids(features37_dir)
    grades, composer_ids = {}, {}
    for seg_id in seg_ids:
        record = read_embedding_npz(features37_dir / f"{seg_id}.npz")
        grades[seg_id] = record.grade
        composer_ids[seg_id] = record.composer_id
    return seg_ids, grades, composer_ids


def main(argv=None, embedder_for_fold=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", type=Path, required=True,
                    help="realaudio_check.py --out-dir (one JSON per piece)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--adapter-root", type=Path, required=True,
        help="the fold_embeddings dir Stage 3.5 downloaded into; fold F's "
             "adapter is read from <adapter-root>/fold{F}/adapter")
    ap.add_argument("--features37-dir", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=None,
                    help="only used to locate --features37-dir")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--model-config", type=Path, default=None)
    ap.add_argument("--n-folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    from claim_measurement.difficulty.bakeoff_paths import (
        features37_dir,
        resolve_paths,
    )

    f37_dir = (args.features37_dir if args.features37_dir is not None
               else features37_dir(resolve_paths(args.data_root).emb_root))
    seg_ids, grades, composer_ids = _load_eval_index(f37_dir)
    fold_of = fold_of_seg_ids(
        seg_ids, np.array([composer_ids[s] for s in seg_ids]),
        args.n_folds, args.seed)

    if embedder_for_fold is None:
        for flag, value in (("--checkpoint", args.checkpoint),
                            ("--repo-root", args.repo_root),
                            ("--model-config", args.model_config)):
            if value is None:
                ap.error(f"{flag} is required when the real backbone is used")

        def embedder_for_fold(fold: int):
            adapter_dir = Path(args.adapter_root) / f"fold{fold}" / "adapter"
            if not adapter_dir.exists():
                raise FileNotFoundError(
                    f"fold {fold}'s adapter is missing: {adapter_dir} does not "
                    f"exist (Stage 3.5 downloads it from that fold's output repo)")
            return build_fold_embedder(adapter_dir, args.checkpoint,
                                        args.repo_root, args.model_config)

    report = extract_audio_embeddings(
        args.cache_dir, args.out_dir, fold_of, grades, composer_ids,
        embedder_for_fold)
    print(f"ok={report.ok} skipped={report.skipped} failed={len(report.failed)}")
    for failure in report.failed[:10]:
        print(f"  FAIL {failure}")
    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
