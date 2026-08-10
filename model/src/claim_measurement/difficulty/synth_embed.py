# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0", "scipy>=1.10.0", "torch>=2.0.0", "peft==0.11.1",
#     "pretty_midi", "mido", "music21", "pandas", "tqdm", "regex", "requests",
#     "filelock", "pyyaml", "safetensors", "tokenizers==0.19.1",
#     "huggingface_hub",
# ]
# ///
"""#167 (#104 S2): embed one arm's MIDIs into the .npz `distribution_shift`
scores.

    <arm>/tx/<seg_id>.mid  ->  arm.npz   (seg_ids + 1920-dim embeddings)

Each piece is embedded by **the adapter that held it out**, which is what makes
the later comparison fold-honest, and adapters are loaded once per fold rather
than once per piece -- the 1.6GB backbone takes far longer to load than the ~1s
an embedding costs.

The `source` arm (B0) is the exception and reads `mid.zip` directly with no
render and no re-transcription. It exists to subtract a confound: arm A passes
through one transcription and the rendered arms pass through two, because
`mid.zip` is itself a transcription of the same performance.

## Environment

Run under the ISOLATED env this file's `# /// script` header defines, never the
shared `model/.venv`: the MoonBeam fork's vendored transformers hard-requires
`tokenizers>=0.19,<0.20` and the shared venv carries a later one, so it dies on
import before any peft code runs. `peft==0.11.1` is the version that WROTE
these adapters.

    uv run --no-project --script synth_embed.py \\
        --arm expressive --out-root .../mirex_s2_shift \\
        --data-root .../model/data \\
        --fold-emb-dir .../phase1_lora/fold_embeddings \\
        --checkpoint .../moonbeam_839M.pt --repo-root .../moonbeam/repo \\
        --model-config .../repo/src/llama_recipes/configs/model_config.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

# Under `uv run --script` this file runs as a loose script, so the package it
# imports from is not on sys.path. __file__-anchored, never CWD-relative, and
# never a parent-walk from a resolved path -- a Hub snapshot is a symlink farm
# and `.resolve()` would leave the tree entirely (#166 cost two jobs to that).
_SRC_ROOT = str(Path(__file__).resolve().parents[2])
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from claim_measurement.difficulty.audio_emb_extract import (  # noqa: E402
    build_fold_embedder,
    fold_of_seg_ids,
    write_notes_midi,
)
from claim_measurement.difficulty.distribution_shift import (  # noqa: E402
    N_FOLDS,
    SEED,
    SOURCE_ARM,
    embed_arm,
    write_arm_npz,
)
from claim_measurement.difficulty.synth_arm import transcription_path  # noqa: E402


def source_midis(seg_ids, manifest_rows, mid_zip: Path, work: Path) -> dict:
    """Materialise the B0 arm's MIDIs from mid.zip.

    Written through `write_notes_midi` rather than extracted verbatim, so the
    B0 arm goes through the identical note-list -> MIDI step the rendered arms
    do. Extracting the raw member instead would leave pedal and multi-track
    structure in place for B0 only, and the control would differ from the arms
    in a second way beyond the one it is meant to isolate.
    """
    from claim_measurement.difficulty.synth_render import read_performance

    midi_name = {row["seg_id"]: row["midi_name"] for row in manifest_rows}
    work.mkdir(parents=True, exist_ok=True)
    out = {}
    with zipfile.ZipFile(mid_zip) as zf:
        for seg_id in seg_ids:
            perf = read_performance(zf.read(midi_name[seg_id]))
            path = work / f"{seg_id}.mid"
            write_notes_midi(perf.notes, path)
            out[seg_id] = path
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True,
                    choices=("expressive", "deadpan", SOURCE_ARM))
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--fold-emb-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--out-npz", type=Path, default=None)
    args = ap.parse_args(argv)

    from claim_measurement.difficulty.bakeoff_npz import read_embedding_npz
    from claim_measurement.difficulty.bakeoff_paths import (
        features37_dir,
        features37_seg_ids,
        resolve_paths,
    )

    paths = resolve_paths(args.data_root)
    f37 = features37_dir(paths.emb_root)
    seg_ids = features37_seg_ids(f37)
    composers = np.array([read_embedding_npz(f37 / f"{s}.npz").composer_id
                          for s in seg_ids])
    fold_by_seg = fold_of_seg_ids(seg_ids, composers, N_FOLDS, SEED)
    manifest_rows = json.loads(Path(paths.manifest).read_text())

    out_npz = args.out_npz or (Path(args.out_root) / f"{args.arm}.npz")

    def embedder_for(fold: int):
        return build_fold_embedder(
            Path(args.fold_emb_dir) / f"fold{fold}" / "adapter",
            args.checkpoint, args.repo_root, args.model_config)

    with tempfile.TemporaryDirectory() as tmp:
        if args.arm == SOURCE_ARM:
            midis = source_midis(seg_ids, manifest_rows,
                                 paths.labels.parent / "mid.zip", Path(tmp))
        else:
            arm_dir = Path(args.out_root) / args.arm
            midis = {s: transcription_path(arm_dir, s) for s in seg_ids}
            missing = [s for s, p in midis.items() if not p.is_file()]
            if missing:
                # Refused rather than scored short: an arm with holes still
                # produces a tau-c, over a different population than the one it
                # is compared against.
                raise SystemExit(
                    f"arm {args.arm!r} is missing {len(missing)} of "
                    f"{len(seg_ids)} transcriptions (e.g. {missing[:3]}); "
                    f"rerun run_s2_arms.sh -- it resumes from the cache")

        vectors = embed_arm(seg_ids, fold_by_seg, midis.__getitem__,
                            embedder_for)

    write_arm_npz(out_npz, vectors)
    print(f"wrote {out_npz} ({len(vectors)} pieces)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
