# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "numpy>=1.24.0",
#     "torch>=2.0.0",
#     # MoonBeam's own fork -- NOT on PyPI, pinned commit chosen by the human
#     # running the real bake-off (guozixunnicolas/moonbeam-midi-foundation-model,
#     # its bundled transformers_minimal fork + custom tokenizer). Uncomment and
#     # pin once that commit is chosen:
#     # "moonbeam @ git+https://github.com/guozixunnicolas/moonbeam-midi-foundation-model",
# ]
# ///
"""MoonBeam-839M extraction, run under an ISOLATED uv-managed Python 3.12 venv
-- NEVER the shared model/.venv (this repo has twice polluted that shared venv
with a competing pretraining stack's pinned deps; see project memory
project_uv_run_mutates_model_venv.md: "uv run --with X --python N" from
inside model/ rebuilds the shared .venv).

SETUP (human-lit, run once, from this file's own directory):
    cd model/src/claim_measurement/difficulty
    uv run --script moonbeam_extract_script.py --help
    # `uv run --script` resolves THIS file's own `# /// script` metadata block
    # into its own cached, ephemeral env keyed to python==3.12.* + the deps
    # above -- never the project's model/.venv. That is different from a bare
    # `uv run` invoked from inside model/, or `uv run --with X`, both of which
    # DO sync the shared project venv (the known gotcha above). Before the
    # real run: uncomment the moonbeam git dependency once its exact
    # commit/tag is chosen, and implement `_real_loader` below against the
    # fork's actual checkpoint-loading API -- undocumented as of #138 Phase 0,
    # this is the human-lit GPU validation step, not this build's.

RUN (human-lit, needs moonbeam_839M.pt, ~3.3GB, and a GPU):
    uv run --script moonbeam_extract_script.py \
        --checkpoint /path/to/moonbeam_839M.pt \
        --sample-manifest /path/to/model/data/results/bakeoff/sample_manifest.json \
        --midi-dir /path/to/model/data/results/amt_gap_curve/transkun_mid \
        --out-dir /path/to/model/data/results/bakeoff/emb/moonbeam \
        --composer-index /path/to/model/data/results/bakeoff/composer_index.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import numpy as np


def _real_loader(checkpoint_path: Path) -> Callable[[Path], np.ndarray]:
    """Build the real per-token-hidden-state loader against MoonBeam's fork.

    NOT implemented in this build: the fork's exact checkpoint/tokenizer API
    is undocumented and can only be nailed down by the human running this
    under the isolated venv against the real 3.3GB checkpoint (#138 Phase 0
    design: "the real GPU forward-pass validation is NOT part of this
    build"). Imports of the fork's packages are deferred to inside this
    function so importing this MODULE never requires them.
    """
    raise NotImplementedError(
        "wire this against the transformers_minimal fork's real checkpoint/"
        "tokenizer API once the isolated venv is set up; see this file's "
        "module docstring"
    )


def main(argv: list[str] | None = None, loader_factory=_real_loader) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # -> model/src
    from claim_measurement.difficulty.bakeoff_sampling import ManifestEntry
    from claim_measurement.difficulty.extract import extract_embeddings
    from claim_measurement.difficulty.moonbeam_backbone import MoonBeamBackbone

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--sample-manifest", type=Path, required=True)
    ap.add_argument("--midi-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--composer-index", type=Path, required=True)
    args = ap.parse_args(argv)

    entries = [ManifestEntry(**e) for e in json.loads(args.sample_manifest.read_text())]
    backbone = MoonBeamBackbone(loader=loader_factory(args.checkpoint))
    report = extract_embeddings(backbone, entries, midi_dir=args.midi_dir,
                                 out_dir=args.out_dir, composer_index_path=args.composer_index)
    print(f"ok={report.ok} failed={len(report.failed)}")
    for f in report.failed[:10]:
        print(f"  FAIL {f}")
    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
