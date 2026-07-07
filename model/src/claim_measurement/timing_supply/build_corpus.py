"""FRONT 7a corpus builder: the 162 baseline_v1 prose docs for the timing supply probe.

Reproduces the EXACT corpus front-5 / G-D measured dynamics supply on
(#101 / #67): every ``chopin_ballade_1`` generator prose doc in ``baseline_v1.jsonl``
with non-empty ``synthesis_text`` AND a locally-present ``skill_eval`` audio wav.
Audio is NOT needed for a claim-supply probe -- it is only the pairing filter that
pins the corpus to the same 162 docs (94 distinct performances) front-5 used, so the
timing supply number is comparable to the dynamics one on the same substrate.

Emits a JSON array of ``{recording_id, run_id, skill_bucket, text}`` for the LLM
timing-claim extractor (``extract_prompt.md``). The extractor is allowed to be an LLM
(Path #1 rule: the CLAIM may be LLM-extracted; the VERDICT/truth label may not).

Run (from the worktree; baseline_v1.jsonl is committed so it is present here):
    uv run python model/src/claim_measurement/timing_supply/build_corpus.py \
        --out /ABS/scratchpad/timing_supply_corpus.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_HERE = Path(__file__).resolve()
# model/src/claim_measurement/timing_supply/build_corpus.py -> parents[4] == repo root
REPO = _HERE.parents[4]

DEFAULT_BASELINE = REPO / "apps/evals/results/baseline_v1.jsonl"
DEFAULT_AUDIO_ROOT = REPO / "model/data/evals/skill_eval/chopin_ballade_1/audio"
PIECE = "chopin_ballade_1"


def build(baseline: Path, audio_root: Path, piece: str,
          require_audio: bool) -> list[dict]:
    """Prose docs for ``piece`` with non-empty text (and, if required, local audio).

    When ``require_audio`` is False the pairing filter is dropped -- use only when the
    audio dir is absent (e.g. a gitignored-data worktree) and the doc set is being
    reproduced from prose alone; the doc count may then exceed 162.
    """
    have = set()
    if audio_root.is_dir():
        have = {p.stem for p in audio_root.glob("*.wav")}
    docs: list[dict] = []
    for line in baseline.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("piece_slug") != piece:
            continue
        text = (r.get("synthesis_text") or "").strip()
        if not text:
            continue
        rid = r["recording_id"]
        if require_audio and have and rid not in have:
            continue
        docs.append({
            "recording_id": rid,
            "run_id": r.get("run_id"),
            "skill_bucket": r.get("skill_bucket"),
            "text": text,
        })
    return docs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="timing_supply.build_corpus")
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    ap.add_argument("--piece", default=PIECE)
    ap.add_argument("--no-require-audio", action="store_true",
                    help="drop the local-audio pairing filter (data-absent worktree)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    docs = build(args.baseline, args.audio_root, args.piece,
                 require_audio=not args.no_require_audio)
    perfs = {d["recording_id"] for d in docs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(docs, indent=2))
    print(f"{len(docs)} prose docs; {len(perfs)} distinct performances -> {args.out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
