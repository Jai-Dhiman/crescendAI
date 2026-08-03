# model/src/follower_eval/piece_id.py
"""Piece-ID stage for the real-audio follower eval (issue #133).

The practice corpus is labeled by the folder a YouTube video was curated into,
and those labels are unreliable -- a clip filed under `fantaisie_impromptu` is
someone sightreading Chopin Op.25/5; one filed `nocturne_op9no2` is actually
op.9 no.1. In production we never trust a label anyway: we identify the piece
from the audio. This module does that from the transcription, against the
10,494-score catalog, so the follower/validator get the score the pianist is
ACTUALLY playing (or we abstain when we can't tell).

APPROACH -- shortlist then verify (the follower is the arbiter):
  1. SHORTLIST candidate scores cheaply: the ngram trigram index
     (`data/fingerprints/ngram_index.json`) votes for melodically-distinctive
     matches, UNION the clip's current folder-label (so a correctly-labeled clip
     is always a candidate even when its texture -- e.g. Bach arpeggios -- has no
     distinctive trigrams).
  2. VERIFY each candidate by running the follower on a WINDOW of the
     transcription against that score and reading coverage + confidence. The
     follower distinguishes "this score matches" (high) from "doesn't" (low)
     where exact ngrams can't.
  3. DECIDE: take the best-verified score if it clears a coverage floor with
     margin over the runner-up; otherwise ABSTAIN -- the honest outcome when the
     true piece isn't in the catalog (Op.25/5-style off-rep) or the clip is a
     mixed practice session.

WHY A WINDOW: identity needs a stable snippet, not the whole clip -- verifying a
5000-note clip against 15 candidates would be minutes. A ~60 s mid-clip window is
fast and matches how production would ID from a snippet.

RUNNING (from the PRIMARY checkout so data/ resolves):

  cd /Users/jdhiman/Documents/crescendai/model
  PYTHONPATH=<worktree>/model/src .venv/bin/python -m follower_eval.piece_id \
    --clips fantaisie_impromptu/JbYGHXsQiqk bach_prelude_c_wtc1/w03EKJjOTJE
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from follower_bench.hmm import TUNED_HMM_PARAMS, follow_hmm
from follower_bench.segments import PerfNote
from follower_eval.realaudio import (
    SCORE_FILENAME_BY_PIECE,
    load_bundle_notes,
    load_score,
)

DEFAULT_NGRAM_INDEX = Path("data/fingerprints/ngram_index.json")

# Verify window + decision thresholds. The decisive signal is CONFIDENCE, not
# coverage: a wrong score can still cover a tonal 30 s window (many pieces share
# notes), but only the RIGHT score gives the follower high forward-backward
# posterior. Decision score = coverage * confidence.
WINDOW_SEC = 60.0
NGRAM_SHORTLIST_K = 12
ACCEPT_COVERAGE = 0.50  # best must cover >= this fraction of the window
ACCEPT_CONFIDENCE = 0.50  # ...AND the follower must be this confident on it
ACCEPT_MARGIN = 0.15  # ...and beat the runner-up's coverage*confidence by this


@dataclass(frozen=True)
class Candidate:
    score_id: str
    source: str  # "ngram" | "label" | both -> "ngram+label"
    ngram_votes: int
    coverage: float  # follower coverage on the verify window
    confidence: float | None


@dataclass(frozen=True)
class PieceIdResult:
    piece_folder: str  # the (unreliable) corpus label
    video_id: str
    decision: str  # a score_id, or "ABSTAIN"
    label_agrees: bool  # did the decision match the folder label?
    n_window_notes: int
    candidates: tuple[Candidate, ...]  # ranked by coverage


def load_ngram_index(path: Path) -> dict:
    return json.loads(path.read_text())


def ngram_shortlist(perf: list[PerfNote], idx: dict, k: int) -> list[tuple[str, int]]:
    """Top-k catalog score_ids by shared pitch-trigram votes (onset-sorted
    consecutive pitch triples). Cheap; strong on melodic material, blind on
    arpeggiated/repetitive textures -- which is why the caller unions the label."""
    from collections import Counter

    pitches = [n.pitch for n in sorted(perf, key=lambda n: n.onset)]
    votes: Counter[str] = Counter()
    for i in range(len(pitches) - 2):
        hits = idx.get(f"{pitches[i]},{pitches[i + 1]},{pitches[i + 2]}")
        if hits:
            for sid, _pos in hits:
                votes[sid] += 1
    return votes.most_common(k)


def _window(perf: list[PerfNote], window_sec: float) -> list[PerfNote]:
    """A mid-clip slice of notes (start at 25% in, span window_sec) -- more likely
    steady playing than the very start (intros/tuning), and enough to identify."""
    if not perf:
        return []
    ordered = sorted(perf, key=lambda n: n.onset)
    lo, hi = ordered[0].onset, ordered[-1].onset
    t0 = lo + 0.25 * (hi - lo)
    win = [n for n in ordered if t0 <= n.onset <= t0 + window_sec]
    return win if len(win) >= 8 else ordered[: max(8, len(ordered) // 4)]


def _verify(
    window: list[PerfNote], score_id: str, scores_root: Path
) -> tuple[float, float | None]:
    """(coverage, median_confidence) of the follower on `window` against
    `score_id`'s score. Loud FileNotFoundError if the score is missing."""
    score_notes, bar_boundaries, _span = load_score(scores_root / f"{score_id}.json")
    est = follow_hmm(
        window, score_notes, TUNED_HMM_PARAMS, bar_boundaries=bar_boundaries
    )
    confs = [m.confidence for m in est.matches if m.confidence is not None]
    coverage = len(est.matches) / len(window) if window else 0.0
    return coverage, (statistics.median(confs) if confs else None)


def identify(
    piece_folder: str,
    video_id: str,
    bundles_root: Path,
    scores_root: Path,
    idx: dict,
    k: int = NGRAM_SHORTLIST_K,
    window_sec: float = WINDOW_SEC,
) -> PieceIdResult:
    """Identify the score a clip is actually playing: ngram-shortlist UNION the
    folder label, follower-verify each on a window, decide best-or-abstain."""
    perf = load_bundle_notes(bundles_root / piece_folder / f"{video_id}.json")
    window = _window(perf, window_sec)

    # the folder label maps to a score FILENAME (rep pieces: bach_prelude_c_wtc1
    # -> bach.prelude.bwv_846); off-catalog folders fall back to the name itself.
    label_sid = SCORE_FILENAME_BY_PIECE.get(piece_folder, f"{piece_folder}.json")
    label_sid = label_sid[:-5] if label_sid.endswith(".json") else label_sid

    ng = dict(ngram_shortlist(perf, idx, k))
    shortlist = set(ng) | {label_sid}
    cands: list[Candidate] = []
    for sid in shortlist:
        if not (scores_root / f"{sid}.json").exists():
            continue  # a candidate with no score file can't be verified
        cov, conf = _verify(window, sid, scores_root)
        src = (
            "ngram+label"
            if (sid in ng and sid == label_sid)
            else ("label" if sid == label_sid else "ngram")
        )
        cands.append(
            Candidate(
                score_id=sid,
                source=src,
                ngram_votes=ng.get(sid, 0),
                coverage=round(cov, 4),
                confidence=round(conf, 4) if conf is not None else None,
            )
        )

    cands.sort(key=_decision_score, reverse=True)
    decision = decide(cands)
    return PieceIdResult(
        piece_folder=piece_folder,
        video_id=video_id,
        decision=decision,
        label_agrees=(decision == label_sid),
        n_window_notes=len(window),
        candidates=tuple(cands),
    )


def _decision_score(c: Candidate) -> float:
    """The right score has BOTH high coverage AND high follower confidence; a
    wrong score can cover a tonal window but never earns confidence."""
    return c.coverage * (c.confidence or 0.0)


def decide(cands: list[Candidate]) -> str:
    """Best candidate's score_id if it clears the coverage + confidence floors and
    beats the runner-up's coverage*confidence by ACCEPT_MARGIN; else 'ABSTAIN'.
    `cands` must already be sorted by _decision_score descending."""
    if not cands:
        return "ABSTAIN"
    best = cands[0]
    runner = _decision_score(cands[1]) if len(cands) > 1 else 0.0
    if (
        best.coverage >= ACCEPT_COVERAGE
        and (best.confidence or 0.0) >= ACCEPT_CONFIDENCE
        and (_decision_score(best) - runner) >= ACCEPT_MARGIN
    ):
        return best.score_id
    return "ABSTAIN"


def result_to_jsonable(r: PieceIdResult) -> dict:
    out = asdict(r)
    out["candidates"] = [asdict(c) for c in r.candidates]
    return out


def _format(results: list[PieceIdResult]) -> str:
    L = [
        "=" * 84,
        "REAL-AUDIO FOLLOWER EVAL (#133) -- PIECE-ID -- "
        "identify the score actually played",
        "=" * 84,
    ]
    agree = sum(1 for r in results if r.label_agrees)
    abstain = sum(1 for r in results if r.decision == "ABSTAIN")
    relabel = sum(1 for r in results if r.decision != "ABSTAIN" and not r.label_agrees)
    L.append(
        f"clips: {len(results)}   label-confirmed: {agree}   "
        f"RE-LABELED: {relabel}   abstain: {abstain}"
    )
    L.append("")
    for r in results:
        L.append(f"{r.piece_folder}/{r.video_id}")
        L.append(
            f"  -> decision: {r.decision}"
            + (
                "  (== label, confirmed)"
                if r.label_agrees
                else (
                    f"  (RE-LABELED, was {r.piece_folder})"
                    if r.decision != "ABSTAIN"
                    else "  (label rejected -> abstain)"
                )
            )
        )
        for c in r.candidates[:4]:
            conf = f"{c.confidence:.2f}" if c.confidence is not None else " n/a"
            L.append(
                f"       cov={c.coverage:.2f} conf={conf} "
                f"votes={c.ngram_votes:<4} [{c.source}] {c.score_id}"
            )
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Piece-ID stage -- identify the score actually played (#133)"
    )
    ap.add_argument(
        "--clips", nargs="+", required=True, help="piece_folder/video_id pairs"
    )
    ap.add_argument(
        "--bundles-root", type=Path, default=Path("data/evals/realaudio_bundles")
    )
    ap.add_argument("--scores-root", type=Path, default=Path("data/scores"))
    ap.add_argument("--ngram-index", type=Path, default=DEFAULT_NGRAM_INDEX)
    ap.add_argument(
        "--k", type=int, default=NGRAM_SHORTLIST_K, help="ngram shortlist size"
    )
    ap.add_argument(
        "--window-sec", type=float, default=WINDOW_SEC, help="verify-window length"
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    idx = load_ngram_index(args.ngram_index)

    # A corpus pass is ~30-60 s/clip, so write after every clip and resume from
    # what --out already holds: a crash on clip 20 must not discard clips 1-19.
    done: list[dict] = []
    if args.out and args.out.exists():
        done = json.loads(args.out.read_text())
        print(f"resuming: {len(done)} clip(s) already in {args.out}")
    seen = {(d["piece_folder"], d["video_id"]) for d in done}

    results = [
        PieceIdResult(
            **{**d, "candidates": tuple(Candidate(**c) for c in d["candidates"])}
        )
        for d in done
    ]
    for i, spec in enumerate(args.clips, 1):
        folder, vid = spec.split("/", 1)
        if (folder, vid) in seen:
            continue
        print(f"[{i}/{len(args.clips)}] {folder}/{vid}", flush=True)
        r = identify(
            folder,
            vid,
            args.bundles_root,
            args.scores_root,
            idx,
            k=args.k,
            window_sec=args.window_sec,
        )
        results.append(r)
        print(f"    -> {r.decision}", flush=True)
        if args.out:
            args.out.write_text(
                json.dumps([result_to_jsonable(x) for x in results], indent=1)
            )
    print(_format(results))
    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
