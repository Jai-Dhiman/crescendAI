# MIREX 2026 — CrescendAI campaign

**Status:** ACTIVE · **Track:** A (Task 1) only · **Submission deadline:** 2026-10-01

Living-doc hub for CrescendAI's MIREX 2026 participation. The fork decision is
made: we are submitting to **Track A only**.

## The track

| Track | MIREX task | Issue | Living doc |
|---|---|---|---|
| **A (Task 1)** | Music Performance Difficulty Prediction | [#104](https://github.com/Jai-Dhiman/crescendAI/issues/104) | [track-a-difficulty-prediction.md](./track-a-difficulty-prediction.md) |

Active sub-issues: [#137](https://github.com/Jai-Dhiman/crescendAI/issues/137)
(Transkun-unlocked features — feature frontier now closed) and
[#138](https://github.com/Jai-Dhiman/crescendAI/issues/138) (end-to-end encoder
fine-tune — the remaining unrefuted lever).

## MIREX 2026 logistics
- **Submission opens:** July 1, 2026. **Closes:** Oct 1, 2026. **Results:** Oct 15, 2026.
- **Venue:** ISMIR 2026, Abu Dhabi, Nov 8-12 (online component). Accepted work → Late-Breaking Demo (LBD).
- **Deliverable:** a self-contained system (Docker/CLI) + a 2-4 page extended-abstract PDF (ISMIR LBD template, non-anonymous). New-for-2026: must disclose training-data size, model size, compute.
- **MIREX home:** https://music-ir.org/mirex/wiki/MIREX_HOME · **Org GitHub:** https://github.com/ismir-mirex/

## How to use these docs
The track doc is a **living document**: append to its Decision Log, keep the
status header current, and post a `STATE:` line to the matching GitHub issue at
each session end (per repo ritual). Fresh sessions: read the track doc + the
linked issue's latest `STATE:` comment before starting.

## Dropped: Track B (Task 2 — Music Evaluation via CMI-RewardBench)

**Dropped 2026-08-03.** Track B was explored in parallel under issues #105, #106,
#107, #122, #123, and #124 (all closed) in a separate gitignored repo, which has
since been deleted along with its cached corpora and its R2 archive.

Why it was dropped:
- **No proprietary asset.** The task is generated-pop-song preference ranking —
  outside the solo-piano domain where CrescendAI's encoders, data, and thesis
  live. It was a credential at best, never a moat.
- **The frozen-encoder ceiling held under four independent attacks.** Head
  architecture, sampler, cross-dataset augmentation (AIME, +6,478 pairs), and
  LoRA fine-tuning of the CLAP audio tower each returned a null or a regression
  against the honest held-out-generator gate.
- **In-distribution parity was not robustness.** Frozen CLAP + a tiny head
  reached statistical parity with the CMI-RM baseline in distribution
  (test 0.790 vs 0.778, McNemar p=0.35), but the leave-one-generator-out gate
  cost ~5.3pp (0.7371), and the newest generator (suno-v5) scored near chance
  at 0.583. LoRA made that gate *worse* (0.6785 → 0.6583) while inflating the
  all-generator number to 0.8065 — an overfit mirage.

The one transferable finding, worth remembering: **CLAP feature granularity, not
head architecture, was the lever.** Mean-pooling three 10s windows beat a single
30s chunk by +3.1pp test, which retroactively exposed an earlier "frozen wall"
as a `CHUNK_SECONDS=30` under-resolution artifact. Pooling operator did not
matter; window length did.
