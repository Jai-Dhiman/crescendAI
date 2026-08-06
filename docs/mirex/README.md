# MIREX 2026 — CrescendAI campaign

**Status:** ACTIVE, submission phase · **Track:** A (Task 1) only · **Submission deadline:** 2026-10-01

Living-doc hub for CrescendAI's MIREX 2026 participation. We are submitting to
**Track A only**.

## The track

| Track | MIREX task | Epic | Living doc |
|---|---|---|---|
| **A (Task 1)** | Music Performance Difficulty Prediction | [#104](https://github.com/Jai-Dhiman/crescendAI/issues/104) | [track-a-difficulty-prediction.md](./track-a-difficulty-prediction.md) |

**Where the research got to:** the hand-crafted feature frontier is closed
([#137](https://github.com/Jai-Dhiman/crescendAI/issues/137), closed) and the
encoder fine-tune — the one unrefuted lever — worked
([#138](https://github.com/Jai-Dhiman/crescendAI/issues/138) →
[#149](https://github.com/Jai-Dhiman/crescendAI/issues/149), both measured).
The remaining work is not modelling: **there is no submission container yet.**
Read the track doc's "Competition contract" section before proposing anything —
it is the fetched task-page text, and three of its clauses are unsatisfied by
the current repo.

## MIREX 2026 logistics
- **Submission opens:** July 1, 2026. **Closes:** Oct 1, 2026. **Results:** Oct 15, 2026.
- **Venue:** ISMIR 2026, Abu Dhabi, Nov 8-12 (online component). Accepted work → Late-Breaking Demo (LBD).
- **Deliverable:** a Docker container with a standardised `WAV path → float` interface, plus a 2-4 page extended-abstract PDF (ISMIR LBD template, non-anonymous). New for 2026: must disclose training-data size, model size, and compute.
- **No baseline, reference container, or starter kit is provided.** The interface details are under-specified; ask the captains rather than guessing.
- **MIREX home:** https://music-ir.org/mirex/wiki/MIREX_HOME · **Task page:** https://music-ir.org/mirex/wiki/2026:Music_Performance_Difficulty_Prediction · **Org GitHub:** https://github.com/ismir-mirex/

## How to use these docs
The track doc is a **living document**: append to its Decision Log, keep the
status header current, and post a `STATE:` line to the matching GitHub issue at
each session end (per repo ritual). Fresh sessions: read the track doc + the
linked issue's latest `STATE:` comment before starting. Active state belongs in
issues, not in new plan or summary files.

`phase1-lora-runbook.md` is the **live retraining procedure** (Stages 0-3.5) —
the submission depends on it, since the final model must be retrained on all
compliant pieces.

## Dropped: Track B (Task 2 — Music Evaluation via CMI-RewardBench)

**Dropped 2026-08-03**, with issues #105, #106, #107, #122, #123, #124 closed
and its separate gitignored repo, cached corpora, and R2 archive deleted. It was
generated-pop-song preference ranking — outside the solo-piano domain where
CrescendAI's encoders, data, and thesis live, so a credential at best and never
a moat. Its frozen-encoder ceiling held under four independent attacks (head
architecture, sampler, cross-dataset augmentation, LoRA fine-tuning of the CLAP
tower), and in-distribution parity turned out not to be robustness: the
leave-one-generator-out gate cost ~5.3pp and the newest generator scored near
chance.

**Its one transferable finding is now Track A open lever #2**, so act on it
there rather than here: *feature window granularity, not head architecture, was
the lever* — mean-pooling three 10s windows beat a single 30s chunk by +3.1pp,
which retroactively exposed an apparent "frozen wall" as a `CHUNK_SECONDS=30`
under-resolution artifact. The pooling *operator* did not matter; window
*length* did.
