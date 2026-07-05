# MIREX 2026 — CrescendAI campaign

**Status:** ACTIVE (spike) · **Created:** 2026-07-03 · **Fork-decision deadline:** ~2026-07-24 · **Submission deadline:** 2026-10-01/02

Living-doc hub for CrescendAI's MIREX 2026 participation. We are running **two tracks in parallel** and will **commit to ONE** (fork-then-converge) by ~July 24 for the Aug-Sep build.

## The two tracks

| Track | MIREX task | Issue | Wheelhouse fit | Living doc |
|---|---|---|---|---|
| **A (Task 1)** | Music Performance Difficulty Prediction | [#104](https://github.com/Jai-Dhiman/crescendAI/issues/104) | **HIGH** — solo-piano audio → difficulty; our core domain | [track-a-difficulty-prediction.md](./track-a-difficulty-prediction.md) |
| **B (Task 2)** | Music Evaluation via CMI-RewardBench | [#105](https://github.com/Jai-Dhiman/crescendAI/issues/105) | **LOW** — generated pop-song preference; no proprietary asset | [track-b-cmi-rewardbench.md](./track-b-cmi-rewardbench.md) |

## Shared MIREX 2026 logistics (both tracks)
- **Submission opens:** July 1, 2026. **Closes:** Oct 1, 2026 (Track B predictions due Oct 2 AOE). **Results:** Oct 15, 2026.
- **Venue:** ISMIR 2026, Abu Dhabi, Nov 8-12 (online component). Accepted work → Late-Breaking Demo (LBD).
- **Deliverable per track:** a self-contained system (Docker/CLI) + a 2-4 page extended-abstract PDF (ISMIR LBD template, non-anonymous). New-for-2026: must disclose training-data size, model size, compute.
- **MIREX home:** https://music-ir.org/mirex/wiki/MIREX_HOME · **Org GitHub:** https://github.com/ismir-mirex/

## How to use these docs
Each track doc is a **living document**: append to its Decision Log, keep the status header current, and post a `STATE:` line to the matching GitHub issue at each session end (per repo ritual). Fresh sessions: read the track doc + the linked issue's latest `STATE:` comment before starting. Brainstorm per track with `/brainstorm`.

## Fork criteria (what we decide by ~July 24)
Commit to the track with the better **expected LBD outcome × strategic value**. Current lean (subject to Track A research): **Track A is the stronger bet** — it uses assets we actually have and is thesis-aligned; Track B is a credential at best (see its doc for the pivot-gate FAIL). Do not let a passing Track-B probe alone trigger a product pivot.
