# Task 2 — RewardBench track  (STUB — CONFIRM SCOPE)

> **!! CONFIRM BEFORE USING !!** This doc assumes "task 2" = the **RewardBench** reward-model track (Track B, per #104's "fork-then-converge with the RewardBench track"). If you actually meant a **second MIREX 2026 task**, rename this file and start a dedicated research pass — none of the difficulty research transfers.
>
> **Status:** STUB — NOT researched. We spent the last two sessions entirely on Task 1 (MIREX difficulty). This track has had zero research passes.
> **Next step:** decide scope (below), then run a `/deep-research` pass to fill this doc the way `research-learnings.md` was filled for Task 1.

## What we know (from #104 only)

- #104 frames the two efforts as a **fork-then-converge**: MIREX difficulty (Track A / Task 1) vs the RewardBench track (Track B / Task 2), with a plan to "commit to ONE by ~July 24 for the Aug-Sep build."
- Running both in parallel (this doc's premise) is a **deliberate departure** from that commit-to-one plan. Note the resource implication: two tracks share one build slot.

## Open scoping questions (answer first)

1. **Is task 2 RewardBench, or a second MIREX task?** (Determines everything below.)
2. If RewardBench: which split/version, and what is the CrescendAI angle — a music/audio reward model, or a general reward model? What existing assets (teacher LLM, judge harness, `apps/evals/`) transfer?
3. What is the published baseline to match/confirm first (mirroring Task 1's "match the baseline" discipline)?
4. Deadline + submission mechanics.

## Research pass to run (once scope confirmed)

Mirror the Task 1 method: (a) problem definition + metric; (b) SOTA + key groups + numbers; (c) datasets + licensing + contamination; (d) approaches + drawbacks; (e) open problems; then (f) map onto CrescendAI assets, name the smallest spike + gate + cost. Persist findings here + a summary comment on the tracking issue.

## Assets that PLAUSIBLY transfer (unverified — for the brainstorm to test)

- The dual-judge / LLM-judge harness in `apps/evals/` (Spearman, dual-judge tests).
- The V6 teacher (glm@WorkersAI) as a reward/judge model.
- Ordinal-dominated training recipe (if the reward signal is ordinal/pairwise — note the SAME within-piece-vs-cross-item pairing caveat that bit Task 1).
