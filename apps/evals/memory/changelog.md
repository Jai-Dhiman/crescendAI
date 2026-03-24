# Memory Eval Autoresearch Changelog

## Iteration 0 -- BASELINE
**Date:** 2026-03-23
**Composite:** 0.771 (reweighted 0.55/0.45, chat extraction unavailable)
**Breakdown:**
- synthesis_recall: 0.833 (33 scenarios, 30 synthetic + 3 realistic with cached responses)
- synthesis_precision: 0.554 (hallucination rate: 0.446)
- temporal_assertion_accuracy: 0.694 (36 assertions, per-category: extraction 0.71, multi_session 0.50, temporal 1.00, knowledge_update 0.50, abstention 1.00)
- chat_extraction: skipped (API not running)
**Matching:** regex + cosine (threshold 0.55)
**Scenarios:** 45 synthesis-eligible (30 synthetic + 15 realistic), 39 temporal-eligible (30 synthetic + 9 realistic)
**Notes:**
- Hallucination rate (0.446) is the primary quality issue -- synthesis creates too many facts not in the expected set
- Multi-session (0.50) and knowledge_update (0.50) temporal categories are weak
- Abstention (1.00) and temporal (1.00) are perfect -- system correctly handles these
- Realistic scenarios with cached responses only (3 of 15 synthesis, unknown temporal) -- need --live run with all scenarios for full picture

## Iteration 1 -- REVERT (-0.033)
**Hypothesis:** Adding multi-session awareness, abstention rules, and anti-hallucination constraints to the synthesis prompt
**Change:** 17 new lines added to SYNTHESIS_SYSTEM: cross-session pattern rules, evidence grounding, abstention on insufficient data, dimension-scope constraints
**Result:** Composite 0.771 -> 0.738. Multi-session improved 0.50->0.62, hallucination rate improved 0.446->0.418. BUT extraction dropped 0.71->0.59, knowledge_update dropped 0.50->0.25. Net negative.
**Why it failed:** The abstention rules ("do NOT create facts from fewer than 3 observations", "do NOT create facts from a single session") were too aggressive. They prevented the system from creating legitimate facts that had sufficient but not overwhelming evidence. The evidence grounding rules also made the system overly cautious about approach and arc facts.
**Lesson:** Need to tune the abstention threshold or apply it only to specific fact types (dimension facts) rather than globally.

## Iteration 2 -- REVERT (-0.063)
**Hypothesis:** Multi-session awareness only (without abstention rules) would keep the multi-session gains without the extraction losses
**Change:** 5 new lines: cross-session pattern rules only. No abstention, no anti-hallucination, no evidence grounding.
**Result:** Composite 0.771 -> 0.708. ALL metrics worse: synth_recall 0.833->0.788, temp_accuracy 0.694->0.611. Multi-session unchanged at 0.50.
**Why it failed:** The multi-session rules alone didn't provide enough signal to overcome LLM response variance. With stochastic LLM outputs, small prompt additions can cause unpredictable shifts in the overall output distribution. The improvement in iteration 1 (multi_session 0.50->0.62) was likely noise.
**Lesson:** Prompt-only changes to the synthesis system may be hitting diminishing returns. The prompt is already well-tuned. Structural changes (matching strategy, pre-synthesis filtering) may be more reliable than prompt tweaks because they reduce variance rather than adding to it.

## Iteration 3 -- REVERT (-0.210)
**Hypothesis:** LLM-as-judge (Llama 8B at temperature 0) would catch semantic matches that regex misses
**Change:** Added _llm_judge_match() with cosine >= 0.30 pre-filter. Judge answers "yes/no" on semantic equivalence. Toggled via EVAL_MATCH_STRATEGY env var.
**Result:** Composite 0.735 -> 0.525. Synthesis recall HALVED from 0.833 to 0.455. Catastrophic regression.
**Why it failed:** The Llama 8B judge is too strict for piano teaching domain. It rejects paraphrases that regex correctly matches. Example: regex matches "flat dynamics" to "limited dynamic contrast" via `(?i)(flat|uniform|limited).*dynamic`, but the judge says "no" because the phrasing differs. The judge lacks domain knowledge about musical equivalences.
**Lesson:** The existing regex + cosine (0.55) matching is actually well-calibrated for this domain. LLM-as-judge would need a much more detailed prompt with piano-specific equivalence examples, or a larger model (70B not 8B). Not worth the cost -- the matching isn't the bottleneck.

## 3 Consecutive Reverts -- Stopping Autoresearch

Three consecutive no-improvements. Per autoresearch discipline, stopping the loop.

**What we learned:**
1. The synthesis prompt is already well-tuned. Prompt tweaks cause unpredictable shifts in LLM output.
2. LLM variance (sigma ~0.036 on composite) dominates small prompt changes.
3. The regex + cosine matching is well-calibrated for the piano teaching domain.
4. The remaining experiments (supersession chains, staleness decay, semantic dedup) are Rust API changes that affect production behavior, not eval measurement. They should be tested with real student data, not synthetic scenarios.

**What WAS valuable:**
1. The eval infrastructure upgrade (Tasks 1-5) is solid: 20 realistic scenarios, JSON output, multi-layer CLI, composite metric.
2. The baseline measurement (3-run average: composite 0.735, synth_recall 0.813, temp_accuracy 0.639) is trustworthy.
3. We now know the variance bounds (sigma ~0.036), so future experiments need delta > 0.05 to be meaningful.
