# STOP Classifier — Removed 2026-05-27

The STOP classifier — a logistic regression on the 6-dim MuQ scores predicting "would a teacher interrupt here" — has been removed from the production pipeline. This doc records why, what it taught us, and what a future "this moment needs attention" gate should look like if we revisit the idea.

## What STOP was

A binary classifier trained on hand-curated masterclass-video moments. Positive class = audio window of student playing immediately before a teacher interrupted; negative class = audio windows where the teacher chose not to interrupt (≥5s gap between consecutive interventions). The deployed model used StandardScaler + 6-input logistic regression with a +1.7 bias term, LOVO-CV AUC ≈ 0.65.

Source code (now deleted):
- `apps/api/src/harness/skills/atoms/classify-stop-moment.ts` — atom tool
- `apps/api/src/wasm/score-analysis/src/stop.rs` — Rust implementation
- `apps/api/src/wasm/score-analysis/pkg/*` — rebuilt without `classify_stop` export
- `apps/config/stop_config.json` — runtime weights
- `stop_probability` field on `TeachingMoment` (Rust struct + TS interface)

Kept as research artifacts:
- `model/data/labels/stop_classifier_weights.json` — historical record of deployed weights
- `model/src/masterclass_experiments/` — training pipeline (orphaned; see "Follow-ups")
- `model/archive/notebooks/masterclass_experiments/03_stop_ablations.ipynb` — ablation notebook

### Behavior swap: deviation-magnitude gate

Where `teaching_moments.rs` previously gated candidate chunks on `stop::classify(...).triggered`, it now gates on whether the chunk's worst dimension is actually below baseline (`deviation < 0`). This preserves the positive-moment fallback (returns "Nice work!" when the student is at-or-above baseline on every dimension of every chunk) using a signal the system already cares about, without re-introducing a learned classifier on top of noisy labels. All 16 cargo tests pass against the new logic, including `no_chunks_above_threshold_returns_positive` and `positive_moment_picks_highest_improvement`.

## Why we removed it

The architecture migrated to post-session synthesis (see [Pipeline Architecture, 2026-03-23](../../docs/architecture.md)), and STOP's original product role evaporated:

| Original STOP job | Status today |
|---|---|
| Real-time interrupt gate during a session | Pipeline doesn't interrupt anymore |
| Filter chunks worth surfacing | Post-session synthesis LLM gets full context |
| "Needs attention" signal in synthesis input | Replaced by deviation magnitude (`selectTeachingMoment`) |

At the time of removal:
- STOP was computed inside `wasm.selectTeachingMoment` and used to gate moment generation (`stop_result.triggered`)
- The `stop_probability` field in the returned `TeachingMoment` was *discarded* by the TS layer
- The `classify-stop-moment` atom was registered as an LLM-callable tool but **no molecule, compound, or synthesis prompt invoked it**
- An eval run (n=12, 2026-05-26) confirmed 100% trigger rate across all skill levels and a negative skill correlation (Spearman ρ = −0.55), meaning the classifier was functionally an "always say stop" with a slight inverted-skill perturbation

## What worked with STOP

These are worth preserving as design lessons even though we removed the classifier itself:

1. **Vertical-slice instrumentation.** The Rust → WASM → TS bridge → atom-registry layering let us build, test, and ship a learned signal end-to-end inside the pipeline. The plumbing is sound; the problem was the signal, not the integration.
2. **Cheap inference at the edge.** 6-input logistic regression runs in microseconds inside WASM. Anything heavier (deep model, embedding lookup) would have been a worse default. Future gates should preserve this latency budget.
3. **Per-dimension interpretability.** Linear coefficients on named dims made the model's behavior diagnosable — we could read off which dimensions pushed toward "stop" and which away. A neural alternative would have hidden the same failure behind a black box.
4. **The atom-tool pattern.** Exposing STOP as a tool the synthesis LLM *could* call (even though it didn't) is the right pattern for new signals: surface them as optional tools first, let prompts opt in, observe which the model actually calls before hard-wiring into critical paths.

## What didn't work

These are the failure modes a successor must avoid:

1. **The label was a noisy proxy.** "Teacher interrupted here" is correlated with "playing needs intervention" but isn't synonymous. Teachers interrupt to praise, to preempt difficult passages, at phrase boundaries for discussion, when running out of session time. Many CONTINUE windows contain mistakes the teacher chose to let pass for pedagogical reasons; many STOP windows contain excellent playing the teacher stopped to discuss. The label ceiling for any model on this data is somewhere near AUC 0.65 — which is roughly where the deployed weights landed.
2. **The +1.7 bias dominated everything.** `sigmoid(1.7) ≈ 0.85` baseline meant the classifier fired on nearly every chunk regardless of input. The negative skill correlation we measured was real but tiny compared to the bias floor. Class-balancing or threshold tuning could have helped, but the underlying label noise would still have capped AUC near 0.65.
3. **The 6-dim bottleneck.** Ablation in notebook 03 showed the 19-dim PercePiano raw scores and 2048-dim pooled MuQ embeddings were comparable or better. The 6-dim composite was chosen for production ergonomics (interpretability, small payload, matches the rest of the pipeline), not signal strength.
4. **No deployment loop.** The weights JSON was exported ad-hoc from a notebook execution. No reproducible training script writes it. If someone wanted to retrain on new data or a different label scheme, the first task would be reconstructing the export.
5. **Decoupled from product need.** STOP was built before post-session synthesis existed. When the architecture shifted, STOP wasn't deprecated alongside it — it stayed wired into the WASM moment-selection gate where it silently became a no-op (because the bias made it always trigger). Nobody noticed until the eval harness was fixed.

## What a future "needs attention" gate should look like

If the product later needs a real-time "is this moment worth flagging" signal — e.g., to highlight specific timestamps in a session replay, to gate which chunks get expensive Tier-1 WASM analysis, or to drive a future real-time intervention feature — start here:

### Label design
- **Be specific about intent.** Don't predict "did a teacher stop" — predict "is there an identifiable issue here that warrants feedback." That requires re-annotating the masterclass corpus with the teacher's *reason* for stopping (correct vs. praise vs. preempt vs. transition), and treating only `correction` as positive.
- **Consider label-free signals first.** Score-aligned deviation magnitude (onset drift, velocity mismatch, pedal misalignment) doesn't need human labels at all and is more directly tied to "objectively wrong" — which is closer to what the product cares about. The bar-analysis path (`analyzeTier1` in WASM) already computes much of this.

### Feature design
- **Don't bottleneck through the 6 dims again** unless ablations justify it. The 19-dim PercePiano or pooled MuQ embedding routes performed comparably or better in notebook 03 and don't lose information.
- **Add temporal context.** STOP looked at one 15s chunk in isolation. A teacher's decision to interrupt depends on what just happened, what's about to happen, and whether the student already failed at this passage. Include a small window of past chunks.

### Deployment design
- **Write a reproducible training script** that takes a config and writes the deployed weights JSON. Not a notebook. The provenance gap on the original deployment is the main reason iteration was hard.
- **Don't bias-pin to "always trigger."** Either calibrate the threshold against a held-out set with an explicit target trigger rate (e.g., 20% of chunks), or use class-balanced training. A gate that fires on everything is not a gate.
- **Surface as an opt-in tool first.** Register it as an atom the synthesis prompt *can* call, observe whether the LLM actually finds it useful, only then promote it into a hard-wired pipeline gate.
- **Eval-driven promotion.** The `eval_chunk` handler now exists. Before any new gate goes to production, it should ship with an eval scenario that measures (a) trigger rate by skill level, (b) correlation with a ground-truth signal (deviation magnitude, expert annotation), and (c) downstream effect on synthesis quality. STOP shipped without any of these.

## Follow-ups

- [ ] **Decide archive policy for `model/data/labels/stop_classifier_weights.json`.** Currently kept as research artifact. If the masterclass-experiments line of work is dead, move to the R2 archive prefix alongside other offloaded research data.
- [ ] **Document the masterclass-experiments retirement.** `model/src/masterclass_experiments/` and notebooks 01–03 are now orphaned. Either repurpose for the next labeling effort or move to `model/archive/`.
