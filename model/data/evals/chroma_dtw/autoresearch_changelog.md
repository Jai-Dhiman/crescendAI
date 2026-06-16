# Chroma-DTW follower autoresearch (#13)

Metric (frozen, lexicographic): maximize `primary` (% chunks |err|<=1.5s), tie-break minimize `mean|err|`.
Guard: verify `regressed == []` AND `cargo test` green. Verify: `just chroma-eval-verify`.

## Iteration 0 -- BASELINE
primary=15.0, mean|err|=18.47s, p90=56.6, max=62.6. g2=0.510, g4=5.56.
Per-chunk attribution: `bach_invention_1` stuck at score=0.0 for all 8 chunks (lock-to-origin,
err grows -4->-63s); `bach_prelude` tracks ~7 chunks then stalls ~38-44s (err -9.8->-22).
Root cause: prior-carry (`prior = prediction`) has no forward pressure; a sticky early
endpoint argmin on self-similar openings freezes the band at the origin (positive feedback).

## Iteration 1 -- KEEP (+5 primary, -10.76 mean) [RATCHETED]
Hypothesis: lock-to-origin is a prior-feedback stall; a forward-marching prior breaks it.
Change: replace `prior = prediction` carry with a dead-reckoned prior from clip start
(`elapsed_audio * tempo_ratio`, tempo_ratio=0.45); band-DTW refines within the window.
Result: primary 15->20, mean 18.47->7.71, p90 56.6->18.9, max 62.6->24.9, g2 .51->.70,
g4 5.6->33.3. regressed=[]. Invention no longer pinned at score=0; tail collapsed.

## Iteration 2 -- REVERT (g4 guard) 
Change: tempo_ratio 0.45->0.50 (read-only sweep peaked here: primary 25, mean 6.49).
Result: primary 25, mean 6.49 BUT g4 33.3->16.7 => regressed. Reverted.

## Iteration 3 -- REVERT (g2+g4 guards) -- but HITS PRIMARY TARGET
Hypothesis: endpoint-only band lets the warping-path MIDPOINT (the reported prediction)
warp back to a look-alike while the endpoint is forced forward -> jitter + teleports.
Change (Rust): when prior>=0, slice score to [prior-back,prior+fwd) and run subseq DTW
on the slice, confining the WHOLE path to the window. Production (prior<0) unchanged.
Result: primary 20->35 (TARGET), mean 7.71->5.89, tail unchanged. BUT g2 .70->.57 and
g4 33.3->27.8 => regressed=['g2','g4']. Reverted per Iron Law.

## STOP -- broken-guard finding (g4)
g4 marks consecutive chunks "continuous" iff |dScore - dAudio| <= 5s, i.e. it expects
score-time to advance at AUDIO rate (tempo ratio 1.0). The eval performances run at
ratio ~0.5 (prelude) - 0.7 (invention), so an accurately-tracking follower has
dScore ~= 6s per 12s audio -> |6-12|=6 > 5 -> flagged discontinuous. g4 thus PENALISES
accurate tracking of any performance slower than ~0.58x notated tempo. This is the
binding constraint blocking iterations 2 & 3; it is a metric mis-specification, not a
follower fault. Per the autoresearch rule (never change a frozen metric mid-loop), the
loop is stopped pending a user decision on g4. The window-slice (iter 3) proves the
follower CAN reach primary=35; it is gated only by g4 (mis-specified) + a g2 regression
(addressable with gentler slicing / cost re-derivation).
Recommended g4 fix: compare dScore against expected dScore (dead-reckon, dAudio*tempo)
or measure backward-jump magnitude / monotonicity, not raw dAudio.

## METRIC CHANGE (user-approved) + RE-BASELINE
g4 redefined: pct of consecutive pairs with implied tempo dScore/dAudio in [0.25,1.5]
(tempo-agnostic teleport/stall detector) -- replaces the ratio-1.0 assumption.
g2 floor HELD at original 0.5098 (NOT iter-1's lucky 0.703): iter-1's g2 ratchet was the
root of (a) smoke breakage (smoke g2=0.5 degenerate single-class AUC) and (b) iter-3's
spurious g2 'regression'. Holding the floor at chance preserves original guard strength.
New frozen baseline: primary=20, g2>=0.5098, g4=66.67. 35 eval tests green; smoke+real exit 0.
Resuming loop against this baseline.

## Iteration 4 -- KEEP (+15 primary) [RATCHETED] (window-slice, re-applied under fixed g4)
primary 20->35, mean 7.71->5.89, g4 66.7->83.3, g2 0.571 (>=floor). regressed=[]. Baseline
ratcheted to primary=35 (g2 floor held 0.5098). The slice that was guard-blocked pre-fix now keeps.

## Iteration 5 -- REVERT (g2 confounded) -- but BEST RESULT (adaptive per-clip tempo)
Change: seed tempo_est=0.45, re-estimate per chunk from cumulative implied tempo (pred/elapsed)
gated to [0.3,1.0]. Fast clips (invention ~0.69) and slow (prelude ~0.5) each converge.
Result: primary 35->40, mean 5.89->2.60, p90 18.9->7.8, MAX 24.9->9.2 (teleport tail ELIMINATED),
g4 83.3->94.4. BUT g2 0.571->0.406 (below the 0.5098 chance floor) => regressed=['g2']. Reverted.
g2 DIAGNOSIS: g2=AUC(cost predicts error) is CONFOUNDED -- cost tracks piece difficulty
(invention chroma cost ~0.5 vs prelude ~0.3, independent of alignment), not alignment error.
mean cost within-tol (0.385) > outside-tol (0.357) => anti-correlated. As accuracy improved,
residual error concentrated in the harder piece and inverted g2. Like g4, g2 mis-specified for
a 2-piece heterogeneous corpus. Fixing it (e.g. per-piece cost z-score, or a cost gate that
isn't AUC-against-tolerance) is the deferred "fix g2 design" lever -> would unlock primary=40.

## STATE: loop at primary=35 (TARGET reached, guard-clean, ratcheted). g2 redesign = next lever.

## METRIC CHANGE #2 (user-approved): g2 redesigned + smoke decoupled
g2 was AUC(path cost, error) -- confounded (cost = per-piece chroma difficulty, not
alignment error; AUC 0.406). Redefined as AUC(|pred - dead_reckon_prior|, error): the
dead-reckon residual is a per-piece-stable confidence signal (AUC 0.65-0.67 both pieces).
Note: under residual-g2 the FIXED-tempo follower scores only 0.407 (its prior systematically
under-shoots the invention so the residual stops discriminating) while the ADAPTIVE follower
scores 0.667 -- the new g2 rewards the adaptive prior, as intended.
Smoke recipe pointed at its own low-guard fixture (smoke_baseline.json) so real g2 can ratchet
above smoke's degenerate single-class AUC (0.5) without breaking the wiring test.

## Iteration 6 -- KEEP (+5 primary) [RATCHETED] (adaptive tempo, now guard-clean)
primary 35->40, mean 5.89->2.60, p90 18.9->7.8, MAX 24.9->9.2 (teleport tail ELIMINATED),
g4 83.3->94.4, g2(residual) 0.667. regressed=[]. Baseline ratcheted to primary=40.

## FINAL: primary 15->40 (2.7x), mean|err| 18.47->2.60s (7x), max 62.6->9.2s. All 5 guards clean.
Kept levers: dead-reckon prior + adaptive per-clip tempo + window-slice DTW. Metric fixes:
g4 (tempo-agnostic continuity) + g2 (dead-reckon-residual confidence). Production WASM path
untouched (banded/adaptive is eval-only). Residual headroom: invention cold-start chunk still
locks at 0; prelude late chunks drift ~7-9s. Next levers (future): smarter cold-start prior,
expand corpus beyond 2 pieces (guards still 2-piece-sensitive).
