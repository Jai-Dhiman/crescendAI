# Molecules

Pedagogical moves. Each molecule chains 2-10 atoms with explicit when-to-invoke instructions. Molecules call atoms, not other molecules. Per the V5 Option B contract, molecules MAY consume prior-molecule artifacts when those artifacts are passed in by the orchestrating compound (the molecule itself never calls another molecule). See `../README.md` for three-tier overview.

**Final size:** 9 molecules.

**Training target.** Molecules are the Qwen finetune data-collection target -- each has a precise input/output spec and per-skill eval signal for atomic RL.

## Diagnosis molecules (write DiagnosisArtifact)
- `voicing-diagnosis` -- detect imbalance between melody and accompaniment voicing in homophonic textures
- `pedal-triage` -- distinguish over-pedaling, under-pedaling, and pedal-timing issues
- `rubato-coaching` -- detect uncompensated rubato (timing deviation without return)
- `phrasing-arc-analysis` -- assess shape of dynamic and timing arc across a marked phrase
- `tempo-stability-triage` -- distinguish drift, intentional rubato, and loss of pulse
- `dynamic-range-audit` -- compare dynamic range used vs asked-for by score
- `articulation-clarity-check` -- identify slur-vs-staccato execution mismatches
- `cross-modal-contradiction-check` -- flag where MuQ dimension and AMT-derived feature disagree (highest-signal teacher diagnostic per the How-to-grep-video wiki)

## Action molecules (write ExerciseArtifact)
- `exercise-proposal` -- generate a targeted exercise from a passed-in DiagnosisArtifact (Option B input contract)

Each molecule declares its atom dependencies in YAML `depends_on`. A molecule's artifact is the interface to compounds; compounds never reach inside a molecule's reasoning.
