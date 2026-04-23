# Molecules

Pedagogical moves. Each molecule chains 2-10 atoms with explicit when-to-invoke instructions. Molecules call atoms, not other molecules. See `../README.md` for three-tier overview.

**Target size:** 8-12 molecules. This is the layer that was previously called "atomic pedagogical skills" before the *Skill Graphs 2* tier correction.

**Training target.** Molecules are the Qwen finetune data-collection target -- each has a precise input/output spec and per-skill eval signal for atomic RL.

Populated by V5 work. Initial candidate list:

- `voicing-diagnosis` -- detect imbalance between melody and accompaniment voicing
- `pedal-triage` -- identify over-pedaling vs under-pedaling vs pedal-timing issues
- `rubato-coaching` -- detect uncompensated rubato (timing deviation without return)
- `phrasing-arc-analysis` -- assess shape of dynamic/timing arc across a phrase
- `tempo-stability-triage` -- flag tempo drift vs intentional rubato vs loss of pulse
- `dynamic-range-audit` -- compare dynamic range used vs asked-for by score
- `articulation-clarity-check` -- identify slurred-vs-staccato execution issues
- `exercise-proposal` -- generate a targeted exercise from a diagnosed issue
- `cross-modal-contradiction-check` -- flag cases where MuQ dim score and AMT-derived feature disagree (highest-signal teacher diagnostic)

Each molecule declares its atom dependencies in YAML `depends_on`. A molecule's artifact is the interface to compounds; compounds never reach inside a molecule's reasoning.
