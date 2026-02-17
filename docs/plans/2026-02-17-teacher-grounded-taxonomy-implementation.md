# Teacher-Grounded Feedback Taxonomy Implementation Plan

**Design doc:** `docs/plans/2026-02-17-teacher-grounded-taxonomy-design.md`

**Goal:** Scale masterclass data to 1000+ STOP moments, derive teacher-grounded feedback dimensions empirically, validate against multiple signals, and produce a training-ready spec for Part 2 (model improvement rewrite).

**Tech Stack:** Rust masterclass pipeline (existing), Python analysis notebooks, sentence-transformers, HDBSCAN, scikit-learn

---

## Dependency Graph

```
Task 1 (expand sources) --> Task 2 (pipeline runs)
Task 2 (pipeline runs) --> Task 3 (open-ended re-extraction)
Task 3 (re-extraction) --> Task 4 (clustering & taxonomy)
Task 4 (taxonomy) --> Task 5 (multi-signal scoring)
Task 5 (scoring) --> Task 6 (PercePiano bridge)
Task 6 (bridge) --> Task 7 (validation & deliverables)
```

**Parallelizable:** Tasks 1 and the open-ended extraction prompt change (part of Task 3) can be developed in parallel with pipeline runs.

---

## Task 1: Expand Masterclass Source Library

**Files:**
- Modify: `tools/masterclass-pipeline/data/sources.yaml`

**Step 1:** Audit current sources -- count videos, estimate yield per video, identify gaps in teacher/composer/era diversity.

**Step 2:** Mine high-yield YouTube channels for additional masterclass videos:
- Berliner Klavierfesttage (~20 additional videos expected)
- Juilliard Master Classes (~15 additional)
- tonebase Piano (~10 additional)
- JMC / Jerusalem Music Centre (~5 additional)
- Royal College of Music masterclasses (~10 additional)
- Saline Royale Academy (free tier, ~5 additional)
- PianoTexas archive (~5 additional)

**Step 3:** Fill era/composer gaps:
- Baroque: Bach, Handel, Scarlatti masterclasses
- Classical: more Haydn, early Beethoven
- 20th century: Bartok, Messiaen, Ligeti, Shostakovich
- Romantic underrepresented: Schumann, Mendelssohn, Grieg

**Step 4:** Verify each new URL (accessible, actually a masterclass format, > 10 min). Add to `sources.yaml` with teacher, piece, composer metadata.

**Target:** 150+ total videos in sources.yaml.

**Step 5:** Run diversity check:
```bash
# Count unique teachers, composers, pieces
python3 -c "
import yaml
with open('tools/masterclass-pipeline/data/sources.yaml') as f:
    cfg = yaml.safe_load(f)
teachers = set(v.get('teacher','') for v in cfg['videos'])
composers = set(v.get('composer','') for v in cfg['videos'])
print(f'Videos: {len(cfg[\"videos\"])}, Teachers: {len(teachers)}, Composers: {len(composers)}')
"
```

---

## Task 2: Run Pipeline on All Sources

**Files:**
- No code changes needed (existing pipeline)

**Step 1:** Run the full pipeline on all sources:
```bash
cd tools/masterclass-pipeline
cargo run -- run --stages discover,download,transcribe,segment,extract
```

**Step 2:** Monitor progress. The pipeline is idempotent -- already-processed videos are skipped.

**Step 3:** After completion, count extracted moments:
```bash
wc -l all_moments.jsonl
```

**Step 4:** Spot-check 10 random moments for extraction quality (feedback_summary makes sense, musical_dimension is reasonable, transcript_text captures the relevant speech).

**Target:** 400+ moments from Phase A+B sources.

---

## Task 3: Open-Ended Re-Extraction

**Files:**
- Modify: `tools/masterclass-pipeline/src/extract.rs` (add open-ended extraction mode)
- Modify: `tools/masterclass-pipeline/src/config.rs` (new prompt variant)

**Step 1:** Add an `--extraction-mode` flag to the pipeline with two values:
- `categorical` (default, current behavior -- forces into 10 predefined categories)
- `open_ended` (new -- asks for free-text musical aspect description)

**Step 2:** Write the open-ended extraction prompt. Key change: replace the `musical_dimension` enum constraint with:

```
"musical_aspect": "In 2-5 words, describe the specific musical aspect the teacher is addressing. Be specific -- e.g., 'left hand voicing balance' rather than just 'voicing'. Do not pick from a predefined list."
```

Keep all other fields (feedback_summary, severity, feedback_type, etc.) unchanged.

**Step 3:** Add a new field `musical_aspect_freetext` to `TeachingMoment` schema alongside the existing `musical_dimension`. This preserves backward compatibility -- old moments keep their categorical label, new extraction adds the free-text.

**Step 4:** Re-extract all moments with `--extraction-mode open_ended`. This re-runs GPT-4o on existing transcripts with the new prompt. Store results alongside originals (don't overwrite).

**Step 5:** Validate: sample 20 moments, verify free-text descriptions are specific and varied (not just reproducing the old 10 categories in different words).

**Target:** 1000+ moments with both categorical and open-ended labels.

---

## Task 4: Clustering & Taxonomy Derivation

**Files:**
- Create: `model/notebooks/model_improvement/01_taxonomy_derivation.ipynb`

**Step 1:** Load all moments with open-ended descriptions.

**Step 2:** Embed descriptions using sentence-transformers (`all-MiniLM-L6-v2` or similar -- small, fast, good for short texts).

**Step 3:** Cluster with HDBSCAN:
```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5, metric='cosine')
labels = clusterer.fit_predict(embeddings)
```

`min_cluster_size=30` enforces the validation gate that every category needs >= 30 moments.

**Step 4:** Visualize clusters with UMAP (2D projection, colored by cluster). Check for:
- Clean separation vs diffuse overlap
- Outlier distribution (HDBSCAN label -1)
- Whether clusters align with or diverge from the 10 predefined categories

**Step 5:** Name clusters by examining the 10 most central descriptions per cluster (closest to centroid). Propose human-readable names.

**Step 6:** Manual review -- merge clusters that are semantically identical but linguistically different. Split clusters that contain clearly distinct subconcepts.

**Step 7:** Build the 2-level hierarchy:
- Level 1 (categories): group clusters that frequently co-occur as primary + secondary dimensions
- Level 2 (specific dims): the individual clusters

**Step 8:** Compare against the existing 10 categories from `config.rs`. Document what survived, what split, what merged, what's new.

---

## Task 5: Multi-Signal Scoring

**Files:**
- Modify: `model/notebooks/model_improvement/01_taxonomy_derivation.ipynb` (add scoring cells)

**Step 1: Teacher frequency signal**

Count moments per cluster. Compute percentage of total. Flag any cluster < 5% as at-risk.

**Step 2: MuQ predictability signal**

For each cluster, identify the closest PercePiano dimension(s) by:
- Semantic similarity between cluster name and PercePiano dimension name
- Correlation between cluster membership (binary: teacher mentioned this) and PercePiano scores for matched segments (requires aligning masterclass audio segments to PercePiano predictions)

Use the MLP probing R2 values from the audit as the predictability score for the mapped PercePiano dims.

**Step 3: STOP contribution signal**

For each candidate dimension:
- Build a PercePiano proxy score (weighted average of mapped dims)
- Run LOVO STOP prediction with and without this proxy
- Measure AUC delta

**Step 4: PercePiano grounding signal**

Compute correlation between the candidate dimension's composite PercePiano score and the original individual PercePiano dimensions. Higher correlation = more training data available.

**Step 5: Selection**

Apply criteria from design doc:
- Keep: frequency > 5% AND (predictability > 0 OR STOP delta > 0)
- Drop: frequency < 3% AND no positive soft signal
- Drop: redundant (r > 0.85 with another kept dimension)

**Step 6:** Document the score card for every candidate (kept and dropped) with justification.

---

## Task 6: PercePiano Bridge

**Files:**
- Create: `model/src/model_improvement/taxonomy.py`
- Modify: `model/notebooks/model_improvement/01_taxonomy_derivation.ipynb` (add bridge cells)

**Step 1:** Define the mapping in code:
```python
TEACHER_TAXONOMY = {
    "dimension_name": {
        "percepiano_dims": ["dim1", "dim2", ...],
        "weights": [0.4, 0.3, ...],  # proportional to max(0, MLP_R2), normalized
        "category": "parent_category",
        "description": "What this dimension measures",
    },
    ...
}
```

**Step 2:** Compute composite labels for all 1,202 PercePiano segments:
```python
def compute_composite_labels(raw_labels, taxonomy):
    """Map 19 PercePiano dims to N teacher dims via weighted aggregation."""
    composites = {}
    for dim_name, spec in taxonomy.items():
        indices = [PERCEPIANO_DIMENSIONS.index(d) for d in spec['percepiano_dims']]
        weights = np.array(spec['weights'])
        composites[dim_name] = raw_labels[:, indices] @ weights
    return composites
```

**Step 3:** Validate composite labels:
- Check inter-dimension correlations (must all be < 0.80)
- Check variance (no composite should be near-constant)
- Visualize distributions (histogram grid, analogous to audit section 3)

**Step 4:** For dimensions without PercePiano proxy, create binary weak labels from masterclass data (teacher flagged this dimension as primary or secondary for a given moment).

---

## Task 7: Validation & Deliverables

**Files:**
- Modify: `model/notebooks/model_improvement/01_taxonomy_derivation.ipynb` (add validation cells)

**Step 1: STOP Preservation Gate**

Re-run the STOP prediction experiment from the audit notebook using composite dimensions:
```python
X_mc_composite = compute_composite_labels(X_mc_quality, TEACHER_TAXONOMY)
auc = leave_one_video_out_cv(X_mc_composite, stop_labels, video_ids, segment_ids)
assert auc >= 0.80, f"STOP AUC {auc} below 0.80 gate"
```

Compare against baselines: all 19 dims (0.814), PCA-4 (0.665), attention-pooled MuQ (0.936).

**Step 2: Independence Gate**

```python
composite_corr = np.corrcoef(composite_matrix.T)
max_offdiag = np.max(np.abs(composite_corr - np.eye(n_dims)))
assert max_offdiag < 0.80, f"Max correlation {max_offdiag} exceeds 0.80 gate"
```

**Step 3: Coverage Gate**

```python
coverage = sum(1 for m in moments if m['cluster'] != -1) / len(moments)
assert coverage >= 0.80, f"Coverage {coverage} below 0.80 gate"
```

**Step 4: Actionability Gate**

For each dimension, verify >= 5 teacher quotes exist in the quote bank. Print the quote bank for manual review.

**Step 5: Data Sufficiency Gate**

Verify 1000+ total moments, minimum 30 per category.

**Step 6: Produce deliverables**

1. Export `TEACHER_TAXONOMY` to `model/src/model_improvement/taxonomy.py`
2. Export composite labels to `model/data/percepiano_cache/composite_labels.json`
3. Export quote bank to `model/data/masterclass_cache/quote_bank.json`
4. Print consolidated validation report (all 5 gates pass/fail with values)

---

## Execution Notes

- **Tasks 1-2:** Can begin immediately. Just sourcing and pipeline execution.
- **Task 3:** Requires a small code change to the extraction pipeline (Rust). Can prepare the prompt change in parallel with Task 2.
- **Tasks 4-7:** Sequential, all in a single notebook. Depends on having 1000+ moments.
- **Total estimated wall-clock:** Dependent on pipeline throughput for 150 videos (download + transcribe + extract). Analysis (Tasks 4-7) is fast once data is ready.
- **All analysis runs locally** on M4/32GB. No GPU needed.
