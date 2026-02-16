# PercePiano Data Quality Audit

## Motivation

Priority signal validation (notebook 01_priority_signal_validation) showed that pure MuQ embeddings (AUC 0.933) dramatically outperform PercePiano's 19 quality dimensions (AUC 0.814) for predicting masterclass STOP moments. Diagnostics revealed high multicollinearity (19 dims collapse to ~5-6 factors) and coefficient instability across folds.

Since the model improvement pipeline trains on PercePiano labels, we need to understand which dimensions are trustworthy, which are redundant/noisy, and whether a reduced set could serve as a better training target.

## Goals

1. Identify which of the 19 dimensions are trustworthy vs noisy/redundant
2. Quantify label informativeness and redundancy structure
3. Determine whether reduced/transformed dimension sets improve downstream utility

## Notebook

`model/notebooks/model_improvement/00_percepiano_data_audit.ipynb` -- runs locally on M4/32GB, no GPU needed.

## Experiments

### 1. Correlation & Redundancy

- Pearson correlation heatmap with hierarchical clustering (seaborn clustermap)
- Variance Inflation Factor (VIF) per dimension -- VIF > 10 = severe redundancy
- Ranked list of dimension pairs by |r|

### 2. Factor Analysis

- PCA on standardized 1202x19 label matrix
- Scree plot + cumulative explained variance
- Factor loadings heatmap (which original dims map to which factors)
- Parallel analysis (Monte Carlo) for statistically justified factor count

### 3. Per-Dimension Distributions

- 4x5 histogram grid for all 19 dims
- Flag: variance < 0.01, skewness > |1|, >50% at floor/ceiling
- Differential entropy per dimension as informativeness score

### 4. MuQ Probing

- Ridge regression per dimension: stats-pooled MuQ (2048-d) -> score
- 4-fold CV using existing folds.json
- R2 bar chart sorted by probing accuracy
- High R2 = audible, Low R2 = symbolic/subjective or noise

### 5. MuQ Residual Analysis

- Linear map: MuQ embeddings -> all 19 dims simultaneously
- PCA on residuals to check if structured or random noise
- Structured residuals = signal symbolic encoders could capture

### 6. UMAP Visualization

- 2D UMAP on stats-pooled MuQ embeddings
- Color by top-5 most/least audible dimensions (from probing)
- Smooth gradients = well-grounded, salt-and-pepper = noisy

### 7. Canonical Correlation Analysis (CCA)

- CCA between MuQ embeddings and 19-dim labels
- Number of significant canonical variates
- Total shared information quantification

### 8. STOP Prediction with Reduced Dims

- Reuse masterclass 98-segment data
- Compare LOVO AUC: all 19 dims vs PCA-reduced vs stable-only vs top-R2-only
- Tests whether dimension reduction helps downstream

### 9. Competition Validation (deferred)

- Placeholder for T2 competition ordinal data
- Spearman correlation per dimension vs placement

## Data Dependencies

- `model/data/percepiano_cache/labels.json` (1,202 segments x 19 dims)
- `model/data/percepiano_cache/muq_embeddings.pt` (pre-extracted frame-level MuQ)
- `model/data/percepiano_cache/folds.json` (4-fold CV splits)
- Masterclass segments from `model/data/masterclass_cache/` (for STOP prediction)
