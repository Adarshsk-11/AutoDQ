# Dataset Quality Index (DQI) — Methodology

## Formula

```
DQI = wC*Completeness + wS*Consistency + wV*Validity + wU*Uniqueness + wB*Balance
```

where the weights sum to 1. This is a **weighted linear aggregation**, the simplest
multi-criteria scoring approach and the one explicitly suggested by the project spec —
deliberately not a learned/black-box formula, so every score is traceable back to its
components (see "Explainability" below).

## Component formulas

### Completeness
```
C = 1 - (missing cells / total cells)
```
Clamped to `[0, 1]`. Cell-level, not row/column-level, so a dataset with many columns
each missing a little looks the same as one column missing a lot proportionally.

### Uniqueness
```
U = 1 - (duplicate rows / total rows)
```
Exact-duplicate detection via `DataFrame.duplicated()`. We additionally report (but do
not score on) *near*-duplicates found after dropping likely ID-like columns (≥98%
unique values), since an auto-incremented id column can mask otherwise-identical rows.

### Validity
Per-column score averaged across all columns, checking only detectable, domain-agnostic
constraints (per the project spec, we do **not** invent domain rules like "age must be
0-120" unless the user supplies them):
- numeric columns: infinite values, and values that are simultaneously (a) far outside
  the IQR fence and (b) repeated often enough to look like a sentinel/placeholder code
  (e.g. `-999`, `9999`) rather than a genuine one-off outlier
- datetime-like columns: strings that fail to parse as dates
- categorical/text columns: null-like string tokens not recognized as missing by the
  parser (`"N/A"`, `"unknown"`, `"-"`, etc.)
- any column: a constant column (single unique value) is penalized as low-utility

### Consistency
Per-column score averaged across all columns, checking structural/formatting
consistency only:
- categorical columns: how many distinct values collapse into the same category once
  case and whitespace differences are normalized (e.g. `"Yes"` vs `"yes "`)
- object-dtype columns: whether the column mixes numeric-looking strings with genuinely
  non-numeric text (a common symptom of inconsistent data entry)

### Balance
Two numbers are computed and both are shown, per an explicit product decision:

1. **Normalized Shannon-entropy ratio** (used as the actual DQI component score):
   ```
   Balance = H(observed) / H(uniform) = -Σ(p_i * ln(p_i)) / ln(n_classes)
   ```
   This generalizes cleanly to any number of classes and is the more defensible metric
   for scoring purposes — a dataset with 5 near-equal classes and one with 2 near-equal
   classes both score close to 1.0, while any skew pulls the ratio toward 0.

2. **Minority:majority ratio** (`e.g. "1:8"`), shown as a human-readable companion
   statistic alongside the entropy score — much more intuitive to say out loud in a
   presentation than an entropy ratio, even though it's not itself the scored quantity
   and doesn't generalize as cleanly beyond binary comparisons.

If no target column is selected or detected, Balance is reported as **unavailable**
(not scored as 0 or 1, either of which would silently distort the DQI). See "Weight
redistribution" below for how this is handled in the final score.

## Weights

**Default weights (general-purpose, explicitly labeled as a configurable heuristic, not
a universally accepted standard):**

| Component | Weight |
|---|---|
| Completeness | 0.25 |
| Consistency | 0.20 |
| Validity | 0.20 |
| Uniqueness | 0.15 |
| Balance | 0.20 |

**Domain presets** (also configurable heuristics — chosen based on which quality
dimension tends to matter more in each domain, not derived from a benchmark study):

| Domain | Completeness | Consistency | Validity | Uniqueness | Balance |
|---|---|---|---|---|---|
| General | 0.25 | 0.20 | 0.20 | 0.15 | 0.20 |
| Healthcare | 0.30 | 0.20 | 0.20 | 0.10 | 0.20 |
| Finance | 0.20 | 0.30 | 0.20 | 0.10 | 0.20 |
| E-commerce | 0.25 | 0.20 | 0.15 | 0.20 | 0.20 |

Healthcare weighting leans on completeness (missing clinical values are especially
costly); finance leans on consistency (transactional/reporting format integrity
matters more than raw missingness there). These are reasonable starting heuristics,
not empirically validated weightings — flagged as such in the UI.

## Weight redistribution when Balance is unavailable

Rather than treating a missing Balance score as 0 (unfairly tanking the DQI of an
otherwise-clean unsupervised/no-target dataset) or 1 (silently overstating quality),
Balance's default weight is redistributed **proportionally** across the remaining four
components:

```
effective_weight[k] = weight[k] + weight[balance] * (weight[k] / sum(weight[c] for c != balance))
```

So with default weights, if Balance (0.20) is unavailable, the remaining 0.80 of weight
(0.25+0.20+0.20+0.15) is scaled up so the four available components still sum to 1.0,
each keeping its relative proportion.

## Interpretation bands

| Range | Label |
|---|---|
| 0–40 | Poor |
| 40–60 | Low |
| 60–80 | Moderate |
| 80–100 | High |

These are explicitly labeled in the UI as *"configurable heuristic ranges"* — not
scientifically universal benchmarks. If AutoDQ is later validated against real datasets
and measured model performance (see `docs/testing.md`, directional-validity tests), these
thresholds could be recalibrated empirically; that calibration work is out of scope for
this MVP.

## What DQI does *not* claim

- DQI is **not** presented as predicting model accuracy directly — no model has been
  trained to learn that relationship in this MVP. "Baseline Model Performance" is a
  separate, independently measured number (see the Baseline ML Engine).
- Class imbalance is **not** treated as proof of societal/fairness bias — it's reported
  as exactly what it is, a distributional property of the target column.
- Leakage detection is heuristic and conservative: findings are always phrased as
  "potential leakage," with the specific reason for the flag stated, never "leakage
  confirmed."
