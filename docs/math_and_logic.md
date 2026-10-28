# The Math Behind AutoDQ's Numbers

This is a single-place reference for exactly how every number AutoDQ shows you gets
computed — from the five quality scores, to the DQI, to issue severities, to the
baseline model metrics. Read top to bottom in pipeline order, or jump to a section.

---

## 1. Completeness

**Formula:**
```
Completeness = 1 - (missing cells / total cells)
```
`total cells = n_rows × n_columns`. Clamped to `[0, 1]`.

**Worked example:** a dataset with 1,000 rows × 10 columns = 10,000 cells. If 300 cells
are `NaN`: `Completeness = 1 - 300/10000 = 0.97` → shown as 97 in the UI (score × 100).

**Why cell-level, not column-level:** a dataset where every column is missing 3% of
values scores the same as one where a single column is missing 30% and the rest are
perfect — both have 3% total missing cells. This is a simplification (arguably the
second case is worse for that one column), so the UI *also* shows per-column missing %
separately in the Dataset Profile page — the single Completeness number is a summary,
not the whole picture.

---

## 2. Uniqueness

**Formula:**
```
Uniqueness = 1 - (duplicate rows / total rows)
```
A "duplicate row" = a row that is byte-for-byte identical to an earlier row across
*all* columns (via pandas' `duplicated(keep="first")`).

**Worked example:** 1,000 rows, 40 are exact duplicates of earlier rows →
`Uniqueness = 1 - 40/1000 = 0.96` → shown as 96.

**Near-duplicates (reported, not scored):** AutoDQ also drops columns that look like
genuine row identifiers (see §7, "ID-like column detection") and re-checks for
duplicates on what's left. This catches the case where every row is unique *only*
because of an auto-incrementing ID column, while the actual data is repeated. This
number is shown alongside Uniqueness for context but does **not** feed into the score
itself.

---

## 3. Validity

Validity is **not** one formula — it's an average of a per-column score, where each
column is checked differently depending on its detected type:

**Numeric columns**, per column:
```
column_score = 1 - (issue_count / non_missing_count)
```
where `issue_count` = infinite values + "sentinel-like" values. A value counts as
sentinel-like if it is *both*:
- outside the range `[Q1 - 6×IQR, Q3 + 6×IQR]` (a much wider fence than the outlier
  check in §7 — deliberately, so we don't double-flag ordinary outliers as validity
  problems), **and**
- repeated at least `max(2, 0.5% of rows)` times — a genuinely one-off extreme value
  is more likely a real outlier than a placeholder code; a *repeated* extreme value
  (like `-999` appearing 40 times) looks like a sentinel/placeholder.

**Datetime-like columns**, per column:
```
column_score = 1 - (unparsable_values / non_missing_count)
```
A value is "unparsable" if `pandas.to_datetime()` can't make sense of it.

**Categorical/text columns**, per column:
```
column_score = 1 - (null_like_token_count / non_missing_count)
```
`null_like_token_count` = values matching (case-insensitive, whitespace-trimmed):
`na`, `n/a`, `null`, `none`, `nan`, `-`, `--`, `?`, `unknown`, `missing`, `n.a.`, `nil`.
These are strings that *look* like missing data but weren't recognized as `NaN` by the
file parser.

**Constant-column penalty:** if a column has exactly one distinct value (and more than
one row), its score is capped at `0.8` regardless of the checks above — a column with
zero information content is a validity concern even if every value is technically
"parseable."

**Dataset-level Validity score** = the plain average of all per-column scores.

---

## 4. Consistency

Also a per-column average, checking structural/formatting issues only (no domain
rules):

**Categorical/boolean/text columns** — casing/whitespace collapse:
```
column_score = 1 - (collapsed_count / raw_distinct_count)
```
where `collapsed_count` = how many distinct values disappear once you lowercase and
trim whitespace. Example: a column with values `"Yes"`, `"yes"`, `"YES "`, `"No"` has
4 raw distinct values but only 2 after normalizing (`"yes"`, `"no"`) → `collapsed_count
= 2`, `raw_distinct_count = 4` → `score = 1 - 2/4 = 0.5`.

**Object-dtype columns that aren't categorical** — mixed-type detection: computes the
fraction of values that look numeric (parse as `float`) vs. don't. If that fraction is
strictly between 5% and 95% (i.e., genuinely mixed, not just a handful of stray values),
the column is penalized proportional to how mixed it is:
```
score = 1 - 2 × min(frac_numeric, 1 - frac_numeric)
```
This peaks in severity at a 50/50 mix and tapers toward 1.0 as the column becomes more
uniformly one type or the other.

**Dataset-level Consistency score** = plain average of all per-column scores.

---

## 5. Balance

Only computed if a target column is selected/detected. Two numbers are always shown
together — **only the first is used in the DQI formula.**

**1. Normalized Shannon entropy ratio (the scored component):**
```
H(observed) = -Σ p_i × ln(p_i)      for each class i, p_i = class_i_count / total
H(uniform)  = ln(n_classes)          the maximum possible entropy for that many classes
Balance     = H(observed) / H(uniform)
```

**Worked example — perfectly balanced binary target** (50/50 split):
`p_1 = p_2 = 0.5`. `H(observed) = -(0.5×ln(0.5) + 0.5×ln(0.5)) = ln(2) ≈ 0.693`.
`H(uniform) = ln(2) ≈ 0.693`. `Balance = 0.693/0.693 = 1.0` → perfectly balanced scores 1.0.

**Worked example — skewed binary target** (900 vs. 100, i.e. 90/10):
`p_1=0.9, p_2=0.1`. `H(observed) = -(0.9×ln(0.9) + 0.1×ln(0.1)) ≈ 0.325`.
`H(uniform) = ln(2) ≈ 0.693`. `Balance = 0.325/0.693 ≈ 0.469`.

**2. Minority:majority ratio** (shown alongside, human-readable, not scored):
```
ratio = majority_class_count / minority_class_count → displayed as "1:ratio"
```
For the 900/100 example above: `900/100 = 9` → displayed as **"1:9"**.

**Why both:** the entropy ratio generalizes cleanly to any number of classes (5 balanced
classes and 2 balanced classes both score ~1.0), but "1:9" is far more intuitive to say
out loud in a presentation than "entropy ratio 0.469" — so both are shown, with the
entropy ratio being the one that actually feeds the DQI formula.

---

## 6. Dataset Quality Index (DQI)

**Formula:**
```
DQI = wC×Completeness + wS×Consistency + wV×Validity + wU×Uniqueness + wB×Balance
```
where the five weights sum to 1. Default weights:

| Component | Weight |
|---|---|
| Completeness | 0.25 |
| Consistency | 0.20 |
| Validity | 0.20 |
| Uniqueness | 0.15 |
| Balance | 0.20 |

Domain presets (Healthcare/Finance/E-commerce) nudge these slightly — see `docs/dqi.md`
for the full table. **These are labeled in the UI as configurable heuristics, not a
scientifically derived formula** — this is a deliberate framing choice for academic
defensibility, not a limitation to hide.

**Weight redistribution when Balance is unavailable** (no target column): rather than
scoring Balance as 0 (unfairly tanking an unsupervised dataset's DQI) or 1 (silently
overstating quality), its weight is redistributed **proportionally** across the other
four:
```
effective_weight[k] = weight[k] + weight[balance] × (weight[k] / sum of the other 4 weights)
```
**Worked example** with default weights, Balance (0.20) unavailable: the other four sum
to 0.80. Completeness's new effective weight = `0.25 + 0.20 × (0.25/0.80) = 0.3125`.
Each of the four grows proportionally so they still sum to 1.0 total.

**Final conversion:**
```
DQI_0_100 = round(DQI × 100, 1)
```

**Interpretation bands** (explicitly labeled "configurable heuristic ranges," not a
universal benchmark):

| Range | Label |
|---|---|
| 0–40 | Poor |
| 40–60 | Low |
| 60–80 | Moderate |
| 80–100 | High |

---

## 7. Issue Detection — thresholds and severity logic

Every issue below is generated from the *same* dataset used to compute the five scores
above — nothing here is hardcoded or random.

### Missing values
One issue per column with any missing values. Severity by percentage:
`≥50% → critical`, `≥20% → high`, `≥5% → medium`, `else → low`.

### Duplicate rows
One issue if `duplicate_pct > 0`. Severity: `≥10% → high`, `≥2% → medium`, `else → low`.

### Class imbalance
```
ratio = majority_class_count / minority_class_count
```
Only flagged if `ratio ≥ 3`. Severity: `≥20 → critical`, `≥8 → high`, `else (3-8) → medium`.

### Outliers (IQR method, numeric columns only)
```
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 × IQR
upper_fence = Q3 + 1.5 × IQR
```
Any value outside `[lower_fence, upper_fence]` counts as an outlier. Skipped entirely if
the column has fewer than 10 non-missing values, or `IQR = 0` (no spread to measure
against). Severity by outlier percentage: `≥10% → high`, `≥3% → medium`, `else → low`.

### Redundant / correlated features
Pairwise Pearson correlation across all numeric columns (rows with any missing value in
either column dropped first, pair-wise). A pair is flagged only if it passes **both**:
```
|correlation| ≥ 0.90                              (the raw practical threshold)
AND
p-value survives Benjamini-Hochberg correction     (see §7a below)
```
Severity: `≥0.97 → high`, `else (0.90-0.97) → medium`.

### Potential leakage — five independent checks, any of which can flag a column:
1. **Name-based**: column name (lowercased) contains `target`, `label`, `outcome`,
   `class_`, `y_true`, or `prediction` → flagged `medium` severity, regardless of any
   numeric relationship.
2. **Numeric target correlation**: if the target itself is numeric, any numeric feature
   with `|correlation with target| ≥ 0.95` **and** surviving Benjamini-Hochberg
   correction across all numeric features tested against the target → flagged `high`
   severity.
3. **Categorical target "purity"**: if the target is categorical, for each categorical
   feature, compute the *weighted group purity* — group rows by the feature's value,
   find the most common target class within each group, weight by group size:
   ```
   purity = Σ over groups [ (size of group / total rows) × (fraction of that group belonging to its most common class) ]
   ```
   Flagged `high` severity if `purity ≥ 0.95` — meaning knowing the feature's value
   almost perfectly tells you the target. Skipped if the feature has more distinct
   values than 90% of rows (too fine-grained to be a meaningful grouping). (This check
   does not currently have a BH-corrected variant — see §7a below for why.)
4. **ID-like feature columns**: see §8 below — flagged `low` severity if a feature
   column is a genuine identifier (near-unique per row, integer-valued, id-like name).
5. **Multivariate combination (model-based probe)**: see §7b below — catches leakage
   that checks 1-4 are structurally unable to see, because each of them only ever
   examines one feature at a time.

All leakage findings are worded as "potential," never "confirmed."

### 7a. Multiple-comparisons correction (Benjamini-Hochberg)

Checks 2 above and the redundancy check both test many pairwise correlations at once
(every numeric-numeric pair for redundancy; every numeric feature against the target
for leakage). Testing `m` independent hypotheses at a fixed significance level means
some will appear "significant" by chance alone as `m` grows — a well-known statistical
issue (the multiple-comparisons problem), and one a fixed correlation threshold alone
does not account for.

**The correction, in plain terms:**
```
1. Compute a p-value for every correlation tested (scipy.stats.pearsonr).
2. Sort all m p-values ascending: p_(1) ≤ p_(2) ≤ ... ≤ p_(m).
3. Find the LARGEST rank k such that p_(k) ≤ (k/m) × alpha   (alpha = 0.05 by default).
4. Every comparison with rank ≤ k is declared "significant"; everything else is not,
   even if its raw p-value alone would look small.
```
A pair/feature must cross **both** the raw correlation threshold (0.90 for redundancy,
0.95 for leakage) **and** be declared significant by this procedure to be flagged.

**Concrete worked demonstration**: 40 independent random numeric columns (n=60 rows,
no genuine relationship between any of them) produce `C(40,2) = 780` pairwise
correlation tests. Checking each against a threshold of 0.35 with no correction flags
**6** pairs as "redundant" — pure chance correlation at that sample size, not a real
relationship. Applying Benjamini-Hochberg correction to the same 780 tests correctly
suppresses all 6 to **0**. On the project's existing test datasets, which have very
few numeric columns (so `m` is small and BH correction is barely restrictive), genuine
strong correlations continue to be detected exactly as before.

**Why this correction is applied to checks 2 (numeric correlation) and redundancy, but
not check 3 (categorical purity)**: correlation naturally comes with a p-value from
a standard statistical test (Pearson's test for `r = 0`). Group purity is a different
kind of statistic without an equivalent standard null-hypothesis test; applying an
equivalent correction there would require a permutation test (repeatedly shuffling the
target and recomputing purity to build a null distribution), which is more
computationally expensive and was scoped out of this implementation — noted here as an
open item, not silently skipped.

### 7b. Multivariate leakage probe (model-based)

Checks 1-4 above only ever look at ONE feature at a time. This structurally cannot
catch leakage that only emerges from a *combination* of features — for example, two
binary features `a` and `b` where the target is exactly `a XOR b`: each of `a` and `b`
individually correlates only weakly with the target (in one constructed test case,
`corr(a, target) ≈ -0.05` and `corr(b, target) ≈ -0.10` — both far below any
univariate threshold), yet together they determine it perfectly.

**How the probe works:**
```
1. Take all usable feature columns (excluding genuine ID-like columns, §8).
   Require at least 3 features -- "a combination" needs more than 2 to be meaningful
   in the sense this probe is checking, though the classic XOR case uses exactly 2
   plus at least one more column to make up the minimum.
2. Split the data once (70/30 train/test -- a single split, not full cross-validation,
   since this is a cheap diagnostic PROBE, not the officially reported baseline metric
   computed later in the pipeline).
3. Fit a single RandomForest (150 trees) on ALL the features together.
4. Measure held-out performance:
     classification -> macro F1,  flagged if score >= 0.97
     regression     -> R²,        flagged if score >= 0.95
5. If the score is suspiciously high AND no individual feature was already flagged by
   checks 1-4 above (a univariate finding already explains the high performance, so a
   second vaguer flag would be redundant noise), report the top 1-3 features by
   RandomForest feature importance as a "potential multivariate leakage" cluster.
```
**Concrete demonstration**: on the constructed XOR case above, the probe achieves
F1 = 1.0 and correctly flags both `a` and `b` together, despite neither individually
crossing any univariate leakage threshold — a real, different class of leakage now
caught that the original univariate-only design could not see by construction.

This remains a heuristic probe (one model fit, not an exhaustive search over every
possible feature subset) and says so explicitly in its own emitted explanation text.

### Distributional imbalance (feature columns, not the target)
For categorical feature columns (excluding the target), if the single most common value
makes up `≥95%` of non-missing rows → flagged `low` severity, worded as "potential
distributional imbalance" — explicitly not a fairness claim.

### Constant columns (zero information)
Any non-target column where every non-missing value is identical → flagged `high`
severity. A constant column carries zero information by mathematical definition — no
model can extract signal from something that never varies. Found via a real user
dataset where 3 of 5 feature columns turned out to be entirely constant (one value
across all 495 rows), which fully explained an otherwise-confusing "Low" model
readiness result. This was previously only reflected as an invisible cap on that
column's Validity sub-score (§3) — now it's also a first-class, visible Issue.

---

## 8. The "ID-like column" heuristic (and the bug it fixed)

Used in three places: near-duplicate detection (§2), the leakage detector's ID-like
check (§7.4), and the baseline model's feature selection (§9). A column is only
classified as a genuine identifier if **both**:

1. It has very high cardinality: `nunique(column) ≥ 0.98 × n_rows`, **and**
2. Either:
   - it's a categorical/text column (high-cardinality text is effectively free text or
     a generated string ID, never a meaningful grouping category), **or**
   - it's a numeric column that is **integer-valued** (not a continuous float) **and**
     its name contains an identifier-suggestive token (`id`, `uuid`, `guid`, `index`,
     `_no`, `number`, `key`).

**Why the second condition matters**: continuous numeric measurements (a price, a
sensor reading, a score) are *naturally* almost entirely unique across rows — that's
normal continuous data, not an identifier. An earlier version of this heuristic only
checked condition 1, which meant every continuous feature got misclassified as an ID
column — silently dropped from the baseline model's training data, and separately,
incorrectly flagged as "potential leakage" by the leakage detector purely for being
continuous. Fixing this (requiring the integer + name-based signal for numeric columns)
was necessary for the baseline model to actually use real features and for leakage
findings to mean something.

---

## 9. Baseline Model / Estimated Dataset Learnability

### Feature selection
All columns except the target, restricted to numeric/categorical/boolean types, minus
any genuine ID-like columns (§8).

### Task type
```
task_type = "regression" if (target is numeric AND target has more than 15 distinct values)
            else "classification"
```

### Preprocessing (inside an sklearn Pipeline, refit per fold — never touches test data)
- Numeric columns: median imputation → standard scaling
- Categorical/boolean columns: most-frequent imputation → one-hot encoding

### Cross-validation fold count
```
folds = min(5, n_rows // 10)
if classification: folds = min(folds, size of the smallest class)
if folds < 2: baseline is reported unavailable (not enough data to CV meaningfully)
```

### Models trained
- Classification: Logistic Regression, Random Forest (200 trees)
- Regression: Linear Regression, Random Forest Regressor (200 trees)

### Metrics (mean ± standard deviation across folds)
- Classification: Accuracy, Precision (macro-averaged), Recall (macro-averaged),
  F1 (macro-averaged) — macro averaging is used uniformly (not "binary" mode) so the
  same code path handles 2-class and multi-class targets identically.
- Regression: R², MAE, RMSE.

### The leakage-aware second run
If any columns were flagged as `potential_leakage` or `redundant_features` (§7), a
second run repeats the whole cross-validation with those columns excluded:
- Every `potential_leakage` column is dropped outright.
- For each `redundant_features` pair, only the *second* column is dropped (the first is
  kept), so the model doesn't lose all signal from a correlated pair — just the
  duplication.

**Readiness label** is computed from whichever run is more trustworthy — the cleaned
run if one exists, otherwise the full-feature run:
```
Classification: primary_metric = mean F1
  ≥0.85 → High, ≥0.65 → Moderate, else → Low
Regression: primary_metric = mean R²
  ≥0.70 → High, ≥0.40 → Moderate, else → Low
```

---

## 10. Recommendations

Not a formula — a direct 1:1 mapping from each detected Issue (§7) to a specific,
templated message that references the actual column name(s) and measured value from
that issue. Recommendations are sorted by severity (`critical → high → medium → low`)
so the highest-impact fixes surface first. There is no recommendation that doesn't
trace back to a specific detected issue — this was an explicit design constraint (see
`docs/architecture.md`).

---

## 11. Feature-Target Informativeness (experimental, separate from DQI)

**Not part of the DQI formula.** Every other section above measures data *hygiene* —
whether the data is well-formed. This section measures something different:
whether the data actually carries predictive signal about the target. A dataset of
pure random noise, with no missing values, no duplicates, consistent formatting, and
perfectly balanced classes, would score near-maximum on every metric in §1-6 while
being completely unlearnable — that gap is exactly what this diagnostic makes visible.
See `docs/limitations_and_future_validation.md` for the full reasoning on why this is
kept separate rather than folded into DQI as a sixth weighted component.

**Formula**, per feature: normalized mutual information with the target, computed via
`sklearn.feature_selection.mutual_info_classif` / `mutual_info_regression`.

**For classification targets**, each feature's score is normalized by the target's own
Shannon entropy (same entropy formula as §5, Balance):
```
normalized_score(feature) = MI(feature, target) / H(target)     clamped to [0, 1]
```
Interpretable as: "what fraction of the target's uncertainty does this feature alone
resolve?" The overall summary score is the maximum across all features.

**For regression targets**, mutual information has no fixed natural maximum (unlike
entropy-based classification), so instead of a false 0-1 "quality" score, each
feature's score is its **share of the total mutual information mass** across all
features:
```
normalized_score(feature) = MI(feature, target) / Σ MI(all features, target)
```
This is a relative ranking (which features matter most to each other), not an
absolute claim about how informative the dataset is overall — the UI and report
explicitly avoid presenting a single top-line regression informativeness score for
this reason.

**Worked demonstration**: given a feature perfectly determining a binary target
(`target = (feature > 0)`) alongside an unrelated pure-noise feature, the informative
feature correctly ranks first with a high normalized score, and the noise feature
ranks near zero. On the demo customer-churn dataset, the diagnostic independently
flags the injected leakage column `retention_flag` as carrying an estimated 100% of
the target's information — the same column the leakage detector (§7) flags via a
completely different method (correlation/purity, not mutual information), which is a
good coherence signal that two independent parts of the pipeline agree.

---

## 12. The Validation Harness (cross-dataset DQI-vs-performance study infrastructure)

`datasets/validation_harness.py` is infrastructure for testing the project's central
empirical claim — that DQI tracks measured model performance — beyond the datasets
built specifically to demonstrate it. Given any CSV and target column, it:

1. Applies controlled, parameterized degradation (missingness, row duplication,
   outliers) at several intensity levels from 0.0 (untouched) to 1.0 (maximally
   degraded by this script's definition).
2. Runs the full AutoDQ pipeline at each level, recording DQI and the baseline
   model's primary metric (F1 for classification, R² for regression).
3. Reports the Pearson correlation between DQI and baseline performance across all
   levels.

**Actual results from running it** (not simulated): on `A_clean.csv`, correlation =
**0.996**. On `demo_customer_churn.csv`, correlation = **0.683**. Both positive,
supporting the hypothesis — though both datasets were built by the project team, so
this is not yet the independent cross-dataset validation described as future work in
`docs/limitations_and_future_validation.md`.

**A bug the harness itself surfaced, worth knowing**: an early version of the
degradation function injected synthetic outliers into every numeric feature column
uniformly, including a near-binary leakage-flag column (`retention_flag`, valued only
0 or 1). A handful of injected extreme values were enough to badly distort that
column's Pearson correlation with the target — correlation is highly sensitive to a
few leverage points, especially in a column with almost no natural variance. This, in
turn, caused AutoDQ's own leakage detector to stop flagging a column it correctly
flagged before degradation, artificially inflating the measured "cleaned" baseline
performance at that degradation level. The fix: skip outlier injection for numeric
columns with 2 or fewer distinct values. This is documented rather than smoothed
over, since finding this kind of instrumentation bug is exactly what running the
harness for real (rather than just describing it) is for.

## 13. Auto-Fix Engine (corrected dataset generation)

Given the "Download Corrected Dataset" button on the Recommendations page, here's
exactly what gets changed, in the order it happens (order matters — each step can
create work for the next one):

1. **Validity cleanup**: null-like string tokens, repeated numeric sentinel values
   (same detection logic as §3), and unparsable dates all become genuine missing
   values (`NaN`/`NaT`), not left as disguised placeholders.
2. **Consistency cleanup**: for categorical columns with casing/whitespace variants,
   each value is remapped to the **most frequent variant after stripping whitespace**
   within its normalized (lowercased, trimmed) group — chosen this way specifically so
   the canonical form itself never carries stray leading/trailing whitespace (an
   earlier version picked the most frequent *raw* variant, which could still be
   `"South "` with a trailing space). Mixed numeric/text object columns matching the
   same 5-95% "mixed" threshold as §4 are coerced to numeric via `pd.to_numeric`,
   turning non-numeric entries into new missing values for step 3 to handle.
3. **Completeness fill**: every remaining missing value — including ones just created
   by steps 1-2 — is filled: **median** for numeric columns, **mode** (most frequent
   value) for categorical/boolean columns, **median date** for datetime columns
   (computed from `pd.to_datetime` on the parseable values). If a column has zero
   real values left after step 1 (every original value turned out to be a null-like
   placeholder), it is honestly reported as unfixable rather than silently left
   broken or filled with an invented value.
4. **Uniqueness fix**: exact duplicate rows are dropped (`drop_duplicates`), done
   *after* steps 1-3 so rows that only differed due to a since-corrected
   inconsistency are correctly recognized as duplicates too.
5. **Balance fix** (last, and the most involved): every class is brought up to the
   majority class's row count via **class-conditional resampling with light
   jitter** — a lightweight approximation to synthetic minority oversampling:
   ```
   For each class with count < majority_count:
     deficit = majority_count - count
     For each non-target column, independently:
       sample `deficit` values WITH REPLACEMENT from that column's own
       distribution WITHIN this class only
       IF the column is numeric:
         add Gaussian jitter ~ N(0, 0.05 * class_subset_std) to each sampled value
     Assemble these independently-sampled columns into `deficit` new rows,
     append to the dataset.
   ```
   **Why independent per-column sampling, not whole-row duplication**: copying an
   entire existing minority row `deficit` times would manufacture exact duplicate
   rows — actively fighting the Uniqueness fix in step 4, and inflating the dataset
   with literal copies rather than varied examples. Sampling each column
   independently from its class-conditional distribution preserves each column's
   marginal distribution within that class while introducing real per-row variation.

   **Why this is NOT full SMOTE, and why that's stated plainly**: true SMOTE
   interpolates between a minority sample and its actual nearest neighbors in
   feature space, which better preserves the *joint* relationships between columns
   (e.g., that a customer with high tenure also tends to have a specific contract
   type). Independent per-column sampling does not preserve those joint
   relationships as faithfully — it can produce combinations that wouldn't co-occur
   in a real record. True SMOTE (or SMOTE-NC for mixed numeric/categorical data, via
   a library like `imbalanced-learn`) would be a direct upgrade path, not
   implemented here because no such library was available in the offline
   environment this MVP was built in. This limitation is stated in the UI's
   fix log itself, not just in this document.

**What this deliberately does NOT touch**: columns flagged as `redundant_features`
or `potential_leakage`/`potential_multivariate_leakage` (§7) are left completely
alone by every step above — the auto-fix engine is scoped to exactly the five DQI
dimensions, matching the project's "analyzer/recommender, not an uncontrolled data
destruction tool" principle. Those findings remain visible on the Issues page for
manual review and decision, not silently resolved by deleting a column.

**Worked demonstration**: run against a deliberately low-quality 850-row synthetic
dataset (heavy missingness, ~47% duplicate rows, ~1:50 class imbalance, inconsistent
casing, sentinel values), the fix pipeline took its DQI from **58.9 ("Low")** to
**96.0 ("High")** — Completeness 0.51→0.90, Consistency 0.88→1.0, Uniqueness
0.53→1.0, Balance 0.13→1.0. Validity stayed capped below 1.0 specifically because one
column (`notes`) turned out to have zero real values after cleanup — correctly
reported as unfixable rather than masked.

## 14. Two real bugs found via actual user datasets, and fixed

**Target auto-detection missed binary numeric targets.** The heuristic that picks a
target column when none is specified only considered columns already typed
`categorical`/`boolean`. Many real binary targets (`passed`, `churned`, `survived`) are
stored as `0`/`1` integers, which schema inference correctly types as `numeric` — so
such a column was entirely invisible to auto-detect, which would instead pick an
unrelated low-cardinality categorical column if one existed, silently testing the wrong
question and producing a confusing "Low" readiness result that had nothing to do with
the intended target. Fixed in `analysis_service.py::detect_default_target` by also
considering numeric columns with exactly 2 distinct values as candidates.

**Constant columns were invisible outside per-column drill-down.** A real uploaded
dataset had 3 of its 5 feature columns entirely constant (one repeated value across
every row) — this was already penalized in the Validity score (§3's constant-column
cap), but never surfaced as a visible Issue, so the reason behind a low readiness score
wasn't discoverable without manually inspecting every column. Fixed by adding
`detect_constant_columns` (§7) as a first-class, high-severity issue type.

---

## Quick reference: where each number lives in the code

| Number | File |
|---|---|
| Completeness | `backend/app/analyzers/completeness.py` |
| Uniqueness | `backend/app/analyzers/uniqueness.py` |
| Validity | `backend/app/analyzers/validity.py` |
| Consistency | `backend/app/analyzers/consistency.py` |
| Balance | `backend/app/analyzers/balance.py` |
| DQI aggregation | `backend/app/metrics/dqi.py` |
| Issue detection (incl. multivariate leakage probe) | `backend/app/services/issue_detection.py` |
| Benjamini-Hochberg correction | `backend/app/utils/stats_utils.py` |
| ID-like column heuristic | `backend/app/utils/column_heuristics.py` |
| Shared ML preprocessing / task-type logic | `backend/app/utils/ml_preprocessing.py` |
| Baseline ML / cross-validation | `backend/app/services/baseline_ml.py` |
| Feature-target informativeness (experimental) | `backend/app/analyzers/informativeness.py` |
| Recommendations | `backend/app/recommendations/engine.py` |
| Auto-fix engine (corrected dataset generation) | `backend/app/services/auto_fix.py` |
| Pipeline orchestration | `backend/app/services/analysis_service.py` |
| Cross-dataset validation harness | `datasets/validation_harness.py` |
