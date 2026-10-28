# Limitations, Threats to Validity, and Future Work

*(Suggested for insertion as a new chapter — e.g. "Limitations and Threats to
Validity" — immediately before the Conclusion chapter of the project report.
Written to match the tone and structure of the existing report.)*

---

## Limitations and Threats to Validity

While the implemented system successfully demonstrates automated dataset profiling,
multi-dimensional quality scoring, issue detection, and a leakage-aware baseline
model comparison, several limitations must be acknowledged for the current
implementation to be evaluated accurately and without overstatement.

### 1. The central predictive claim has not yet been empirically validated across independent datasets

The project's core hypothesis — that the Dataset Quality Index (DQI) reflects a
dataset's reliability and correlates with downstream model performance — has so far
been demonstrated only on synthetic datasets constructed specifically for this
purpose (Datasets A–H, described in the testing documentation). These datasets were
deliberately degraded by the project team along dimensions the team itself defined
(missing values, duplication, imbalance, redundancy, leakage), and evaluated using
the same scoring rules the team implemented.

This constitutes a demonstration of internal consistency — the system behaves as
designed — but not yet independent empirical validation. A rigorous validation would
require:

1. Selecting a benchmark suite of real-world, independently-sourced datasets (e.g.
   from UCI Machine Learning Repository or Kaggle) not constructed by the project
   team.
2. Applying controlled, parameterized degradations to each (e.g. systematically
   varying missingness from 0–50% in defined increments, imbalance ratios from 1:1
   to 1:50, redundancy correlation from 0 to 0.99).
3. Measuring both the DQI and baseline model performance (accuracy/F1/R²) at each
   degradation level, across all datasets.
4. Testing whether DQI and model performance move together consistently *across*
   datasets of different domains and structures, not only within the datasets the
   scoring rules were designed against.

Without this cross-dataset validation, the directional relationship demonstrated in
this project (worse data quality measurably lowering both DQI and baseline model
performance on the same synthetic datasets) should be understood as a proof of
concept for the mechanism, not yet as generalized empirical evidence.

### 2. Data Quality Index reflects data hygiene, not data informativeness — **DIAGNOSTIC ADDED, NOT MERGED INTO DQI**

*Status: an experimental feature-target informativeness diagnostic has been
implemented (`backend/app/analyzers/informativeness.py`), computing normalized mutual
information between each feature and the target. It is deliberately kept separate
from the DQI score rather than folded in as a sixth weighted component — mixing a
hygiene metric with an informativeness metric would make a low DQI ambiguous between
"messy data" and "no signal," which require different fixes, and would require
recalibrating every weight and documented example built around the existing
five-component formula this late in the project.*

The diagnostic is shown in the UI and exported report as a clearly labeled,
separate panel. It has already surfaced one useful cross-validation result: on the
demo customer-churn dataset, it independently identifies the injected leakage-like
`retention_flag` column as carrying an estimated 100% of the target's information —
agreeing with the (differently-computed) leakage detector via mutual information
rather than correlation, which is a good coherence signal between two independent
parts of the pipeline.

The underlying conceptual gap remains as originally stated below: a dataset of pure
random noise, with no missing values, no duplicates, consistent formatting, and
perfectly balanced classes, still scores near-maximum DQI, because DQI's five
components measure hygiene, not signal. The informativeness diagnostic makes this
gap visible and measurable rather than closing it — whether and how it should
eventually become a full sixth DQI component (with its own weight, requiring
recalibration of all default weights) remains future work, not resolved here.

### 3. Fixed detection thresholds do not account for multiple-comparisons effects — **IMPLEMENTED**

*Status: resolved. The redundancy detector and the numeric-target branch of the leakage
detector now compute a p-value for every pairwise correlation tested (via
`scipy.stats.pearsonr`) and apply Benjamini-Hochberg false discovery rate correction
across all comparisons in that batch, in addition to the original raw threshold. A
column pair must cross both the raw threshold AND survive BH correction to be flagged.*

Concrete demonstration: on a synthetic dataset of 40 independent random numeric
columns (n=60 rows, no real relationships between any of them), testing all 780
pairwise correlations against a threshold of 0.35 flags 6 pairs as "redundant" under
the original uncorrected approach, purely from chance correlation at that sample size.
With Benjamini-Hochberg correction applied, all 6 are correctly suppressed. On the
existing test datasets (which have few numeric columns), genuine strong correlations
(e.g. in `F_redundant_features.csv`) continue to be detected correctly — the
correction only changes behavior when there are enough comparisons for chance
correlations to become likely.

See `backend/app/utils/stats_utils.py` (the BH implementation) and the updated
`detect_redundancy` / `detect_leakage` in `backend/app/services/issue_detection.py`.

### 4. Leakage detection is limited to univariate signals — **PARTIALLY ADDRESSED**

*Status: a model-based multivariate leakage probe has been added
(`detect_multivariate_leakage` in `backend/app/services/issue_detection.py`), which
catches a meaningfully different class of leakage than the univariate checks above:
features that are each only weakly correlated with the target individually, but that
jointly determine it. This does not solve the general multivariate leakage problem
(which would require exhaustive subset search or a more principled importance-based
method), but demonstrably extends coverage beyond individual-feature correlation.*

Concrete demonstration: constructing a target defined as `target = (a != b)` (an XOR
of two binary features `a` and `b`) produces individual correlations of `corr(a,
target) ≈ -0.05` and `corr(b, target) ≈ -0.10` — both far below any univariate
leakage threshold, so the existing univariate checks correctly find nothing. The new
probe trains a fast RandomForest on all features and observes it achieves a perfect
F1 = 1.0 on held-out data, then flags `a` and `b` together as a potential
multivariate leakage cluster — a class of leakage the univariate checks are
structurally unable to detect, now caught.

This remains a heuristic probe (a single quick model fit, not an exhaustive search
over feature subsets) and is explicitly labeled as such in its emitted explanation.
Exhaustive or more principled multivariate leakage detection remains future work.

---

## Future Work: Toward Empirical Validation

Building directly on the limitations above, items 3, 4, and part of item 2 (the
informativeness diagnostic) have since moved from proposed future work to implemented
and tested features — described in the updated Limitations sections above. The
remaining open items are:

1. **Cross-dataset validation study — infrastructure built, full study still pending
   real external data.** A validation harness (`datasets/validation_harness.py`) now
   exists: given any CSV and target column, it applies controlled, parameterized
   degradation (missingness, duplication, outliers) across several intensity levels
   and reports the correlation between DQI and cross-validated baseline model
   performance. Running it against the two datasets available in this project
   (`A_clean.csv`, a synthetic dataset built by the project team, and
   `demo_customer_churn.csv`, a semi-synthetic dataset also built by the project
   team) produced Pearson correlations of 0.996 and 0.683 respectively — both
   positive, supporting the core hypothesis, though on datasets the team itself
   constructed rather than independently-sourced ones. The harness is ready to run
   against real UCI/Kaggle datasets the moment they're available; that step —
   assembling 5–10 independently-sourced real datasets and running the harness
   against them — remains the one genuinely unfinished piece of empirical validation.

   Worth noting honestly: building and running this harness surfaced a real bug in
   the harness itself, not in AutoDQ. An early version injected synthetic outliers
   into every numeric feature column indiscriminately, including a near-binary
   leakage-flag column (`retention_flag`, valued 0 or 1) — a handful of extreme
   injected values were enough to badly distort that column's Pearson correlation
   with the target (correlation is highly sensitive to leverage points), which in
   turn caused AutoDQ's own leakage detector to stop flagging a column it had
   correctly flagged before degradation, artificially inflating the measured
   "cleaned" baseline performance at that degradation level. Restricting outlier
   injection to columns with more than two distinct values fixed this. This is
   flagged here deliberately, not smoothed over: it is a concrete example of the
   kind of validation-tooling bug that a real cross-dataset study needs to guard
   against, and finding it early is a better outcome than a validation study whose
   own instrumentation quietly produces misleading numbers.

2. **Empirical calibration of interpretation bands.** Still future work, and now
   properly gated on item 1 above: recalibrating the 0–40/40–60/60–80/80–100
   thresholds requires the real cross-dataset study, not just the two
   project-constructed datasets currently available.

3. **Extending informativeness from a diagnostic to a certified sixth DQI
   dimension.** The mutual-information-based informativeness panel now exists and
   is shown separately from DQI (see Limitation 2, updated above). Whether and how
   it should become a weighted sixth DQI component — which would require
   recalibrating all five existing default weights — remains an open design
   question, deliberately not decided in this MVP.

These extensions are deliberately scoped as future work rather than included in the
current implementation, consistent with the project's phased approach: the MVP
prioritizes a working, explainable, and honestly-scoped system over premature
sophistication, with the architecture (documented in the accompanying technical
documentation) designed to accommodate each of these extensions without requiring
a redesign of the existing pipeline.
