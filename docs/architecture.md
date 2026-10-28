# Architecture

## Pipeline

```
Dataset Ingestion  ->  Dataset Profiler  ->  Quality Metrics Engine  ->  Issue Detection Engine
    ->  DQI Engine  ->  Recommendation Engine  ->  Baseline ML Engine  ->  Report/Visualization API
```

Each stage is a separate, independently testable module under `backend/app/`. The
orchestrator (`services/analysis_service.py::run_full_analysis`) wires them together
and is the single function the API layer calls — meaning the entire engine can be
exercised and unit-tested without starting a web server (see `backend/tests/test_core_engine.py`).

## Why no database

The spec is explicit: don't introduce a database if the MVP doesn't need one. A single
in-memory dict (`app/utils/store.py`) keyed by a generated dataset id is sufficient for a
single-user demo session. Swapping in SQLite/Postgres later only touches `store.py` and
the two lines in `api/routes.py` that call it — the analysis engine itself has zero
knowledge of how datasets are persisted.

## Why this module boundary

- **Analyzers** (`completeness.py`, `consistency.py`, `validity.py`, `uniqueness.py`,
  `balance.py`) each take a DataFrame (+ schema/target where relevant) and return a
  dataclass with a `.score` in `[0, 1]` plus an `.explanation`. Adding a sixth quality
  dimension later means adding one more analyzer module and one line in `dqi.py`/
  `analysis_service.py` — nothing else changes.
- **Issue detectors** (`services/issue_detection.py`) are separate functions per issue
  type (`detect_missing_value_issues`, `detect_outliers`, `detect_redundancy`,
  `detect_leakage`, `detect_bias_signals`, `detect_imbalance_issue`,
  `detect_duplicate_issue`), all returning a common `Issue` dataclass
  (type/severity/columns/measured_value/explanation). New detectors just append to
  `detect_all_issues`.
- **Recommendations** are generated purely from the `Issue` list (`recommendations/engine.py`)
  — there is no path by which a recommendation can appear without a corresponding
  detected issue, satisfying the "no recommendations disconnected from actual analysis"
  requirement.
- **Baseline ML** is isolated in `services/baseline_ml.py` so it can be swapped for a
  more sophisticated model later without touching quality metrics or issue detection.
  It shares a common "is this column a genuine identifier?" heuristic
  (`utils/column_heuristics.py`) with `issue_detection.py` and `analyzers/uniqueness.py`
  rather than each reimplementing its own cardinality check — see "A bug this shared
  heuristic fixed" below for why that mattered.

### A bug this shared heuristic fixed

Each of these three modules originally had its own copy of "if a column is >=98% unique,
treat it as an ID column." That heuristic is wrong for continuous numeric features —
a price, a sensor reading, a score — which are *naturally* almost entirely unique across
rows. The practical effect: the baseline model was silently dropping real continuous
features from training (thinking they were IDs), and the leakage detector was flagging
every continuous feature as "potential leakage" purely for being continuous. Consolidating
the check into `column_heuristics.py::is_id_like_column` — which only classifies a column
as ID-like if it's high-cardinality *and* either integer-valued with an identifier-like
name, or a high-cardinality categorical/text column — fixed all three call sites at once
and made the fix impossible to regress independently in one of them later.

## Frontend structure

Single-page app with client-side tab navigation (`App.jsx` + `components/Nav.jsx`) rather
than a router — the whole demo is one uploaded dataset's analysis, so full client-side
routing would be over-engineering for a 2-3 day MVP. Pages are plain components that
receive the already-fetched `analysis` object as a prop; charts are small, focused
Recharts wrappers under `charts/`.

## API design

REST, matching the spec's suggested endpoints:

```
POST /api/datasets/upload
GET  /api/datasets                        (session history / recent datasets)
GET  /api/datasets/{id}
POST /api/datasets/{id}/analyze
GET  /api/datasets/{id}/profile
GET  /api/datasets/{id}/metrics
GET  /api/datasets/{id}/informativeness    (experimental, separate from /metrics' dqi field)
GET  /api/datasets/{id}/issues
GET  /api/datasets/{id}/recommendations
GET  /api/datasets/{id}/model-performance
GET  /api/datasets/{id}/report?format=json|html
POST /api/datasets/{id}/clean              (auto-fix -> new cleaned dataset + before/after DQI)
GET  /api/datasets/{id}/download           (CSV export -- works for original or cleaned datasets)
```

`/analyze` runs (and caches) the full pipeline; the four `GET` endpoints after it slice
the cached result rather than recomputing, so a dashboard with several panels doesn't
re-run the baseline model on every page load.

## Extensibility for non-tabular data (future)

The ingestion layer already isolates "get me a DataFrame + schema" from everything
downstream. Adding image/text support later means writing a new ingestion path that
also produces a DataFrame-like representation (e.g. one row per image/document with
extracted features) — the quality metrics, issue detection, and DQI layers would not
need to change, since they only assume a tabular structure with typed columns.

## Why HTML export instead of PDF

The report export (`services/report_export.py`, `GET /api/datasets/{id}/report`)
generates a self-contained HTML document rather than a PDF. This was a deliberate
choice, not an oversight: PDF generation would need a library like `reportlab` or
`weasyprint`, and this MVP was built in a sandbox with no network access to install or
test one. Shipping an untested PDF-generation code path risked a broken export at demo
time, which is worse than an honestly-scoped one. The HTML report already contains
everything (DQI breakdown, issues, recommendations, baseline results) and converts to a
PDF via any browser's Print → Save as PDF — swapping in a real PDF library later only
touches `report_export.py` and the `format=` branch in the report route.

## Why cross-validation and the leakage-aware baseline comparison live in `baseline_ml.py`

`run_baseline()` now does two things beyond the original single-split MVP: it uses
stratified k-fold CV instead of one split (so metrics are reported as mean ± std, which
is both more statistically defensible and explicitly requested by the project spec),
and it accepts the already-detected `Issue` list to compute which columns to exclude
for a second, "cleaned" run. This keeps the dependency direction one-way — issue
detection doesn't know or care about the baseline model, but the baseline model can
consume its output — so the baseline module stays swappable (e.g. for a more
sophisticated model later) without the issue-detection layer needing to change.

## Why `ml_preprocessing.py` and `stats_utils.py` exist as separate shared utilities

Two additions moved detection beyond the original MVP's univariate-only, uncorrected
approach: multiple-comparisons correction (Benjamini-Hochberg) and a model-based
multivariate leakage probe. Both live in `services/issue_detection.py`, but both also
need machinery that `services/baseline_ml.py` already had (a preprocessing pipeline
builder, task-type detection). Rather than duplicate that logic in
`issue_detection.py` — which risks the two modules silently drifting apart on what
counts as "numeric" vs. "categorical" preprocessing, or what counts as a
classification vs. regression task — it was factored out into
`utils/ml_preprocessing.py`, imported by both. `utils/stats_utils.py` holds a small,
dependency-free Benjamini-Hochberg implementation (no assumption that `statsmodels`
would be available), used by both the redundancy detector and the numeric-correlation
branch of the leakage detector. This follows the same pattern as
`column_heuristics.py` above: shared correctness-critical logic lives in one place,
not copy-pasted into each caller.

## Why the multivariate leakage probe is a single train/test split, not full cross-validation

This is a deliberate, documented exception to "prefer cross-validation" elsewhere in
this project. The probe in `detect_multivariate_leakage()` exists purely to answer a
yes/no diagnostic question ("does *some* combination of features predict the target
suspiciously well, even though no single feature does?") — it is not the reported
baseline metric (that's still the fully cross-validated `run_baseline()`, computed
separately, later in the pipeline). Running full CV here as well would roughly double
the pipeline's total training cost for a check that only needs to clear a coarse
suspicion threshold (F1 ≥ 0.97 / R² ≥ 0.95), not report a precise number.

## Why the informativeness diagnostic is not part of DQI

`analyzers/informativeness.py` computes normalized mutual information between each
feature and the target, and is fully wired into the pipeline (`analysis_service.py`),
the API (`GET /api/datasets/{id}/informativeness`, and inside `/metrics`), the
frontend (a dedicated Overview page panel, visually distinct from the DQI section),
and the exported report. It is deliberately **not** one of DQI's five weighted
components. Two reasons: first, conceptually, DQI measures whether data is
well-formed (hygiene) while this measures whether it carries predictive signal
(informativeness) — collapsing both into one number would make a low DQI ambiguous
between "your data is messy" and "your data has no signal," which call for different
fixes. Second, practically, folding in a sixth component this late would require
recalibrating all five existing default weights (and every already-written test,
doc example, and report screenshot built around the current five-component formula)
for a feature that is explicitly labeled experimental. See
`docs/limitations_and_future_validation.md`, Limitation 2, for the full reasoning.

## The validation harness (`datasets/validation_harness.py`)

Infrastructure for testing whether DQI actually tracks measured model performance —
the project's central empirical claim — beyond the datasets purpose-built to
demonstrate it. It applies parameterized degradation to any dataset at several
intensity levels and reports the correlation between DQI and cross-validated baseline
performance across those levels. It is intentionally kept as a standalone script
under `datasets/`, not part of the FastAPI application: it's a research/validation
tool for the project team, not a feature end users interact with. Running it against
the two datasets available in this project produced positive correlations (0.996 and
0.683) — see `docs/math_and_logic.md` §12 for the full results, including a real bug
the harness's own degradation logic surfaced and how it was fixed.
