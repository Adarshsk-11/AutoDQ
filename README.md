# AutoDQ

**A Data-Centric Framework for Evaluating Dataset Reliability and Predicting Model Performance**

AutoDQ lets you upload a tabular dataset (CSV, XLSX, or JSON) and automatically answers:
what does it look like, how complete/consistent/valid/unique/balanced is it, what specific
issues does it have (missing values, duplicates, imbalance, outliers, redundant features,
potential leakage), what should you fix, and how well does a baseline ML model actually
perform on it — all without writing a line of Python.

This repo is the 2–3 day MVP implementation of the AutoDQ major project.

---

## Project overview

- **Backend**: FastAPI + pandas/NumPy/SciPy/scikit-learn. A modular analysis engine
  (ingestion → profiling → quality metrics → issue detection → DQI → recommendations →
  baseline ML) that is fully unit-tested independently of the web server.
- **Frontend**: React + Vite + Tailwind, talking to the backend over a small REST API.
- **No database.** Datasets and analysis results are held in an in-memory store for the
  lifetime of the backend process — sufficient for a single-user MVP demo, and the
  architecture doesn't preclude adding persistence later.

## Architecture

```
backend/app/
    main.py                 FastAPI app + CORS
    api/routes.py            All REST endpoints
    services/
        ingestion.py          CSV/XLSX/JSON loading, schema detection, validation
        profiler.py           Descriptive stats, distributions, histograms
        issue_detection.py    Outlier/Redundancy/Leakage/Bias detectors
        baseline_ml.py        sklearn Pipeline baseline (classification/regression)
        analysis_service.py   Orchestrates the full pipeline end-to-end
    analyzers/
        completeness.py  consistency.py  validity.py  uniqueness.py  balance.py
    metrics/dqi.py            Weighted DQI aggregation + explainability
    recommendations/engine.py Issue -> actionable recommendation mapping
    schemas/                  Pydantic request/response models
    utils/store.py            In-memory dataset store

frontend/src/
    App.jsx, pages/*.jsx      6-page flow: Upload, Overview, Issues, Profile, Model, Recommendations
    charts/*.jsx              Recharts-based DQI gauge, histograms, bar charts
    services/api.js           Thin fetch wrapper around the backend API

datasets/
    generate_datasets.py      Generates synthetic test datasets A-H + the demo dataset
    A_clean.csv ... H_combined_problems.csv
    demo_customer_churn.csv   Polished demo dataset with documented injected issues

backend/tests/test_core_engine.py   27 unit tests covering every analyzer/detector
```

See `docs/architecture.md`, `docs/dqi.md`, and `docs/testing.md` for details.

## Setup & running it

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # optional -- defaults work with no .env at all
python -m uvicorn app.main:app --reload
```

The API runs at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

Run the test suite:

```bash
pytest tests/ -v
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env             # optional -- defaults to http://localhost:8000
npm run dev
```

Open the printed local URL (default `http://localhost:5173`). The frontend expects the
backend at `http://localhost:8000` by default; override with a `.env` file setting
`VITE_API_URL=http://your-backend-host:port`.

### Generating the synthetic/demo datasets

Already generated and committed under `datasets/`, but you can regenerate them:

```bash
cd datasets
python3 generate_datasets.py
```

## Supported file types

CSV (`.csv`), Excel (`.xlsx`, `.xls`), JSON (`.json` — list-of-records, dict-of-columns,
or a single object are all handled). Max file size for this MVP: 50MB.

## Dataset Quality Index (DQI)

```
DQI = wC*Completeness + wS*Consistency + wV*Validity + wU*Uniqueness + wB*Balance
```

Full formulas and reasoning in `docs/dqi.md`. Short version:

| Metric | Formula |
|---|---|
| Completeness | `1 - (missing cells / total cells)` |
| Uniqueness | `1 - (duplicate rows / total rows)` |
| Balance | Normalized Shannon-entropy ratio of the target class distribution, plus a minority:majority ratio shown alongside it for readability |
| Validity | Average per-column score checking for unparsable dates, sentinel/placeholder values, null-like string tokens, constant columns |
| Consistency | Average per-column score checking for casing/whitespace category collapse and mixed-type columns |

**Default weights** (labeled explicitly in the UI as configurable heuristics, not a
universal standard): Completeness 0.25, Consistency 0.20, Validity 0.20, Uniqueness 0.15,
Balance 0.20. Domain presets (General/Healthcare/Finance/E-commerce) adjust these slightly.

If no target column is available, Balance's weight is redistributed proportionally across
the other four metrics rather than silently scored as 0 or 1.

**Interpretation bands** (0-40 Poor, 40-60 Low, 60-80 Moderate, 80-100 High) are explicitly
labeled in the UI as "configurable heuristic ranges," not scientifically validated
universal thresholds.

## Issue detection

| Issue | Method |
|---|---|
| Missing values | Per-column missing percentage, severity scaled by % |
| Duplicate rows | Exact row duplication + near-duplicate detection ignoring ID-like columns |
| Class imbalance | Minority:majority ratio on the target column |
| Outliers | IQR fence (1.5×IQR) per numeric column |
| Redundant features | Pairwise Pearson correlation ≥ 0.90 among numeric columns |
| Potential leakage | Near-perfect correlation/group-purity with target, ID-like features, suspicious column names |
| Distributional imbalance | A single categorical value dominating (≥95%) a feature column |

All leakage/bias findings are reported as "potential," never "confirmed" — see
`docs/dqi.md` for the reasoning.

## Baseline Model Performance

For a detected/selected target column, AutoDQ trains a real baseline model (Logistic
Regression + Random Forest for classification; Linear Regression + Random Forest for
regression) using **stratified k-fold cross-validation** (classification) / k-fold CV
(regression) rather than a single train/test split — every metric is reported as
**mean ± standard deviation across folds**, which is both more statistically defensible
and what the project spec calls for ("preferably cross-validation"). Preprocessing
(imputation/scaling/encoding) runs inside an sklearn `Pipeline` + `ColumnTransformer`
refit on each training fold only, preventing the train/test leakage this project is
meant to help users detect elsewhere.

**The baseline runs twice** when AutoDQ's own issue detector has flagged potential
leakage or redundant features: once on all usable columns, and once excluding the
flagged ones. This turns "we detected a leakage feature" into a demonstrated effect —
e.g. on the synthetic leakage test dataset, the with-leakage run scores a suspiciously
perfect F1 = 1.00, while the cleaned run drops to a realistic ~0.91, visibly proving the
leakage detector's value rather than just asserting it.

This is explicitly labeled "Baseline Model Performance" / "Estimated Dataset
Learnability" — AutoDQ does **not** claim the DQI score itself predicts model accuracy,
since no such learned relationship has been trained in this MVP.

## Exporting a report

From the top bar after analyzing a dataset: **Export Report** opens a self-contained,
printable HTML report (DQI breakdown, issues, recommendations, baseline results) —
use your browser's Print → Save as PDF to get a PDF version. **Export JSON** downloads
the full analysis payload for programmatic use. (We intentionally didn't add a
PDF-generation library for this MVP — see `docs/architecture.md` for why.)

## Dataset history

`GET /api/datasets` lists every dataset uploaded in the current backend session
(id, filename, shape, whether it's been analyzed, and its DQI if so). The Upload page
shows this as a "Recent datasets this session" panel so you can reopen a previous
upload without re-selecting the file.

## Beyond the core MVP: multiple-comparisons correction, multivariate leakage, and an informativeness diagnostic

Three additions go beyond the original MVP scope, aimed at closing real methodological
gaps rather than adding surface features:

- **Multiple-comparisons-aware thresholds.** The redundancy and numeric-leakage
  detectors now apply Benjamini-Hochberg false discovery rate correction across all
  pairwise correlations tested, not just a bare threshold. Demonstrated concretely:
  on 40 independent random numeric columns (780 pairwise tests), a naive threshold
  flags 6 pairs as "redundant" purely from chance correlation — BH correction
  suppresses all 6, while still catching genuine strong correlations on normal-sized
  datasets. See `docs/math_and_logic.md` §7a.
- **Multivariate leakage detection.** A model-based probe (`detect_multivariate_leakage`)
  catches leakage that only emerges from a *combination* of features — something the
  original univariate correlation/purity checks are structurally unable to see.
  Demonstrated on a constructed XOR case: two features individually correlate at
  -0.05 and -0.10 with the target (invisible to univariate checks) but together
  determine it perfectly; the probe catches this via a quick model fit. See
  `docs/math_and_logic.md` §7b.
- **Feature-target informativeness diagnostic.** A separate, experimental panel
  (mutual information between each feature and the target) makes visible a real
  conceptual gap in DQI: the five components all measure data hygiene, not whether
  the data carries predictive signal. Deliberately kept **out** of the DQI formula
  rather than merged in — see `docs/limitations_and_future_validation.md`,
  Limitation 2, and `docs/math_and_logic.md` §11.
- **A cross-dataset validation harness** (`datasets/validation_harness.py`) —
  infrastructure for testing whether DQI tracks measured performance beyond the
  datasets built to demonstrate it. Actual runs so far: correlation 0.996 on
  `A_clean.csv`, 0.683 on `demo_customer_churn.csv` — both positive, though both
  datasets were built by the project team rather than independently sourced. See
  `docs/math_and_logic.md` §12 for full results, including a real bug the harness
  itself surfaced and how it was fixed.

Full reasoning, worked examples, and honest discussion of what's still open (a true
cross-dataset study against independently-sourced real datasets) are in
`docs/limitations_and_future_validation.md`.

## Auto-Fix: generating a corrected dataset

Below the recommendations list, a **Download Corrected Dataset** button algorithmically
fixes exactly the five DQI dimensions — Completeness, Consistency, Validity, Uniqueness,
Balance — and produces a new, downloadable CSV plus a before/after DQI comparison shown
in the app. It deliberately does **not** touch columns flagged as redundant or leaked;
those stay visible in Issues for manual review rather than being silently dropped.

What it does, in order:
1. **Validity**: null-like string tokens (`"N/A"`, `"unknown"`, etc.), repeated numeric
   sentinel values, and unparsable dates are converted to genuine missing values.
2. **Consistency**: categorical casing/whitespace variants (`"Yes"` / `"yes "` / `"YES"`)
   are canonicalized to a single consistent form; mixed numeric/text columns are coerced
   to numeric where the column is mostly numeric-like.
3. **Completeness**: every remaining missing value (including ones just created by steps
   1–2) is filled — median for numeric, mode for categorical, median date for datetime.
4. **Uniqueness**: exact duplicate rows are dropped.
5. **Balance**: every class is brought up to the majority class's count via
   class-conditional resampling — each column's synthetic value is sampled from that
   class's own empirical distribution (not copied whole-row, which would manufacture
   exact duplicates), with light Gaussian jitter on numeric columns. **This is a
   lightweight approximation, not full SMOTE** (no nearest-neighbor library was available
   in this environment) — documented honestly as such in `docs/math_and_logic.md` §13.

If a column turns out to have zero real (non-placeholder) values left after validity
cleaning, it is honestly reported as unfixable in the log rather than silently left
broken or filled with a fabricated value — see the fix log's "could NOT be filled"
entries for an example of this in practice.

`POST /api/datasets/{id}/clean` runs this and returns a new `cleaned_dataset_id` (the
corrected dataset is saved as its own entry, never mutating the original), a full
before/after DQI comparison, and the complete fix log. `GET /api/datasets/{id}/download`
streams any dataset (original or cleaned) as a CSV file.

## Limitations (MVP scope)

Deferred, per the project plan, to keep the 2–3 day build achievable:
- Image / NLP dataset analysis
- Distributed / large-scale processing
- Deep learning baselines
- Sophisticated, research-grade fairness analysis (we distinguish class imbalance
  from "potential distributional imbalance," but do not make protected-attribute
  fairness claims unless such a column is explicitly provided)
- LLM-based recommendation generation (recommendations are rule-based, tied 1:1 to
  detected issues)
- Authentication, multi-user permissions, and a persistent database
- A DQI-predicts-accuracy learned model (would require calibration against many
  real datasets, out of scope for this MVP; see `docs/dqi.md`)

## Future work

- Persist datasets/results (SQLite or similar) instead of the in-memory store
- Calibrate DQI against a benchmark suite of real datasets + trained models to move
  from heuristic interpretation bands to empirically validated ones
- Add Isolation Forest as an alternative outlier detector for larger datasets
- Support image/text dataset profiling as a separate analyzer track
- Learned recommendation ranking based on which fixes historically improved F1/R² most

## Demo instructions

1. Start the backend and frontend as above.
2. Upload `datasets/demo_customer_churn.csv`.
3. Select target column `churn` (or leave on auto-detect — it will find it).
4. Click **Analyze Dataset**.
5. Walk through Overview → Issues → Dataset Profile → Model Readiness → Recommendations.
6. To show directional validity live, also upload `datasets/A_clean.csv` (target
   `target`) and compare its DQI/F1 against the churn dataset or `H_combined_problems.csv`.

The demo dataset's intentionally injected issues are documented in `docs/testing.md`.
