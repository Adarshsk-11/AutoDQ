"""Unit tests for the AutoDQ core engine, run against the synthetic
datasets A-H (spec section 17). Verifies AutoDQ actually detects the
problems each dataset was designed to contain, and that nothing crashes
on the edge cases in spec section 15.

Run: pytest tests/test_core_engine.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.analyzers.balance import analyze_balance
from app.analyzers.completeness import analyze_completeness
from app.analyzers.consistency import analyze_consistency
from app.analyzers.uniqueness import analyze_uniqueness
from app.analyzers.validity import analyze_validity
from app.metrics.dqi import compute_dqi
from app.services.analysis_service import run_full_analysis
from app.services.ingestion import detect_schema
from app.services.issue_detection import detect_all_issues

DATASETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")


def load(name):
    return pd.read_csv(os.path.join(DATASETS_DIR, name))


def analyze(df, target="target"):
    schema = detect_schema(df)
    return run_full_analysis(df, schema, target_column=target)


# ---------------------------------------------------------------------
# Dataset A: clean, high quality
# ---------------------------------------------------------------------
def test_dataset_a_clean_scores_high():
    df = load("A_clean.csv")
    result = analyze(df)
    assert result.completeness.score == 1.0
    assert result.uniqueness.score == 1.0
    assert result.dqi.dqi_0_100 >= 75
    issue_types = {i.issue_type for i in result.issues}
    assert "missing_values" not in issue_types
    assert "duplicate_rows" not in issue_types


# ---------------------------------------------------------------------
# Dataset B: missing values
# ---------------------------------------------------------------------
def test_dataset_b_detects_missing_values():
    df = load("B_missing_values.csv")
    result = analyze(df)
    assert result.completeness.score < 0.95
    issue_types = {i.issue_type for i in result.issues}
    assert "missing_values" in issue_types
    recs = [r for r in result.recommendations if r.issue_type == "missing_values"]
    assert len(recs) > 0


# ---------------------------------------------------------------------
# Dataset C: class imbalance
# ---------------------------------------------------------------------
def test_dataset_c_detects_imbalance():
    df = load("C_class_imbalance.csv")
    result = analyze(df)
    assert result.balance.available
    assert result.balance.score < 0.8  # entropy ratio should reflect skew
    issue_types = {i.issue_type for i in result.issues}
    assert "class_imbalance" in issue_types


# ---------------------------------------------------------------------
# Dataset D: duplicates
# ---------------------------------------------------------------------
def test_dataset_d_detects_duplicates():
    df = load("D_duplicates.csv")
    result = analyze(df)
    assert result.uniqueness.score < 0.95
    issue_types = {i.issue_type for i in result.issues}
    assert "duplicate_rows" in issue_types


# ---------------------------------------------------------------------
# Dataset E: outliers
# ---------------------------------------------------------------------
def test_dataset_e_detects_outliers():
    df = load("E_outliers.csv")
    result = analyze(df)
    issue_types = {i.issue_type for i in result.issues}
    assert "outliers" in issue_types


# ---------------------------------------------------------------------
# Dataset F: redundant/correlated features
# ---------------------------------------------------------------------
def test_dataset_f_detects_redundancy():
    df = load("F_redundant_features.csv")
    result = analyze(df)
    issue_types = {i.issue_type for i in result.issues}
    assert "redundant_features" in issue_types
    redund = [i for i in result.issues if i.issue_type == "redundant_features"][0]
    assert redund.measured_value >= 0.9


# ---------------------------------------------------------------------
# Dataset G: potential leakage
# ---------------------------------------------------------------------
def test_dataset_g_detects_leakage():
    df = load("G_leakage.csv")
    result = analyze(df)
    issue_types = {i.issue_type for i in result.issues}
    assert "potential_leakage" in issue_types
    leak = [i for i in result.issues if i.issue_type == "potential_leakage"]
    assert any("target_echo" in i.columns for i in leak)


# ---------------------------------------------------------------------
# Dataset H: combination of problems -- DQI should be markedly lower than A
# ---------------------------------------------------------------------
def test_dataset_h_combined_scores_lower_than_clean():
    df_a = load("A_clean.csv")
    df_h = load("H_combined_problems.csv")
    result_a = analyze(df_a)
    result_h = analyze(df_h)
    assert result_h.dqi.dqi_0_100 < result_a.dqi.dqi_0_100
    issue_types = {i.issue_type for i in result_h.issues}
    # combined dataset should trip at least several distinct detectors
    assert len(issue_types & {
        "missing_values", "duplicate_rows", "class_imbalance",
        "outliers", "redundant_features", "potential_leakage"
    }) >= 4


# ---------------------------------------------------------------------
# Directional validity: degrading a dataset should not improve its DQI
# ---------------------------------------------------------------------
def test_directional_validity_missing_values_lower_dqi():
    df_a = load("A_clean.csv")
    result_a = analyze(df_a)
    df_b = load("B_missing_values.csv")
    result_b = analyze(df_b)
    assert result_b.dqi.dqi_0_100 < result_a.dqi.dqi_0_100


# ---------------------------------------------------------------------
# Component-level unit tests
# ---------------------------------------------------------------------
def test_completeness_formula():
    df = pd.DataFrame({"a": [1, 2, None, 4], "b": [1, None, None, 4]})
    result = analyze_completeness(df)
    # 3 missing out of 8 cells
    assert result.score == pytest.approx(1 - 3 / 8, abs=1e-6)


def test_uniqueness_formula():
    df = pd.DataFrame({"a": [1, 2, 2, 3], "b": ["x", "y", "y", "z"]})
    result = analyze_uniqueness(df)
    assert result.duplicate_rows == 1
    assert result.score == pytest.approx(1 - 1 / 4, abs=1e-6)


def test_balance_unavailable_without_target():
    df = pd.DataFrame({"a": [1, 2, 3]})
    result = analyze_balance(df, None)
    assert not result.available


def test_balance_perfectly_balanced_binary():
    df = pd.DataFrame({"y": ["a", "b"] * 50})
    result = analyze_balance(df, "y")
    assert result.available
    assert result.score == pytest.approx(1.0, abs=1e-6)
    assert result.minority_majority_ratio == "1:1.0"


def test_dqi_weights_sum_and_redistribution():
    dqi = compute_dqi(completeness=1.0, consistency=1.0, validity=1.0, uniqueness=1.0, balance=None)
    assert dqi.dqi_0_100 == 100.0
    balance_component = [c for c in dqi.components if c.name == "balance"][0]
    assert not balance_component.available
    assert balance_component.effective_weight == 0.0
    # remaining weights should have absorbed balance's default weight
    other_weights = sum(c.effective_weight for c in dqi.components if c.name != "balance")
    assert other_weights == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------
# Baseline ML: cross-validation + leakage-aware feature comparison
# ---------------------------------------------------------------------
def test_baseline_runs_cross_validation_with_mean_and_std():
    df = load("A_clean.csv")
    result = analyze(df)
    assert result.baseline.available
    assert result.baseline.cv_folds >= 2
    assert result.baseline.full_features is not None
    for model_result in result.baseline.full_features.results:
        if "error" in model_result.metrics:
            continue
        for metric_name, stats in model_result.metrics.items():
            assert "mean" in stats and "std" in stats


def test_baseline_excludes_leakage_feature_and_shows_both_runs():
    df = load("G_leakage.csv")
    result = analyze(df)
    assert result.baseline.available
    assert "target_echo" in result.baseline.excluded_columns
    assert result.baseline.cleaned_features is not None
    # the leaked feature should inflate full-feature performance above the cleaned run
    assert result.baseline.full_features.primary_metric_mean >= result.baseline.cleaned_features.primary_metric_mean


def test_baseline_excludes_one_of_redundant_pair():
    df = load("F_redundant_features.csv")
    result = analyze(df)
    assert result.baseline.available
    assert set(result.baseline.excluded_columns) & {"feature_1", "feature_1_copy"}
    # only one of the pair should be dropped, not both
    assert not {"feature_1", "feature_1_copy"}.issubset(set(result.baseline.excluded_columns))


def test_baseline_no_exclusions_on_clean_dataset():
    df = load("A_clean.csv")
    result = analyze(df)
    assert result.baseline.excluded_columns == []
    assert result.baseline.cleaned_features is None


# ---------------------------------------------------------------------
# Multiple-comparisons correction (Benjamini-Hochberg) on redundancy/leakage
# ---------------------------------------------------------------------
def test_bh_correction_suppresses_false_positives_in_wide_random_data():
    from app.services.issue_detection import detect_redundancy
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({f"col_{i}": rng.normal(0, 1, n) for i in range(40)})
    column_types = {c: "numeric" for c in df.columns}
    # Lenient threshold that pure noise crosses by chance at this sample size --
    # isolates whether BH correction, not the threshold itself, does the work.
    flagged = detect_redundancy(df, column_types, threshold=0.35, alpha=0.05)
    assert flagged == []


def test_genuine_redundancy_still_detected_with_bh_correction():
    # Regression check: BH correction must not suppress a REAL, strong
    # correlation just because many pairs were tested.
    df = load("F_redundant_features.csv")
    result = analyze(df)
    issue_types = {i.issue_type for i in result.issues}
    assert "redundant_features" in issue_types


# ---------------------------------------------------------------------
# Multivariate leakage probe (catches what univariate correlation cannot)
# ---------------------------------------------------------------------
def test_multivariate_leakage_catches_xor_combination():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    target = (a != b).astype(int)  # XOR: fully determined by (a,b) TOGETHER
    df = pd.DataFrame({
        "feat_a": a, "feat_b": b,
        "noise1": rng.normal(0, 1, n), "noise2": rng.normal(0, 1, n),
        "target": target,
    })
    # Individually, feat_a/feat_b should NOT trip univariate leakage --
    # confirming this really is a case univariate checks miss by construction.
    assert abs(df["feat_a"].corr(df["target"])) < 0.3
    assert abs(df["feat_b"].corr(df["target"])) < 0.3

    result = analyze(df)
    issue_types = {i.issue_type for i in result.issues}
    assert "potential_multivariate_leakage" in issue_types
    mv_issue = [i for i in result.issues if i.issue_type == "potential_multivariate_leakage"][0]
    assert "feat_a" in mv_issue.columns and "feat_b" in mv_issue.columns


def test_multivariate_probe_stays_quiet_when_univariate_already_flagged():
    # If a univariate leakage column already explains high performance, the
    # multivariate probe should not ALSO fire a redundant, vaguer flag.
    df = load("G_leakage.csv")
    result = analyze(df)
    issue_types = {i.issue_type for i in result.issues}
    assert "potential_leakage" in issue_types
    assert "potential_multivariate_leakage" not in issue_types


def test_multivariate_probe_quiet_on_clean_dataset():
    df = load("A_clean.csv")
    result = analyze(df)
    issue_types = {i.issue_type for i in result.issues}
    assert "potential_multivariate_leakage" not in issue_types


# ---------------------------------------------------------------------
# Informativeness diagnostic (separate from DQI by design)
# ---------------------------------------------------------------------
def test_informativeness_not_part_of_dqi_components():
    df = load("A_clean.csv")
    result = analyze(df)
    dqi_component_names = {c.name for c in result.dqi.components}
    assert "informativeness" not in dqi_component_names
    assert len(result.dqi.components) == 5  # still exactly the original five


def test_informativeness_ranks_real_signal_above_pure_noise():
    from app.services.ingestion import detect_schema
    from app.services.analysis_service import run_full_analysis
    rng = np.random.default_rng(1)
    n = 300
    informative = rng.normal(0, 1, n)
    target = (informative > 0).astype(int)  # perfectly determined by `informative`
    pure_noise = rng.normal(0, 1, n)  # unrelated to target

    df = pd.DataFrame({"informative_feature": informative, "noise_feature": pure_noise, "target": target})
    schema = detect_schema(df)
    result = run_full_analysis(df, schema, target_column="target")

    assert result.informativeness.available
    ranked_columns = [f.column for f in result.informativeness.top_features]
    assert ranked_columns.index("informative_feature") < ranked_columns.index("noise_feature")


def test_informativeness_unavailable_without_target():
    from app.services.ingestion import detect_schema
    from app.services.analysis_service import run_full_analysis
    df = pd.DataFrame({"a": range(50), "b": range(50)})
    schema = detect_schema(df)
    result = run_full_analysis(df, schema, target_column=None)
    assert not result.informativeness.available


# ---------------------------------------------------------------------
# Target auto-detection: must consider binary 0/1 numeric columns too,
# not just categorical-typed ones (a real bug found via a user's dataset:
# a 0/1-encoded target was invisible to auto-detect, which picked an
# unrelated low-cardinality categorical decoy column instead).
# ---------------------------------------------------------------------
def test_autodetect_finds_binary_numeric_target_over_unrelated_categorical():
    from app.services.analysis_service import detect_default_target
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "feature_1": rng.normal(size=n),
        "unrelated_category": rng.choice(["A", "B", "C"], size=n),  # 3 classes, no relation to anything
        "passed": rng.integers(0, 2, size=n),  # 0/1 numeric -- should be preferred (2 classes < 3)
    })
    schema = detect_schema(df)
    column_types = {s.name: s.dtype for s in schema}
    assert column_types["passed"] == "numeric"  # confirm it's really typed numeric, not categorical
    assert detect_default_target(df, column_types) == "passed"


# ---------------------------------------------------------------------
# Constant-column detection: a column with a single value carries zero
# information and should be a first-class, visible Issue -- found via a
# real user dataset where 3 of 5 columns were entirely constant, which
# fully explained an otherwise-confusing "Low" model readiness result.
# ---------------------------------------------------------------------
def test_constant_column_detected_as_issue():
    df = pd.DataFrame({
        "region": ["Karnataka"] * 50,
        "status": ["Shortlisted"] * 50,
        "score": range(50),
        "target": (["a"] * 25 + ["b"] * 25),
    })
    result = analyze(df, target="target")
    constant_issues = [i for i in result.issues if i.issue_type == "constant_column"]
    flagged_columns = {i.columns[0] for i in constant_issues}
    assert flagged_columns == {"region", "status"}
    assert "score" not in flagged_columns and "target" not in flagged_columns


# ---------------------------------------------------------------------
# Auto-fix engine: algorithmic correction of the five DQI dimensions
# ---------------------------------------------------------------------
def test_auto_fix_improves_dqi_on_low_quality_dataset():
    from app.services.auto_fix import apply_fixes
    df = pd.DataFrame({
        "age": [25, 30, None, 45, 50, 29, 31, 60, 22, 40] * 20,
        "status": ["Active", "active ", "ACTIVE", "inactive", "Inactive"] * 40,
        "target": (["yes"] * 15 + ["no"] * 185),
    })
    schema = detect_schema(df)
    column_types = {s.name: s.dtype for s in schema}
    before = run_full_analysis(df, schema, target_column="target")

    result = apply_fixes(df, column_types, target_column="target")
    cleaned_schema = detect_schema(result.cleaned_df)
    after = run_full_analysis(result.cleaned_df, cleaned_schema, target_column="target")

    assert after.dqi.dqi_0_100 > before.dqi.dqi_0_100
    assert after.completeness.score >= before.completeness.score
    assert after.consistency.score >= before.consistency.score
    assert after.balance.score > before.balance.score


def test_auto_fix_is_noop_on_already_clean_dataset():
    from app.services.auto_fix import apply_fixes
    df = load("A_clean.csv")
    schema = detect_schema(df)
    column_types = {s.name: s.dtype for s in schema}
    result = apply_fixes(df, column_types, target_column="target")
    assert len(result.fix_log) == 1
    assert "already clean" in result.fix_log[0].lower()
    assert len(result.cleaned_df) == len(df)


def test_auto_fix_removes_duplicate_rows():
    from app.services.auto_fix import apply_fixes
    df = load("D_duplicates.csv")
    schema = detect_schema(df)
    column_types = {s.name: s.dtype for s in schema}
    original_dupes = df.duplicated().sum()
    assert original_dupes > 0
    result = apply_fixes(df, column_types, target_column="target")
    assert result.cleaned_df.duplicated().sum() == 0


def test_auto_fix_balances_target_to_majority_count():
    from app.services.auto_fix import apply_fixes
    df = load("C_class_imbalance.csv")
    schema = detect_schema(df)
    column_types = {s.name: s.dtype for s in schema}
    result = apply_fixes(df, column_types, target_column="target")
    final_counts = result.cleaned_df["target"].value_counts()
    # every class should now be equal to the (pre-fix) majority count
    assert final_counts.nunique() == 1


def test_auto_fix_does_not_touch_redundant_or_leakage_columns():
    # Auto-fix is scoped to exactly the five DQI dimensions -- it must not
    # silently drop a column flagged as redundant or leaky.
    from app.services.auto_fix import apply_fixes
    df = load("G_leakage.csv")
    schema = detect_schema(df)
    column_types = {s.name: s.dtype for s in schema}
    result = apply_fixes(df, column_types, target_column="target")
    assert "target_echo" in result.cleaned_df.columns


def test_auto_fix_honestly_reports_column_with_zero_real_data():
    from app.services.auto_fix import apply_fixes
    df = pd.DataFrame({
        "notes": ["N/A", "unknown", "-", None, "null"] * 20,
        "value": range(100),
        "target": (["a"] * 50 + ["b"] * 50),
    })
    schema = detect_schema(df)
    column_types = {s.name: s.dtype for s in schema}
    result = apply_fixes(df, column_types, target_column="target")
    assert result.cleaned_df["notes"].isna().all()
    assert any("could not be filled" in line.lower() or "could not" in line.lower() for line in result.fix_log)


def test_auto_fix_canonical_casing_has_no_stray_whitespace():
    from app.services.auto_fix import apply_fixes
    df = pd.DataFrame({
        "status": ["active", "Active", "ACTIVE ", " active", "inactive"] * 20,
        "value": range(100),
        "target": (["a"] * 50 + ["b"] * 50),
    })
    schema = detect_schema(df)
    column_types = {s.name: s.dtype for s in schema}
    result = apply_fixes(df, column_types, target_column="target")
    for v in result.cleaned_df["status"].unique():
        assert v == v.strip(), f"value {v!r} still has stray whitespace"


# ---------------------------------------------------------------------
# Error-handling / edge cases (spec section 15) -- must never crash
# ---------------------------------------------------------------------
def test_empty_dataframe_does_not_crash():
    df = pd.DataFrame()
    schema = detect_schema(df)
    result = run_full_analysis(df, schema)
    assert result.dqi.dqi_0_100 == 0.0


def test_no_target_column_does_not_crash():
    # No categorical candidate exists for auto-detection here (numeric-only
    # frame), so balance analysis should gracefully report unavailability.
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    schema = detect_schema(df)
    result = run_full_analysis(df, schema, target_column=None)
    assert not result.balance.available
    assert "no target column" in result.balance.explanation.lower()


def test_all_missing_column_does_not_crash():
    df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
    schema = detect_schema(df)
    result = run_full_analysis(df, schema)
    assert result.dqi.dqi_0_100 >= 0


def test_single_class_target_does_not_crash():
    df = pd.DataFrame({"a": range(50), "y": ["only_class"] * 50})
    schema = detect_schema(df)
    result = run_full_analysis(df, schema, target_column="y")
    assert not result.balance.available
    assert not result.baseline.available


def test_constant_column_does_not_crash():
    df = pd.DataFrame({"a": [5] * 20, "b": range(20)})
    schema = detect_schema(df)
    result = run_full_analysis(df, schema)
    assert result.dqi.dqi_0_100 >= 0


def test_tiny_dataset_does_not_crash():
    df = pd.DataFrame({"a": [1, 2], "y": ["x", "y"]})
    schema = detect_schema(df)
    result = run_full_analysis(df, schema, target_column="y")
    assert result.dqi.dqi_0_100 >= 0
    assert not result.baseline.available  # too few rows for a baseline


def test_all_numeric_dataset_does_not_crash():
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 5)), columns=list("abcde"))
    schema = detect_schema(df)
    result = run_full_analysis(df, schema)
    assert result.dqi.dqi_0_100 >= 0


def test_all_categorical_dataset_does_not_crash():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "c1": rng.choice(list("XYZ"), 50),
        "c2": rng.choice(list("PQR"), 50),
    })
    schema = detect_schema(df)
    result = run_full_analysis(df, schema)
    assert result.dqi.dqi_0_100 >= 0


def test_datetime_column_does_not_crash():
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=30),
        "value": range(30),
    })
    schema = detect_schema(df)
    result = run_full_analysis(df, schema)
    assert result.dqi.dqi_0_100 >= 0


def test_correlation_undefined_for_constant_columns_does_not_crash():
    df = pd.DataFrame({"a": [1.0] * 30, "b": [2.0] * 30, "c": range(30)})
    schema = detect_schema(df)
    result = run_full_analysis(df, schema)
    assert result.dqi.dqi_0_100 >= 0


def test_json_malformed_structure_handled():
    from app.services.ingestion import load_dataset, IngestionError
    bad_json = b'{"just_a_string_value": "not a table"}'
    # Single flat object -> treated as one-row dataset, not a crash
    result = load_dataset(bad_json, "data.json")
    assert result.df.shape[0] == 1


def test_unsupported_file_type_raises_ingestion_error():
    from app.services.ingestion import load_dataset, IngestionError
    with pytest.raises(IngestionError):
        load_dataset(b"some bytes", "malware.exe")


def test_empty_file_raises_ingestion_error():
    from app.services.ingestion import load_dataset, IngestionError
    with pytest.raises(IngestionError):
        load_dataset(b"", "empty.csv")
