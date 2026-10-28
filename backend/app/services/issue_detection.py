"""Issue detection engine.

Detects: missing values, duplicate rows, class imbalance, outliers/noise,
redundant/correlated features, and potential data leakage. Each issue
carries a type, severity, affected column(s), a measured value, and a
human-readable explanation, per spec section F.

We are deliberately conservative in language: "potential leakage" not
"leakage confirmed"; "potential distributional imbalance" not proof of
societal bias.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from app.utils.column_heuristics import is_id_like_column
from app.utils.ml_preprocessing import build_preprocessor, determine_task_type, map_encoded_importances_to_columns
from app.utils.stats_utils import benjamini_hochberg

Severity = Literal["critical", "high", "medium", "low"]


@dataclass
class Issue:
    issue_type: str
    severity: Severity
    columns: list[str]
    measured_value: Any
    explanation: str


def detect_missing_value_issues(df: pd.DataFrame) -> list[Issue]:
    issues = []
    if df.shape[0] == 0:
        return issues
    pct = (df.isna().mean() * 100).round(2)
    for col, p in pct.items():
        if p == 0:
            continue
        if p >= 50:
            sev: Severity = "critical"
        elif p >= 20:
            sev = "high"
        elif p >= 5:
            sev = "medium"
        else:
            sev = "low"
        issues.append(Issue(
            issue_type="missing_values",
            severity=sev,
            columns=[col],
            measured_value=float(p),
            explanation=f"'{col}' contains {p}% missing values.",
        ))
    return issues


def detect_duplicate_issue(df: pd.DataFrame) -> list[Issue]:
    if df.shape[0] == 0:
        return []
    dup_count = int(df.duplicated(keep="first").sum())
    if dup_count == 0:
        return []
    pct = round(dup_count / df.shape[0] * 100, 2)
    sev: Severity = "high" if pct >= 10 else ("medium" if pct >= 2 else "low")
    return [Issue(
        issue_type="duplicate_rows",
        severity=sev,
        columns=[],
        measured_value=pct,
        explanation=f"{dup_count:,} duplicate rows detected ({pct}% of the dataset).",
    )]


def detect_imbalance_issue(class_counts: dict[str, int], target_column: str | None) -> list[Issue]:
    if not target_column or not class_counts or len(class_counts) < 2:
        return []
    counts = sorted(class_counts.values())
    minority, majority = counts[0], counts[-1]
    if majority == 0:
        return []
    ratio = majority / minority
    if ratio < 3:
        return []
    sev: Severity = "critical" if ratio >= 20 else ("high" if ratio >= 8 else "medium")
    return [Issue(
        issue_type="class_imbalance",
        severity=sev,
        columns=[target_column],
        measured_value=round(ratio, 2),
        explanation=(
            f"The target classes in '{target_column}' have roughly a 1:{round(ratio, 1)} imbalance "
            f"({minority:,} vs {majority:,} records)."
        ),
    )]


def detect_outliers(df: pd.DataFrame, column_types: dict[str, str]) -> list[Issue]:
    """IQR-based outlier detection on numeric columns. Simple, cheap, and
    robust; we avoid Isolation Forest here to keep the MVP fast and
    deterministic, noting it as a documented extension point."""
    issues = []
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    for col in numeric_cols:
        series = df[col].dropna()
        n = len(series)
        if n < 10:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        pct = round(len(outliers) / n * 100, 2)
        if pct == 0:
            continue
        sev: Severity = "high" if pct >= 10 else ("medium" if pct >= 3 else "low")
        issues.append(Issue(
            issue_type="outliers",
            severity=sev,
            columns=[col],
            measured_value=pct,
            explanation=(
                f"'{col}' contains {len(outliers):,} statistical outliers ({pct}%) outside the "
                f"IQR fence [{round(lower, 2)}, {round(upper, 2)}]. Note: statistical outliers are "
                "not automatically incorrect -- investigate before removing."
            ),
        ))
    return issues


def detect_redundancy(df: pd.DataFrame, column_types: dict[str, str], threshold: float = 0.9, alpha: float = 0.05) -> list[Issue]:
    """Pairwise Pearson correlation among numeric columns, flagged only if a
    pair BOTH crosses the raw correlation threshold AND survives
    Benjamini-Hochberg false discovery rate correction across all pairs
    tested. The raw threshold alone doesn't account for the number of
    comparisons being made: as the number of numeric columns grows, the
    number of pairs tested grows quadratically, and some will cross a fixed
    threshold by chance alone even with no real relationship. See
    docs/limitations_and_future_validation.md, Limitation 3."""
    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    if len(numeric_cols) < 2:
        return []
    numeric_df = df[numeric_cols].dropna(axis=0, how="any")
    if len(numeric_df) < 3:
        return []

    pairs = [(a, b) for i, a in enumerate(numeric_cols) for b in numeric_cols[i + 1:]]
    if not pairs:
        return []

    correlations: list[float] = []
    p_values: list[float] = []
    for col_a, col_b in pairs:
        try:
            r, p = pearsonr(numeric_df[col_a], numeric_df[col_b])
        except Exception:
            r, p = float("nan"), 1.0
        correlations.append(r)
        p_values.append(p if not pd.isna(p) else 1.0)

    significant = benjamini_hochberg(p_values, alpha=alpha)

    issues = []
    for (col_a, col_b), value, p, sig in zip(pairs, correlations, p_values, significant):
        if pd.isna(value) or abs(value) < threshold:
            continue
        if not sig:
            # Crosses the raw threshold but doesn't survive multiple-comparisons
            # correction across all pairs tested -- likely noise at this scale,
            # not flagged.
            continue
        sev: Severity = "high" if abs(value) >= 0.97 else "medium"
        issues.append(Issue(
            issue_type="redundant_features",
            severity=sev,
            columns=[col_a, col_b],
            measured_value=round(float(value), 4),
            explanation=(
                f"'{col_a}' and '{col_b}' have a correlation of {round(float(value), 2)} "
                f"(significant after Benjamini-Hochberg correction across {len(pairs)} pairwise tests, "
                f"p={p:.2e}). Consider removing one of them to reduce redundancy."
            ),
        ))
    return issues


def detect_leakage(
    df: pd.DataFrame, column_types: dict[str, str], target_column: str | None, threshold: float = 0.95
) -> list[Issue]:
    """Practical, conservative leakage heuristics:
    - numeric features near-perfectly correlated with a numeric target
    - categorical/numeric features whose groups near-perfectly predict a
      categorical target (very high "purity" per group)
    - ID-like columns included as features (near-unique per row)
    - suspicious feature names (contains 'target', 'label', 'outcome', etc.
      but isn't the target itself)

    We never claim leakage is "confirmed" -- only "potential", with the
    reason for the flag stated explicitly.
    """
    if not target_column or target_column not in df.columns:
        return []

    issues: list[Issue] = []
    target = df[target_column]
    feature_cols = [c for c in df.columns if c != target_column]

    suspicious_name_tokens = ("target", "label", "outcome", "class_", "y_true", "prediction")
    for col in feature_cols:
        lowered = col.lower()
        if any(tok in lowered for tok in suspicious_name_tokens):
            issues.append(Issue(
                issue_type="potential_leakage",
                severity="medium",
                columns=[col],
                measured_value=None,
                explanation=(
                    f"Feature '{col}' has a name suggestive of target-derived information. "
                    "Investigate whether it was available at prediction time."
                ),
            ))

    if column_types.get(target_column) == "numeric":
        numeric_features = [c for c in feature_cols if column_types.get(c) == "numeric"]
        pairs_tested = []
        p_values = []
        correlations = []
        for col in numeric_features:
            paired = df[[col, target_column]].dropna()
            if len(paired) < 5:
                continue
            try:
                r, p = pearsonr(paired[col], paired[target_column])
            except Exception:
                continue
            if pd.isna(r):
                continue
            pairs_tested.append(col)
            correlations.append(r)
            p_values.append(p if not pd.isna(p) else 1.0)

        # Same multiple-comparisons correction as detect_redundancy above --
        # testing every numeric feature against the target is itself a batch
        # of comparisons, so a raw threshold alone isn't enough at scale.
        significant = benjamini_hochberg(p_values, alpha=0.05) if p_values else []
        for col, corr, p, sig in zip(pairs_tested, correlations, p_values, significant):
            if abs(corr) >= threshold and sig:
                issues.append(Issue(
                    issue_type="potential_leakage",
                    severity="high",
                    columns=[col],
                    measured_value=round(float(corr), 4),
                    explanation=(
                        f"'{col}' has an unusually strong correlation ({round(float(corr), 3)}) with "
                        f"target '{target_column}' (Benjamini-Hochberg corrected p={p:.2e} across "
                        f"{len(pairs_tested)} features tested) and may contain target-derived information."
                    ),
                ))
    else:
        # Categorical target: check per-group "purity" for categorical features
        categorical_features = [c for c in feature_cols if column_types.get(c) in ("categorical", "boolean")]
        for col in categorical_features:
            paired = df[[col, target_column]].dropna()
            if len(paired) < 10 or paired[col].nunique() < 2:
                continue
            if paired[col].nunique() > 0.9 * len(paired):
                continue  # too high-cardinality to be a meaningful grouping
            purity = paired.groupby(col)[target_column].apply(
                lambda g: g.value_counts(normalize=True).iloc[0] if len(g) else 0
            )
            weighted_purity = (purity * paired.groupby(col).size() / len(paired)).sum()
            if weighted_purity >= threshold:
                issues.append(Issue(
                    issue_type="potential_leakage",
                    severity="high",
                    columns=[col],
                    measured_value=round(float(weighted_purity), 4),
                    explanation=(
                        f"'{col}' almost perfectly determines target '{target_column}' "
                        f"(weighted group purity {round(float(weighted_purity), 3)}) and may contain "
                        "target-related information."
                    ),
                ))

    # ID-like feature columns (genuine identifiers only -- a continuous
    # numeric feature with high cardinality is NOT, by itself, an ID column;
    # see app.utils.column_heuristics for the distinction).
    n_rows = len(df)
    for col in feature_cols:
        if n_rows > 20 and is_id_like_column(col, df[col], column_types.get(col, ""), n_rows):
            issues.append(Issue(
                issue_type="potential_leakage",
                severity="low",
                columns=[col],
                measured_value=None,
                explanation=(
                    f"'{col}' is almost entirely unique per row (ID-like) and provides no generalizable "
                    "signal; including it as a feature risks the model memorizing identifiers."
                ),
            ))

    return issues


def detect_multivariate_leakage(
    df: pd.DataFrame,
    column_types: dict[str, str],
    target_column: str | None,
    already_flagged_columns: set[str],
) -> list[Issue]:
    """Model-based probe for leakage that only emerges from a COMBINATION of
    features -- something the univariate correlation/purity checks in
    detect_leakage() cannot catch by construction, since each of those checks
    only ever looks at one feature at a time. See
    docs/limitations_and_future_validation.md, Limitation 4.

    Trains a single, fast RandomForest on a single held-out split (NOT the
    full cross-validated official baseline model computed later in the
    pipeline, which would be redundant work here -- this is a cheap
    diagnostic probe, not the reported metric). If it achieves suspiciously
    high held-out performance, AND no individual feature was already flagged
    by the univariate checks, we flag the top contributing features as a
    potential multivariate leakage cluster.

    If a univariate check already flagged a column, we skip this: that
    column's flag already explains the suspiciously good performance, and a
    second, vaguer "combination of features" flag would just be noise on top
    of an already-actionable finding.

    This is still a heuristic probe, not an exhaustive subset search over all
    feature combinations (which is computationally expensive) -- stated
    explicitly in the emitted explanation, not just in code comments.
    """
    if not target_column or target_column not in df.columns:
        return []

    working = df.dropna(subset=[target_column])
    if len(working) < 40:
        return []

    feature_cols = [
        c for c in working.columns
        if c != target_column
        and column_types.get(c) in ("numeric", "categorical", "boolean")
        and not is_id_like_column(c, working[c], column_types.get(c, ""), len(working))
    ]
    # Need at least 3 features for "a combination" to be a meaningful concept.
    if len(feature_cols) < 3:
        return []

    y = working[target_column]
    task_type = determine_task_type(y, column_types, target_column)

    stratify = None
    if task_type == "classification":
        counts = y.value_counts()
        if (counts < 2).any() or y.nunique() < 2:
            return []
        stratify = y

    X = working[feature_cols]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0, stratify=None
        )

    preprocessor = build_preprocessor(feature_cols, column_types)
    model = (
        RandomForestClassifier(n_estimators=150, random_state=0, n_jobs=-1)
        if task_type == "classification"
        else RandomForestRegressor(n_estimators=150, random_state=0, n_jobs=-1)
    )
    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    try:
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
    except Exception:
        return []

    if task_type == "classification":
        score = f1_score(y_test, preds, average="macro", zero_division=0)
        threshold_score = 0.97
        metric_name = "macro F1"
    else:
        score = r2_score(y_test, preds)
        threshold_score = 0.95
        metric_name = "R\u00b2"

    if score < threshold_score:
        return []  # not suspiciously high -- nothing to flag

    if already_flagged_columns:
        # A univariate finding already explains this; avoid a redundant,
        # vaguer second flag on top of an already-actionable one.
        return []

    fitted_preprocessor = pipe.named_steps["prep"]
    importances = pipe.named_steps["model"].feature_importances_
    per_column_importance = map_encoded_importances_to_columns(
        fitted_preprocessor, importances, feature_cols, column_types
    )
    top_features = sorted(per_column_importance.items(), key=lambda kv: kv[1], reverse=True)[:3]
    top_names = [c for c, imp in top_features if imp > 0]
    if not top_names:
        return []

    return [Issue(
        issue_type="potential_multivariate_leakage",
        severity="high",
        columns=top_names,
        measured_value=round(float(score), 4),
        explanation=(
            f"No single feature showed strong leakage individually, but a quick diagnostic model using "
            f"a combination of features (top contributors: {', '.join(top_names)}) predicts the target with "
            f"an unusually high {metric_name} of {round(float(score), 3)} on held-out data. This may indicate "
            "leakage arising from a COMBINATION of features rather than any one column. This is a heuristic "
            "probe, not an exhaustive check of every feature combination -- investigate these columns together."
        ),
    )]


def detect_constant_columns(df: pd.DataFrame, target_column: str | None) -> list[Issue]:
    """Flags columns with exactly one distinct non-missing value across every
    row. A constant column carries zero information by definition -- no
    algorithm can extract signal from something that never varies. This
    already reduces the Validity score for that column (see
    analyzers/validity.py's constant-column penalty), but that only shows up
    as a capped sub-score, invisible unless you drill into per-column detail.
    This makes it a first-class, visible Issue instead -- surfaced directly
    from a real dataset where 3 of 5 columns turned out to be constant, which
    fully explained an otherwise-confusing "Low" model readiness result."""
    issues = []
    if df.shape[0] < 2:
        return issues
    for col in df.columns:
        if col == target_column:
            continue  # a constant target is already caught by detect_imbalance_issue/baseline unavailability
        non_null = df[col].dropna()
        if len(non_null) < 2:
            continue
        if non_null.nunique() == 1:
            value = non_null.iloc[0]
            issues.append(Issue(
                issue_type="constant_column",
                severity="high",
                columns=[col],
                measured_value=str(value),
                explanation=(
                    f"'{col}' has the same value ('{value}') in every row and carries zero "
                    "information. It cannot contribute to any model and should not be relied on "
                    "as a feature -- consider dropping it or checking whether this reflects a data "
                    "export problem (e.g. a filter that was accidentally applied before export)."
                ),
            ))
    return issues


def detect_bias_signals(df: pd.DataFrame, column_types: dict[str, str], target_column: str | None) -> list[Issue]:
    """Distinguishes class imbalance (handled separately) from broader
    distributional skew in categorical feature columns, reported cautiously
    as 'potential distributional imbalance' rather than a fairness claim."""
    issues = []
    categorical_cols = [
        c for c, t in column_types.items()
        if t in ("categorical", "boolean") and c != target_column
    ]
    for col in categorical_cols:
        series = df[col].dropna()
        if len(series) < 20 or series.nunique() < 2:
            continue
        top_share = series.value_counts(normalize=True).iloc[0]
        if top_share >= 0.95:
            issues.append(Issue(
                issue_type="distributional_imbalance",
                severity="low",
                columns=[col],
                measured_value=round(float(top_share), 4),
                explanation=(
                    f"'{col}' is dominated by a single category ({round(top_share * 100, 1)}% of rows). "
                    "This is a potential distributional imbalance, not a confirmed fairness issue -- "
                    "review whether this reflects the real-world population or a sampling artifact."
                ),
            ))
    return issues


def detect_all_issues(
    df: pd.DataFrame,
    column_types: dict[str, str],
    target_column: str | None,
    class_counts: dict[str, int] | None = None,
) -> list[Issue]:
    issues: list[Issue] = []
    issues += detect_missing_value_issues(df)
    issues += detect_duplicate_issue(df)
    issues += detect_imbalance_issue(class_counts or {}, target_column)
    issues += detect_outliers(df, column_types)
    issues += detect_redundancy(df, column_types)

    leakage_issues = detect_leakage(df, column_types, target_column)
    issues += leakage_issues

    already_flagged_columns: set[str] = set()
    for issue in leakage_issues:
        already_flagged_columns.update(issue.columns)
    issues += detect_multivariate_leakage(df, column_types, target_column, already_flagged_columns)

    issues += detect_bias_signals(df, column_types, target_column)
    issues += detect_constant_columns(df, target_column)
    return issues
