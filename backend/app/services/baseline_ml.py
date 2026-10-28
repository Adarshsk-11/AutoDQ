"""Baseline model / dataset-learnability evaluation.

Deliberately NOT framed as "DQI predicts model accuracy" -- we only report
the actual measured performance of a real baseline model trained on the
dataset, labeled "Baseline Model Performance" per spec section I / 12.

Two upgrades over the initial MVP cut:

1. Stratified k-fold cross-validation (classification) / k-fold (regression)
   instead of a single train/test split, reported as mean +/- std per
   metric. A single split is noisy on small/medium datasets; CV gives a
   more defensible number and is explicitly what the project spec asks for
   ("preferably cross-validation").

2. The baseline is run TWICE: once on all usable feature columns, and once
   excluding columns AutoDQ's own issue detector flagged as potential
   leakage or redundant. This is important for credibility -- if we let a
   leakage-flagged column stay in the model, the reported accuracy will
   look artificially perfect and silently contradict the tool's own
   findings. Showing both numbers side by side turns "we detected a
   leakage feature" into a demonstrated, not just claimed, effect on model
   reliability.

Uses sklearn Pipeline + ColumnTransformer inside cross_validate, so
preprocessing (imputation, scaling, encoding) is refit on each training
fold only -- this avoids the train/test leakage the project itself is
meant to help users detect.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from app.utils.column_heuristics import is_id_like_column
from app.utils.ml_preprocessing import build_preprocessor, determine_task_type

MIN_ROWS_FOR_BASELINE = 30
MAX_CV_FOLDS = 5
MIN_CV_FOLDS = 2


@dataclass
class ModelResult:
    model_name: str
    # metric_name -> {"mean": float, "std": float}, or {"error": str} on failure
    metrics: dict[str, dict[str, float] | str]


@dataclass
class FeatureSetResult:
    label: str
    feature_columns: list[str]
    n_features: int
    results: list[ModelResult] = field(default_factory=list)
    primary_metric_mean: float | None = None


@dataclass
class BaselineResult:
    available: bool
    task_type: str | None = None  # "classification" | "regression"
    target_column: str | None = None
    cv_folds: int = 0
    n_rows_used: int = 0
    full_features: FeatureSetResult | None = None
    cleaned_features: FeatureSetResult | None = None  # None if nothing was excluded
    excluded_columns: list[str] = field(default_factory=list)
    readiness: str | None = None  # High / Moderate / Low
    reason_unavailable: str = ""


def _readiness_from_metrics(task_type: str, primary_metric: float) -> str:
    if task_type == "classification":
        if primary_metric >= 0.85:
            return "High"
        if primary_metric >= 0.65:
            return "Moderate"
        return "Low"
    else:
        if primary_metric >= 0.7:
            return "High"
        if primary_metric >= 0.4:
            return "Moderate"
        return "Low"


def _determine_leakage_and_redundancy_exclusions(issues, target_column: str | None) -> list[str]:
    """Derive which feature columns to drop for the 'cleaned' baseline run,
    based on AutoDQ's own issue detection output.

    - potential_leakage / potential_multivariate_leakage: drop the flagged
      column(s) outright.
    - redundant_features: for each correlated pair, keep the first column
      and drop the second, so we remove the redundancy without losing all
      signal from that pair.
    """
    if not issues:
        return []

    excluded: set[str] = set()
    for issue in issues:
        if issue.issue_type in ("potential_leakage", "potential_multivariate_leakage"):
            for col in issue.columns:
                if col != target_column:
                    excluded.add(col)
        elif issue.issue_type == "redundant_features" and len(issue.columns) == 2:
            a, b = issue.columns
            if a not in excluded and b not in excluded:
                excluded.add(b)  # keep a, drop b
    return sorted(excluded)


def _determine_cv_folds(n_rows: int, min_class_count: int | None = None) -> int | None:
    folds = min(MAX_CV_FOLDS, n_rows // 10)
    if min_class_count is not None:
        folds = min(folds, min_class_count)
    if folds < MIN_CV_FOLDS:
        return None
    return folds


def _run_cv_for_feature_set(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str],
    column_types: dict[str, str],
    task_type: str,
    cv_folds: int,
    label: str,
) -> FeatureSetResult:
    preprocessor = build_preprocessor(feature_cols, column_types)
    results: list[ModelResult] = []
    primary_metric_mean: float | None = None

    if task_type == "classification":
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro",
        }
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        }
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        scoring = {
            "r2": "r2",
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
        }
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
        }

    for name, model in models.items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        try:
            scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=None, error_score="raise")
            metrics: dict[str, dict[str, float]] = {}
            for metric_name, scoring_key in scoring.items():
                raw = scores[f"test_{metric_name}"]
                # neg_* scorers are negated by sklearn convention; flip back for readability
                if scoring_key.startswith("neg_"):
                    raw = -raw
                metrics[metric_name] = {"mean": round(float(np.mean(raw)), 4), "std": round(float(np.std(raw)), 4)}
            results.append(ModelResult(model_name=name, metrics=metrics))

            primary_key = "f1" if task_type == "classification" else "r2"
            candidate = metrics[primary_key]["mean"]
            if primary_metric_mean is None or candidate > primary_metric_mean:
                primary_metric_mean = candidate
        except Exception as exc:
            results.append(ModelResult(model_name=name, metrics={"error": str(exc)}))

    return FeatureSetResult(
        label=label,
        feature_columns=feature_cols,
        n_features=len(feature_cols),
        results=results,
        primary_metric_mean=primary_metric_mean,
    )


def run_baseline(
    df: pd.DataFrame,
    target_column: str | None,
    column_types: dict[str, str],
    issues: list | None = None,
) -> BaselineResult:
    if not target_column or target_column not in df.columns:
        return BaselineResult(available=False, reason_unavailable="No target column selected.")

    working = df.dropna(subset=[target_column]).copy()
    if len(working) < MIN_ROWS_FOR_BASELINE:
        return BaselineResult(
            available=False,
            reason_unavailable=(
                f"Only {len(working)} rows with a non-missing target; at least "
                f"{MIN_ROWS_FOR_BASELINE} are needed for a meaningful cross-validated baseline."
            ),
        )

    all_feature_cols = [c for c in working.columns if c != target_column]
    all_feature_cols = [
        c for c in all_feature_cols
        if column_types.get(c) in ("numeric", "categorical", "boolean")
        and not is_id_like_column(c, working[c], column_types.get(c, ""), len(working))
    ]
    if not all_feature_cols:
        return BaselineResult(available=False, reason_unavailable="No usable feature columns after excluding ID-like/unsupported columns.")

    y = working[target_column]
    task_type = determine_task_type(y, column_types, target_column)

    min_class_count = None
    if task_type == "classification":
        class_counts = y.value_counts()
        if (class_counts < 2).any() or y.nunique() < 2:
            return BaselineResult(
                available=False,
                reason_unavailable="At least one class has fewer than 2 samples, or only one class exists -- cannot run stratified cross-validation.",
            )
        min_class_count = int(class_counts.min())

    cv_folds = _determine_cv_folds(len(working), min_class_count)
    if cv_folds is None:
        return BaselineResult(
            available=False,
            reason_unavailable=(
                "Not enough rows (or too few samples in the smallest class) to run at least "
                f"{MIN_CV_FOLDS}-fold cross-validation."
            ),
        )

    X_full = working[all_feature_cols]
    full_result = _run_cv_for_feature_set(
        X_full, y, all_feature_cols, column_types, task_type, cv_folds, label="All features"
    )

    excluded_columns = _determine_leakage_and_redundancy_exclusions(issues, target_column)
    excluded_columns = [c for c in excluded_columns if c in all_feature_cols]
    cleaned_result: FeatureSetResult | None = None
    cleaned_feature_cols = [c for c in all_feature_cols if c not in excluded_columns]

    if excluded_columns and cleaned_feature_cols:
        X_cleaned = working[cleaned_feature_cols]
        cleaned_result = _run_cv_for_feature_set(
            X_cleaned, y, cleaned_feature_cols, column_types, task_type, cv_folds,
            label="Excluding flagged leakage/redundant features",
        )

    # Readiness is based on the cleaned run when available (the more honest
    # number), falling back to the full-feature run otherwise.
    primary_source = cleaned_result if cleaned_result else full_result
    readiness = _readiness_from_metrics(task_type, primary_source.primary_metric_mean or 0.0)

    return BaselineResult(
        available=True,
        task_type=task_type,
        target_column=target_column,
        cv_folds=cv_folds,
        n_rows_used=len(working),
        full_features=full_result,
        cleaned_features=cleaned_result,
        excluded_columns=excluded_columns,
        readiness=readiness,
    )
