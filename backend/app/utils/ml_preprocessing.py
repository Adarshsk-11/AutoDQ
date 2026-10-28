"""Shared sklearn preprocessing/task-type logic used by both the official
baseline model (app.services.baseline_ml) and the multivariate leakage probe
(app.services.issue_detection). Factored out so both stay consistent and
neither silently drifts from the other.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(feature_cols: list[str], column_types: dict[str, str]) -> ColumnTransformer:
    numeric_cols = [c for c in feature_cols if column_types.get(c) == "numeric"]
    categorical_cols = [c for c in feature_cols if column_types.get(c) in ("categorical", "boolean")]

    transformers = []
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    return ColumnTransformer(transformers, remainder="drop")


def determine_task_type(y: pd.Series, column_types: dict[str, str], target_column: str) -> str:
    """Regression if the target is numeric with more than 15 distinct values,
    classification otherwise. Shared so the baseline model and the
    multivariate leakage probe never disagree on what kind of task this is."""
    target_type = column_types.get(target_column)
    return "regression" if target_type == "numeric" and y.nunique() > 15 else "classification"


def map_encoded_importances_to_columns(
    preprocessor: ColumnTransformer, importances, feature_cols: list[str], column_types: dict[str, str]
) -> dict[str, float]:
    """A fitted ColumnTransformer + OneHotEncoder expands categorical columns
    into many output columns (e.g. 'region_North', 'region_South'). This maps
    per-output-column feature importances back to a single importance per
    ORIGINAL column, by summing the importances of all encoded columns that
    came from it -- so the leakage probe can report 'region' as a candidate,
    not 'region_North' / 'region_South' / ... separately."""
    try:
        output_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: can't recover names (older sklearn) -- return empty mapping
        # rather than guessing, which would misattribute importance.
        return {}

    aggregated: dict[str, float] = {c: 0.0 for c in feature_cols}
    for name, importance in zip(output_names, importances):
        # names look like "num__feature_1" or "cat__region_North"
        matched = None
        for col in feature_cols:
            prefix_num = f"num__{col}"
            prefix_cat = f"cat__{col}_"
            if name == prefix_num or name.startswith(prefix_cat):
                matched = col
                break
        if matched:
            aggregated[matched] += float(importance)
    return aggregated
