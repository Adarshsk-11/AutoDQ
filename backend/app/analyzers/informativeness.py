"""Feature-target informativeness -- an EXPERIMENTAL diagnostic, deliberately
kept separate from the Dataset Quality Index.

DQI's five components (Completeness, Consistency, Validity, Uniqueness,
Balance) all measure data HYGIENE: whether the data is well-formed. None of
them measure whether the data actually carries predictive signal about the
target -- a dataset of pure random noise, with no missing values, no
duplicates, consistent formatting, and perfectly balanced classes, would
score near-maximum on the current DQI despite being entirely unlearnable.

This module measures that different property -- informativeness -- via
mutual information between each feature and the target. It is intentionally
NOT folded into the DQI formula: mixing a hygiene metric with an
informativeness metric would make DQI harder to interpret (a low DQI could
then mean either "messy data" or "no signal," which are different problems
requiring different fixes), and would also require recalibrating every
weight, test, and documented example built around the current five-component
formula. See docs/limitations_and_future_validation.md, Limitation 2, for the
full reasoning and for the case that a dedicated informativeness dimension
belongs in a future version rather than retrofitted into this one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import OrdinalEncoder

from app.utils.column_heuristics import is_id_like_column
from app.utils.ml_preprocessing import determine_task_type

MIN_ROWS_FOR_INFORMATIVENESS = 20


@dataclass
class FeatureInformativeness:
    column: str
    normalized_mi: float  # meaning depends on task_type -- see module docstring on InformativenessResult


@dataclass
class InformativenessResult:
    available: bool
    task_type: str | None = None
    overall_score: float | None = None
    top_features: list[FeatureInformativeness] = field(default_factory=list)
    explanation: str = ""
    reason_unavailable: str = ""
    note: str = (
        "Experimental diagnostic, deliberately separate from DQI. Measures how much "
        "predictive signal features carry about the target, which is a different "
        "property from data hygiene (completeness/consistency/validity/uniqueness/"
        "balance). Not included in the DQI score -- see docs/limitations_and_future_validation.md."
    )


def _entropy_of_target(y: pd.Series) -> float:
    counts = y.value_counts(normalize=True)
    return float(-sum(p * log(p) for p in counts if p > 0))


def analyze_informativeness(
    df: pd.DataFrame, column_types: dict[str, str], target_column: str | None
) -> InformativenessResult:
    if not target_column or target_column not in df.columns:
        return InformativenessResult(available=False, reason_unavailable="No target column selected.")

    working = df.dropna(subset=[target_column])
    if len(working) < MIN_ROWS_FOR_INFORMATIVENESS:
        return InformativenessResult(
            available=False,
            reason_unavailable=f"Only {len(working)} rows with a non-missing target; at least {MIN_ROWS_FOR_INFORMATIVENESS} are needed.",
        )

    feature_cols = [
        c for c in working.columns
        if c != target_column
        and column_types.get(c) in ("numeric", "categorical", "boolean")
        and not is_id_like_column(c, working[c], column_types.get(c, ""), len(working))
    ]
    if not feature_cols:
        return InformativenessResult(available=False, reason_unavailable="No usable feature columns after excluding ID-like/unsupported columns.")

    y = working[target_column]
    task_type = determine_task_type(y, column_types, target_column)

    if task_type == "classification" and y.nunique() < 2:
        return InformativenessResult(available=False, reason_unavailable="Target has only one class.")

    # Build a numeric matrix: numeric columns imputed with the median as-is;
    # categorical/boolean columns ordinal-encoded (missing values become their
    # own category) so sklearn's mutual_info_* functions can treat them as
    # discrete features via the discrete_features mask.
    X_columns = []
    discrete_mask = []
    for c in feature_cols:
        col = working[c]
        if column_types.get(c) == "numeric":
            filled = col.fillna(col.median())
            X_columns.append(filled.to_numpy().reshape(-1, 1))
            discrete_mask.append(False)
        else:
            filled = col.fillna("__missing__").astype(str)
            encoded = OrdinalEncoder().fit_transform(filled.to_numpy().reshape(-1, 1))
            X_columns.append(encoded)
            discrete_mask.append(True)
    X = np.hstack(X_columns)

    try:
        if task_type == "classification":
            y_encoded = OrdinalEncoder().fit_transform(y.astype(str).to_numpy().reshape(-1, 1)).ravel()
            mi = mutual_info_classif(X, y_encoded, discrete_features=discrete_mask, random_state=42)
        else:
            mi = mutual_info_regression(X, y.to_numpy(), discrete_features=discrete_mask, random_state=42)
    except Exception as exc:
        return InformativenessResult(available=False, reason_unavailable=f"Mutual information computation failed: {exc}")

    mi = np.clip(mi, 0, None)  # MI estimates can be slightly negative due to estimator noise

    if task_type == "classification":
        # Normalize each feature's MI by the target's own entropy -- gives a
        # score in roughly [0, 1], interpretable as "fraction of the target's
        # uncertainty this feature alone resolves."
        h_y = _entropy_of_target(y)
        normalized = (mi / h_y).clip(0, 1) if h_y > 0 else np.zeros_like(mi)
        overall_score = float(np.max(normalized)) if len(normalized) else None
    else:
        # Regression MI has no natural fixed maximum, so instead of a false
        # 0-1 "quality" score, we report each feature's SHARE of the total
        # mutual information mass across all features -- a relative ranking,
        # not an absolute quality claim. overall_score is intentionally left
        # unset for regression to avoid implying a false absolute scale.
        total = mi.sum()
        normalized = (mi / total) if total > 0 else np.zeros_like(mi)
        overall_score = None

    ranked = sorted(zip(feature_cols, normalized), key=lambda kv: kv[1], reverse=True)
    top_features = [FeatureInformativeness(column=c, normalized_mi=round(float(v), 4)) for c, v in ranked[:8]]

    if task_type == "classification":
        best = top_features[0] if top_features else None
        if best and best.normalized_mi >= 0.5:
            level = "strong"
        elif best and best.normalized_mi >= 0.2:
            level = "moderate"
        else:
            level = "weak"
        explanation = (
            f"The single most informative feature ('{best.column}') resolves an estimated "
            f"{round(best.normalized_mi * 100, 1)}% of the target's uncertainty ({level} signal)."
            if best else "No feature showed measurable mutual information with the target."
        )
    else:
        best = top_features[0] if top_features else None
        explanation = (
            f"'{best.column}' carries the largest share of the detectable feature-target signal "
            f"({round(best.normalized_mi * 100, 1)}% of total mutual information among features); "
            "this is a relative ranking, not an absolute 0-100 score, since regression mutual "
            "information has no fixed maximum."
            if best else "No feature showed measurable mutual information with the target."
        )

    return InformativenessResult(
        available=True,
        task_type=task_type,
        overall_score=round(overall_score, 4) if overall_score is not None else None,
        top_features=top_features,
        explanation=explanation,
    )
