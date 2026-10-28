"""Validity: whether values satisfy detectable, domain-agnostic constraints.

We deliberately do NOT encode domain-specific rules (e.g. "age must be
0-120") unless the user supplies them -- see spec section 6. What we CAN
detect without domain knowledge:

- numeric columns: infinities, and values that are extreme enough to look
  like sentinel/placeholder codes (e.g. -999, 9999) relative to the rest
  of the column's distribution.
- datetime columns (or datetime-like text): unparsable date strings.
- categorical columns: null-like string tokens that weren't recognized as
  NaN by the parser ("N/A", "null", "none", "-", "unknown", "?", etc.).
- any column: constant columns (zero information content) are flagged as a
  validity/utility concern.

Each column gets a per-column validity score; the dataset-level score is the
row-weighted (well, cell-weighted) average across evaluated columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

NULL_LIKE_TOKENS = {
    "na", "n/a", "null", "none", "nan", "-", "--", "?", "unknown", "missing", "n.a.", "nil",
}


@dataclass
class ColumnValidity:
    column: str
    score: float
    issue_count: int
    detail: str


@dataclass
class ValidityResult:
    score: float
    columns: list[ColumnValidity] = field(default_factory=list)
    explanation: str = ""


def _validate_numeric(series: pd.Series) -> ColumnValidity:
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return ColumnValidity(series.name, 1.0, 0, "No non-missing values to validate.")

    inf_count = int(np.isinf(non_null.to_numpy(dtype=float, na_value=0)).sum())

    # Sentinel-style outlier codes: values whose magnitude sits far beyond
    # the IQR fence AND repeat suspiciously often (a single extreme value is
    # more likely a genuine outlier than a placeholder code).
    q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
    iqr = q3 - q1
    sentinel_count = 0
    if iqr > 0:
        lower, upper = q1 - 6 * iqr, q3 + 6 * iqr
        extreme = non_null[(non_null < lower) | (non_null > upper)]
        value_counts = extreme.value_counts()
        repeated_sentinels = value_counts[value_counts >= max(2, int(0.005 * n))]
        sentinel_count = int(repeated_sentinels.sum())

    issue_count = inf_count + sentinel_count
    score = max(0.0, 1 - issue_count / n)
    details = []
    if inf_count:
        details.append(f"{inf_count} infinite value(s)")
    if sentinel_count:
        details.append(f"{sentinel_count} value(s) resembling placeholder/sentinel codes")
    detail = "; ".join(details) if details else "No validity issues detected."
    return ColumnValidity(series.name, round(score, 4), issue_count, detail)


def _validate_datetime_like(series: pd.Series) -> ColumnValidity:
    non_null = series.dropna()
    n = len(non_null)
    if n == 0:
        return ColumnValidity(series.name, 1.0, 0, "No non-missing values to validate.")
    parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
    unparsable = int(parsed.isna().sum())
    score = max(0.0, 1 - unparsable / n)
    detail = (
        f"{unparsable} value(s) could not be parsed as valid dates."
        if unparsable else "All values parsed as valid dates."
    )
    return ColumnValidity(series.name, round(score, 4), unparsable, detail)


def _validate_categorical(series: pd.Series) -> ColumnValidity:
    non_null = series.dropna().astype(str)
    n = len(non_null)
    if n == 0:
        return ColumnValidity(series.name, 1.0, 0, "No non-missing values to validate.")
    null_like = int(non_null.str.strip().str.lower().isin(NULL_LIKE_TOKENS).sum())
    score = max(0.0, 1 - null_like / n)
    detail = (
        f"{null_like} value(s) are null-like placeholders (e.g. 'N/A', 'unknown') "
        "not recognized as missing by the parser."
        if null_like else "No null-like placeholder tokens detected."
    )
    return ColumnValidity(series.name, round(score, 4), null_like, detail)


def analyze_validity(df: pd.DataFrame, column_types: dict[str, str]) -> ValidityResult:
    if df.shape[0] == 0 or df.shape[1] == 0:
        return ValidityResult(score=0.0, columns=[], explanation="Validity could not be computed: dataset is empty.")

    results: list[ColumnValidity] = []
    for col in df.columns:
        ctype = column_types.get(col, "unknown")
        series = df[col]
        if series.dropna().empty:
            results.append(ColumnValidity(col, 0.5, 0, "Column is entirely missing; validity is indeterminate."))
            continue
        if ctype == "numeric":
            results.append(_validate_numeric(series))
        elif ctype == "datetime":
            results.append(_validate_datetime_like(series))
        elif ctype in ("categorical", "boolean", "text"):
            results.append(_validate_categorical(series))
        else:
            results.append(ColumnValidity(col, 1.0, 0, "Column type unrecognized; skipped."))

        # Constant-column penalty: zero information content is a validity concern
        if series.dropna().nunique() == 1 and len(series.dropna()) > 1:
            cv = results[-1]
            results[-1] = ColumnValidity(
                col, min(cv.score, 0.8), cv.issue_count,
                cv.detail + " Column is constant (single unique value) across all rows."
            )

    overall = round(sum(r.score for r in results) / len(results), 4) if results else 0.0
    flagged = [r for r in results if r.score < 0.95]
    if not flagged:
        explanation = "All columns pass basic validity checks (no unparsable values, sentinels, or placeholder nulls detected)."
    else:
        names = ", ".join(r.column for r in flagged[:4])
        explanation = f"{len(flagged)} column(s) have detectable validity concerns, notably: {names}."

    return ValidityResult(score=overall, columns=results, explanation=explanation)
