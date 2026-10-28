"""Consistency: structural/statistical consistency, without domain rules.

Detectable, domain-agnostic consistency issues:
- mixed types within a single column (e.g. numbers stored as strings mixed
  with real strings in an "object" column)
- inconsistent categorical casing/formatting that likely represents the same
  category (e.g. "Male" / "male" / "MALE ")
- inconsistent free-text formats where a column looks numeric-like but has
  inconsistent separators/units embedded (best-effort heuristic)

As with validity, we do not invent domain rules -- only structural checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ColumnConsistency:
    column: str
    score: float
    issue_count: int
    detail: str


@dataclass
class ConsistencyResult:
    score: float
    columns: list[ColumnConsistency] = field(default_factory=list)
    explanation: str = ""


def _mixed_type_score(series: pd.Series) -> ColumnConsistency:
    non_null = series.dropna()
    n = len(non_null)
    if n == 0 or series.dtype != object:
        return ColumnConsistency(series.name, 1.0, 0, "Not applicable (non-object dtype).")

    type_tags = non_null.map(lambda v: type(v).__name__)
    # Within object columns, check how many values are "numeric-looking strings"
    # mixed with genuinely non-numeric strings -- a common real-world mess.
    def _is_numeric_string(v: object) -> bool:
        s = str(v).strip()
        if not s:
            return False
        try:
            float(s.replace(",", ""))
            return True
        except ValueError:
            return False

    numeric_like = non_null.map(_is_numeric_string)
    frac_numeric = numeric_like.mean()
    mixed = 0.05 < frac_numeric < 0.95
    issue_count = int(numeric_like.sum()) if mixed else 0
    score = 1.0 if not mixed else round(1 - min(frac_numeric, 1 - frac_numeric) * 2, 4)
    score = max(0.0, min(1.0, score))
    detail = (
        f"Column mixes numeric-looking values ({frac_numeric:.0%}) with non-numeric text; "
        "likely inconsistent data entry." if mixed else "No mixed-type inconsistency detected."
    )
    return ColumnConsistency(series.name, score, issue_count, detail)


def _categorical_casing_score(series: pd.Series) -> ColumnConsistency:
    non_null = series.dropna().astype(str)
    n = len(non_null)
    if n == 0:
        return ColumnConsistency(series.name, 1.0, 0, "No values to check.")

    raw_unique = non_null.nunique()
    normalized_unique = non_null.str.strip().str.lower().nunique()

    if raw_unique <= normalized_unique:
        return ColumnConsistency(series.name, 1.0, 0, "No casing/whitespace inconsistency detected.")

    collapsed = raw_unique - normalized_unique
    # Fraction of "extra" distinct values purely due to formatting
    score = max(0.0, 1 - collapsed / raw_unique)
    detail = (
        f"{collapsed} of {raw_unique} distinct values collapse to the same category once "
        "case/whitespace differences are normalized (e.g. 'Yes' vs 'yes ')."
    )
    return ColumnConsistency(series.name, round(score, 4), collapsed, detail)


def analyze_consistency(df: pd.DataFrame, column_types: dict[str, str]) -> ConsistencyResult:
    if df.shape[0] == 0 or df.shape[1] == 0:
        return ConsistencyResult(score=0.0, columns=[], explanation="Consistency could not be computed: dataset is empty.")

    results: list[ColumnConsistency] = []
    for col in df.columns:
        ctype = column_types.get(col, "unknown")
        series = df[col]
        if series.dropna().empty:
            results.append(ColumnConsistency(col, 1.0, 0, "Column is entirely missing; skipped."))
            continue

        if ctype in ("categorical", "boolean", "text"):
            results.append(_categorical_casing_score(series))
        elif series.dtype == object:
            results.append(_mixed_type_score(series))
        else:
            results.append(ColumnConsistency(col, 1.0, 0, "No structural consistency issues detected."))

    overall = round(sum(r.score for r in results) / len(results), 4) if results else 0.0
    flagged = [r for r in results if r.score < 0.95]
    if not flagged:
        explanation = "No structural consistency issues detected (casing, formatting, or mixed types)."
    else:
        names = ", ".join(r.column for r in flagged[:4])
        explanation = f"{len(flagged)} column(s) show formatting/type inconsistencies, notably: {names}."

    return ConsistencyResult(score=overall, columns=results, explanation=explanation)
