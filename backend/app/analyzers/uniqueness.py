"""Uniqueness: proportion of non-duplicate records.

U = 1 - (duplicate rows / total rows), clamped to [0, 1].

We also report near-duplicates (identical after ignoring an obvious ID-like
column) as a secondary, non-scoring signal, since exact-duplicate detection
alone can miss records that differ only by an auto-incremented id.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.utils.column_heuristics import is_id_like_column


@dataclass
class UniquenessResult:
    score: float  # in [0, 1]
    total_rows: int
    duplicate_rows: int
    duplicate_pct: float
    near_duplicate_rows: int
    explanation: str


def analyze_uniqueness(df: pd.DataFrame, column_types: dict[str, str] | None = None) -> UniquenessResult:
    total_rows = df.shape[0]
    if total_rows == 0:
        return UniquenessResult(
            score=0.0,
            total_rows=0,
            duplicate_rows=0,
            duplicate_pct=0.0,
            near_duplicate_rows=0,
            explanation="Uniqueness could not be computed: dataset has no rows.",
        )

    duplicate_mask = df.duplicated(keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    score = max(0.0, min(1.0, 1 - (duplicate_rows / total_rows)))
    duplicate_pct = round(duplicate_rows / total_rows * 100, 2)

    # Near-duplicates: drop likely genuine ID-like columns, then re-check.
    # Continuous numeric feature columns are deliberately NOT treated as
    # ID-like here -- see app.utils.column_heuristics for why.
    column_types = column_types or {}
    id_like_cols = [
        c for c in df.columns
        if is_id_like_column(c, df[c], column_types.get(c, ""), total_rows)
    ]
    near_duplicate_rows = duplicate_rows
    if id_like_cols and len(id_like_cols) < df.shape[1]:
        reduced = df.drop(columns=id_like_cols)
        near_duplicate_rows = int(reduced.duplicated(keep="first").sum())

    if duplicate_rows == 0 and near_duplicate_rows == 0:
        explanation = "No duplicate rows detected."
    else:
        explanation = (
            f"{duplicate_rows:,} of {total_rows:,} rows ({duplicate_pct}%) are exact duplicates."
        )
        if near_duplicate_rows > duplicate_rows:
            explanation += (
                f" An additional {near_duplicate_rows - duplicate_rows:,} rows are duplicates "
                f"once likely ID-like columns ({', '.join(id_like_cols)}) are ignored."
            )

    return UniquenessResult(
        score=round(score, 4),
        total_rows=total_rows,
        duplicate_rows=duplicate_rows,
        duplicate_pct=duplicate_pct,
        near_duplicate_rows=near_duplicate_rows,
        explanation=explanation,
    )
