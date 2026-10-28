"""Completeness: proportion of non-missing cells in the dataset.

C = 1 - (missing cells / total cells), clamped to [0, 1].
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CompletenessResult:
    score: float  # in [0, 1]
    total_cells: int
    missing_cells: int
    missing_pct: float
    per_column_missing_pct: dict[str, float]
    worst_columns: list[tuple[str, float]]  # top offenders, sorted desc
    explanation: str


def analyze_completeness(df: pd.DataFrame) -> CompletenessResult:
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return CompletenessResult(
            score=0.0,
            total_cells=0,
            missing_cells=0,
            missing_pct=0.0,
            per_column_missing_pct={},
            worst_columns=[],
            explanation="Completeness could not be computed: dataset is empty.",
        )

    missing_cells = int(df.isna().sum().sum())
    score = max(0.0, min(1.0, 1 - (missing_cells / total_cells)))

    per_col = (df.isna().mean() * 100).round(2)
    per_col_dict = per_col.to_dict()
    worst = sorted(per_col_dict.items(), key=lambda kv: kv[1], reverse=True)[:5]
    worst = [(c, p) for c, p in worst if p > 0]

    missing_pct = round(missing_cells / total_cells * 100, 2)
    if missing_cells == 0:
        explanation = "No missing values detected across any column."
    else:
        top = ", ".join(f"{c} ({p}%)" for c, p in worst[:3])
        explanation = (
            f"{missing_pct}% of all cells are missing ({missing_cells:,} of {total_cells:,}). "
            f"Most affected columns: {top}."
        )

    return CompletenessResult(
        score=round(score, 4),
        total_cells=total_cells,
        missing_cells=missing_cells,
        missing_pct=missing_pct,
        per_column_missing_pct=per_col_dict,
        worst_columns=worst,
        explanation=explanation,
    )
