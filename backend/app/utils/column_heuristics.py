"""Shared heuristic for detecting genuinely identifier-like columns.

IMPORTANT DISTINCTION this exists to fix: a column having high cardinality
(nunique close to the row count) does NOT by itself mean it's an ID column.
Continuous numeric measurements (e.g. a sensor reading, a price, a score)
are *expected* to be almost entirely unique across rows -- that's normal
continuous data, not an identifier.

A column is only genuinely "ID-like" if BOTH:
  1. it has very high cardinality (>= 98% unique), AND
  2. either its values are integer-like (not continuous floats) with a
     name suggestive of an identifier, OR it's a high-cardinality
     categorical/text column (a free-text or generated-string identifier
     column, which is never a meaningful category for grouping).

Continuous numeric columns (floats without an id-like name) are excluded
from this classification entirely, since applying the naive cardinality-only
rule to them would incorrectly strip real predictive features out of the
baseline model and incorrectly flag them as "potential leakage" purely for
being continuous.
"""
from __future__ import annotations

import pandas as pd

ID_LIKE_NAME_TOKENS = ("id", "uuid", "guid", "index", "_no", "number", "key")


def is_id_like_column(name: str, series: pd.Series, column_type: str, n_rows: int) -> bool:
    if n_rows == 0:
        return False

    high_cardinality = series.nunique(dropna=True) >= 0.98 * n_rows
    if not high_cardinality:
        return False

    if column_type in ("categorical", "boolean", "text"):
        # High-cardinality categorical/text is effectively a free-text or
        # generated-string identifier -- never a meaningful grouping category.
        return True

    if column_type == "numeric":
        non_null = series.dropna()
        if non_null.empty:
            return False
        is_integer_like = pd.api.types.is_integer_dtype(series) or (non_null % 1 == 0).all()
        name_suggests_id = any(tok in name.lower() for tok in ID_LIKE_NAME_TOKENS)
        return bool(is_integer_like and name_suggests_id)

    return False
