"""Dataset profiling: shape, dtypes, descriptive stats, distributions."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.services.ingestion import ColumnSchema


@dataclass
class NumericProfile:
    column: str
    count: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    skewness: float
    kurtosis: float
    histogram: dict  # {"bins": [...], "counts": [...]}


@dataclass
class CategoricalProfile:
    column: str
    count: int
    n_unique: int
    top_values: dict[str, int]  # value -> count, top 10


@dataclass
class DatasetProfile:
    n_rows: int
    n_columns: int
    memory_bytes: int
    column_types: dict[str, str]
    numeric_profiles: list[NumericProfile] = field(default_factory=list)
    categorical_profiles: list[CategoricalProfile] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_profile(df: pd.DataFrame, schema: list[ColumnSchema]) -> DatasetProfile:
    column_types = {s.name: s.dtype for s in schema}
    warnings: list[str] = []

    numeric_profiles = []
    categorical_profiles = []

    for col, ctype in column_types.items():
        series = df[col]
        non_null = series.dropna()

        if ctype == "numeric" and not non_null.empty:
            try:
                numeric_profiles.append(_profile_numeric(col, non_null))
            except Exception:
                warnings.append(f"Could not compute numeric statistics for '{col}'.")
        elif ctype in ("categorical", "boolean", "text") and not non_null.empty:
            categorical_profiles.append(_profile_categorical(col, non_null))
        elif non_null.empty:
            warnings.append(f"Column '{col}' is entirely missing; skipped in detailed profiling.")

    return DatasetProfile(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        memory_bytes=int(df.memory_usage(deep=True).sum()),
        column_types=column_types,
        numeric_profiles=numeric_profiles,
        categorical_profiles=categorical_profiles,
        warnings=warnings,
    )


def _profile_numeric(col: str, series: pd.Series) -> NumericProfile:
    values = series.astype(float)
    n = len(values)
    skew = float(values.skew()) if n >= 3 else 0.0
    kurt = float(values.kurtosis()) if n >= 4 else 0.0

    counts, bin_edges = np.histogram(values, bins=min(20, max(3, int(np.sqrt(n)))))
    return NumericProfile(
        column=col,
        count=n,
        mean=round(float(values.mean()), 4),
        std=round(float(values.std()) if n > 1 else 0.0, 4),
        min=round(float(values.min()), 4),
        q1=round(float(values.quantile(0.25)), 4),
        median=round(float(values.median()), 4),
        q3=round(float(values.quantile(0.75)), 4),
        max=round(float(values.max()), 4),
        skewness=round(skew, 4) if np.isfinite(skew) else 0.0,
        kurtosis=round(kurt, 4) if np.isfinite(kurt) else 0.0,
        histogram={"bins": [round(float(b), 4) for b in bin_edges], "counts": [int(c) for c in counts]},
    )


def _profile_categorical(col: str, series: pd.Series) -> CategoricalProfile:
    value_counts = series.astype(str).value_counts().head(10)
    return CategoricalProfile(
        column=col,
        count=len(series),
        n_unique=int(series.nunique()),
        top_values={str(k): int(v) for k, v in value_counts.items()},
    )
