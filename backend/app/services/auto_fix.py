"""Auto-fix engine: algorithmic corrections targeting the five DQI dimensions
(Completeness, Consistency, Validity, Uniqueness, Balance).

Deliberately scoped to exactly these five, matching the DQI formula itself --
NOT touching redundant/correlated features, outliers, or leakage-flagged
columns. Those remain analyzer/recommender findings only: AutoDQ's own
design principle (see the original project spec) is "analyzer/recommender,
not an uncontrolled data destruction tool," and silently dropping a column a
user might still need is a materially different, less reversible action than
filling in a missing cell. Those three issue types stay visible in the
Issues/Recommendations pages, unmodified by this pipeline.

Order of operations matters here and is deliberate:
  1. Validity fixes first (null-like tokens, numeric sentinels, unparsable
     dates all become genuine NaN/NaT) -- this makes the true extent of
     missingness visible before completeness fixes act on it.
  2. Consistency fixes (casing/whitespace canonicalization, mixed-type
     column coercion) -- also produces additional NaNs where coercion fails.
  3. Completeness fixes (median/mode/median-date imputation) LAST among the
     per-cell fixes, so it fills in everything steps 1-2 revealed, not just
     the originally-missing cells.
  4. Uniqueness fix (drop exact duplicate rows) -- after cell-level fixes,
     so rows that only differed by a since-corrected inconsistency are
     correctly recognized as duplicates too.
  5. Balance fix (class-conditional resampling) LAST of all -- operates on
     the already-cleaned data, and deliberately after deduplication so
     synthetic minority rows are compared against a clean baseline.

ID-like columns (see app.utils.column_heuristics) are left untouched by
every per-cell fix: filling in or normalizing an identifier doesn't make
sense and risks breaking whatever the identifier is meant to reference.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.utils.column_heuristics import is_id_like_column

NULL_LIKE_TOKENS = {
    "na", "n/a", "null", "none", "nan", "-", "--", "?", "unknown", "missing", "n.a.", "nil",
}


@dataclass
class FixResult:
    cleaned_df: pd.DataFrame
    fix_log: list[str] = field(default_factory=list)


def _fix_validity(df: pd.DataFrame, column_types: dict[str, str], id_like: set[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    log = []
    for col in df.columns:
        if col in id_like:
            continue
        ctype = column_types.get(col)
        series = df[col]

        if ctype in ("categorical", "boolean", "text"):
            non_null = series.dropna()
            if non_null.empty:
                continue
            null_like_mask = non_null.astype(str).str.strip().str.lower().isin(NULL_LIKE_TOKENS)
            n_null_like = int(null_like_mask.sum())
            if n_null_like > 0:
                idx_to_null = non_null[null_like_mask].index
                df.loc[idx_to_null, col] = np.nan
                log.append(f"'{col}': replaced {n_null_like} null-like placeholder value(s) (e.g. 'N/A', 'unknown') with missing.")

        elif ctype == "numeric":
            non_null = series.dropna()
            if len(non_null) < 2:
                continue
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower, upper = q1 - 6 * iqr, q3 + 6 * iqr
                extreme = non_null[(non_null < lower) | (non_null > upper)]
                value_counts = extreme.value_counts()
                repeated_sentinels = value_counts[value_counts >= max(2, int(0.005 * len(non_null)))]
                if len(repeated_sentinels) > 0:
                    sentinel_values = set(repeated_sentinels.index)
                    mask = df[col].isin(sentinel_values)
                    n_sentinel = int(mask.sum())
                    df.loc[mask, col] = np.nan
                    log.append(f"'{col}': replaced {n_sentinel} repeated sentinel/placeholder value(s) (e.g. {sorted(sentinel_values)[:3]}) with missing.")

        elif ctype == "datetime":
            non_null = series.dropna()
            if non_null.empty:
                continue
            parsed = pd.to_datetime(non_null, errors="coerce", format="mixed")
            unparsable_mask = parsed.isna()
            n_unparsable = int(unparsable_mask.sum())
            if n_unparsable > 0:
                idx_to_null = non_null[unparsable_mask].index
                df.loc[idx_to_null, col] = np.nan
                log.append(f"'{col}': replaced {n_unparsable} unparsable date value(s) with missing.")

    return df, log


def _fix_consistency(df: pd.DataFrame, column_types: dict[str, str], id_like: set[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    log = []
    for col in df.columns:
        if col in id_like:
            continue
        ctype = column_types.get(col)
        series = df[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        if ctype in ("categorical", "boolean", "text"):
            as_str = non_null.astype(str)
            stripped = as_str.str.strip()
            normalized = stripped.str.lower()
            raw_unique = as_str.nunique()
            normalized_unique = normalized.nunique()
            if raw_unique > normalized_unique:
                # Canonicalize using the STRIPPED values' mode -- using the
                # raw (unstripped) mode here would risk the canonical form
                # itself still carrying stray leading/trailing whitespace.
                canonical_map = (
                    stripped.groupby(normalized)
                    .agg(lambda vals: vals.value_counts().idxmax())
                )
                remapped = normalized.map(canonical_map)
                df.loc[non_null.index, col] = remapped
                collapsed = raw_unique - normalized_unique
                log.append(f"'{col}': normalized casing/whitespace, collapsing {collapsed} redundant variant(s) into a single canonical form per value.")

        elif series.dtype == object and ctype not in ("categorical", "boolean", "text", "datetime"):
            # Mixed-type object column: if it's mostly numeric-looking,
            # coerce the whole column to numeric (non-numeric entries become
            # NaN, picked up by the completeness fix below).
            def _is_numeric_string(v):
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
            if 0.05 < frac_numeric < 0.95:
                coerced = pd.to_numeric(df[col].astype(str).str.replace(",", "", regex=False), errors="coerce")
                n_coerced_to_nan = int(coerced.isna().sum() - df[col].isna().sum())
                df[col] = coerced
                log.append(f"'{col}': converted mixed numeric/text column to numeric ({n_coerced_to_nan} non-numeric value(s) became missing, to be imputed below).")

    return df, log


def _fix_completeness(df: pd.DataFrame, column_types: dict[str, str], id_like: set[str]) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()
    log = []
    for col in df.columns:
        if col in id_like:
            continue
        series = df[col]
        n_missing = int(series.isna().sum())
        if n_missing == 0:
            continue

        ctype = column_types.get(col)
        # Re-check dtype AFTER consistency fixes may have coerced a column
        # to numeric -- trust the live pandas dtype over the original schema
        # label for this decision.
        is_numeric_now = pd.api.types.is_numeric_dtype(df[col])

        if ctype == "datetime":
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
            median_date = parsed.median()
            if pd.notna(median_date):
                df[col] = parsed.fillna(median_date)
                log.append(f"'{col}': filled {n_missing} missing date(s) with the column median ({median_date.date()}).")
            else:
                log.append(f"'{col}': could NOT be filled -- every value in this column is missing or unparsable as a date, so there is no real data left to impute from. This column carries no usable information; consider dropping it.")
        elif is_numeric_now:
            median_value = df[col].median()
            if pd.notna(median_value):
                df[col] = df[col].fillna(median_value)
                log.append(f"'{col}': filled {n_missing} missing value(s) with the median ({round(float(median_value), 4)}).")
            else:
                log.append(f"'{col}': could NOT be filled -- every value in this column is missing, so there is no real data left to impute from. This column carries no usable information; consider dropping it.")
        else:
            mode_series = df[col].mode(dropna=True)
            if not mode_series.empty:
                mode_value = mode_series.iloc[0]
                df[col] = df[col].fillna(mode_value)
                log.append(f"'{col}': filled {n_missing} missing value(s) with the most frequent value ('{mode_value}').")
            else:
                log.append(f"'{col}': could NOT be filled -- every value in this column is missing (in this dataset, all of its original values turned out to be null-like placeholders such as 'N/A'/'unknown', leaving nothing real to impute from). This column carries no usable information; consider dropping it.")

    return df, log


def _fix_uniqueness(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    before = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    n_dropped = before - len(df)
    log = [f"Dropped {n_dropped} exact duplicate row(s)."] if n_dropped > 0 else []
    return df, log


def _fix_balance(
    df: pd.DataFrame, target_column: str | None, column_types: dict[str, str], id_like: set[str], seed: int = 42
) -> tuple[pd.DataFrame, list[str]]:
    """Bring every class up to the majority class's count via lightweight,
    class-conditional resampling: each non-target column's synthetic value
    is independently sampled from that column's OWN empirical distribution
    within the minority class (not copied whole-row, which would manufacture
    exact duplicate rows), with small Gaussian jitter added to numeric
    columns for additional variation.

    This is explicitly a lightweight approximation, not full SMOTE (which
    would require a nearest-neighbor library not available in this
    environment) -- it preserves each column's per-class marginal
    distribution but does not preserve joint correlations between columns as
    faithfully as true SMOTE would. Documented here and in
    docs/math_and_logic.md so this isn't overclaimed as more sophisticated
    than it is.
    """
    if not target_column or target_column not in df.columns:
        return df, []

    counts = df[target_column].value_counts()
    if len(counts) < 2:
        return df, []

    majority_count = int(counts.max())
    rng = np.random.default_rng(seed)
    synthetic_frames = [df]
    log = []

    for cls, count in counts.items():
        deficit = majority_count - int(count)
        if deficit <= 0:
            continue
        class_subset = df[df[target_column] == cls]
        synthetic_cols = {}
        for col in df.columns:
            if col == target_column:
                synthetic_cols[col] = [cls] * deficit
                continue
            sampled = class_subset[col].dropna()
            if sampled.empty:
                sampled = df[col].dropna()
            if sampled.empty:
                synthetic_cols[col] = [np.nan] * deficit
                continue
            values = sampled.sample(n=deficit, replace=True, random_state=int(rng.integers(0, 1_000_000))).to_numpy()
            if col not in id_like and column_types.get(col) == "numeric" and pd.api.types.is_numeric_dtype(values):
                std = float(np.nanstd(sampled.to_numpy(dtype=float))) if len(sampled) > 1 else 0.0
                if std > 0:
                    jitter = rng.normal(0, 0.05 * std, size=deficit)
                    values = values.astype(float) + jitter
            synthetic_cols[col] = values
        synthetic_frames.append(pd.DataFrame(synthetic_cols))
        log.append(
            f"Target '{target_column}', class '{cls}': added {deficit} synthetic row(s) via class-conditional "
            "resampling with light numeric jitter (lightweight approximation, not full SMOTE) to match the majority class count."
        )

    if len(synthetic_frames) == 1:
        return df, []

    combined = pd.concat(synthetic_frames, ignore_index=True)
    return combined, log


def apply_fixes(
    df: pd.DataFrame,
    column_types: dict[str, str],
    target_column: str | None,
) -> FixResult:
    n_rows = len(df)
    id_like = {c for c in df.columns if is_id_like_column(c, df[c], column_types.get(c, ""), n_rows)}

    fix_log: list[str] = []

    working, log = _fix_validity(df, column_types, id_like)
    fix_log += log

    working, log = _fix_consistency(working, column_types, id_like)
    fix_log += log

    working, log = _fix_completeness(working, column_types, id_like)
    fix_log += log

    working, log = _fix_uniqueness(working)
    fix_log += log

    working, log = _fix_balance(working, target_column, column_types, id_like)
    fix_log += log

    if not fix_log:
        fix_log.append("No fixable issues were detected -- the dataset was already clean across all five dimensions.")

    return FixResult(cleaned_df=working, fix_log=fix_log)
