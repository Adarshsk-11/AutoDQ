"""Validation harness for testing whether DQI tracks measured model
performance as data quality is progressively degraded.

This is infrastructure for the cross-dataset validation study described in
docs/limitations_and_future_validation.md (Limitation 1 / Future Work item 1).
It does NOT itself constitute the full validation -- that requires running
this harness against several independently-sourced, real-world datasets,
which requires network access this development environment didn't have.
What this script provides is the machinery to do that the moment a real
dataset is available: point it at any CSV + target column, and it will
apply controlled, parameterized degradation at several intensity levels,
measure DQI and cross-validated baseline performance at each level, and
report whether they move together.

Usage:
    python validation_harness.py <path_to_csv> <target_column> [--levels 5]

Example (using a dataset already in this repo):
    python validation_harness.py A_clean.csv target --levels 5

Output: a table of degradation level -> DQI -> baseline F1/R2, plus the
Pearson correlation between DQI and performance across levels (the core
empirical question this harness exists to answer).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

from app.services.ingestion import detect_schema  # noqa: E402
from app.services.analysis_service import run_full_analysis  # noqa: E402


def degrade_dataset(df: pd.DataFrame, target_column: str, intensity: float, seed: int = 0) -> pd.DataFrame:
    """Apply a controlled degradation at the given intensity (0.0 = no
    degradation, 1.0 = maximum degradation applied by this function).
    Combines several degradation types proportionally so a single
    "intensity" knob sweeps across missingness, duplication, and outliers
    together -- a simple, reproducible stand-in for "how messy is this
    dataset" that scales smoothly for the correlation test below.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    feature_cols = [c for c in out.columns if c != target_column]

    # 1. Missingness: up to 40% of cells in feature columns at full intensity
    missing_frac = 0.40 * intensity
    if missing_frac > 0:
        for col in feature_cols:
            idx = rng.choice(out.index, size=int(missing_frac * n), replace=False)
            out.loc[idx, col] = np.nan

    # 2. Duplicate rows: up to 30% of rows re-appended at full intensity
    dup_frac = 0.30 * intensity
    if dup_frac > 0:
        dupes = out.sample(frac=dup_frac, random_state=seed)
        out = pd.concat([out, dupes], ignore_index=True)

    # 3. Outliers: inject extreme values into up to 10% of numeric cells.
    # Skip binary/near-binary numeric columns (e.g. 0/1 flags) -- injecting a
    # wild leverage-point value into a column with almost no natural variance
    # distorts its correlation with everything else far more than a
    # realistic "outlier" would, and would misrepresent what this harness is
    # trying to measure (an earlier version of this script did this and
    # produced a misleading result: it accidentally destroyed a genuine
    # leakage relationship's correlation by corrupting the flag column that
    # carried it, rather than realistically degrading the data).
    outlier_frac = 0.10 * intensity
    if outlier_frac > 0:
        numeric_cols = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(out[c]) and out[c].dropna().nunique() > 2
        ]
        for col in numeric_cols:
            idx = rng.choice(out.index, size=int(outlier_frac * len(out)), replace=False)
            out.loc[idx, col] = out[col].dropna().mean() + out[col].dropna().std() * rng.choice([8, -8], size=len(idx))

    return out


def run_harness(csv_path: str, target_column: str, levels: int = 5) -> pd.DataFrame:
    base_df = pd.read_csv(csv_path)
    if target_column not in base_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {csv_path}. Columns: {list(base_df.columns)}")

    rows = []
    for i in range(levels):
        intensity = i / (levels - 1) if levels > 1 else 0.0
        degraded = degrade_dataset(base_df, target_column, intensity, seed=i)
        schema = detect_schema(degraded)
        result = run_full_analysis(degraded, schema, target_column=target_column)

        dqi = result.dqi.dqi_0_100
        primary_metric = None
        if result.baseline.available:
            source = result.baseline.cleaned_features or result.baseline.full_features
            primary_metric = source.primary_metric_mean

        rows.append({
            "intensity": round(intensity, 2),
            "n_rows": degraded.shape[0],
            "dqi": dqi,
            "baseline_primary_metric": primary_metric,
            "task_type": result.baseline.task_type,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to the dataset CSV to degrade and evaluate")
    parser.add_argument("target_column", help="Name of the target column")
    parser.add_argument("--levels", type=int, default=5, help="Number of degradation levels to test (default: 5)")
    args = parser.parse_args()

    results = run_harness(args.csv_path, args.target_column, levels=args.levels)
    print(results.to_string(index=False))
    print()

    valid = results.dropna(subset=["baseline_primary_metric"])
    if len(valid) >= 3:
        corr = valid["dqi"].corr(valid["baseline_primary_metric"])
        print(f"Pearson correlation between DQI and baseline performance across {len(valid)} levels: {corr:.3f}")
        if corr > 0.5:
            print("-> DQI and measured model performance move together on this dataset (supports the core hypothesis).")
        elif corr > 0:
            print("-> Weak positive relationship -- worth testing on more datasets before drawing a conclusion.")
        else:
            print("-> No positive relationship observed on this dataset/run -- report this honestly if it occurs; "
                  "do not discard or re-run until a positive result appears.")
    else:
        print("Not enough levels produced a valid baseline metric to compute a correlation.")


if __name__ == "__main__":
    main()
