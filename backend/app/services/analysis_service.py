"""Orchestrates the full AutoDQ pipeline:

Ingestion -> Profiler -> Quality Metrics -> Issue Detection -> DQI ->
Recommendations -> Baseline ML

This is the single entry point the API layer calls; it is fully testable
without FastAPI or a running server.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from app.analyzers.balance import analyze_balance
from app.analyzers.completeness import analyze_completeness
from app.analyzers.consistency import analyze_consistency
from app.analyzers.informativeness import analyze_informativeness
from app.analyzers.uniqueness import analyze_uniqueness
from app.analyzers.validity import analyze_validity
from app.metrics.dqi import compute_dqi
from app.recommendations.engine import generate_recommendations
from app.services.baseline_ml import run_baseline
from app.services.ingestion import ColumnSchema
from app.services.issue_detection import detect_all_issues
from app.services.profiler import build_profile


@dataclass
class AnalysisResult:
    profile: Any
    completeness: Any
    consistency: Any
    validity: Any
    uniqueness: Any
    balance: Any
    dqi: Any
    issues: list
    recommendations: list
    baseline: Any
    informativeness: Any
    warnings: list[str] = field(default_factory=list)


def detect_default_target(df: pd.DataFrame, column_types: dict[str, str]) -> str | None:
    """Heuristic target auto-detection: prefer a low-cardinality categorical
    column named like a common target, else the lowest-cardinality
    categorical column with 2-20 classes.

    Also considers binary (exactly 2 distinct values) NUMERIC columns as
    candidates. Many real-world binary targets (e.g. "passed", "churned",
    "survived") are stored as 0/1 integers, which schema inference correctly
    types as "numeric" rather than "categorical" -- without this, such
    columns would be entirely invisible to auto-detection, and a genuinely
    unrelated low-cardinality categorical column (if one happens to exist)
    would get picked instead, silently testing the wrong question."""
    common_names = {"target", "label", "class", "y", "outcome", "churn", "survived", "diagnosis", "default", "fraud"}
    categorical_candidates = [c for c in df.columns if column_types.get(c) in ("categorical", "boolean")]
    binary_numeric_candidates = [
        c for c in df.columns
        if column_types.get(c) == "numeric" and df[c].nunique(dropna=True) == 2
    ]

    for c in categorical_candidates + binary_numeric_candidates:
        if c.lower() in common_names:
            return c

    scored = []
    for c in categorical_candidates:
        n_unique = df[c].nunique(dropna=True)
        if 2 <= n_unique <= 20:
            scored.append((n_unique, c))
    for c in binary_numeric_candidates:
        scored.append((2, c))

    if scored:
        scored.sort()
        return scored[0][1]
    return None


def run_full_analysis(
    df: pd.DataFrame,
    schema: list[ColumnSchema],
    target_column: str | None = None,
    domain: str = "general",
    ingestion_warnings: list[str] | None = None,
) -> AnalysisResult:
    column_types = {s.name: s.dtype for s in schema}
    warnings = list(ingestion_warnings or [])

    profile = build_profile(df, schema)

    if target_column and target_column not in df.columns:
        warnings.append(f"Requested target column '{target_column}' not found; falling back to auto-detection.")
        target_column = None
    if not target_column:
        target_column = detect_default_target(df, column_types)
        if target_column:
            warnings.append(f"No target column specified; auto-detected '{target_column}' as the likely target.")
        else:
            warnings.append("No target column specified or detected; balance and leakage analysis will be limited.")

    completeness = analyze_completeness(df)
    consistency = analyze_consistency(df, column_types)
    validity = analyze_validity(df, column_types)
    uniqueness = analyze_uniqueness(df, column_types)
    balance = analyze_balance(df, target_column)

    dqi = compute_dqi(
        completeness=completeness.score,
        consistency=consistency.score,
        validity=validity.score,
        uniqueness=uniqueness.score,
        balance=balance.score if balance.available else None,
        domain=domain,
    )

    issues = detect_all_issues(
        df, column_types, target_column,
        class_counts=balance.class_counts if balance.available else None,
    )
    recommendations = generate_recommendations(issues)

    baseline = run_baseline(df, target_column, column_types, issues=issues)

    # Informativeness is computed and returned alongside the DQI, but is
    # deliberately NOT one of its five weighted components -- see
    # app.analyzers.informativeness for why hygiene and informativeness are
    # kept as separate, independently-interpretable signals.
    informativeness = analyze_informativeness(df, column_types, target_column)

    return AnalysisResult(
        profile=profile,
        completeness=completeness,
        consistency=consistency,
        validity=validity,
        uniqueness=uniqueness,
        balance=balance,
        dqi=dqi,
        issues=issues,
        recommendations=recommendations,
        baseline=baseline,
        informativeness=informativeness,
        warnings=warnings,
    )


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses (and nested structures) into plain
    JSON-serializable Python objects for the API layer."""
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    return obj


def analysis_result_to_dict(result: AnalysisResult) -> dict:
    return _to_jsonable(result)
