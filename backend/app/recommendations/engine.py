"""Recommendation engine.

Every recommendation is generated from an actual detected Issue -- never
a hardcoded/generic tip disconnected from the analyzed dataset (spec
section 23). Recommendations are ranked by issue severity so the most
impactful fixes surface first.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.issue_detection import Issue

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@dataclass
class Recommendation:
    issue_type: str
    severity: str
    columns: list[str]
    message: str


def _recommend_for_issue(issue: Issue) -> str:
    cols = ", ".join(issue.columns) if issue.columns else "the dataset"

    if issue.issue_type == "missing_values":
        col = issue.columns[0]
        pct = issue.measured_value
        if pct >= 50:
            return (
                f"'{col}' contains {pct}% missing values. This column may be unusable as-is -- "
                "consider dropping it or investigating whether it can be reconstructed from another source."
            )
        return f"'{col}' contains {pct}% missing values. Consider median/mode imputation, or KNN imputation for numeric columns."

    if issue.issue_type == "duplicate_rows":
        return (
            f"{issue.measured_value}% of rows are exact duplicates. Consider removing duplicate rows "
            "after confirming they aren't legitimate repeated observations."
        )

    if issue.issue_type == "class_imbalance":
        col = issue.columns[0]
        return (
            f"The target classes in '{col}' have roughly a 1:{issue.measured_value} imbalance. "
            "Consider class weighting, oversampling (e.g. SMOTE), or undersampling the majority class."
        )

    if issue.issue_type == "outliers":
        col = issue.columns[0]
        return (
            f"'{col}' contains {issue.measured_value}% statistical outliers. Investigate extreme values "
            "before training -- they may be genuine, data-entry errors, or sensor faults."
        )

    if issue.issue_type == "redundant_features":
        a, b = issue.columns
        return f"'{a}' and '{b}' have correlation {issue.measured_value}. Consider removing one of them to reduce redundancy."

    if issue.issue_type == "potential_leakage":
        col = issue.columns[0]
        return (
            f"'{col}' has an unusually strong association with the target and may contain "
            "target-related information. Investigate for possible leakage before training."
        )

    if issue.issue_type == "distributional_imbalance":
        col = issue.columns[0]
        return (
            f"'{col}' is dominated by a single category. Review whether this reflects the true "
            "population or a sampling artifact before using it as a feature."
        )

    if issue.issue_type == "constant_column":
        col = issue.columns[0]
        return (
            f"'{col}' has the same value in every row and carries zero information. Consider "
            "dropping it, or check whether this reflects a data export problem (e.g. a filter "
            "that was accidentally applied before export)."
        )

    return f"Review {cols}: {issue.explanation}"


def generate_recommendations(issues: list[Issue]) -> list[Recommendation]:
    ranked = sorted(issues, key=lambda i: _SEVERITY_ORDER.get(i.severity, 99))
    return [
        Recommendation(
            issue_type=i.issue_type,
            severity=i.severity,
            columns=i.columns,
            message=_recommend_for_issue(i),
        )
        for i in ranked
    ]
