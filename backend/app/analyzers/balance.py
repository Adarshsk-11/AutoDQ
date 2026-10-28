"""Balance: class-imbalance assessment for a classification target.

Per user decision, we report BOTH:
  - a normalized Shannon-entropy ratio as the actual DQI component score
    (Balance = H(observed) / H(uniform)), which generalizes cleanly to
    multiclass and is the more defensible metric for scoring purposes, and
  - the minority:majority ratio as a human-readable companion statistic
    (e.g. "1:8 imbalance"), which is far more intuitive to communicate.

If no target column is selected/detected, Balance is reported as
unavailable rather than silently defaulting to a misleading score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from math import log

import pandas as pd


@dataclass
class BalanceResult:
    available: bool
    score: float | None = None  # entropy ratio in [0, 1]
    class_counts: dict[str, int] = field(default_factory=dict)
    minority_majority_ratio: str | None = None  # e.g. "1:8"
    minority_majority_ratio_value: float | None = None  # e.g. 0.125
    n_classes: int = 0
    explanation: str = ""


def analyze_balance(df: pd.DataFrame, target_column: str | None) -> BalanceResult:
    if not target_column or target_column not in df.columns:
        return BalanceResult(
            available=False,
            explanation="Balance analysis unavailable because no target column was selected.",
        )

    series = df[target_column].dropna()
    if series.empty:
        return BalanceResult(
            available=False,
            explanation=f"Balance analysis unavailable: target column '{target_column}' has no non-missing values.",
        )

    counts = series.value_counts()
    n_classes = len(counts)

    if n_classes < 2:
        return BalanceResult(
            available=False,
            class_counts={str(k): int(v) for k, v in counts.items()},
            n_classes=n_classes,
            explanation=(
                f"Balance analysis unavailable: target column '{target_column}' has only "
                f"{n_classes} distinct class, so imbalance is not meaningful (this may also "
                "indicate the wrong column was selected as target, or a regression target)."
            ),
        )

    total = int(counts.sum())
    probs = (counts / total).to_numpy()

    # Shannon entropy of the observed distribution, normalized by max possible entropy (uniform)
    entropy = -sum(p * log(p) for p in probs if p > 0)
    max_entropy = log(n_classes)
    entropy_ratio = entropy / max_entropy if max_entropy > 0 else 1.0
    score = round(max(0.0, min(1.0, entropy_ratio)), 4)

    minority_count = int(counts.min())
    majority_count = int(counts.max())
    ratio_value = round(minority_count / majority_count, 4) if majority_count > 0 else None
    # Express as "1:N" using majority/minority (rounded to 1 decimal for readability)
    ratio_str = f"1:{round(majority_count / minority_count, 1)}" if minority_count > 0 else "undefined"

    if score >= 0.9:
        level = "well balanced"
    elif score >= 0.7:
        level = "mildly imbalanced"
    elif score >= 0.4:
        level = "moderately imbalanced"
    else:
        level = "severely imbalanced"

    explanation = (
        f"Target '{target_column}' has {n_classes} classes with a minority:majority ratio of "
        f"{ratio_str} ({minority_count:,} vs {majority_count:,} records). The dataset is {level} "
        f"(normalized entropy score {score})."
    )

    return BalanceResult(
        available=True,
        score=score,
        class_counts={str(k): int(v) for k, v in counts.items()},
        minority_majority_ratio=ratio_str,
        minority_majority_ratio_value=ratio_value,
        n_classes=n_classes,
        explanation=explanation,
    )
