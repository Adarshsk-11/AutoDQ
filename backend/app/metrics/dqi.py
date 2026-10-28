"""Dataset Quality Index (DQI) computation.

DQI = wC*C + wS*S + wV*V + wU*U + wB*B, weights sum to 1.

Weights are explicitly labeled DEFAULT / CONFIGURABLE heuristics, not a
universal standard (spec section 6). If Balance is unavailable (no target
selected), its weight is redistributed proportionally across the remaining
four components rather than silently treating a missing metric as 0 or 1 --
either of those would distort the score in a misleading direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

DOMAIN_PRESETS: dict[str, dict[str, float]] = {
    "general": {"completeness": 0.25, "consistency": 0.20, "validity": 0.20, "uniqueness": 0.15, "balance": 0.20},
    "healthcare": {"completeness": 0.30, "consistency": 0.20, "validity": 0.20, "uniqueness": 0.10, "balance": 0.20},
    "finance": {"completeness": 0.20, "consistency": 0.30, "validity": 0.20, "uniqueness": 0.10, "balance": 0.20},
    "ecommerce": {"completeness": 0.25, "consistency": 0.20, "validity": 0.15, "uniqueness": 0.20, "balance": 0.20},
}

INTERPRETATION_BANDS = [
    (0, 40, "Poor"),
    (40, 60, "Low"),
    (60, 80, "Moderate"),
    (80, 101, "High"),
]


@dataclass
class ComponentContribution:
    name: str
    score: float | None
    weight: float
    effective_weight: float  # after redistribution if unavailable
    contribution: float
    available: bool


@dataclass
class DQIResult:
    dqi_0_1: float
    dqi_0_100: float
    interpretation: str
    domain_preset: str
    weights_used: dict[str, float]
    components: list[ComponentContribution] = field(default_factory=list)
    note: str = (
        "Weights are default, configurable heuristics -- not a universally standardized "
        "formula. Interpretation thresholds are configurable heuristic ranges."
    )


def _interpret(score_0_100: float) -> str:
    for low, high, label in INTERPRETATION_BANDS:
        if low <= score_0_100 < high:
            return label
    return "High"


def compute_dqi(
    completeness: float,
    consistency: float,
    validity: float,
    uniqueness: float,
    balance: float | None,
    domain: str = "general",
    custom_weights: dict[str, float] | None = None,
) -> DQIResult:
    weights = dict(custom_weights) if custom_weights else dict(DOMAIN_PRESETS.get(domain, DOMAIN_PRESETS["general"]))

    scores = {
        "completeness": completeness,
        "consistency": consistency,
        "validity": validity,
        "uniqueness": uniqueness,
        "balance": balance,
    }

    balance_available = balance is not None
    effective_weights = dict(weights)
    if not balance_available:
        redistribute = weights["balance"]
        remaining_keys = [k for k in weights if k != "balance"]
        remaining_total = sum(weights[k] for k in remaining_keys)
        for k in remaining_keys:
            share = weights[k] / remaining_total if remaining_total > 0 else 1 / len(remaining_keys)
            effective_weights[k] = weights[k] + redistribute * share
        effective_weights["balance"] = 0.0

    components = []
    total = 0.0
    for name in ("completeness", "consistency", "validity", "uniqueness", "balance"):
        score = scores[name]
        available = score is not None
        eff_w = effective_weights[name]
        contribution = (score * eff_w) if available else 0.0
        total += contribution
        components.append(ComponentContribution(
            name=name,
            score=round(score, 4) if available else None,
            weight=round(weights[name], 4),
            effective_weight=round(eff_w, 4),
            contribution=round(contribution, 4),
            available=available,
        ))

    dqi_0_1 = max(0.0, min(1.0, total))
    dqi_0_100 = round(dqi_0_1 * 100, 1)

    return DQIResult(
        dqi_0_1=round(dqi_0_1, 4),
        dqi_0_100=dqi_0_100,
        interpretation=_interpret(dqi_0_100),
        domain_preset=domain,
        weights_used=weights,
        components=components,
    )
