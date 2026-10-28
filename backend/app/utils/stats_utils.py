"""Benjamini-Hochberg false discovery rate correction.

Used by the redundancy and leakage detectors (see app.services.issue_detection)
to avoid the multiple-comparisons problem: testing many pairwise correlations
independently means some will cross a fixed threshold by chance alone, even
with no real relationship, especially as the number of numeric columns grows
(n columns -> n(n-1)/2 pairs tested). A fixed correlation threshold alone
doesn't account for this; BH correction controls the expected proportion of
false positives among everything flagged as significant.

This is a small, dependency-free implementation (statsmodels wasn't assumed
available) of the standard BH step-up procedure.
"""
from __future__ import annotations


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Given a list of p-values (in any order), return a same-length list of
    booleans indicating which ones are significant after BH correction at
    the given false discovery rate `alpha`.

    Standard BH step-up procedure:
      1. Sort p-values ascending, rank them 1..m.
      2. Find the largest rank k where p_(k) <= (k/m) * alpha.
      3. All p-values with rank <= k are declared significant.
    """
    m = len(p_values)
    if m == 0:
        return []

    indexed = sorted(range(m), key=lambda i: p_values[i])
    threshold_rank = -1
    for rank, idx in enumerate(indexed, start=1):
        p = p_values[idx]
        if p <= (rank / m) * alpha:
            threshold_rank = rank  # keep updating; BH takes the LARGEST qualifying rank

    significant = [False] * m
    if threshold_rank == -1:
        return significant

    for rank, idx in enumerate(indexed, start=1):
        if rank <= threshold_rank:
            significant[idx] = True
    return significant
