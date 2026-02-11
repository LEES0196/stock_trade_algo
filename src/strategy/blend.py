"""Combine multiple allocation dictionaries.

The `blend` function takes two allocation dictionaries and returns a
weighted combination.  It optionally enforces maximum weights per
ticker and renormalizes to ensure weights sum to one.
"""

from __future__ import annotations

from typing import Dict, Optional

def blend(
    alloc1: Dict[str, float],
    alloc2: Dict[str, float],
    w1: float = 0.6,
    w2: float = 0.4,
    caps: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Blend two allocation dictionaries.

    Parameters
    ----------
    alloc1, alloc2 : Dict[str, float]
        Weight dictionaries to combine.
    w1, w2 : float
        Weights applied to the first and second allocations respectively.  They
        do not need to sum to one; the result is normalized after combination.
    caps : Optional[Dict[str, float]]
        A dictionary of cap values.  Recognized keys are:
        - ``max_weight``: maximum allowed weight per ticker
        - ``min_weight``: minimum allowed weight per ticker (unused here but
          included for completeness)
        - ``cash_cap``: maximum allowed weight for the cash ticker

    Returns
    -------
    Dict[str, float]
        Blended and normalized allocation.
    """
    # Merge keys from both allocations
    result: Dict[str, float] = {}
    keys = set(alloc1) | set(alloc2)
    for k in keys:
        result[k] = alloc1.get(k, 0.0) * w1 + alloc2.get(k, 0.0) * w2
    # Normalize to sum to one
    total = sum(result.values()) or 1.0
    result = {k: v / total for k, v in result.items()}

    if caps:
        max_weight = caps.get("max_weight")
        cash_cap = caps.get("cash_cap")
        cash_key = next((k for k in result if k.upper() == "CASH"), None)
        # Apply individual max cap
        if max_weight is not None:
            result = {k: min(v, max_weight) for k, v in result.items()}
        # Apply cash cap separately if specified
        if cash_cap is not None and cash_key:
            result[cash_key] = min(result[cash_key], cash_cap)
        # Re‑normalize after caps
        s = sum(result.values()) or 1.0
        result = {k: v / s for k, v in result.items()}
    return result