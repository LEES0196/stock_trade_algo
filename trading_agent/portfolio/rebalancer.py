"""
Threshold-based rebalancing logic.

Logic
-----
Trigger a rebalance if abs(current_weight - target_weight) > threshold.

Mathematical Note
-----------------
This is a band policy minimizing turnover: for each asset i, rebalance when
|w_i - w_i^*| > delta. This approximates an L^0/L^1 transaction cost penalty by
avoiding small, frequent trades.
"""

from __future__ import annotations

from typing import Dict


def needs_rebalance(current: Dict[str, float], target: Dict[str, float], threshold: float) -> bool:
    """Return True if any asset deviates from target by more than threshold.

    Parameters
    ----------
    current : Dict[str, float]
        Current portfolio weights.
    target : Dict[str, float]
        Target portfolio weights.
    threshold : float
        Absolute deviation threshold (e.g., 0.05 for 5%).
    """

    for k, tgt in target.items():
        cur = current.get(k, 0.0)
        if abs(cur - tgt) > threshold:
            return True
    return False

