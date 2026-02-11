"""Score‑based allocation algorithm (Algorithm 2).

This algorithm assigns a score to each selected asset based on the
fraction of lookback horizons where the return is positive.  A fixed
score is assigned to cash.  Weights are proportional to scores.
"""

from __future__ import annotations

from typing import Dict, Iterable

import pandas as pd

def run_algorithm_2(
    full_table: pd.DataFrame,
    score_set: Iterable[str],
    cash_score: float = 0.15,
    horizon_count: int = 12,
    cash_ticker: str = "CASH",
) -> Dict[str, float]:
    """Compute a score‑based allocation.

    Parameters
    ----------
    full_table : pd.DataFrame
        DataFrame indexed by ticker with columns for each lookback (e.g. "1M".."12M").
    score_set : Iterable[str]
        Tickers to evaluate in this algorithm.
    cash_score : float
        A fixed score assigned to the cash position.
    horizon_count : int
        Number of lookback columns to consider (default 12).  If fewer are present,
        only those columns will be used.
    cash_ticker : str
        Symbol representing cash.

    Returns
    -------
    Dict[str, float]
        A mapping from ticker to allocation weight summing to 1.0.
    """
    scores: Dict[str, float] = {}

    # Determine which columns to examine based on available data
    available_cols = [c for c in full_table.columns if c.endswith("M")]
    cols_to_use = [c for c in available_cols if int(c.rstrip("M")) <= horizon_count]
    for ticker in score_set:
        if ticker not in full_table.index:
            continue
        row = full_table.loc[ticker]
        # Count how many horizons have positive returns
        pos_count = sum(1 for col in cols_to_use if row.get(col, float("nan")) > 0)
        if horizon_count > 0:
            scores[ticker] = pos_count / float(len(cols_to_use)) if cols_to_use else 0.0
        else:
            scores[ticker] = 0.0

    total_score = sum(scores.values()) + cash_score
    if total_score == 0:
        # Avoid division by zero; allocate entirely to cash
        return {cash_ticker: 1.0}
    alloc: Dict[str, float] = {t: s / total_score for t, s in scores.items()}
    alloc[cash_ticker] = cash_score / total_score
    return alloc