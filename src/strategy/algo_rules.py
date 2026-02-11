"""Rule‑based allocation algorithm (Algorithm 1).

This algorithm implements a simple momentum switch: if any aggressive
asset exhibits positive momentum over a specified horizon, allocate
100 percent to the best performing aggressive asset.  Otherwise,
allocate equally across the top N passive assets by a shorter horizon,
replacing any with negative returns by cash.  The behaviour mirrors
the description in the provided PDF and notebook.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

def run_algorithm_1(
    full_table: pd.DataFrame,
    aggressive: Iterable[str],
    passive: Iterable[str],
    aggressive_pos_col: str = "12M",
    passive_rank_col: str = "6M",
    passive_top_n: int = 3,
    cash_ticker: str = "CASH",
) -> Dict[str, float]:
    """Compute a rule‑based allocation.

    Parameters
    ----------
    full_table : pd.DataFrame
        DataFrame indexed by ticker with columns for each lookback (e.g. "1M".."12M").
    aggressive : Iterable[str]
        Tickers considered "aggressive"; these will be evaluated on a longer horizon.
    passive : Iterable[str]
        Tickers considered "passive"; used when aggressive momentum is negative.
    aggressive_pos_col : str
        Column used to test if aggressive assets have positive momentum (default "12M").
    passive_rank_col : str
        Column used to rank passive assets (default "6M").
    passive_top_n : int
        Number of passive assets to include in fallback allocation.
    cash_ticker : str
        Symbol representing cash; receives weight if passive returns are negative.

    Returns
    -------
    Dict[str, float]
        A mapping from ticker to allocation weight summing to 1.0.
    """
    # Filter tables to only the relevant tickers
    agg_df = full_table.loc[full_table.index.intersection(aggressive)]
    pas_df = full_table.loc[full_table.index.intersection(passive)]

    alloc: Dict[str, float] = {}

    # Identify aggressive assets with positive long‑term return
    if aggressive_pos_col in agg_df.columns:
        positive_aggs = agg_df[agg_df[aggressive_pos_col] > 0]
    else:
        positive_aggs = pd.DataFrame()
    # If any aggressive asset is positive, allocate 100% to the best one
    if not positive_aggs.empty:
        # Pick the ticker with the highest value in the aggressive column
        best = positive_aggs[aggressive_pos_col].idxmax()
        alloc[best] = 1.0
        return alloc

    # Otherwise, rank passive assets by the selected lookback
    if passive_rank_col not in pas_df.columns or pas_df.empty:
        # No passive data; fall back entirely to cash
        alloc[cash_ticker] = 1.0
        return alloc

    top = pas_df.sort_values(passive_rank_col, ascending=False).head(passive_top_n)
    # Equal weight across the top N
    equal_w = 1.0 / passive_top_n if passive_top_n > 0 else 0.0
    cash_add = 0.0
    for ticker, row in top.iterrows():
        if row[passive_rank_col] > 0:
            alloc[ticker] = alloc.get(ticker, 0.0) + equal_w
        else:
            cash_add += equal_w
    if cash_add > 0:
        alloc[cash_ticker] = cash_add
    # Normalize to sum to 1 (in case some passives were negative and removed)
    total = sum(alloc.values())
    if total > 0:
        alloc = {k: v / total for k, v in alloc.items()}
    return alloc