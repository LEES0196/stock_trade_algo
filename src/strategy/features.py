"""Feature engineering for the auto portfolio strategy.

This module contains helpers to compute monthly return features from daily
prices.  The primary function builds a DataFrame of 1–N month returns
for each ticker.  An auxiliary function extracts the most recent set of
features for use in the allocation algorithms.
"""

from __future__ import annotations

import pandas as pd
from typing import Dict

def monthly_returns(prices: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """Compute rolling monthly percentage returns for a price DataFrame.

    The returned DataFrame has a multi‑index on the columns where the top
    level corresponds to the lookback horizon (e.g. "1M", "2M", …) and the
    lower level is the ticker.  Each row corresponds to a month end.

    Parameters
    ----------
    prices : pd.DataFrame
        Daily (or higher frequency) price series indexed by date.
    months : int
        Number of monthly lookback periods to compute (default 12).

    Returns
    -------
    pd.DataFrame
        A DataFrame with a DateTimeIndex and columns of multi‑index
        (horizon, ticker), containing monthly percentage returns.
    """
    # Resample to month‑end and drop incomplete rows
    # Use 'ME' (month end) to avoid pandas deprecation warnings for 'M'
    month_end = prices.resample("ME").last().dropna(how="all")

    feats: Dict[str, pd.DataFrame] = {}
    for k in range(1, months + 1):
        # Percentage change over k months
        feats[f"{k}M"] = month_end.pct_change(k)
    # Concatenate along columns, using the horizon label as the top level
    out = pd.concat(feats, axis=1)
    return out

def latest_snapshot(monthly: pd.DataFrame) -> pd.DataFrame:
    """Extract the most recent set of monthly return features.

    Given the full multi‑indexed DataFrame of monthly returns, this
    function returns a simple DataFrame with tickers as the index and
    columns named by the lookback (e.g. "1M", "2M" …), containing the
    latest values.  This form is convenient for ranking and scoring
    algorithms.

    Parameters
    ----------
    monthly : pd.DataFrame
        Output from `monthly_returns`.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by ticker with columns for each lookback.
    """
    if monthly.empty:
        return pd.DataFrame()
    # Take the last row (most recent month end) as a Series with MultiIndex
    # where the levels are (horizon, ticker), then pivot to a simple table
    # of tickers x horizons.
    last_series = monthly.iloc[-1]
    table = last_series.unstack(level=0)
    table.index.name = "Ticker"
    return table
