"""
Feature engineering and preprocessing utilities.

This module computes technical indicators and merges optional macro features.

Mathematical Note
-----------------
We produce a simple moving average (SMA) and volatility (rolling std) as
baseline statistical features used by the hybrid manager's voting logic.
"""

from __future__ import annotations

import pandas as pd


def add_technical_indicators(df: pd.DataFrame, sma_window: int = 20, vol_window: int = 20) -> pd.DataFrame:
    """Add basic technical indicators (SMA, volatility) to OHLCV data.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV with columns like 'close', 'volume'.
    sma_window : int
        Window for simple moving average of close.
    vol_window : int
        Window for rolling volatility (standard deviation of returns).

    Returns
    -------
    pd.DataFrame
        Original data with 'sma' and 'volatility' columns added.
    """

    out = df.copy()
    out["return"] = out["close"].pct_change()
    out["sma"] = out["close"].rolling(window=sma_window, min_periods=1).mean()
    out["volatility"] = out["return"].rolling(window=vol_window, min_periods=2).std().fillna(0.0)
    return out

