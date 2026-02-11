"""Data access utilities for the auto portfolio strategy.

This module provides a function for downloading historical price data
using the `yfinance` package.  Results can optionally be cached to
speed up repeated runs on the same universe and time frame.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
import yfinance as yf

def _parquet_available() -> bool:
    """Return True if a parquet engine is available for pandas.

    We avoid importing optional dependencies at module import time; instead we
    check lazily and fall back to CSV caching if parquet isn't available.
    """
    try:
        # pandas checks for one of these to be importable
        import pyarrow  # type: ignore  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # type: ignore  # noqa: F401
            return True
        except Exception:
            return False

def download_prices(
    tickers: List[str],
    period: str = "5y",
    interval: str = "1d",
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Download historical close prices for a list of tickers.

    Parameters
    ----------
    tickers : List[str]
        A list of ticker symbols accepted by Yahoo Finance.  Synthetic tickers
        such as "CASH" will be ignored.
    period : str, optional
        The lookback period to download, passed directly to yfinance (e.g. "5y").
    interval : str, optional
        The sampling frequency (e.g. "1d" for daily).
    cache_dir : Optional[str], optional
        If provided, downloaded data will be written to this directory as
        Parquet files and reused on subsequent calls.

    Returns
    -------
    pd.DataFrame
        A DataFrame indexed by timestamp with columns for each ticker's close
        price.  Missing values are dropped entirely to avoid downstream
        propagation of NaNs.
    """

    # Remove synthetic tickers from the download list
    symbols = [t for t in tickers if t.upper() != "CASH"]
    if not symbols:
        return pd.DataFrame()

    # Use a simple cache keyed by tickers, period and interval
    cache_path: Optional[Path] = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        use_parquet = _parquet_available()
        ext = "parquet" if use_parquet else "csv"
        key = f"{'_'.join(sorted(symbols))}_{period}_{interval}.{ext}"
        cache_path = Path(cache_dir) / key
        if cache_path.exists():
            if use_parquet:
                return pd.read_parquet(cache_path)
            else:
                return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    # Download from Yahoo; auto_adjust ensures dividends/splits are accounted for
    data = yf.download(
        tickers=symbols,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns a multi‑index DataFrame when multiple tickers are used
    if isinstance(data, pd.Series):
        data = data.to_frame()
    # We only care about close prices
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data
    # Drop rows where all tickers are NaN
    data = data.dropna(how="all")

    if cache_path is not None:
        if cache_path.suffix == ".parquet" and _parquet_available():
            data.to_parquet(cache_path)
        else:
            # Fallback to CSV caching to avoid optional parquet dependency
            data.to_csv(cache_path)

    return data
