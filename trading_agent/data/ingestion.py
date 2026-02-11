"""
Data ingestion utilities for market data.

This module provides a thin abstraction for fetching historical prices using
`yfinance`. In production, this can be extended to support Alpaca/Polygon/
other providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:  # Optional import; only used when running ingestion
    import yfinance as yf
except Exception:  # pragma: no cover - import guard
    yf = None  # type: ignore


@dataclass
class IngestionConfig:
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1d"


def fetch_yfinance(cfg: IngestionConfig) -> pd.DataFrame:
    """Fetch OHLCV data using yfinance.

    Parameters
    ----------
    cfg : IngestionConfig
        Configuration including symbol, start/end dates, and interval.

    Returns
    -------
    pd.DataFrame
        Historical OHLCV indexed by DatetimeIndex.
    """

    if yf is None:
        raise RuntimeError("yfinance not available; install per requirements.txt")

    data = yf.download(cfg.symbol, start=cfg.start, end=cfg.end, interval=cfg.interval, auto_adjust=True)
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("No data returned from yfinance")
    data = data.rename(columns=str.lower)
    data.index = pd.to_datetime(data.index)
    return data

