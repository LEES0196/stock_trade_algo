"""Strategy orchestration for the auto portfolio project.

This module coordinates data download, feature generation and the two
allocation algorithms before blending them into a final portfolio.  The
primary entry point is the ``run_all`` function, which accepts a
configuration dictionary (parsed from YAML) and returns a DataFrame of
the computed weights along with the evaluation timestamp.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

from .data import download_prices
from .features import monthly_returns, latest_snapshot
from .algo_rules import run_algorithm_1
from .algo_scores import run_algorithm_2
from .blend import blend
from .nn_predictor import nn_allocation

def _build_full_table(prices: pd.DataFrame, months: int) -> pd.DataFrame:
    """Helper to construct a ticker x lookback table from price data."""
    mret = monthly_returns(prices, months=months)
    # Extract the most recent row and pivot into a simple table
    table = latest_snapshot(mret)
    # Ensure the index name is set to 'Ticker'
    table.index.name = "Ticker"
    return table

def run_all(cfg: Dict) -> pd.DataFrame:
    """Execute the full strategy pipeline.

    Parameters
    ----------
    cfg : Dict
        Configuration dictionary loaded from YAML.  See ``config/base.yaml`` for
        an example of the expected structure.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ``timestamp`` and ``weight`` and the ticker as
        the index.  The index is sorted in descending order of weight.
    """
    uni = cfg.get("universe", {})
    # Compose the list of tickers to download (aggressive + passive)
    tickers: List[str] = list(uni.get("aggressive", [])) + list(uni.get("passive", []))
    prices = download_prices(tickers, **cfg.get("data", {}))

    # If model config requests NN, use NN-based allocation path
    model_cfg = cfg.get("model", {}) or {}
    model_type = str(model_cfg.get("type", "")).lower()
    if model_type == "nn":
        aggressive = list(uni.get("aggressive", []))
        passive = list(uni.get("passive", []))
        alloc = nn_allocation(
            prices,
            aggressive=aggressive,
            passive=passive,
            data_interval=str(cfg.get("data", {}).get("interval", "1d")),
            time_interval=str(model_cfg.get("time_interval", "5d")),
            lookback_window=int(model_cfg.get("lookback_window", 60)),
            confidence_threshold=float(model_cfg.get("confidence_interval", 0.7)),
            cash_ticker=str(uni.get("cash_ticker", "CASH")),
            model_path=model_cfg.get("model_path"),
            train=bool(model_cfg.get("train", False)),
        )
        df = pd.Series(alloc, name="weight").to_frame()
        df["timestamp"] = pd.Timestamp.utcnow().normalize()
        return df[["timestamp", "weight"]].sort_values("weight", ascending=False)

    # Legacy path: feature table + two algorithms blended
    features_cfg = cfg.get("features", {})
    months = features_cfg.get("months", 12)
    full_table = _build_full_table(prices, months)
    alloc_cfg = cfg.get("allocation", {})
    # Algorithm 1 parameters
    algo1_cfg = alloc_cfg.get("algo1", {})
    a1 = run_algorithm_1(
        full_table,
        uni.get("aggressive", []),
        uni.get("passive", []),
        aggressive_pos_col=algo1_cfg.get("aggressive_pos_col", "12M"),
        passive_rank_col=algo1_cfg.get("passive_rank_col", "6M"),
        passive_top_n=algo1_cfg.get("passive_top_n", 3),
        cash_ticker=uni.get("cash_ticker", "CASH"),
    )
    # Algorithm 2 parameters
    algo2_cfg = alloc_cfg.get("algo2", {})
    a2 = run_algorithm_2(
        full_table,
        uni.get("score_set", []),
        cash_score=algo2_cfg.get("cash_score", 0.15),
        horizon_count=features_cfg.get("months", 12),
        cash_ticker=uni.get("cash_ticker", "CASH"),
    )
    # Blend allocations
    blend_cfg = alloc_cfg.get("blend", {})
    caps_cfg = alloc_cfg.get("caps", {})
    final_alloc = blend(
        a1,
        a2,
        w1=blend_cfg.get("w1", 0.6),
        w2=blend_cfg.get("w2", 0.4),
        caps=caps_cfg or None,
    )
    # Convert to DataFrame with timestamp
    df = pd.Series(final_alloc, name="weight").to_frame()
    df["timestamp"] = pd.Timestamp.utcnow().normalize()
    return df[["timestamp", "weight"]].sort_values("weight", ascending=False)
