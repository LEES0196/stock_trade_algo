"""
Portfolio optimizer using Riskfolio-Lib.

Goal: Maximize Sharpe Ratio subject to user constraints from config.yaml.

Mathematical Note
-----------------
We solve: max_w (mu^T w - r_f) / sqrt(w^T Sigma w) subject to constraints.
Riskfolio-Lib provides high-level wrappers to set up the problem; here we
expose a helper to compute target weights.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:  # Optional heavy import
    import riskfolio as rp
except Exception:  # pragma: no cover - import guard
    rp = None  # type: ignore


def optimize_weights(historical_returns: pd.DataFrame, constraint_matrix: Optional[np.ndarray] = None) -> pd.Series:
    """Optimize portfolio weights to maximize Sharpe Ratio.

    Parameters
    ----------
    historical_returns : pd.DataFrame
        Asset returns with columns as tickers and rows as dates.
    constraint_matrix : Optional[np.ndarray]
        Additional linear constraints A w <= b encoded into Riskfolio-Lib
        structures; omitted in this stub.

    Returns
    -------
    pd.Series
        Optimal weights per asset (summing to 1), uniform if riskfolio missing.
    """

    n_assets = historical_returns.shape[1]
    if n_assets == 0:
        return pd.Series(dtype=float)

    if rp is None:  # Fallback: equal weights
        w = np.repeat(1.0 / n_assets, n_assets)
        return pd.Series(w, index=historical_returns.columns)

    y = historical_returns.dropna()
    port = rp.Portfolio(returns=y)
    port.assets_stats(method_mu="hist", method_cov="hist")
    model = "Classic"  # mean-variance
    rm = "MV"  # volatility risk measure
    obj = "Sharpe"
    hist = True
    w = port.optimization(model=model, rm=rm, obj=obj, hist=hist)
    return w.iloc[:, 0]

