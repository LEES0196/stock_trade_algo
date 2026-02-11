"""
Risk Guardrails: enforce hard constraints such as Max Drawdown.

check_risk(order) returns False if a prospective trade violates configured
limits based on current portfolio state.

Mathematical Note
-----------------
Max Drawdown (MDD) is defined as the maximum observed decline from a peak to
trough of the equity curve. If current drawdown exceeds the configured bound
MDD_max, we block new risk-increasing trades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PortfolioState:
    equity: float
    peak_equity: float

    @property
    def drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)


def check_risk(order: object, portfolio: PortfolioState, max_drawdown: float) -> bool:
    """Return True if trade is allowed under risk constraints.

    Parameters
    ----------
    order : object
        The order under consideration (unused in this simple guardrail).
    portfolio : PortfolioState
        Current equity and peak equity.
    max_drawdown : float
        Maximum allowed drawdown (e.g., 0.2 for 20%).

    Returns
    -------
    bool
        False if current drawdown exceeds the limit; True otherwise.
    """

    return portfolio.drawdown <= max_drawdown

