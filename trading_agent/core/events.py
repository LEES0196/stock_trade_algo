"""
Event definitions for the event-driven trading agent.

This module defines a minimal set of events used across the system to
coordinate flow between components. Events are immutable dataclasses that
travel over the EventBus to decouple producers and consumers.

Mathematical/Architecture Note
------------------------------
- MarketEvent: conveys new market state x_t (e.g., OHLCV) that downstream
  components use to produce signals s_t. This aligns with a discrete-time
  state-space view where observations arrive in an event stream.
- SignalEvent: represents a decision signal s_t in [-1, 1] where -1 = short,
  0 = flat, 1 = long. It is an aggregation of multiple estimators
  (e.g., TFT, FinBERT, moving averages) and can be interpreted as a vote or
  continuous score. Keeping it continuous supports thresholding and risk.
- OrderEvent: converts a signal to an executable intent (side, size). Sizing
  may come from an RL policy a_t in [-1,1] mapping to position weight.
- FillEvent: confirms execution; enables PnL accounting and drawdown tracking.

All events include type hints and docstrings for clarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional


class Event:
    """Base class for all events.

    Serves as a marker type. Specific event payloads are defined in
    subclasses. Events should be treated as immutable messages.
    """


@dataclass(frozen=True)
class MarketEvent(Event):
    """New market data has arrived.

    Attributes
    ----------
    symbol: str
        The trading symbol (e.g., "AAPL").
    timestamp: datetime
        The timestamp of the observation.
    data: Dict[str, Any]
        Arbitrary payload such as OHLCV fields, indicators, etc.
    """

    symbol: str
    timestamp: datetime
    data: Dict[str, Any]


@dataclass(frozen=True)
class SignalEvent(Event):
    """A strategy signal derived from market data.

    Attributes
    ----------
    symbol: str
        Target symbol for the signal.
    strength: float
        Continuous signal in [-1, 1]: -1 short, +1 long.
    source: str
        Which component originated the signal (e.g., "hybrid_manager").
    meta: Optional[Dict[str, Any]]
        Optional metadata such as sub-signal contributions.
    """

    symbol: str
    strength: float
    source: str
    meta: Optional[Dict[str, Any]] = None


@dataclass(frozen=True)
class OrderEvent(Event):
    """An order intent produced by an execution policy.

    Attributes
    ----------
    symbol: str
        Symbol to trade.
    side: str
        "buy" or "sell".
    quantity: float
        Number of units or position weight if interpreted by the broker.
    order_type: str
        "market" or "limit". For simplicity default to market.
    limit_price: Optional[float]
        Price for limit orders, if used.
    """

    symbol: str
    side: str
    quantity: float
    order_type: str = "market"
    limit_price: Optional[float] = None


@dataclass(frozen=True)
class FillEvent(Event):
    """A confirmation of order execution from the broker/backtester.

    Attributes
    ----------
    symbol: str
        Symbol filled.
    filled_qty: float
        Executed quantity.
    fill_price: float
        Execution price.
    timestamp: datetime
        Fill time.
    commission: float
        Commission or fees paid.
    """

    symbol: str
    filled_qty: float
    fill_price: float
    timestamp: datetime
    commission: float = 0.0

