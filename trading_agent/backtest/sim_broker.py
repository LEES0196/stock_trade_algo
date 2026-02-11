"""
Simple backtesting broker that executes orders at the latest market price and
emits FillEvents. Maintains portfolio state (cash, positions, equity) used by
risk guardrails.

Execution Model
---------------
- Price: executes at last seen close price with optional slippage.
- Sizing: interprets OrderEvent.quantity as target weight in [0, 1], with side
  determining sign (buy=+weight, sell=-weight). Trades to reach target weight.
- Cash/Positions: updates cash and shares; allows shorting for simplicity.

Mathematical Notes
------------------
Equity_t = Cash_t + sum_i(Price_{t,i} * Shares_{t,i})
Drawdown_t = max(0, 1 - Equity_t / PeakEquity_t)
TradeValue = SharesFilled * FillPrice; Cash_{t+} = Cash_t - TradeValue - Commission(|TradeValue|)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict

import numpy as np

from ..core.event_bus import EventBus
from ..core.events import MarketEvent, OrderEvent, FillEvent
from ..strategies.risk_guardrail import PortfolioState


@dataclass
class SimBroker:
    bus: EventBus
    initial_cash: float = 100000.0
    commission_rate: float = 0.0  # proportional commission on trade value
    slippage_bps: float = 0.0     # slippage in basis points
    cash: float = field(init=False)
    positions: Dict[str, float] = field(default_factory=dict)  # symbol -> shares
    last_price: Dict[str, float] = field(default_factory=dict)
    portfolio_state: PortfolioState = field(init=False)

    def __post_init__(self) -> None:
        self.cash = float(self.initial_cash)
        self.portfolio_state = PortfolioState(equity=self.initial_cash, peak_equity=self.initial_cash)
        # Subscriptions
        self.bus.subscribe(MarketEvent, self._on_market)
        self.bus.subscribe(OrderEvent, self._on_order)

    async def _on_market(self, event: MarketEvent) -> None:
        price = float(event.data.get("close", 0.0))
        self.last_price[event.symbol] = price
        # Mark-to-market equity update
        equity = self.cash
        for sym, qty in self.positions.items():
            px = self.last_price.get(sym, price if sym == event.symbol else 0.0)
            equity += qty * px
        self.portfolio_state.equity = float(equity)
        self.portfolio_state.peak_equity = float(max(self.portfolio_state.peak_equity, equity))

    async def _on_order(self, event: OrderEvent) -> None:
        symbol = event.symbol
        if symbol not in self.last_price:
            return  # cannot execute without a price
        price = self.last_price[symbol]
        # Interpret quantity as target weight; side sets sign
        target_weight = float(np.clip(event.quantity, 0.0, 1.0))
        target_weight = +target_weight if event.side.lower() == "buy" else -target_weight

        # Current state
        equity = max(self.portfolio_state.equity, 1e-8)
        current_qty = float(self.positions.get(symbol, 0.0))
        current_weight = (current_qty * price) / equity
        delta_weight = target_weight - current_weight

        # Desired trade in shares
        desired_dollars = delta_weight * equity
        desired_qty = desired_dollars / max(price, 1e-8)
        if desired_qty == 0:
            return

        # Apply slippage
        slip = self.slippage_bps * 1e-4
        direction = 1.0 if desired_qty > 0 else -1.0
        fill_price = price * (1.0 + direction * slip)

        # Cash constraint for buys (ignore for shorts for simplicity)
        if desired_qty > 0:
            max_affordable_qty = self.cash / (fill_price * (1.0 + self.commission_rate) + 1e-12)
            trade_qty = float(min(desired_qty, max_affordable_qty))
        else:
            trade_qty = float(desired_qty)

        if trade_qty == 0.0:
            return

        trade_value = trade_qty * fill_price
        commission = self.commission_rate * abs(trade_value)

        # Update state
        self.cash -= trade_value + commission
        self.positions[symbol] = current_qty + trade_qty

        # Publish FillEvent
        await self.bus.publish(
            FillEvent(
                symbol=symbol,
                filled_qty=trade_qty,
                fill_price=fill_price,
                timestamp=datetime.utcnow(),
                commission=commission,
            )
        )

