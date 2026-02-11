"""
Main orchestrator for the Autonomous Algorithmic Trading Agent.

Pipeline (Event-Driven)
-----------------------
1) Fetch Data -> publish MarketEvent
2) Strategy consumes MarketEvent -> compute ensemble SignalEvent
3) Risk Guardrail checks -> allow/deny
4) Execution sizing -> OrderEvent
5) (Future) Broker/backtest returns FillEvent -> update state

This file wires modules together with a minimal simulated loop. It does not
attempt to download data or train models at import time.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from .core.event_bus import EventBus
from .core.events import MarketEvent, SignalEvent, OrderEvent, FillEvent
from .models.alpha_tft import TFTModel
from .models.sentiment_bert import FinBERTSentiment
from .models.execution_ppo import PPOExecutor
from .strategies.hybrid_manager import moving_average_signal, hybrid_vote
from .strategies.risk_guardrail import PortfolioState, check_risk
from .backtest.sim_broker import SimBroker
from .data.ingestion import IngestionConfig, fetch_yfinance
from .data.processor import add_technical_indicators


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


async def main() -> None:
    cfg = load_config(Path(__file__).with_name("config").joinpath("config.yaml"))

    bus = EventBus()

    # Initialize models
    tft = TFTModel(in_features=3)
    bert = FinBERTSentiment()
    exec_agent = PPOExecutor()

    # Portfolio state (toy)
    portfolio = PortfolioState(equity=cfg.get("capital", 100000.0), peak_equity=cfg.get("capital", 100000.0))
    broker = SimBroker(bus=bus, initial_cash=cfg.get("capital", 100000.0))

    # Handlers
    async def on_market(event: MarketEvent) -> None:
        """Handle new market data: compute ensemble signal and publish it."""

        price = float(event.data.get("close", 0.0))
        volume = float(event.data.get("volume", 0.0))
        sma = float(event.data.get("sma", price))
        vol = float(event.data.get("volatility", 0.0))

        # Signals
        tft_sig = np.tanh((tft.predict(np.array([[price, volume, sma]], dtype=float)) - price) / max(price, 1e-6))
        bert_score = bert.score(["placeholder headline for sentiment"])  # toy
        bert_sig = float(bert_score.positive - bert_score.negative)
        ma_sig = moving_average_signal(price, sma)

        final_sig = hybrid_vote(
            tft_signal=float(tft_sig),
            bert_signal=bert_sig,
            ma_signal=ma_sig,
            volatility=vol,
            vol_threshold=float(cfg.get("volatility_threshold", 0.02)),
        )

        meta = {"price": price, "volume": volume}
        await bus.publish(SignalEvent(symbol=event.symbol, strength=final_sig, source="hybrid_manager", meta=meta))

    async def on_signal(event: SignalEvent) -> None:
        """Risk check and execution sizing -> publish OrderEvent if allowed."""

        # Convert signal to side and size via executor
        side = "buy" if event.strength >= 0 else "sell"
        weight = exec_agent.suggest_weight(price=event.meta.get("price", 0.0) if event.meta else 0.0,
                                           volume=event.meta.get("volume", 0.0) if event.meta else 0.0,
                                           balance=portfolio.equity)
        order = OrderEvent(symbol=event.symbol, side=side, quantity=abs(weight))

        allowed = check_risk(order, portfolio, max_drawdown=float(cfg.get("max_drawdown", 0.2)))
        if allowed:
            print(f"[EXECUTE] {order.side} {order.quantity:.3f} {order.symbol}")
            await bus.publish(order)
        else:
            print("[BLOCKED] Risk guardrail blocked the trade due to drawdown.")

    async def on_fill(event: FillEvent) -> None:
        # Log fills and show updated portfolio equity from broker state
        print(f"[FILL] {event.symbol} qty={event.filled_qty:.4f} @ {event.fill_price:.2f} fee={event.commission:.2f}")
        # Sync local portfolio view with broker's state
        portfolio.equity = broker.portfolio_state.equity
        portfolio.peak_equity = broker.portfolio_state.peak_equity
        dd = portfolio.drawdown
        print(f"[PORTFOLIO] equity={portfolio.equity:.2f} peak={portfolio.peak_equity:.2f} dd={dd:.2%}")

    # Subscriptions
    bus.subscribe(MarketEvent, on_market)
    bus.subscribe(SignalEvent, on_signal)
    bus.subscribe(FillEvent, on_fill)

    # YFinance-driven feed replacing simulation
    async def feed_yfinance() -> None:
        dcfg = cfg.get("data", {})
        icfg = IngestionConfig(
            symbol=str(dcfg.get("symbol", "AAPL")),
            start=dcfg.get("start"),
            end=dcfg.get("end"),
            interval=str(dcfg.get("interval", "1d")),
        )
        df = fetch_yfinance(icfg)
        df = add_technical_indicators(df)
        for ts, row in df.iterrows():
            data = {
                "close": float(row.get("close", 0.0)),
                "volume": float(row.get("volume", 0.0)),
                "sma": float(row.get("sma", float(row.get("close", 0.0)))),
                "volatility": float(row.get("volatility", 0.0)),
            }
            # Use timestamp from data
            ts_py = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else datetime.utcnow()
            await bus.publish(MarketEvent(symbol=icfg.symbol, timestamp=ts_py, data=data))

    # Run dispatcher and feed concurrently
    runner = asyncio.create_task(bus.run())
    await feed_yfinance()
    # Give the bus time to process all events
    await asyncio.sleep(0.5)
    bus.stop()
    await asyncio.sleep(0.1)
    runner.cancel()
    try:
        await runner
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(main())
