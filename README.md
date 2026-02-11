# Autonomous Algorithmic Trading Agent (EDA Scaffold)

This repository provides a modular, event-driven scaffold for an autonomous algorithmic trading agent. It separates ingestion, modeling, strategy, risk, execution, and portfolio optimization into focused components to avoid a monolithic script.

## Project Structure

```
/trading_agent
    /config
        config.yaml          # User inputs: Max Drawdown, Capital, Risk Tolerance
    /core
        events.py            # Event classes (MarketEvent, SignalEvent, OrderEvent, FillEvent)
        event_bus.py         # Async Event Bus/Queue handler (pub/sub)
    /data
        ingestion.py         # YFinance wrapper
        processor.py         # Feature Engineering (SMA, Volatility)
    /models
        alpha_tft.py         # TFTModel: inference-only placeholder
        sentiment_bert.py    # FinBERT wrapper with keyword fallback
        execution_ppo.py     # PPOExecutor + custom Gym env
    /strategies
        hybrid_manager.py    # Ensemble voting (ML + MA) with volatility gating
        risk_guardrail.py    # Max Drawdown guardrail
    /portfolio
        rebalancer.py        # Threshold-based rebalance trigger
        optimizer.py         # Riskfolio-Lib Sharpe maximization
    main_agent.py            # Orchestrator wiring the event loop (simulated feed + backtest broker)
    /backtest
        sim_broker.py       # Simple broker executing orders and emitting FillEvents
```

## Event-Driven Flow

- MarketEvent: New market state x_t (OHLCV + features)
- Strategy: Produce SignalEvent s_t in [-1, 1]
- Risk: Block or allow based on drawdown constraint
- Execution: Map signal to OrderEvent (target weight/side)
- Fill: (Future) Confirm execution, update PnL/drawdown state

Core bus: `trading_agent/core/event_bus.py` implements async pub/sub by event type.

## Key Components

- Core
  - `trading_agent/core/events.py`: Base `Event` + `MarketEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`.
  - `trading_agent/core/event_bus.py`: Asyncio queue dispatcher with type subscriptions.
- Data
  - `trading_agent/data/ingestion.py`: `yfinance` fetch wrapper.
  - `trading_agent/data/processor.py`: Adds SMA and realized volatility features.
- Models
  - `trading_agent/models/alpha_tft.py`: `TFTModel.predict(lookback)` — inference-only placeholder (falls back to naive last price if torch unavailable).
  - `trading_agent/models/sentiment_bert.py`: FinBERT sentiment with safe keyword fallback when models aren’t available.
  - `trading_agent/models/execution_ppo.py`: `PPOExecutor` and a custom Gym env with obs = (price, volume, balance), action = target weight in [-1, 1]. Falls back to a heuristic if SB3 is unavailable.
- Strategies
  - `trading_agent/strategies/hybrid_manager.py`: Voting logic: if volatility > threshold, ignore ML and rely on mean reversion (MA). Else take median of TFT, FinBERT, and MA signals.
  - `trading_agent/strategies/risk_guardrail.py`: `check_risk(order)` blocks trades when current drawdown > `config.max_drawdown`.
- Portfolio
  - `trading_agent/portfolio/optimizer.py`: Riskfolio-Lib Sharpe maximization with equal-weight fallback.
  - `trading_agent/portfolio/rebalancer.py`: Rebalance trigger when `|w - w*| > threshold`.
- Orchestration
  - `trading_agent/main_agent.py`: Loads `config.yaml`, initializes EventBus and components, and simulates a small feed to demonstrate the event flow.

## Configuration

Edit `trading_agent/config/config.yaml`:

- `capital`: starting equity (e.g., 100000)
- `max_drawdown`: max allowed drawdown (e.g., 0.2 = 20%)
- `risk_tolerance`: qualitative flag
- `volatility_threshold`: gating for the ensemble (ignore ML above this)
- `rebalancing_threshold`: drift band for rebalancing
- `ppo`: PPO hyperparameters (learning rate, gamma, etc.)
- `data`: symbol, lookback, interval, start/end for ingestion

### Parameter Details

- `capital` (float)
  - Units: quote currency of the instrument (e.g., USD for US equities like AAPL from yfinance).
  - Meaning: initial cash used by the backtest broker to size trades and compute equity/PnL.
  - Typical values: 10_000 to 1_000_000.

- `max_drawdown` (float in [0, 1])
  - Purpose: hard risk constraint; blocks orders once current drawdown exceeds this threshold.
  - Calculation: drawdown = max(0, 1 - equity / peak_equity).
  - Typical range: 0.1–0.3 (10–30%). Lower is stricter.

- `risk_tolerance` (string)
  - Values: low | medium | high (free-form).
  - Current use: informational placeholder for future risk scaling (e.g., volatility target, position caps).

- `volatility_threshold` (float)
  - Units: realized volatility estimate on the return series (per bar).
  - Role: volatility gating in the hybrid manager; if vol > threshold, ignore ML (TFT/BERT) and rely on mean reversion.
  - Typical daily values: 0.01–0.05 (1–5%); tune to bar interval.

- `rebalancing_threshold` (float in [0, 1])
  - Role: threshold-band policy; trigger rebalance when |w_current − w_target| > threshold for any asset.
  - Typical range: 0.02–0.10. Larger bands reduce churn; smaller bands track targets closely.

- `ppo` (mapping)
  - `learning_rate`: step size for PPO optimizer (e.g., 3e-4).
  - `gamma`: discount factor in [0,1] (e.g., 0.99).
  - `n_steps`: rollout length per update (e.g., 2048).
  - `batch_size`: minibatch size for updates (e.g., 64).
  - Note: PPO training is not wired in the orchestrator yet; used when adding a training routine.

- `data` (mapping)
  - `symbol`: yfinance ticker (e.g., AAPL, MSFT).
  - `lookback`: reserved window length for models (not enforced in the orchestrator loop yet).
  - `interval`: yfinance bar size (e.g., 1d, 1h, 1wk, 1mo). Intraday history may be limited by provider.
  - `start`, `end`: ISO dates YYYY-MM-DD defining the data window. Omit `end` to fetch up to the most recent.

Currency Note
-------------
- All monetary quantities (capital, cash, equity, PnL, commission) are expressed in the instrument’s quote currency. For US equities, this is USD.

## Installation

- Install Python 3.10+
- `pip install -r requirements.txt`
  - Heavy deps are optional at runtime; modules include safe fallbacks:
    - TFT: falls back to naive forecast
    - FinBERT: falls back to keyword sentiment
    - PPO/SB3: falls back to heuristic sizing

## Run (YFinance + Backtest)

- `python -m trading_agent.main_agent`
  - Downloads historical data via `yfinance` using `trading_agent/data/ingestion.py` and generates features via `processor.py`.
  - Event flow: MarketEvent (per bar) -> SignalEvent -> Risk check -> OrderEvent -> FillEvent.
  - Backtesting broker executes at bar close with optional slippage/commission and updates equity/drawdown.

## Mathematical Notes (Brief)

- Voting: `s_t = median(s_TFT, s_BERT, s_MA)` if volatility ≤ threshold, else `s_t = s_MA`.
- Drawdown: `DD_t = max(0, 1 - equity_t / peak_equity_t)`; block if `DD_t > MDD_max`.
- Rebalancing: trigger when `|w_i - w_i*| > δ` (threshold/band policy).
- Optimization: maximize Sharpe `((μᵀw - r_f) / √(wᵀΣw))` with Riskfolio-Lib.

## Next Steps

- Replace the simulated feed with `yfinance` ingestion and `processor` features.
- Add a broker/backtester to emit `FillEvent` and update portfolio state.
- Integrate training loops for TFT and PPO; persist models and configs.
- Expand risk (position limits, volatility targeting), logging, and monitoring.
