"""
Streamlit Web UI for Autonomous Trading Agent.

Run: streamlit run trading_agent/webui.py
Or:  python -m streamlit run trading_agent/webui.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add parent directory to path for imports when running directly
_this_file = Path(__file__).resolve()
_project_root = _this_file.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import trading agent components (absolute imports)
from trading_agent.data.ingestion import IngestionConfig, fetch_yfinance
from trading_agent.data.processor import add_technical_indicators
from trading_agent.models.execution_ppo import (
    PPOExecutor,
    TradingEnvironment,
    TradingEnvConfig,
    train_ppo,
    evaluate_model,
    load_training_data,
    MODEL_PATH,
)
from trading_agent.portfolio.optimizer import optimize_weights
from trading_agent.portfolio.rebalancer import needs_rebalance
from trading_agent.strategies.hybrid_manager import moving_average_signal, hybrid_vote
from trading_agent.models.alpha_tft import TFTModel
from trading_agent.models.sentiment_bert import FinBERTSentiment

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


# Page configuration
st.set_page_config(
    page_title="Trading Agent Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .profit {
        color: #00c853;
    }
    .loss {
        color: #ff1744;
    }
</style>
""", unsafe_allow_html=True)


def load_data(symbol: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Load and process market data."""
    cfg = IngestionConfig(symbol=symbol, start=start, end=end, interval=interval)
    df = fetch_yfinance(cfg)
    df = add_technical_indicators(df)
    return df


@st.cache_resource
def get_models():
    """Load ML models (cached)."""
    tft = TFTModel(in_features=3)
    bert = FinBERTSentiment()
    ppo = PPOExecutor()
    return tft, bert, ppo


def create_price_chart(df: pd.DataFrame, symbol: str) -> go.Figure:
    """Create interactive price chart with indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{symbol} Price', 'Volume', 'Volatility')
    )

    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        ),
        row=1, col=1
    )

    # SMA
    if 'sma' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['sma'], name='SMA(20)', line=dict(color='orange')),
            row=1, col=1
        )

    # Volume
    colors = ['red' if row['open'] > row['close'] else 'green' for _, row in df.iterrows()]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )

    # Volatility
    if 'volatility' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['volatility'], name='Volatility', fill='tozeroy'),
            row=3, col=1
        )

    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def create_portfolio_chart(history: List[Dict]) -> go.Figure:
    """Create portfolio value chart."""
    if not history:
        return go.Figure()

    df = pd.DataFrame(history)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=('Portfolio Value', 'Position')
    )

    fig.add_trace(
        go.Scatter(
            x=df['step'],
            y=df['portfolio_value'],
            name='Portfolio Value',
            fill='tozeroy',
            line=dict(color='blue')
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Bar(
            x=df['step'],
            y=df['position'],
            name='Position',
            marker_color=['green' if p > 0 else 'red' for p in df['position']]
        ),
        row=2, col=1
    )

    fig.update_layout(height=500, showlegend=True)
    return fig


def page_home():
    """Home/Overview page."""
    st.title("📈 Autonomous Trading Agent Dashboard")

    st.markdown("""
    Welcome to the Trading Agent Dashboard. This system provides:

    - **Real-time Market Analysis**: View price charts, technical indicators, and market signals
    - **PPO Training**: Train reinforcement learning models for trade execution
    - **Backtesting**: Test strategies on historical data
    - **Portfolio Optimization**: Optimize multi-asset portfolios using Sharpe maximization
    - **Live Simulation**: Run simulated trading with trained models
    """)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        model_exists = MODEL_PATH.exists()
        st.metric("PPO Model", "Trained" if model_exists else "Not Trained")

    with col2:
        st.metric("Strategy", "Hybrid (TFT + BERT + MA)")

    with col3:
        st.metric("Risk Management", "Max Drawdown Guard")

    with col4:
        st.metric("Execution", "PPO-Based Sizing")

    st.divider()

    # Quick market overview
    st.subheader("Quick Market View")

    col1, col2 = st.columns([1, 3])

    with col1:
        symbol = st.text_input("Symbol", value="AAPL")
        days = st.slider("Days", 30, 365, 90)

    with col2:
        if st.button("Load Data", type="primary"):
            with st.spinner("Loading market data..."):
                end = datetime.now().strftime("%Y-%m-%d")
                start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

                try:
                    df = load_data(symbol, start, end)
                    st.session_state['market_data'] = df
                    st.session_state['symbol'] = symbol
                    st.success(f"Loaded {len(df)} data points for {symbol}")
                except Exception as e:
                    st.error(f"Error loading data: {e}")

    if 'market_data' in st.session_state:
        df = st.session_state['market_data']
        symbol = st.session_state.get('symbol', 'AAPL')

        # Display metrics
        col1, col2, col3, col4 = st.columns(4)

        current_price = float(df['close'].iloc[-1])
        prev_price = float(df['close'].iloc[-2]) if len(df) > 1 else current_price
        change = (current_price - prev_price) / prev_price * 100

        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{change:+.2f}%")
        with col2:
            st.metric("Volume", f"{int(df['volume'].iloc[-1]):,}")
        with col3:
            volatility = float(df['volatility'].iloc[-1]) if 'volatility' in df.columns else 0
            st.metric("Volatility", f"{volatility:.4f}")
        with col4:
            sma = float(df['sma'].iloc[-1]) if 'sma' in df.columns else current_price
            signal = "BUY" if current_price > sma else "SELL"
            st.metric("MA Signal", signal)

        # Chart
        fig = create_price_chart(df, symbol)
        st.plotly_chart(fig, use_container_width=True)


def page_training():
    """PPO Training page."""
    st.title("🎯 PPO Model Training")

    st.markdown("""
    Train a PPO (Proximal Policy Optimization) agent for trade execution sizing.
    The agent learns to determine optimal position weights based on market conditions.
    """)

    # Training configuration
    st.subheader("Training Configuration")

    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input("Training Symbol", value="AAPL")
        train_start = st.date_input("Training Start", value=datetime(2015, 1, 1))
        train_end = st.date_input("Training End", value=datetime(2023, 12, 31))

    with col2:
        timesteps = st.number_input("Training Timesteps", min_value=10000, max_value=1000000, value=50000, step=10000)
        learning_rate = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-2, value=3e-4, format="%.5f")

    st.subheader("Evaluation Configuration")
    col1, col2 = st.columns(2)

    with col1:
        eval_start = st.date_input("Evaluation Start", value=datetime(2024, 1, 1))
        eval_end = st.date_input("Evaluation End", value=datetime(2024, 12, 31))

    with col2:
        run_eval = st.checkbox("Run evaluation after training", value=True)

    # Training button
    if st.button("Start Training", type="primary"):
        if PPO is None:
            st.error("stable-baselines3 is not installed. Please install it first.")
            return

        progress_bar = st.progress(0)
        status_text = st.empty()

        with st.spinner("Training in progress..."):
            status_text.text("Loading training data...")
            progress_bar.progress(10)

            try:
                # Training
                status_text.text("Training PPO model...")
                progress_bar.progress(20)

                model = train_ppo(
                    symbol=symbol,
                    start=train_start.strftime("%Y-%m-%d"),
                    end=train_end.strftime("%Y-%m-%d"),
                    total_timesteps=timesteps,
                    learning_rate=learning_rate,
                    verbose=0,
                )

                progress_bar.progress(80)
                st.success(f"Training completed! Model saved to {MODEL_PATH}")

                # Evaluation
                if run_eval:
                    status_text.text("Evaluating model...")
                    results = evaluate_model(
                        model,
                        symbol=symbol,
                        start=eval_start.strftime("%Y-%m-%d"),
                        end=eval_end.strftime("%Y-%m-%d"),
                        n_episodes=5,
                    )

                    progress_bar.progress(100)
                    status_text.text("Done!")

                    # Display results
                    st.subheader("Evaluation Results")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Return", f"{results['mean_return']:.2%}")
                    with col2:
                        st.metric("Std Return", f"{results['std_return']:.2%}")
                    with col3:
                        st.metric("Mean Sharpe", f"{results['mean_sharpe']:.2f}")
                    with col4:
                        st.metric("Std Sharpe", f"{results['std_sharpe']:.2f}")

            except Exception as e:
                st.error(f"Training failed: {e}")
                progress_bar.progress(0)

    # Model status
    st.divider()
    st.subheader("Model Status")

    if MODEL_PATH.exists():
        st.success(f"Trained model available at: {MODEL_PATH}")

        # Load and test
        if st.button("Test Model Prediction"):
            executor = PPOExecutor()
            weight = executor.suggest_weight(
                price=150.0,
                volume=50000000,
                balance=100000,
                position=0.0,
                recent_return=0.01,
                volatility=0.02
            )
            st.info(f"Sample prediction: Target position weight = {weight:.4f}")
    else:
        st.warning("No trained model found. Please train a model first.")


def page_backtest():
    """Backtesting page."""
    st.title("📊 Backtesting")

    st.markdown("""
    Run backtests on historical data to evaluate strategy performance.
    """)

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input("Symbol", value="AAPL", key="bt_symbol")
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1), key="bt_start")
        end_date = st.date_input("End Date", value=datetime(2023, 12, 31), key="bt_end")

    with col2:
        initial_capital = st.number_input("Initial Capital", value=100000, step=10000)
        transaction_cost = st.number_input("Transaction Cost (%)", value=0.1, step=0.01) / 100
        use_trained_model = st.checkbox("Use Trained PPO Model", value=True)

    if st.button("Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                # Load data
                df = load_data(
                    symbol,
                    start_date.strftime("%Y-%m-%d"),
                    end_date.strftime("%Y-%m-%d")
                )

                # Initialize
                tft, bert, ppo = get_models()
                if not use_trained_model:
                    ppo.model = None  # Use heuristic

                # Backtest loop
                portfolio_value = initial_capital
                position = 0.0
                history = []

                progress = st.progress(0)

                for i, (ts, row) in enumerate(df.iterrows()):
                    progress.progress((i + 1) / len(df))

                    price = float(row['close'])
                    volume = float(row['volume'])
                    sma = float(row.get('sma', price))
                    vol = float(row.get('volatility', 0.0))
                    ret = float(row.get('return', 0.0))

                    # Get signal
                    tft_sig = np.tanh((tft.predict(np.array([[price, volume, sma]])) - price) / max(price, 1e-6))
                    ma_sig = moving_average_signal(price, sma)

                    final_sig = hybrid_vote(
                        tft_signal=float(tft_sig),
                        bert_signal=0.0,  # Skip BERT for speed
                        ma_signal=ma_sig,
                        volatility=vol,
                        vol_threshold=0.02,
                    )

                    # Get position size from PPO
                    target_position = ppo.suggest_weight(
                        price=price,
                        volume=volume,
                        balance=portfolio_value,
                        position=position,
                        recent_return=ret,
                        volatility=vol
                    )

                    # Adjust based on signal direction
                    target_position = target_position * np.sign(final_sig) if final_sig != 0 else 0

                    # Calculate PnL
                    if i > 0:
                        prev_price = float(df.iloc[i-1]['close'])
                        price_return = (price - prev_price) / prev_price
                        pnl = position * portfolio_value * price_return
                        cost = abs(target_position - position) * portfolio_value * transaction_cost
                        portfolio_value = portfolio_value + pnl - cost

                    position = target_position

                    history.append({
                        'step': i,
                        'date': ts,
                        'price': price,
                        'portfolio_value': portfolio_value,
                        'position': position,
                        'signal': final_sig
                    })

                progress.progress(100)

                # Results
                st.subheader("Backtest Results")

                total_return = (portfolio_value - initial_capital) / initial_capital

                history_df = pd.DataFrame(history)
                daily_returns = history_df['portfolio_value'].pct_change().dropna()
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
                max_dd = (history_df['portfolio_value'] / history_df['portfolio_value'].cummax() - 1).min()

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Return", f"{total_return:.2%}")
                with col2:
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{max_dd:.2%}")
                with col4:
                    st.metric("Final Value", f"${portfolio_value:,.2f}")

                # Charts
                fig = create_portfolio_chart(history)
                st.plotly_chart(fig, use_container_width=True)

                # Trade log
                st.subheader("Trade History")
                st.dataframe(history_df.tail(20), use_container_width=True)

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def page_portfolio():
    """Portfolio Optimization page."""
    st.title("💼 Portfolio Optimization")

    st.markdown("""
    Optimize portfolio weights using Mean-Variance optimization (Sharpe Maximization).
    """)

    # Asset selection
    st.subheader("Asset Selection")

    default_assets = "AAPL,MSFT,GOOGL,AMZN,META"
    assets_input = st.text_input("Assets (comma-separated)", value=default_assets)
    assets = [a.strip().upper() for a in assets_input.split(",")]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=datetime(2023, 1, 1), key="po_start")
    with col2:
        end_date = st.date_input("End Date", value=datetime(2023, 12, 31), key="po_end")

    if st.button("Optimize Portfolio", type="primary"):
        with st.spinner("Loading data and optimizing..."):
            try:
                # Load returns for all assets
                returns_dict = {}

                progress = st.progress(0)
                for i, asset in enumerate(assets):
                    progress.progress((i + 1) / len(assets))
                    cfg = IngestionConfig(
                        symbol=asset,
                        start=start_date.strftime("%Y-%m-%d"),
                        end=end_date.strftime("%Y-%m-%d"),
                        interval="1d"
                    )
                    df = fetch_yfinance(cfg)
                    returns_dict[asset] = df['close'].pct_change().dropna()

                # Combine into DataFrame
                returns_df = pd.DataFrame(returns_dict).dropna()

                # Optimize
                optimal_weights = optimize_weights(returns_df)

                # Display results
                st.subheader("Optimal Weights")

                col1, col2 = st.columns(2)

                with col1:
                    # Pie chart
                    fig = px.pie(
                        values=optimal_weights.values,
                        names=optimal_weights.index,
                        title="Portfolio Allocation"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Bar chart
                    fig = px.bar(
                        x=optimal_weights.index,
                        y=optimal_weights.values,
                        title="Weight Distribution"
                    )
                    fig.update_layout(xaxis_title="Asset", yaxis_title="Weight")
                    st.plotly_chart(fig, use_container_width=True)

                # Weights table
                st.subheader("Allocation Details")
                weights_df = pd.DataFrame({
                    'Asset': optimal_weights.index,
                    'Weight': optimal_weights.values,
                    'Weight (%)': optimal_weights.values * 100
                })
                st.dataframe(weights_df, use_container_width=True)

                # Portfolio stats
                st.subheader("Portfolio Statistics")

                portfolio_returns = (returns_df * optimal_weights).sum(axis=1)

                col1, col2, col3 = st.columns(3)
                with col1:
                    ann_return = portfolio_returns.mean() * 252
                    st.metric("Expected Annual Return", f"{ann_return:.2%}")
                with col2:
                    ann_vol = portfolio_returns.std() * np.sqrt(252)
                    st.metric("Annual Volatility", f"{ann_vol:.2%}")
                with col3:
                    sharpe = ann_return / ann_vol if ann_vol > 0 else 0
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")

                # Cumulative returns chart
                fig = px.line(
                    x=portfolio_returns.index,
                    y=(1 + portfolio_returns).cumprod(),
                    title="Cumulative Portfolio Returns"
                )
                fig.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return")
                st.plotly_chart(fig, use_container_width=True)

                # Rebalancing check
                st.subheader("Rebalancing Check")

                # Simulate current weights (random drift)
                current = {asset: w + np.random.uniform(-0.05, 0.05) for asset, w in optimal_weights.items()}
                total = sum(current.values())
                current = {k: v/total for k, v in current.items()}

                threshold = st.slider("Rebalancing Threshold", 0.01, 0.20, 0.05)

                target = dict(optimal_weights)
                needs_rebal = needs_rebalance(current, target, threshold)

                if needs_rebal:
                    st.warning("Rebalancing needed! Current weights deviate from target.")
                else:
                    st.success("Portfolio is within tolerance. No rebalancing needed.")

                # Show comparison
                compare_df = pd.DataFrame({
                    'Asset': list(current.keys()),
                    'Current': list(current.values()),
                    'Target': [target[a] for a in current.keys()],
                    'Deviation': [current[a] - target[a] for a in current.keys()]
                })
                st.dataframe(compare_df, use_container_width=True)

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                import traceback
                st.code(traceback.format_exc())


def page_simulation():
    """Live Trading Simulation page."""
    st.title("🎮 Live Trading Simulation")

    st.markdown("""
    Run a live trading simulation with the trained models.
    This simulates trading decisions in real-time on recent market data.
    """)

    # Configuration
    col1, col2 = st.columns(2)

    with col1:
        symbol = st.text_input("Symbol", value="AAPL", key="sim_symbol")
        initial_capital = st.number_input("Initial Capital", value=100000, step=10000, key="sim_capital")

    with col2:
        speed = st.slider("Simulation Speed (seconds per step)", 0.1, 2.0, 0.5)
        show_signals = st.checkbox("Show Signal Details", value=True)

    # Initialize session state
    if 'sim_running' not in st.session_state:
        st.session_state.sim_running = False
    if 'sim_history' not in st.session_state:
        st.session_state.sim_history = []

    col1, col2, col3 = st.columns(3)

    with col1:
        start_btn = st.button("Start Simulation", type="primary", disabled=st.session_state.sim_running)
    with col2:
        stop_btn = st.button("Stop Simulation", disabled=not st.session_state.sim_running)
    with col3:
        reset_btn = st.button("Reset")

    if reset_btn:
        st.session_state.sim_history = []
        st.session_state.sim_running = False
        st.rerun()

    if stop_btn:
        st.session_state.sim_running = False
        st.rerun()

    if start_btn:
        st.session_state.sim_running = True
        st.session_state.sim_history = []

        # Load recent data
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")

        try:
            df = load_data(symbol, start, end)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            st.session_state.sim_running = False
            st.stop()

        # Initialize models
        tft, bert, ppo = get_models()

        # Simulation state
        portfolio_value = initial_capital
        position = 0.0

        # UI elements
        metrics_placeholder = st.empty()
        chart_placeholder = st.empty()
        signals_placeholder = st.empty()
        log_placeholder = st.empty()

        # Run simulation
        for i, (ts, row) in enumerate(df.iterrows()):
            if not st.session_state.sim_running:
                break

            price = float(row['close'])
            volume = float(row['volume'])
            sma = float(row.get('sma', price))
            vol = float(row.get('volatility', 0.0))
            ret = float(row.get('return', 0.0))

            # Get signals
            tft_sig = float(np.tanh((tft.predict(np.array([[price, volume, sma]])) - price) / max(price, 1e-6)))
            ma_sig = moving_average_signal(price, sma)

            final_sig = hybrid_vote(
                tft_signal=tft_sig,
                bert_signal=0.0,
                ma_signal=ma_sig,
                volatility=vol,
                vol_threshold=0.02,
            )

            # Get position from PPO
            target_position = ppo.suggest_weight(
                price=price,
                volume=volume,
                balance=portfolio_value,
                position=position,
                recent_return=ret,
                volatility=vol
            )

            target_position = target_position * np.sign(final_sig) if final_sig != 0 else 0

            # Calculate PnL
            if i > 0:
                prev_price = float(df.iloc[i-1]['close'])
                price_return = (price - prev_price) / prev_price
                pnl = position * portfolio_value * price_return
                portfolio_value = portfolio_value + pnl

            position = target_position

            # Record history
            st.session_state.sim_history.append({
                'step': i,
                'date': ts,
                'price': price,
                'portfolio_value': portfolio_value,
                'position': position,
                'signal': final_sig,
                'tft_signal': tft_sig,
                'ma_signal': ma_sig
            })

            # Update UI
            with metrics_placeholder.container():
                col1, col2, col3, col4 = st.columns(4)
                pnl_pct = (portfolio_value - initial_capital) / initial_capital * 100

                with col1:
                    st.metric("Portfolio Value", f"${portfolio_value:,.2f}", f"{pnl_pct:+.2f}%")
                with col2:
                    st.metric("Current Price", f"${price:.2f}")
                with col3:
                    st.metric("Position", f"{position:.2f}")
                with col4:
                    st.metric("Signal", f"{final_sig:.3f}")

            with chart_placeholder.container():
                if len(st.session_state.sim_history) > 1:
                    fig = create_portfolio_chart(st.session_state.sim_history)
                    st.plotly_chart(fig, use_container_width=True, key=f"sim_chart_{i}")

            if show_signals:
                with signals_placeholder.container():
                    st.write(f"**Step {i}** | TFT: {tft_sig:.3f} | MA: {ma_sig:.3f} | Final: {final_sig:.3f}")

            time.sleep(speed)

        st.session_state.sim_running = False
        st.success("Simulation completed!")

        # Final results
        if st.session_state.sim_history:
            final_value = st.session_state.sim_history[-1]['portfolio_value']
            total_return = (final_value - initial_capital) / initial_capital

            history_df = pd.DataFrame(st.session_state.sim_history)
            daily_returns = history_df['portfolio_value'].pct_change().dropna()
            sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 and daily_returns.std() > 0 else 0

            st.subheader("Simulation Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", f"{total_return:.2%}")
            with col2:
                st.metric("Final Value", f"${final_value:,.2f}")
            with col3:
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")


def main():
    """Main application."""
    # Sidebar navigation
    st.sidebar.title("Navigation")

    pages = {
        "Home": page_home,
        "PPO Training": page_training,
        "Backtesting": page_backtest,
        "Portfolio Optimization": page_portfolio,
        "Live Simulation": page_simulation,
    }

    selection = st.sidebar.radio("Go to", list(pages.keys()))

    # Model status in sidebar
    st.sidebar.divider()
    st.sidebar.subheader("Model Status")

    if MODEL_PATH.exists():
        st.sidebar.success("PPO Model: Trained")
    else:
        st.sidebar.warning("PPO Model: Not trained")

    # Run selected page
    pages[selection]()


if __name__ == "__main__":
    main()
