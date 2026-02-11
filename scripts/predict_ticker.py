#!/usr/bin/env python3
"""Predict single-ticker price movement probability using the NN predictor.

Examples:
  python scripts/predict_ticker.py --ticker NVDA --period 5y --interval 1d \
      --time-interval 5d --lookback 60 --confidence 0.7 --train

If --train is omitted, the script will attempt to load a model from --model-path.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.strategy.data import download_prices
from src.strategy.nn_predictor import (
    train_or_load_model,
    predict_probabilities,
    _parse_time_interval,
)


def main():
    ap = argparse.ArgumentParser(description="Predict ticker movement with NN")
    ap.add_argument("--ticker", required=True, help="Ticker symbol, e.g., NVDA")
    ap.add_argument("--period", default="5y", help="yfinance period, e.g., 1y, 5y, max")
    ap.add_argument("--interval", default="1d", help="yfinance interval, e.g., 1d, 1h")
    ap.add_argument("--time-interval", default="5d", help="Forecast horizon relative to --interval, e.g., 5d")
    ap.add_argument("--lookback", type=int, default=60, help="Number of past steps for features")
    ap.add_argument("--confidence", type=float, default=0.7, help="Confidence threshold for favorable classification")
    ap.add_argument(
        "--model-path",
        default="data/models/nn_model.pkl",
        help="Path to save/load model weights",
    )
    ap.add_argument("--train", action="store_true", help="Train the model on this ticker's history before predicting")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs when --train is set")
    args = ap.parse_args()

    ticker = args.ticker.upper().strip()

    # Download historical prices
    prices = download_prices([ticker], period=args.period, interval=args.interval)

    # Determine horizon in steps and train/load model
    horizon_steps = _parse_time_interval(args.time_interval, args.interval)
    model = train_or_load_model(
        prices,
        lookback_window=args.lookback,
        horizon_steps=horizon_steps,
        model_path=args.model_path,
        train=args.train,
        epochs=args.epochs,
    )

    probs = predict_probabilities(model, prices, lookback_window=args.lookback)
    p = float(probs.get(ticker, 0.0))
    favorable = p >= float(args.confidence)

    print("Ticker:", ticker)
    print("Period:", args.period, "Interval:", args.interval)
    print("Lookback:", args.lookback, "Horizon steps:", horizon_steps)
    print("Probability positive:", f"{p:.4f}")
    print("Confidence threshold:", f"{args.confidence:.2f}")
    print("Decision:", "FAVORABLE" if favorable else "UNFAVORABLE")


if __name__ == "__main__":
    main()

