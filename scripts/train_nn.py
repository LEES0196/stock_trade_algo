#!/usr/bin/env python3
"""Train the simple NN classifier on historical prices and save weights.

Usage:
  python scripts/train_nn.py --config config/base.yaml

It will respect the `model` section for lookback, time_interval,
and model_path. If `data.cache_dir` is set and writable, prices will
be cached by the strategy downloader.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml

import pandas as pd

from src.strategy.data import download_prices
from src.strategy.nn_predictor import train_or_load_model, _parse_time_interval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/base.yaml", help="Path to YAML config")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    uni = cfg.get("universe", {})
    tickers = list(uni.get("aggressive", [])) + list(uni.get("passive", []))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})

    prices = download_prices(tickers, **data_cfg)

    lookback = int(model_cfg.get("lookback_window", 60))
    horizon_steps = _parse_time_interval(str(model_cfg.get("time_interval", "5d")), str(data_cfg.get("interval", "1d")))
    path = model_cfg.get("model_path", "data/models/nn_model.pkl")

    print(f"Training NN on {len(prices.columns)} tickers, lookback={lookback}, horizon_steps={horizon_steps}")
    model = train_or_load_model(prices, lookback, horizon_steps, path, train=True, epochs=args.epochs)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    main()

