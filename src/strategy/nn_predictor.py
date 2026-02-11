"""Neural network–based stock movement predictor (pure NumPy).

This module provides a lightweight, dependency-free neural predictor
that can be trained on historical prices to classify whether the
cumulative return over a future horizon (``time_interval``) is positive.

It exposes helpers to:
- build training samples from price data
- train a small MLP (one-hidden-layer) via Adam
- predict per-ticker probabilities for the latest window
- derive an allocation based on a confidence threshold

The implementation avoids heavy ML frameworks to keep the project
portable, while remaining "well trainable" for small/medium datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import math
import os
import pickle

import numpy as np
import pandas as pd


# -------------------------
# Utility parsing and math
# -------------------------

def _parse_time_interval(time_interval: str, data_interval: str) -> int:
    """Convert a time interval (e.g., "5d") and data interval (e.g., "1d") to steps.

    Supports daily inputs. If units differ or are missing, falls back to integer.
    """
    def _to_steps(token: str) -> Tuple[int, str]:
        token = token.strip().lower()
        num = ""
        unit = ""
        for ch in token:
            if ch.isdigit():
                num += ch
            else:
                unit += ch
        n = int(num) if num else 1
        unit = unit or "d"
        return n, unit

    n_h, u_h = _to_steps(time_interval)
    n_d, u_d = _to_steps(data_interval)
    # Only handle same units for simplicity; otherwise approximate to days
    to_days = {"d": 1, "day": 1, "days": 1, "wk": 5, "w": 5, "mo": 21, "m": 21}
    h_days = n_h * to_days.get(u_h, 1)
    d_days = n_d * to_days.get(u_d, 1)
    steps = max(1, h_days // d_days)
    return steps


def _log_returns(prices: pd.Series) -> np.ndarray:
    s = prices.dropna().astype(float)
    return np.log(s.values[1:] / s.values[:-1])


# -------------------------
# Simple MLP binary classifier
# -------------------------

@dataclass
class MLPBinaryConfig:
    input_dim: int
    hidden_dim: int = 32
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    seed: int = 42


class MLPBinary:
    def __init__(self, cfg: MLPBinaryConfig):
        rng = np.random.default_rng(cfg.seed)
        self.cfg = cfg
        self.W1 = rng.normal(0, 0.1, size=(cfg.input_dim, cfg.hidden_dim))
        self.b1 = np.zeros((cfg.hidden_dim,))
        self.W2 = rng.normal(0, 0.1, size=(cfg.hidden_dim, 1))
        self.b2 = np.zeros((1,))
        # Adam states
        self.mW1 = np.zeros_like(self.W1)
        self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1)
        self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2)
        self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2)
        self.vb2 = np.zeros_like(self.b2)
        self.t = 0

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # clamp for stability
        x = np.clip(x, -30, 30)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        Z1 = X @ self.W1 + self.b1
        A1 = self._relu(Z1)
        logits = A1 @ self.W2 + self.b2  # shape (N,1)
        return logits, (X, Z1, A1)

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        # y shape (N,), values in {0,1}
        logits, cache = self.forward(X)
        p = self._sigmoid(logits).reshape(-1)
        # Binary cross-entropy
        eps = 1e-12
        loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        # Gradients
        N = X.shape[0]
        dlogits = (p - y).reshape(-1, 1) / N  # shape (N,1)
        Xc, Z1, A1 = cache
        dW2 = A1.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dA1 = dlogits @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = Xc.T @ dZ1
        db1 = dZ1.sum(axis=0)
        return loss, (dW1, db1, dW2, db2)

    def _adam_update(self, W, dW, m, v):
        self.t += 1
        b1, b2, eps, lr = self.cfg.beta1, self.cfg.beta2, self.cfg.eps, self.cfg.lr
        m[:] = b1 * m + (1 - b1) * dW
        v[:] = b2 * v + (1 - b2) * (dW * dW)
        m_hat = m / (1 - b1 ** self.t)
        v_hat = v / (1 - b2 ** self.t)
        W[:] = W - lr * m_hat / (np.sqrt(v_hat) + eps)

    def step(self, grads):
        dW1, db1, dW2, db2 = grads
        self._adam_update(self.W1, dW1, self.mW1, self.vW1)
        self._adam_update(self.b1, db1, self.mb1, self.vb1)
        self._adam_update(self.W2, dW2, self.mW2, self.vW2)
        self._adam_update(self.b2, db2, self.mb2, self.vb2)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 30, batch_size: int = 128, verbose: bool = False):
        N = X.shape[0]
        for ep in range(1, epochs + 1):
            idx = np.random.permutation(N)
            Xs, ys = X[idx], y[idx]
            batches = math.ceil(N / batch_size)
            ep_loss = 0.0
            for b in range(batches):
                s = b * batch_size
                e = min(N, s + batch_size)
                loss, grads = self.loss_and_grads(Xs[s:e], ys[s:e])
                ep_loss += loss * (e - s)
                self.step(grads)
            if verbose:
                print(f"[MLP] epoch {ep:03d} loss={ep_loss / N:.4f}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(X)
        return self._sigmoid(logits).reshape(-1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "cfg": self.cfg,
                    "W1": self.W1,
                    "b1": self.b1,
                    "W2": self.W2,
                    "b2": self.b2,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "MLPBinary":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        model = cls(obj["cfg"])  # type: ignore[arg-type]
        model.W1 = obj["W1"]
        model.b1 = obj["b1"]
        model.W2 = obj["W2"]
        model.b2 = obj["b2"]
        return model


# -------------------------
# Dataset preparation
# -------------------------

def build_samples(
    prices: pd.DataFrame,
    lookback_window: int,
    horizon_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct features X and labels y across all tickers.

    X: concatenated windows of past ``lookback_window`` log-returns
    y: 1 if sum of next ``horizon_steps`` log-returns > 0 else 0
    """
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for col in prices.columns:
        r = _log_returns(prices[col])
        if len(r) < lookback_window + horizon_steps + 1:
            continue
        # rolling windows
        for i in range(lookback_window, len(r) - horizon_steps):
            past = r[i - lookback_window : i]
            fut = r[i : i + horizon_steps]
            Xs.append(past)
            ys.append(1.0 if fut.sum() > 0 else 0.0)
    if not Xs:
        return np.zeros((0, lookback_window), dtype=float), np.zeros((0,), dtype=float)
    X = np.vstack(Xs)
    y = np.array(ys, dtype=float)
    # standardize features for stability
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - mu) / sigma
    return X, y


def latest_windows(prices: pd.DataFrame, lookback_window: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for col in prices.columns:
        r = _log_returns(prices[col])
        if len(r) >= lookback_window:
            w = r[-lookback_window:]
            # simple standardization by window stats
            mu = w.mean()
            sd = w.std() + 1e-8
            out[col] = ((w - mu) / sd).astype(float)
    return out


# -------------------------
# Allocation via NN
# -------------------------

def train_or_load_model(
    prices: pd.DataFrame,
    lookback_window: int,
    horizon_steps: int,
    model_path: Optional[str],
    train: bool,
    hidden_dim: int = 32,
    epochs: int = 30,
) -> MLPBinary:
    """Train a model or load from disk if available and training is disabled."""
    if (not train) and model_path and os.path.exists(model_path):
        return MLPBinary.load(model_path)

    X, y = build_samples(prices, lookback_window, horizon_steps)
    if X.shape[0] == 0:
        # Fallback: tiny dummy model
        model = MLPBinary(MLPBinaryConfig(input_dim=lookback_window, hidden_dim=hidden_dim))
        return model
    model = MLPBinary(MLPBinaryConfig(input_dim=lookback_window, hidden_dim=hidden_dim))
    model.fit(X, y, epochs=epochs, batch_size=256, verbose=False)
    if model_path:
        try:
            model.save(model_path)
        except Exception:
            # ignore persistence errors (e.g., read-only env)
            pass
    return model


def predict_probabilities(
    model: MLPBinary,
    prices: pd.DataFrame,
    lookback_window: int,
) -> Dict[str, float]:
    """Return per-ticker probability of positive future return."""
    windows = latest_windows(prices, lookback_window)
    probs: Dict[str, float] = {}
    if not windows:
        return probs
    names = list(windows.keys())
    X = np.vstack([windows[n] for n in names])
    probs_vec = model.predict_proba(X)
    for n, p in zip(names, probs_vec):
        probs[n] = float(p)
    return probs


def nn_allocation(
    prices: pd.DataFrame,
    aggressive: list[str],
    passive: list[str],
    data_interval: str,
    time_interval: str,
    lookback_window: int,
    confidence_threshold: float,
    cash_ticker: str,
    model_path: Optional[str] = None,
    train: bool = False,
) -> Dict[str, float]:
    """Compute allocation using NN-based predictions.

    Strategy:
    - Train or load an MLP classifier on all tickers
    - Compute probability that each ticker's future cumulative return over
      ``time_interval`` is positive
    - Select tickers whose probability >= confidence_threshold; equal-weight them
    - If none meet the threshold:
        * if any have p > 0.5, pick the top one
        * else allocate fully to cash
    Caps and blending can be applied by the caller if desired.
    """
    horizon_steps = _parse_time_interval(time_interval, data_interval)
    universe = list(dict.fromkeys(list(aggressive) + list(passive)))
    # Restrict prices to universe
    df = prices.loc[:, prices.columns.intersection(universe)]
    # Train/load model
    model = train_or_load_model(
        df, lookback_window=lookback_window, horizon_steps=horizon_steps, model_path=model_path, train=train
    )
    probs = predict_probabilities(model, df, lookback_window)
    # Filter to only those present
    probs = {k: v for k, v in probs.items() if k in universe}
    # Selection
    selected = [k for k, v in probs.items() if v >= confidence_threshold]
    alloc: Dict[str, float] = {}
    if selected:
        w = 1.0 / len(selected)
        for k in selected:
            alloc[k] = w
        return alloc
    # No one passed the threshold; pick top if p>0.5
    if probs:
        best = max(probs.items(), key=lambda kv: kv[1])
        if best[1] > 0.5:
            alloc[best[0]] = 1.0
            return alloc
    # Fallback to cash
    alloc[cash_ticker] = 1.0
    return alloc

