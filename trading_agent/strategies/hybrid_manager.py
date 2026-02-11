"""
Hybrid Manager: Ensemble voting across ML and statistical signals.

Voting Logic
------------
- Inputs: TFT forecast, FinBERT sentiment, Moving Average signal.
- If realized volatility > threshold, ignore ML (TFT, FinBERT) and rely on
  mean reversion (statistical signal), which is often more robust in high
  volatility regimes.

Mathematical Note
-----------------
Let s_t^TFT in [-1,1], s_t^BERT in [-1,1] (positive sentiment bullish), and
  s_t^MA in [-1,1] (e.g., price above/below SMA). Define volatility v_t.
Decision rule:
  if v_t > v_thr: s_t = s_t^MA
  else: s_t = median(s_t^TFT, s_t^BERT, s_t^MA)
Clipping to [-1,1] ensures bounded action space for downstream risk.
"""

from __future__ import annotations

from statistics import median
from typing import Dict


def moving_average_signal(price: float, sma: float) -> float:
    """Return mean-reversion signal in [-1,1] from price vs SMA.

    s = sign(sma - price): if price above SMA => short tilt, else long.
    """

    if sma == 0:
        return 0.0
    raw = (sma - price) / max(abs(sma), 1e-6)
    return float(max(-1.0, min(1.0, raw)))


def hybrid_vote(tft_signal: float, bert_signal: float, ma_signal: float, volatility: float, vol_threshold: float) -> float:
    """Combine signals with volatility-aware gating.

    Parameters
    ----------
    tft_signal, bert_signal, ma_signal : float
        Individual signals in [-1,1].
    volatility : float
        Realized volatility estimate.
    vol_threshold : float
        Threshold above which ML signals are ignored.

    Returns
    -------
    float
        Final ensemble signal in [-1,1].
    """

    if volatility > vol_threshold:
        return float(max(-1.0, min(1.0, ma_signal)))
    s = median([tft_signal, bert_signal, ma_signal])
    return float(max(-1.0, min(1.0, s)))

