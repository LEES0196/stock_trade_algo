"""
FinBERT sentiment wrapper.

This module wraps a transformers pipeline to score news/text sentiment for a
given symbol. In constrained environments without model weights, it falls back
to a rule-based sentiment on keywords.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

try:  # Optional heavy import
    from transformers import pipeline
except Exception:  # pragma: no cover - import guard
    pipeline = None  # type: ignore


@dataclass
class SentimentScore:
    positive: float
    neutral: float
    negative: float


class FinBERTSentiment:
    """FinBERT sentiment analysis with safe fallback.

    Returns a dict-like score in [0,1] that sums to 1 across classes.
    """

    def __init__(self) -> None:
        self._pipe = None
        if pipeline is not None:  # pragma: no cover - transformers dependency
            try:
                self._pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            except Exception:
                self._pipe = None

    def score(self, texts: List[str]) -> SentimentScore:
        if self._pipe is None:
            # Simple keyword fallback
            text = " ".join(texts).lower()
            pos = sum(word in text for word in ["beat", "surge", "record", "bullish", "upgrade"]) * 1.0
            neg = sum(word in text for word in ["miss", "plunge", "downgrade", "bearish", "fraud"]) * 1.0
            total = pos + neg + 1e-6
            return SentimentScore(positive=pos / total, neutral=1.0 / total * 0.0, negative=neg / total)

        out = self._pipe(texts)  # pragma: no cover - transformers dependency
        # Aggregate probabilities across items
        pos = sum(d["score"] for d in out if d["label"].lower().startswith("positive"))
        neg = sum(d["score"] for d in out if d["label"].lower().startswith("negative"))
        neutral = sum(d["score"] for d in out if d["label"].lower().startswith("neutral"))
        total = max(pos + neg + neutral, 1e-6)
        return SentimentScore(positive=pos / total, neutral=neutral / total, negative=neg / total)

