"""
Temporal Fusion Transformer (TFT) wrapper for alpha generation.

We provide an inference-only skeleton that accepts lookback data and produces
next-step price forecasts. The training loop is intentionally omitted and can
be integrated via PyTorch Lightning in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # Optional heavy imports
    import torch
    from torch import nn
except Exception:  # pragma: no cover - import guard
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class TFTConfig:
    lookback: int = 60
    hidden_size: int = 64
    dropout: float = 0.1


class SimpleTFT(nn.Module):  # type: ignore[misc]
    """A tiny placeholder mimicking TFT-style encoder-decoder behavior.

    This is NOT a full TFT. It simply demonstrates an architecture that maps a
    sequence of features to a scalar forecast using gated residual blocks.
    """

    def __init__(self, in_features: int, cfg: TFTConfig) -> None:  # pragma: no cover - lightweight stub
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, cfg.hidden_size),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_size, 1),
        )

    def forward(self, x):  # x: (batch, features)
        return self.net(x)


class TFTModel:
    """Inference-only TFT wrapper.

    Methods
    -------
    predict(lookback_data): np.ndarray
        Returns a 1-step-ahead forecast for price given recent features.

    Mathematical Note
    -----------------
    Given feature matrix X_{t-L+1:t} we produce \hat{y}_{t+1} = f_theta(X).
    Here f_theta is a shallow NN placeholder for the full TFT.
    """

    def __init__(self, in_features: int, cfg: Optional[TFTConfig] = None) -> None:
        self.cfg = cfg or TFTConfig()
        if torch is None or nn is None:
            self.model = None
        else:  # pragma: no cover - requires torch
            self.model = SimpleTFT(in_features, self.cfg)
            self.model.eval()

    def predict(self, lookback_data: np.ndarray) -> float:
        """Produce a scalar price forecast from lookback features.

        Parameters
        ----------
        lookback_data : np.ndarray
            Array of shape (L, F) or (F,) with recent features.

        Returns
        -------
        float
            A price delta or level forecast (placeholder returns level).
        """

        if self.model is None:
            # Fallback deterministic baseline: last price as naive forecast
            x = np.asarray(lookback_data)
            last_price = float(x[-1, 0]) if x.ndim == 2 else float(x[0])
            return last_price

        x = np.asarray(lookback_data)
        if x.ndim == 2:
            x = x[-1]  # simple last-step summarization for the stub
        x_t = torch.from_numpy(x.astype("float32")).unsqueeze(0)
        with torch.no_grad():  # pragma: no cover - requires torch
            y = self.model(x_t).squeeze(0).squeeze(-1)
            return float(y.item())

