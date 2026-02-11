"""
PPO-based execution policy using stable-baselines3.

We define a minimal custom environment with observation space comprising
(price, volume, account_balance) and a continuous action space in [-1, 1]
representing target position weight. The RL training loop is omitted, but the
environment and agent wrapper are provided to support integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:  # Optional heavy imports
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
except Exception:  # pragma: no cover - import guard
    gym = None  # type: ignore
    spaces = None  # type: ignore
    PPO = None  # type: ignore


@dataclass
class ExecEnvState:
    price: float
    volume: float
    balance: float


class ExecutionEnv(gym.Env):  # type: ignore[misc]
    """Toy environment for execution sizing.

    Observation: np.array([price, volume, balance])
    Action: target weight in [-1, 1]

    Reward (placeholder): zero unless overridden during training; the purpose
    here is to provide the interface for PPO.
    """

    metadata = {"render_modes": []}

    def __init__(self, initial_state: ExecEnvState) -> None:  # pragma: no cover - sb3 dependency
        assert spaces is not None
        super().__init__()
        self.state = initial_state
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._step_count = 0

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):  # type: ignore[override]
        super().reset(seed=seed)
        self._step_count = 0
        obs = np.array([self.state.price, self.state.volume, self.state.balance], dtype=np.float32)
        return obs, {}

    def step(self, action: np.ndarray):  # type: ignore[override]
        # Placeholder dynamics: no actual environment progression
        self._step_count += 1
        reward = 0.0
        terminated = False
        truncated = self._step_count >= 1
        info = {"action": float(action[0])}
        obs = np.array([self.state.price, self.state.volume, self.state.balance], dtype=np.float32)
        return obs, reward, terminated, truncated, info


class PPOExecutor:
    """Wrapper around stable-baselines3 PPO for trade execution sizing.

    Methods
    -------
    suggest_weight(obs): float
        Given (price, volume, balance), return target weight in [-1, 1].
    """

    def __init__(self) -> None:
        self.model = None

    def load_or_init(self, env: ExecutionEnv) -> None:  # pragma: no cover - sb3 dependency
        if PPO is None:
            self.model = None
            return
        if self.model is None:
            self.model = PPO("MlpPolicy", env, verbose=0)

    def suggest_weight(self, price: float, volume: float, balance: float) -> float:
        """Return a continuous weight in [-1, 1]. If PPO unavailable, use heuristic.

        Heuristic: proportional to normalized momentum sign with balance cap.
        """

        if self.model is None:  # Fallback heuristic
            # Simple placeholder: prefer smaller weights when balance is low
            norm_balance = np.tanh(balance / 1e5)
            weight = float(np.clip(0.1 * norm_balance, -1.0, 1.0))
            return weight

        obs = np.array([[price, volume, balance]], dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)  # pragma: no cover - sb3 dependency
        return float(np.clip(action[0], -1.0, 1.0))

