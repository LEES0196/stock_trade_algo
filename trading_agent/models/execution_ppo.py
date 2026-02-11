"""
PPO-based execution policy using stable-baselines3.

We define a trading environment with observation space comprising
(price, volume, account_balance, position, returns, volatility) and a continuous
action space in [-1, 1] representing target position weight.

Training
--------
Run: python -m trading_agent.models.execution_ppo --train

Inference
---------
The PPOExecutor class loads a trained model and provides suggest_weight().
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

try:  # Optional heavy imports
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
except Exception:  # pragma: no cover - import guard
    gym = None  # type: ignore
    spaces = None  # type: ignore
    PPO = None  # type: ignore
    DummyVecEnv = None  # type: ignore
    EvalCallback = None  # type: ignore
    BaseCallback = None  # type: ignore


# Default model save path
MODEL_DIR = Path(__file__).parent.parent / "trained_models"
MODEL_PATH = MODEL_DIR / "ppo_executor.zip"


@dataclass
class TradingEnvConfig:
    """Configuration for the trading environment."""
    initial_balance: float = 100000.0
    transaction_cost: float = 0.001  # 0.1% per trade
    max_position: float = 1.0  # Maximum position as fraction of portfolio
    reward_scaling: float = 1.0
    episode_length: int = 252  # ~1 trading year


class TradingEnvironment(gym.Env):
    """
    A realistic trading environment for training PPO execution policy.

    Observation Space (6 features):
        - normalized_price: price / initial_price
        - normalized_volume: volume / mean_volume
        - normalized_balance: balance / initial_balance
        - current_position: position weight in [-1, 1]
        - recent_return: last period return
        - volatility: rolling volatility

    Action Space:
        - Continuous [-1, 1]: target position weight
          -1 = full short, 0 = no position, 1 = full long

    Reward:
        - PnL from position change minus transaction costs
        - Scaled by reward_scaling factor
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        price_data: np.ndarray,
        volume_data: np.ndarray,
        config: Optional[TradingEnvConfig] = None,
    ) -> None:
        assert spaces is not None, "gymnasium not available"
        super().__init__()

        self.config = config or TradingEnvConfig()
        self.price_data = price_data
        self.volume_data = volume_data
        self.n_steps = len(price_data)

        # Precompute statistics for normalization
        self.initial_price = price_data[0]
        self.mean_volume = np.mean(volume_data) + 1e-8

        # Precompute returns and volatility
        self.returns = np.zeros(self.n_steps)
        self.returns[1:] = (price_data[1:] - price_data[:-1]) / (price_data[:-1] + 1e-8)

        # Rolling volatility (20-period)
        self.volatility = np.zeros(self.n_steps)
        for i in range(20, self.n_steps):
            self.volatility[i] = np.std(self.returns[i-20:i])

        # Observation: [norm_price, norm_volume, norm_balance, position, return, volatility]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1, -1, 0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        # Action: target position weight in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.position = 0.0  # Current position weight
        self.portfolio_value = self.config.initial_balance
        self.entry_price = 0.0

    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        price = self.price_data[self.current_step]
        volume = self.volume_data[self.current_step]

        obs = np.array([
            price / self.initial_price,  # normalized price
            volume / self.mean_volume,    # normalized volume
            self.portfolio_value / self.config.initial_balance,  # normalized portfolio
            self.position,                # current position
            np.clip(self.returns[self.current_step], -1, 1),  # recent return
            np.clip(self.volatility[self.current_step], 0, 1),  # volatility
        ], dtype=np.float32)

        return obs

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Random start point (leave room for episode)
        max_start = max(20, self.n_steps - self.config.episode_length - 1)
        self.current_step = self.np_random.integers(20, max_start) if max_start > 20 else 20
        self.start_step = self.current_step

        # Reset portfolio state
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.portfolio_value = self.config.initial_balance
        self.entry_price = self.price_data[self.current_step]

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment."""
        target_position = float(np.clip(action[0], -1, 1))

        current_price = self.price_data[self.current_step]

        # Calculate position change and transaction cost
        position_change = target_position - self.position
        transaction_cost = abs(position_change) * self.portfolio_value * self.config.transaction_cost

        # Move to next step
        self.current_step += 1
        next_price = self.price_data[self.current_step]

        # Calculate PnL from position
        price_return = (next_price - current_price) / (current_price + 1e-8)
        position_pnl = target_position * self.portfolio_value * price_return

        # Update portfolio value
        old_value = self.portfolio_value
        self.portfolio_value = self.portfolio_value + position_pnl - transaction_cost
        self.position = target_position

        # Reward: log return of portfolio value (encourages growth, penalizes drawdown)
        reward = np.log(self.portfolio_value / old_value + 1e-8) * self.config.reward_scaling

        # Episode termination
        steps_taken = self.current_step - self.start_step
        terminated = False
        truncated = (
            steps_taken >= self.config.episode_length or
            self.current_step >= self.n_steps - 1 or
            self.portfolio_value <= 0.1 * self.config.initial_balance  # Bankruptcy
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "position": self.position,
            "transaction_cost": transaction_cost,
            "pnl": position_pnl,
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        """Print current state."""
        print(f"Step {self.current_step}: Price={self.price_data[self.current_step]:.2f}, "
              f"Position={self.position:.2f}, Portfolio=${self.portfolio_value:.2f}")


class ProgressCallback(BaseCallback):
    """Callback to print training progress."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get recent episode rewards if available
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep["r"] for ep in self.model.ep_info_buffer])
                mean_length = np.mean([ep["l"] for ep in self.model.ep_info_buffer])
                print(f"Step {self.n_calls}: Mean Reward={mean_reward:.4f}, Mean Length={mean_length:.0f}")
        return True


def create_env_from_data(
    price_data: np.ndarray,
    volume_data: np.ndarray,
    config: Optional[TradingEnvConfig] = None
) -> TradingEnvironment:
    """Factory function to create trading environment."""
    return TradingEnvironment(price_data, volume_data, config)


def load_training_data(
    symbol: str = "AAPL",
    start: str = "2015-01-01",
    end: str = "2023-12-31",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load historical data for training."""
    from ..data.ingestion import IngestionConfig, fetch_yfinance

    cfg = IngestionConfig(symbol=symbol, start=start, end=end, interval="1d")
    df = fetch_yfinance(cfg)

    price_data = df["close"].values.astype(np.float32).flatten()
    volume_data = df["volume"].values.astype(np.float32).flatten()

    return price_data, volume_data


def train_ppo(
    symbol: str = "AAPL",
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    total_timesteps: int = 100000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    gamma: float = 0.99,
    save_path: Optional[Path] = None,
    verbose: int = 1,
) -> PPO:
    """
    Train a PPO agent for trade execution.

    Parameters
    ----------
    symbol : str
        Stock ticker symbol
    start, end : str
        Date range for training data
    total_timesteps : int
        Total training steps
    learning_rate : float
        PPO learning rate
    n_steps : int
        Number of steps per rollout
    batch_size : int
        Minibatch size for updates
    gamma : float
        Discount factor
    save_path : Path
        Where to save the trained model
    verbose : int
        Verbosity level

    Returns
    -------
    PPO
        Trained PPO model
    """
    if PPO is None:
        raise RuntimeError("stable-baselines3 not available")

    print(f"Loading training data for {symbol} from {start} to {end}...")
    price_data, volume_data = load_training_data(symbol, start, end)
    print(f"Loaded {len(price_data)} data points")

    # Create environment
    env_config = TradingEnvConfig(
        initial_balance=100000.0,
        transaction_cost=0.001,
        episode_length=252,
    )

    def make_env():
        return create_env_from_data(price_data, volume_data, env_config)

    # Vectorized environment for stable-baselines3
    vec_env = DummyVecEnv([make_env])

    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        verbose=verbose,
    )

    # Training callback
    callback = ProgressCallback(check_freq=5000, verbose=verbose)

    # Train
    print(f"Starting training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # Save model
    save_path = save_path or MODEL_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    print(f"Model saved to {save_path}")

    return model


def evaluate_model(
    model: PPO,
    symbol: str = "AAPL",
    start: str = "2024-01-01",
    end: str = "2024-12-31",
    n_episodes: int = 5,
) -> dict:
    """
    Evaluate a trained PPO model on held-out data.

    Returns
    -------
    dict
        Evaluation metrics including mean return, Sharpe ratio, etc.
    """
    print(f"Loading evaluation data for {symbol} from {start} to {end}...")
    price_data, volume_data = load_training_data(symbol, start, end)

    env_config = TradingEnvConfig(
        initial_balance=100000.0,
        transaction_cost=0.001,
        episode_length=min(252, len(price_data) - 21),
    )
    env = create_env_from_data(price_data, volume_data, env_config)

    episode_returns = []
    episode_sharpes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        daily_returns = []
        prev_value = env.config.initial_balance

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track daily return
            curr_value = info["portfolio_value"]
            daily_ret = (curr_value - prev_value) / prev_value
            daily_returns.append(daily_ret)
            prev_value = curr_value

        # Calculate metrics
        total_return = (env.portfolio_value - env.config.initial_balance) / env.config.initial_balance
        episode_returns.append(total_return)

        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
        else:
            sharpe = 0.0
        episode_sharpes.append(sharpe)

        print(f"Episode {ep+1}: Return={total_return:.2%}, Sharpe={sharpe:.2f}, "
              f"Final Value=${env.portfolio_value:.2f}")

    results = {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_sharpe": np.mean(episode_sharpes),
        "std_sharpe": np.std(episode_sharpes),
    }

    print(f"\nEvaluation Summary:")
    print(f"  Mean Return: {results['mean_return']:.2%} +/- {results['std_return']:.2%}")
    print(f"  Mean Sharpe: {results['mean_sharpe']:.2f} +/- {results['std_sharpe']:.2f}")

    return results


class PPOExecutor:
    """Wrapper around stable-baselines3 PPO for trade execution sizing.

    Methods
    -------
    suggest_weight(price, volume, balance): float
        Given market state, return target weight in [-1, 1].
    load(path): Load a trained model
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model = None
        self._last_position = 0.0

        # Try to load trained model
        path = model_path or MODEL_PATH
        if path.exists():
            self.load(path)

    def load(self, path: Path) -> None:
        """Load a trained PPO model."""
        if PPO is None:
            print("Warning: stable-baselines3 not available, using heuristic fallback")
            return
        try:
            self.model = PPO.load(str(path))
            print(f"Loaded PPO model from {path}")
        except Exception as e:
            print(f"Warning: Could not load model from {path}: {e}")
            self.model = None

    def load_or_init(self, env: TradingEnvironment) -> None:
        """Initialize a new PPO model (for training)."""
        if PPO is None:
            self.model = None
            return
        if self.model is None:
            self.model = PPO("MlpPolicy", env, verbose=0)

    def suggest_weight(
        self,
        price: float,
        volume: float,
        balance: float,
        position: float = 0.0,
        recent_return: float = 0.0,
        volatility: float = 0.0,
    ) -> float:
        """Return a continuous weight in [-1, 1].

        Parameters
        ----------
        price : float
            Current price
        volume : float
            Current volume
        balance : float
            Current portfolio value
        position : float
            Current position weight
        recent_return : float
            Recent price return
        volatility : float
            Recent volatility

        Returns
        -------
        float
            Target position weight in [-1, 1]
        """
        if self.model is None:
            # Fallback heuristic: momentum-based with risk scaling
            momentum = np.sign(recent_return) * min(abs(recent_return) * 10, 0.5)
            vol_scale = max(0.1, 1.0 - volatility * 10)  # Reduce position in high vol
            weight = float(np.clip(momentum * vol_scale, -1.0, 1.0))
            return weight

        # Construct observation (normalized)
        obs = np.array([[
            price / 100.0,  # Rough normalization
            volume / 1e7,   # Rough normalization
            balance / 100000.0,
            position,
            np.clip(recent_return, -1, 1),
            np.clip(volatility, 0, 1),
        ]], dtype=np.float32)

        action, _ = self.model.predict(obs, deterministic=True)
        return float(np.clip(action[0], -1.0, 1.0))


def main():
    """CLI entry point for training and evaluation."""
    parser = argparse.ArgumentParser(description="PPO Execution Agent Training")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate existing model")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol")
    parser.add_argument("--train-start", type=str, default="2015-01-01", help="Training start date")
    parser.add_argument("--train-end", type=str, default="2023-12-31", help="Training end date")
    parser.add_argument("--eval-start", type=str, default="2024-01-01", help="Evaluation start date")
    parser.add_argument("--eval-end", type=str, default="2024-12-31", help="Evaluation end date")
    parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--model-path", type=str, default=None, help="Model save/load path")

    args = parser.parse_args()

    model_path = Path(args.model_path) if args.model_path else MODEL_PATH

    if args.train:
        model = train_ppo(
            symbol=args.symbol,
            start=args.train_start,
            end=args.train_end,
            total_timesteps=args.timesteps,
            learning_rate=args.lr,
            save_path=model_path,
        )

        if args.evaluate:
            print("\n" + "="*50)
            print("Running evaluation on held-out data...")
            evaluate_model(
                model,
                symbol=args.symbol,
                start=args.eval_start,
                end=args.eval_end,
            )

    elif args.evaluate:
        if not model_path.exists():
            print(f"Error: No model found at {model_path}. Train first with --train")
            return

        model = PPO.load(str(model_path))
        evaluate_model(
            model,
            symbol=args.symbol,
            start=args.eval_start,
            end=args.eval_end,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
