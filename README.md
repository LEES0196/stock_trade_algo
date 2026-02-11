# Auto Portfolio Strategy (Baseline)

This repository contains a simple, modular Python implementation of a momentum‑tilt investment strategy. It is built from the high‑level design outlined in the provided PDF document and exposes a small Dash (Plotly) application so non‑technical users can run the strategy through a graphical interface.

**Important disclaimer:**  The code in this repository is for educational and research purposes only and does **not** constitute financial advice.  It does not take into account your personal financial circumstances and should not be used to make live trading decisions without consulting a qualified professional.

## Repository structure

```
auto_portfolio/
├── config/
│   └── base.yaml           # Default configuration describing tickers, lookbacks and weights
├── requirements.txt        # Minimal Python dependencies
├── src/
│   ├── app/
│   │   ├── __init__.py
│   │   └── dash_app.py     # Dash UI for running the strategy from a YAML config
│   └── strategy/
│       ├── __init__.py
│       ├── data.py         # Download and cache price data using yfinance
│       ├── features.py     # Build monthly return features
│       ├── algo_rules.py   # Algorithm 1: rule‑based aggressive/passive logic
│       ├── algo_scores.py  # Algorithm 2: score‑based logic
│       ├── blend.py        # Combine the two allocation schemes
│       ├── allocate.py     # Orchestrate the full pipeline
│       └── io.py           # I/O helper for saving decisions to CSV
└── README.md               # This file
```

## Getting started

1. **Install dependencies**.  Use a fresh virtual environment and install the packages listed in `requirements.txt`:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Adjust the configuration**. The file `config/base.yaml` defines the universe of tickers, lookback periods and blending weights. Edit this YAML file to suit your needs; you can copy it to a new file and paste it into the Dash UI as text.

3. **Run the Dash app**. Launch the interface by executing:

   ```bash
   python -m src.app.dash_app
   ```

   Your browser opens a Dash page. You can now configure the strategy in two ways:
   - Use the form controls (recommended): pick period/interval, edit tickers, and adjust numeric options with up/down arrows (e.g., Passive top N). Click "Run (Form)".
   - Advanced: paste/edit the full YAML and click "Run from YAML".

4. **Unit tests (optional)**.  You can add your own tests under a `tests/` folder and run them with pytest.  Writing tests helps you verify that modifications to the strategy behave as expected.

## Extending the strategy

This baseline is intentionally minimal to make it easy to experiment.  To extend or modify the strategy:

* **Change the universe** by editing the `aggressive`, `passive` and `score_set` lists in the YAML config.
* **Modify lookbacks or weights** using the form controls (months, passive_top_n with arrows, cash_score slider, blend) or by editing the YAML directly.
* **Implement different algorithms** by editing `algo_rules.py` or `algo_scores.py`.  Each function accepts a pandas DataFrame of monthly returns and should return a dictionary mapping tickers to weights.
* **Add constraints** such as maximum weight caps by modifying the `blend()` function or adding new parameters.

Always test your changes on historical data and consider the risks of any strategy before using it in practice.

## Neural network predictor (optional)

You can switch from the baseline momentum allocation to a neural network–based predictor that classifies whether the cumulative return over a future horizon will be positive, then allocates across assets that meet a confidence threshold.

- Enable it by adding a `model` section in `config/base.yaml` (already included):

  - `model.type`: set to `"nn"` to enable the neural path
  - `model.lookback_window`: number of past steps used as features
  - `model.time_interval`: forecast horizon relative to `data.interval` (e.g., `"5d"` for 5 days)
  - `model.confidence_interval`: probability threshold [0,1] to include an asset in the allocation
  - `model.train`: if `true`, the model will train on-the-fly before predicting; otherwise, it will try to load from `model_path`
  - `model.model_path`: path to save/load the trained model weights (pickle)

To pre-train a model and save weights:

```
python scripts/train_nn.py --config config/base.yaml --epochs 50
```

Notes:
- The NN is implemented in pure NumPy for portability and keeps training time modest on small datasets.
- Allocation logic: equal-weight assets whose probability ≥ `confidence_interval`. If none pass the threshold, pick the highest-probability asset if > 0.5; otherwise allocate to cash.
- The NN path is fully optional; omitting `model.type: nn` leaves legacy algorithms unchanged.

### Single-ticker prediction (CLI)

Predict the probability that a single ticker's cumulative return over a horizon will be positive:

```
python scripts/predict_ticker.py --ticker NVDA --period 5y --interval 1d --time-interval 5d --lookback 60 --confidence 0.7 --train
```

- `--ticker`: e.g., `NVDA`.
- `--time-interval`: forecast horizon relative to `--interval`.
- `--confidence`: threshold to label as favorable; prints decision.
- Use `--train` the first time to fit on the downloaded history (or pre-train via `scripts/train_nn.py`).

### Single-ticker prediction (Dash)

The Dash app now includes a "Single‑Ticker NN Prediction" section:

- Enter a `Ticker` (e.g., NVDA), select `Period`, `Interval`, `Time horizon`, `Lookback`, and `Confidence`.
- Optionally check `Train` to fit on the latest downloaded history before predicting.
- Click `Predict` to see the probability and decision inline.

Below the prediction controls, the app also plots an estimated price movement over the chosen horizon using a simple GBM simulation calibrated to recent returns (mean and volatility). The shaded area reflects the selected confidence level.
