# 003_Advanced_Trading_Strategies

Pairs trading framework using a Kalman Filter to estimate hedge ratios and a backtester that supports train / test / validation splits and z‑score threshold optimization.

This project implements:
- Identification of cointegrated pairs (helpers in cointegration.py)
- Kalman Filter based dynamic hedge ratio estimation (kalman_filter.py)
- A backtesting engine for pairs trading with realistic costs (backtest.py)
- Performance metrics (metrics.py) and plotting utilities (visualization.py)
- Utilities for data splitting, standardization, and portfolio accounting (utils.py)
- A simple CLI entry point (main.py) that runs optimization on the training set and evaluates on test/validation

---

## Table of contents
- Features
- Requirements
- Setup
- Data requirements & format
- Configuration / Hyperparameters
- Usage examples
- Outputs & interpretation
- Notes, caveats and troubleshooting
- License

---

## Features
- Dynamic hedge ratio estimation using a Kalman Filter
- Strategy signals based on z-score of spread
- Transaction costs, borrow rate and commission modeling
- Train / Test / Validation split for parameter selection (e.g., optimal z-score)
- Metrics computed per split and overall (Sharpe, Sortino, drawdown, win rate, etc.)
- Plotting of hedge ratios, spread, signals, portfolio values and trade returns

---

## Requirements
- Python 3.8+
- Recommended Python packages:
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - statsmodels (for cointegration tests)
  - scipy
  - (optional) tqdm

Install packages with pip when necessary, e.g.:
```bash
pip install numpy pandas matplotlib seaborn statsmodels scipy
```
If a `requirements.txt` is added to the repo you can install with:
```bash
pip install -r requirements.txt
```

---

## Setup

1. Clone the repository:
```bash
git clone https://github.com/luisejdm/003_Advanced_Trading_Strategies.git
cd 003_Advanced_Trading_Strategies
```

2. (Optional) Create and activate a virtual environment:
```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. Install required packages (see Requirements section).

4. Place your historical price data in the `data/` directory (create it if it doesn't exist) or provide a path to your DataFrame when adapting the code.

---

## Data requirements & format

The code expects a pandas DataFrame (or CSV that can be loaded to a DataFrame) with the following properties:

- Index: Date-like (convertible to pandas datetime) and sorted ascending (oldest -> newest).
- Columns: one column per ticker (string column names), each containing the asset price (typically adjusted close).
- No multi-index is expected; simple columns are used and referenced by column name.

Example CSV structure (first row is a header with ticker names):

date, AAPL, MSFT, GOOG
2020-01-02, 300.35, 157.70, 1352.83
2020-01-03, 297.43, 155.75, 1367.37
...

Load example:
```python
import pandas as pd
data = pd.read_csv("data/prices.csv", parse_dates=["date"], index_col="date")
```

Notes:
- Missing values: the code uses standard pandas operations. It's advisable to forward/backward fill or drop rows with NaNs for the selected pair before backtesting.
- The cointegration routines expect enough historical data (multi-year daily series is typical). The backtest defaults to a 252-day rolling window.

---

## Configuration / Hyperparameters

Key parameters (defaults shown in `main.py` and `config.py`):
- initial_capital: 1_000_000 (USD)
- commission: 0.125 / 100 (0.125%)
- borrow_rate: 0.25 / 100 (0.25%)
- invest_fraction: 0.8 (fraction of capital used in new positions)
- z_threshold: threshold(s) used when entering positions (main runs a grid of values to optimize)
- z_close_threshold: z distance at which to close positions (default 0.1)
- window: rolling window in days for mu/sigma (default 252 trading days)
- Kalman filter hyperparameters: p, q, r — process/measurement/estimate error covariances (tunable in main.py)

In `main.py` you'll find variables near the top that you can change to adapt runs:
- choose whether to use a precomputed best pair or find the best pair from the dataset (use_best_pair)
- initial_capital / commission / borrow_rate / invest_fraction
- grid for z_threshold (array)

The BacktestConfig dataclass (config.py) centralizes most options:
```python
from config import BacktestConfig

config = BacktestConfig(
    initial_capital=1_000_000,
    commission=0.00125,
    borrow_rate=0.0025,
    invest_fraction=0.8,
    z_threshold=1.75,
    window=252,
    z_close_threshold=0.1
)
```

---

## Running

The repository contains `main.py` which orchestrates:
- pair selection (via cointegration modules)
- standardization and plotting of the chosen pair
- grid search over z_threshold values on the training set
- final evaluation on test and validation sets
- plotting and printing of metrics and summary

To run:
```bash
python main.py
```

Optional edits before running:
- Provide your price dataset inside the code or modify `main.py` to load a different CSV/DataFrame path.
- Tweak hyperparameters described above.
- If you want to focus on a specific pair, set `use_best_pair = True` and provide `best_pair = ('TICKER_X', 'TICKER_Y')` (see `main.py`).

---

## Outputs & interpretation

- Console prints:
  - Best pair info (symbol names and cointegration p-value)
  - Performance metrics per split (Train / Test / Validation / Overall) such as Sharpe, Sortino, Max Drawdown, ROI, Win rate.
  - Backtest summary (initial vs final capital, total ROI, total borrowed amount/borrow costs).

- Plots (via visualization.py):
  - Estimated hedge ratio over time (Kalman predictions)
  - Portfolio value vs time with entry/exit signal overlays
  - Spread and z-score with signal thresholds
  - Distribution / returns of trades

- Backtest internals:
  - The backtester returns detailed daily portfolio results, lists of open/closed positions, and predicted hedge ratios — useful for post-processing or building additional reports.

---

## Files of interest
- main.py — main entry point and parameter sweep logic
- backtest.py — backtesting engine and Position dataclass
- kalman_filter.py — Kalman Filter implementation used for dynamic hedge ratio estimation
- cointegration.py — functions to find cointegrated pairs
- sectors.py — (helper) sector grouping utilities used by pair-finding logic
- utils.py — data splitting, standardization and portfolio helpers
- metrics.py — performance metric calculations
- visualization.py — plotting functions
- prints.py — console printing helpers
- config.py — BacktestConfig dataclass holding default configuration
- LICENSE — MIT License

---

## Common adjustments & tips
- Data frequency: code assumes daily data. If using intraday or weekly, review the window and borrow rate scaling (daily_borrow_rate = borrow_rate / 252 in code).
- Handling NaNs: ensure price series for chosen tickers have no NaNs in the evaluation windows or prefill/drop appropriately.
- Kalman hyperparameters p, q, r: these can materially affect hedge ratio smoothness and performance — tune on train/test splits.
- Invest sizing: invest_fraction and conversion to integer number of shares (flooring) can create rounding effects; consider running experiments at larger capital scales or using notional shares if desired.

---

## License
MIT License — see LICENSE file.

---

Author: Luis Jiménez (repository owner)
