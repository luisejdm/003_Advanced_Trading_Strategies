from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter
from config import BacktestConfig
from visualization import plot_estimations, plot_portfolio_value
from metrics import get_metrics

@dataclass
class Position:
    """
    Represents a trading position.
    Attributes:
        n_shares (float): The number of shares in the position.
        price (pd.Series): The entry price of the position.
        sl (float): The stop-loss price.
        tp (float): The take-profit price.
        time (pd.Series): The time the position was opened.
        is_win (bool): Indicates if the position was closed at a profit.
        type (str): The type of position ('long' or 'short').
    """
    n_shares: float
    price: float
    sl: float
    tp: float
    time: pd.Timestamp
    is_win: bool = None
    type: str = None


def run_backtest(
        data: pd.DataFrame,  config: BacktestConfig, x: str, y: str,
        p: float, q: float, r: float
):
    """
    Backtest a trading strategy on historical data.
    Args:
        data (pd.DataFrame): The historical price data for backtesting.
        config (BacktestConfig): Configuration for the backtest.
        x (str): The ticker symbol for the first asset.
        y (str): The ticker symbol for the second asset.
        p (float): Initial estimate covariance for Kalman Filter.
        q (float): Process noise covariance for Kalman Filter.
        r (float): Measurement noise covariance for Kalman Filter.
    Returns:
        metrics (dict): A dictionary containing performance metrics.
        n_long_trades (int): The number of long trades executed.
        n_short_trades (int): The number of short trades executed.
        portfolio_value (list): The portfolio value over time.
        final_capital (float): The final capital after backtesting.
    """
    # Extract config parameters
    capital = config.initial_capital
    commission = config.commission
    z_threshold = config.z_threshold
    exec_lag = config.exec_lag
    borrow_rate = config.borrow_rate
    daily_borrow_rate = borrow_rate / 252
    window = config.window
    z_close_threshold = config.z_close_threshold
    invest_frac = config.invest_fraction

    # Prepare data
    data = data.copy()
    columns = [x, y]
    data = data[columns]

    # Initial guesses for first kalman filter iteration
    x0 = data[x].iloc[window-1]
    y0 = data[y].iloc[window-1]
    w0 = np.array([0, y0 / x0])

    # Initialize Kalman Filter
    kf = KalmanFilter(w0, p, q, r)

    portfolio_values = []
    positions = []
    pending_postitions = [] # Signals to be executed next day

    n_long_trades = 0
    n_short_trades = 0

    # Store predicted hedge ratios
    w_pred = []
    current_signal = None

    cash = capital
    x_shares = 0
    y_shares = 0

    # Backtesting loop
    for i in range(window, len(data)):
        # Rolling window mu and sigma calculation
        w_data = data.iloc[i-window:i]
        w_spread = w_data[y] - (kf.coef()[0] + kf.coef()[1]*w_data[x])
        #w_stand_spread = (w_spread - w_spread.mean()) / w_spread.std()
        #mu, sigma = w_stand_spread.mean(), w_stand_spread.std()
        mu, sigma = w_spread.mean(), w_spread.std()

        # Current prices
        x_t = data[x].iloc[i]
        y_t = data[y].iloc[i]

        # Predict hedge ratio using Kalman Filter
        w_t = kf.predict(x_t, y_t)
        w_pred.append(w_t)
        alpha_t, beta_t = w_t.ravel()

        # Calculate z-score of the spread
        actual_spread = y_t - (alpha_t + beta_t * x_t)
        z_score = (actual_spread - mu) / sigma

        # Generate trading signals
        if np.isfinite(z_score):
            # Close position
            if np.abs(z_score) <= z_close_threshold:
                today_signal = 0  # Close positions

            # Open Positions
            elif z_score > z_threshold:
                today_signal = -1 # Short spread (y short, x long)
            elif z_score < -z_threshold:
                today_signal = 1 # Long spread (y long, x short)

            # If no signal, maintain current position
            else:
                today_signal = current_signal
        else:
            today_signal = None

        # Save signal for next day execution
        if today_signal is not None:
            j = i + exec_lag
            if j < len(data):
                exec_date = data.index[j]
                pending_postitions.append((exec_date, today_signal))

        # Check for signals to execute today
        exec_flag = None # Initialize signal to execute today
        today = data.index[i]
        if pending_postitions and any(d == today for d, _ in pending_postitions):
            signal = [pending_signal for (d, pending_signal) in pending_postitions if d == today]
            exec_flag = signal[-1]
            # Remove executed signals
            pending_postitions = [(d, pf) for (d, pf) in pending_postitions if d != today]

        current_portfolio_value = y_shares * y_t + x_shares * x_t + cash
        total_equity = current_portfolio_value + cash

        target_y_shares = 0
        target_x_shares = 0
        if exec_flag is not None:
            # Close positions
            budget_for_trade = invest_frac * total_equity
            budget_per_asset = budget_for_trade / 2

            # Calculate number of shares to trade
            n_y = np.floor(budget_per_asset / y_t)
            n_x = np.floor(budget_per_asset / (np.abs(beta_t) * x_t)) # Adjust for hedge ratio

            # Rebalance positions based on signal
            if exec_flag == 1:
                # Long Spread (y long, x short)
                target_y_shares = n_y
                target_x_shares = -beta_t * n_x
                n_long_trades += 1
            elif exec_flag == -1:
                # Short Spread (y short, x long)
                target_y_shares = -n_y
                target_x_shares = beta_t * n_x
                n_short_trades += 1
            elif exec_flag == 0:
                # Close Positions
                target_y_shares = 0
                target_x_shares = 0

        # Execute trades and update cash
        delta_y = target_y_shares - y_shares
        delta_x = target_x_shares - x_shares
        traded_value = (np.abs(delta_y) * y_t) + (np.abs(delta_x) * x_t)
        commission_cost = traded_value * commission

        cash -= (delta_y * y_t) + (delta_x * x_t) + commission_cost

        y_shares = target_y_shares
        x_shares = target_x_shares

        # Borrowing cost for short positions
        short_value = abs(min(0, y_shares * y_t)) + abs(min(0, x_shares * x_t))
        borrow_cost = short_value * daily_borrow_rate
        cash -= borrow_cost

        portfolio_value = y_shares * y_t + x_shares * x_t
        total_equity = portfolio_value + cash
        portfolio_values.append(total_equity)

        # Update signal if was executed
        if exec_flag is not None:
            current_signal = exec_flag

    metrics = get_metrics(portfolio_values)

    return metrics, w_pred, portfolio_values














