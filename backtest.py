from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from kalman_filter import KalmanFilter
from config import BacktestConfig
from visualization import plot_w

@dataclass
class Position:
    """
    Represents a trading position.
    Attributes:
        ticker (str): The ticker symbol of the asset.
        quantity (float): The quantity of the asset held in the position.
        price (pd.Series): The entry price of the position.
        sl (float): The stop-loss price.
        tp (float): The take-profit price.
        time (pd.Series): The time the position was opened.
        is_win (bool): Indicates if the position was closed at a profit.
        type (str): The type of position ('long' or 'short').
    """
    ticker: str
    quantity: float
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

    data = data.copy()

    columns = [x, y]
    data = data[columns]


    # Initial guesses for first kalman filter iteration
    x0 = data[x].iloc[window-1]
    y0 = data[y].iloc[window-1]
    w0 = np.array([0, y0 / x0])



    # Initialize Kalman Filter
    kf = KalmanFilter(w0, p, q, r)

    # Store portfolio value over time
    portfolio_value = []

    # Store predicted hedge ratios
    w_pred = []
    flag = None

    # Backtesting loop
    for i in range(window, len(data)):
        # Rolling window mu and sigma calculation
        w_data = data.iloc[i-window:i]
        w_spread = w_data[y] - (kf.coef()[0] + kf.coef()[1]*w_data[x])
        w_stand_spread = (w_spread - w_spread.mean()) / w_spread.std()
        mu, sigma = w_stand_spread.mean(), w_stand_spread.std()

        # Current prices
        x_t = data[x].iloc[i]
        y_t = data[y].iloc[i]

        # Predict hedge ratio using Kalman Filter
        w_t = kf.predict(x_t, y_t)
        w_pred.append(w_t)

        # Calculate z-score of the spread
        actual_spread = y_t - (w_t[0 ] + w_t[1]*x_t)
        z_score = (actual_spread - mu) / sigma

        # # Determine trading signal based on z-score
        # if z_score > z_threshold:
        #     flag = -1  # Short spread
        # elif z_score < -z_threshold:
        #     flag = 1   # Long spread
        # else:
        #     flag = 0   # No position


    # Plot predicted hedge ratios
    plot_w(w_pred, data.index[window:])















