from dataclasses import dataclass
import pandas as pd
import numpy as np

from kalman_filter import KalmanFilter
from config import BacktestConfig
from metrics import get_metrics
from utils import get_portfolio_value, filter_positions, get_individual_trade_returns


@dataclass
class Position:
    """
    Represents a trading position.
    Attributes:
        type (str): The type of position ('long' or 'short').
        ticker (str): The asset ticker.
        entry_date (pd.Timestamp): The date the position was entered.
        entry_price (float): The price at which the position was entered.
        n_shares (int): The number of shares in the position.
        borrow_cost (float): The accumulated borrow cost for short positions.
        commission_cost (float): The total commission cost for the position.
        exit_date (pd.Timestamp, optional): The date the position was exited.
        exit_price (float, optional): The price at which the position was exited.
        is_win (bool, optional): Indicates if the position was profitable.
    """
    type: str  # 'long' or 'short'
    ticker: str # Asset ticker
    entry_date: pd.Timestamp
    entry_price: float
    n_shares: int
    borrow_cost: float = 0.0
    commission_cost: float = 0.0
    exit_date: pd.Timestamp = None
    exit_price: float = None
    is_win: bool = None

def run_backtest(
        data: pd.DataFrame,  config: BacktestConfig, x: str, y: str,
        p: float, q: float, r: float,
        last_train_date, last_test_date
):
    """
    Run a backtest of the pairs trading strategy using a Kalman Filter for hedge ratio estimation.
    Args:
        data (pd.DataFrame): DataFrame containing price data for the two assets.
        config (BacktestConfig): Configuration parameters for the backtest.
        x (str): The column name for asset X.
        y (str): The column name for asset Y.
        p (float): Process noise covariance for the Kalman Filter.
        q (float): Measurement noise covariance for the Kalman Filter.
        r (float): Estimate error covariance for the Kalman Filter.
        last_train_date: The last date of the training data.
        last_test_date: The last date of the test data.
    Returns:
        metrics (dict): Performance metrics of the backtest.
        w_pred (list): Predicted hedge ratios over time.
        portfolio_values (list): Portfolio values over time.
        portfolio_results (pd.DataFrame): Detailed daily portfolio results.
    """
    # Extract config parameters
    capital = float(config.initial_capital)
    commission = float(config.commission)
    z_threshold = float(config.z_threshold)
    borrow_rate = float(config.borrow_rate)
    daily_borrow_rate = borrow_rate / 252.0
    window = int(config.window)
    z_close_threshold = float(config.z_close_threshold)
    invest_frac = float(config.invest_fraction)

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

    w_preds = []
    z_scores = []

    portfolio_value = []
    active_long_positions = []
    active_short_positions = []

    closed_long_positions = []
    closed_short_positions = []

    signals = []
    current_plot_signal = 0

    # Backtesting loop
    for i in range(window, len(data)):
        # Rolling window mu and sigma calculation
        w_data = data.iloc[i-window:i]
        w_spread = w_data[y] - (kf.coef()[0] + kf.coef()[1]*w_data[x])
        mu, sigma = w_spread.mean(), w_spread.std()

        # Current prices
        x_t = data[x].iloc[i]
        y_t = data[y].iloc[i]

        # Predict hedge ratio using Kalman Filter
        w_t = kf.predict(x_t, y_t)
        w_preds.append(w_t)
        alpha_t, beta_t = w_t[0], w_t[1]

        # Calculate z-score of the spread
        actual_spread = y_t - (alpha_t + beta_t * x_t)
        z_score = (actual_spread - mu) / sigma
        z_scores.append(z_score)

        if abs(z_score) <= z_close_threshold:
            day_signal = 0
        elif z_score > z_threshold:
            day_signal = -1
        elif z_score < -z_threshold:
            day_signal = 1
        else:
            day_signal = current_plot_signal  # No change
        signals.append(day_signal)
        current_plot_signal = day_signal

        for position in active_long_positions.copy():
            # Check for long position exit
            if abs(z_score) < z_close_threshold:
                # If x was longed
                if position.ticker == x:
                    pnl = x_t - position.entry_price
                    position.is_win = pnl > 0
                    capital += x_t * position.n_shares* (1 - commission)
                    position.exit_date = data.index[i]
                    position.exit_price = x_t
                    position.commission_cost += x_t * position.n_shares * commission
                    # Remove from active and add to closed
                    closed_long_positions.append(position)
                # If y was longed
                elif position.ticker == y:
                    pnl = y_t - position.entry_price
                    position.is_win = pnl > 0
                    capital += y_t * position.n_shares * (1 - commission)
                    position.exit_date = data.index[i]
                    position.exit_price = y_t
                    position.commission_cost += y_t * position.n_shares * commission
                    # Remove from active and add to closed
                    closed_long_positions.append(position)
                active_long_positions.remove(position)

        # Apply daily borrow cost to active short positions before checking for exits
        for position in active_short_positions.copy():
            if position.ticker == x:
                current_value = position.n_shares * x_t
                borrow_cost = current_value * daily_borrow_rate
                capital -= borrow_cost
                position.borrow_cost += borrow_cost
            elif position.ticker == y:
                current_value = position.n_shares * y_t
                borrow_cost = current_value * daily_borrow_rate
                capital -= borrow_cost
                position.borrow_cost += borrow_cost

        # Check for short position exit
        for position in active_short_positions.copy():
            if abs(z_score) < z_close_threshold:
                # If x was shorted
                if position.ticker == x:
                    pnl = (position.entry_price - x_t) * position.n_shares
                    position.is_win = pnl > 0
                    exit_commision = (x_t * position.n_shares) * commission
                    capital += pnl - exit_commision
                    position.commission_cost += exit_commision
                    position.exit_date = data.index[i]
                    position.exit_price = x_t
                    # Remove from active and add to closed
                    closed_short_positions.append(position)
                # If y was shorted
                elif position.ticker == y:
                    pnl = (position.entry_price - y_t) * position.n_shares
                    position.is_win = pnl > 0
                    exit_commision = (y_t * position.n_shares) * commission
                    capital += pnl - exit_commision
                    position.commission_cost += exit_commision
                    position.exit_date = data.index[i]
                    position.exit_price = y_t
                    # Remove from active and add to closed
                    closed_short_positions.append(position)
                active_short_positions.remove(position)


        # Check Long spread if there was a change between yesterday and today
        if z_scores[-1] < -z_threshold and (len(z_scores) >= 2 and z_scores[-2] >= -z_threshold):
            # Determine investment using beta_t estimated for today
            invest_amount_x = (capital * invest_frac) / (1 + abs(beta_t))
            invest_amount_y = invest_amount_x * abs(beta_t)
            n_shares_x = int(np.floor(invest_amount_x / x_t))
            n_shares_y = int(np.floor(invest_amount_y / y_t))

            # Trading costs
            x_cost = n_shares_x * x_t * commission
            y_cost = n_shares_y * y_t * (1+commission)
            total_cost = x_cost + y_cost

            # Check if there is enough capital
            if capital > total_cost > 0:
                capital -= total_cost
                # Open short position on x
                short_x_position = Position(
                    type='short',
                    ticker=x,
                    entry_date=data.index[i],
                    entry_price=x_t,
                    n_shares=n_shares_x
                )
                short_x_position.commission_cost += x_cost
                active_short_positions.append(short_x_position)
                # Open long position on y
                long_y_position = Position(
                    type='long',
                    ticker=y,
                    entry_date=data.index[i],
                    entry_price=y_t,
                    n_shares=n_shares_y
                )
                long_y_position.commission_cost += (n_shares_y * y_t * commission)
                active_long_positions.append(long_y_position)

        # Check Short spread if there was a change between yesterday and today
        if z_scores[-1] > z_threshold and (len(z_scores) >= 2 and z_scores[-2] <= z_threshold):
            # Determine investment using beta_t estimated for today
            invest_amount_x = (capital * invest_frac) / (1 + abs(beta_t))
            invest_amount_y = invest_amount_x * abs(beta_t)
            n_shares_x = int(np.floor(invest_amount_x / x_t))
            n_shares_y = int(np.floor(invest_amount_y / y_t))

            # Trading costs
            x_cost = n_shares_x * x_t * (1+commission)
            y_cost = n_shares_y * y_t * commission
            total_cost = x_cost + y_cost

            # Check if there is enough capital
            if capital > total_cost > 0:
                capital -= total_cost
                # Open long position on x
                long_x_position = Position(
                    type='long',
                    ticker=x,
                    entry_date=data.index[i],
                    entry_price=x_t,
                    n_shares=n_shares_x
                )
                long_x_position.commission_cost += (n_shares_x * x_t * commission)
                active_long_positions.append(long_x_position)
                # Open short position on y
                short_y_position = Position(
                    type='short',
                    ticker=y,
                    entry_date=data.index[i],
                    entry_price=y_t,
                    n_shares=n_shares_y
                )
                short_y_position.commission_cost += y_cost
                active_short_positions.append(short_y_position)

        # Record portfolio value at the end of the day
        current_port_val = get_portfolio_value(
            capital, active_long_positions, active_short_positions, x, y, x_t, y_t
        )
        portfolio_value.append(current_port_val)

    # Calculate the portfolio value at the end of the backtest with all active positions
    last_price_x = data[x].iloc[-1]
    last_price_y = data[y].iloc[-1]

    for position in active_long_positions:
        if position.ticker == x:
            position.is_win = last_price_x > position.entry_price
            capital += last_price_x * position.n_shares # No commsion since position isn't actualy closed
            closed_long_positions.append(position)
        elif position.ticker == y:
            position.is_win = last_price_y > position.entry_price
            capital += last_price_y * position.n_shares # No commsion since position isn't actualy closed
            closed_long_positions.append(position)
    active_long_positions = []

    for position in active_short_positions:
        if position.ticker == x:
            position.is_win = last_price_x < position.entry_price
            pnl = (position.entry_price - last_price_x) * position.n_shares
            capital += pnl + position.entry_price * position.n_shares
            closed_short_positions.append(position)
        elif position.ticker == y:
            position.is_win = last_price_y < position.entry_price
            pnl = (position.entry_price - last_price_y) * position.n_shares
            capital += pnl + position.entry_price * position.n_shares
            closed_short_positions.append(position)
    active_short_positions = []

    # Separate results for training, testing and validation
    if last_train_date is not None and last_test_date is not None:
        portfolio_values = pd.Series(portfolio_value, index=data.index[window:])
        train_portfolio_values = portfolio_values.loc[portfolio_values.index <= last_train_date]
        test_portfolio_values = portfolio_values.loc[
            (portfolio_values.index > last_train_date) &
            (portfolio_values.index <= last_test_date)
        ]
        val_portfolio_values = portfolio_values.loc[portfolio_values.index >= last_test_date]

        capitals = {
            'Train': train_portfolio_values.iloc[-1],
            'Test': test_portfolio_values.iloc[-1],
            'Validation': val_portfolio_values.iloc[-1],
        }


        train_closed_long_positions = filter_positions(closed_long_positions,
                                                       end=last_train_date)
        train_closed_short_positions = filter_positions(closed_short_positions,
                                                        end=last_train_date)
        test_closed_long_positions = filter_positions(closed_long_positions,
                                                      start=last_train_date,
                                                      end=last_test_date)
        test_closed_short_positions = filter_positions(closed_short_positions,
                                                       start=last_train_date,
                                                       end=last_test_date)
        val_closed_long_positions = filter_positions(closed_long_positions,
                                                     start=last_test_date)
        val_closed_short_positions = filter_positions(closed_short_positions,
                                                      start=last_test_date)

        metrics = {
            'Train': get_metrics(
                train_portfolio_values.tolist(),
                train_closed_long_positions,
                train_closed_short_positions
            ),
            'Test': get_metrics(
                test_portfolio_values.tolist(),
                test_closed_long_positions,
                test_closed_short_positions
            ),
            'Validation': get_metrics(
                val_portfolio_values.tolist(),
                val_closed_long_positions,
                val_closed_short_positions
            ),
            'Overall': get_metrics(
                portfolio_value,
                closed_long_positions,
                closed_short_positions
            )
        }
        all_position_returns = get_individual_trade_returns(closed_long_positions + closed_short_positions)
    else:
        capitals = {'Final': capital}
        metrics = get_metrics(portfolio_value, closed_long_positions, closed_short_positions)
        all_position_returns = get_individual_trade_returns(closed_long_positions + closed_short_positions)

    dates = data.index[window:]
    signal_series = pd.Series(np.asarray(signals), index=dates, name='Signal')
    zscore_series = pd.Series(np.asarray(z_scores), index=dates, name='Z_Score')

    all_closed_positions = closed_long_positions + closed_short_positions

    return metrics, w_preds, portfolio_value, signal_series, zscore_series, all_position_returns, capitals, all_closed_positions