import pandas as pd
import numpy as np


def get_sharpe(data: pd.DataFrame) -> float:
    """
    Calculate the Sharpe ratio of the portfolio.
    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.

    Returns:
        float: The Sharpe ratio of the portfolio.
    """
    mean = data.rets.mean()
    std = data.rets.std()

    annual_rets = mean * 252
    annual_std = std * np.sqrt(252)

    return annual_rets / annual_std if annual_std != 0 else 0


def get_sortino(data: pd.DataFrame) -> float:
    """
    Calculate the Sortino ratio of the portfolio.
    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.

    Returns:
        float: The Sortino ratio of the portfolio.
    """
    mean = data.rets.mean()
    std = data.rets.std()
    down_risk = data.rets[data.rets < 0].fillna(0).std()

    annual_rets = mean * 252
    annual_std = std * np.sqrt(252)
    annual_down_risk = down_risk * np.sqrt(252)

    return annual_rets / annual_down_risk if annual_std != 0 else 0


def get_maximum_drawdown(data: pd.DataFrame) -> float:
    """
    Calculate the maximum drawdown of the portfolio.
    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.

    Returns:
        float: The maximum drawdown of the portfolio.
    """
    roll_max = data['Value'].cummax()
    max_drawdown = (roll_max - data['Value']) / roll_max
    return max_drawdown.max()


def get_calmar(data: pd.DataFrame) -> float:
    """
    Calculate the Calmar ratio of the portfolio.
    Args:
        data (pd.DataFrame): A DataFrame containing the portfolio values over time.
    Returns:
        calmar_ratio (float): The Calmar ratio of the portfolio.
    """
    mean = data.rets.mean()
    annual_rets = mean * 252
    max_drawdown = get_maximum_drawdown(data)
    return annual_rets / max_drawdown if max_drawdown != 0 else 0


def get_win_rate(closed_positions: list) -> float:
    """
    Calculate the win rate of closed positions.
    Args:
        closed_positions (list): A list of closed Position objects.
    Returns:
        float: The win rate of the closed positions.
    """
    if not closed_positions:
        return 0

    wins = sum(1 for position in closed_positions if position.is_win)
    return wins / len(closed_positions)


def get_total_borrowed_amount(closed_positions: list) -> float:
    """
    Calculate the total borrowed amount from closed positions.
    Args:
        closed_positions (list): A list of closed Position objects.
    Returns:
        float: The total borrowed amount.
    """
    total_borrowed_amount = sum(pos.borrow_cost for pos in closed_positions)
    return total_borrowed_amount


def get_total_commission_cost(closed_positions: list) -> float:
    """
    Calculate the total commission cost from closed positions.
    Args:
        closed_positions (list): A list of closed Position objects.
    Returns:
        float: The total commission cost.
    """
    total_commission_cost = sum(pos.commission_cost for pos in closed_positions)
    return total_commission_cost


def get_avg_return_per_trade(
        closed_long_positions: list, closed_short_positions: list) -> tuple:
    """
    Calculate the average return per trade from closed positions.
    Args:
        closed_long_positions (list): A list of closed long Position objects.
        closed_short_positions (list): A list of closed short Position objects.
    Returns:
        long_avg_return (float): The average return for long positions.
        short_avg_return (float): The average return for short positions.
        total_avg_return (float): The overall average return.
    """
    long_avg_return = np.mean([
        (pos.exit_price - pos.entry_price) / pos.entry_price for pos in closed_long_positions
        if pos.exit_price is not None
    ])
    short_avg_return = np.mean([
        (pos.entry_price - pos.exit_price) / pos.entry_price for pos in closed_short_positions
        if pos.exit_price is not None
    ])
    total_avg_return = (long_avg_return + short_avg_return) / 2
    return long_avg_return, short_avg_return, total_avg_return


def get_profit_factor(positions: list) -> float:
    """
    Calculate the Profit Factor = Total Gross Profit / Total Gross Loss.
    Args:
        positions (list): List of closed Position objects with entry and exit prices.
    Returns:
        float: Profit factor value.
    """
    gross_profit = sum(
        max((pos.exit_price - pos.entry_price) * pos.n_shares, 0)
        if pos.type == 'long'
        else max((pos.entry_price - pos.exit_price) * pos.n_shares, 0)
        for pos in positions if pos.exit_price is not None
    )
    gross_loss = sum(
        abs(min((pos.exit_price - pos.entry_price) * pos.n_shares, 0))
        if pos.type == 'long'
        else abs(min((pos.entry_price - pos.exit_price) * pos.n_shares, 0))
        for pos in positions if pos.exit_price is not None
    )

    return gross_profit / gross_loss if gross_loss > 0 else np.nan


def get_metrics(
        portfolio_value: list, closed_long_positions: list, closed_short_positions: list
) -> dict:
    """
    Calculate various performance metrics for the backtest.
    Args:
        portfolio_value (list): A DataFrame containing the portfolio values over time.
        closed_long_positions (list): A list of closed long Position objects.
        closed_short_positions (list): A list of closed short Position objects.
    Returns:
        metrics (dict): A dictionary containing various performance metrics.
    """
    df = pd.DataFrame({'Value': portfolio_value})
    df['rets'] = df.Value.pct_change()
    df.dropna(inplace=True)

    long_avg_return, short_avg_return, total_avg_return = get_avg_return_per_trade(
        closed_long_positions, closed_short_positions
    )

    metrics = {
        'Sharpe': get_sharpe(df),
        'Sortino': get_sortino(df),
        'Maximum Drawdown': get_maximum_drawdown(df),
        'Calmar': get_calmar(df),
        'Number of long trades': len(closed_long_positions),
        'Number of short trades': len(closed_short_positions),
        'Total borrowed amount': get_total_borrowed_amount(closed_long_positions + closed_short_positions),
        'Total commission cost': get_total_commission_cost(closed_long_positions + closed_short_positions),
        'Win rate on long positions': get_win_rate(closed_long_positions),
        'Win rate on short positions': get_win_rate(closed_short_positions),
        'General win rate': get_win_rate(closed_long_positions + closed_short_positions),
        'Average return per long trade': long_avg_return,
        'Average return per short trade': short_avg_return,
        'Average return per trade': total_avg_return,
        'Profit Factor': get_profit_factor(closed_long_positions + closed_short_positions)
    }
    return metrics