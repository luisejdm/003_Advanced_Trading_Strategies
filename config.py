from dataclasses import dataclass

@dataclass
class BacktestConfig:
    """
        Configuration for backtesting a trading strategy.
        Attributes:
            initial_capital (float): The initial capital for the backtest.
            commission (float): The commission rate per trade (as a decimal).
            borrow_rate (float): The borrow rate for short selling (as a decimal).
            invest_fraction (float): The fraction of capital to invest in each trade.
            z_threshold (float): The z-score threshold for entering trades.
            window (int): The rolling window size for dynamic z_score. (default is 252 trading days)
            z_close_threshold (float): The z-score threshold for closing trades.
        """
    initial_capital: float = 1_000_000
    commission: float = 0.125 / 100
    borrow_rate: float = 0.25 / 100
    invest_fraction: float = 0.8
    z_threshold: float = 1.75
    window: int = 252,
    z_close_threshold: float = 0.1
