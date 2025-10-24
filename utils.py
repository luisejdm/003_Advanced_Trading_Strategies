import pandas as pd

def train_test_validation(
        data, trian_size=0.6, test_size=0.2, validation_size=0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training, testing, and validation sets.
    Params:
        data (pd.DataFrame): The input data to be cleaned.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The cleaned and split data.
    """
    n = len(data)
    train_end = int(n*trian_size)
    test_end = int(n*(1 - validation_size))

    train_data = data.iloc[:train_end].copy()
    test_data = data.iloc[train_end:test_end].copy()
    validation_data = data.iloc[test_end:].copy()

    return train_data, test_data, validation_data


def standarize_pair(data: pd.DataFrame) -> pd.DataFrame:
    """
    Estandarize the stock prices for plotting purposes.
    Args:
        data (pd.DataFrame): DataFrame containing stock prices.
    Returns:
        pd.DataFrame: Estandarized stock prices.
    """
    return (data - data.mean()) / data.std()


def get_portfolio_value(
        capital: float, long_positions: list, short_positions: list,
        x: str, y: str, current_x_price: float, current_y_price: float
) -> float:
    """
    Estimate the portfolio value for graphing purposes.
    Args:
        capital (float): The current capital available.
        long_positions (list): A list of active long positions.
        short_positions (list): A list of active short positions.
        x (str): Ticker symbol for stock X.
        y (str): Ticker symbol for stock Y.
        current_x_price (float): The current price of stock X.
        current_y_price (float): The current price of stock Y.
    Returns:
        float: The total portfolio value.
    """
    value = capital

    # For long positions
    for position in long_positions:
        if position.ticker == x:
            value += current_x_price * position.n_shares # No commission since position isn't actualy closed
        elif position.ticker == y:
            value += current_y_price * position.n_shares # No commission since position isn't actualy closed

    # For short positions
    for position in short_positions:
        if position.ticker == x:
            pnl = (position.entry_price - current_x_price) * position.n_shares
            value += pnl
        elif position.ticker == y:
            pnl = (position.entry_price - current_y_price) * position.n_shares
            value += pnl

    return value


def filter_positions(positions, start=None, end=None) -> list:
    """
    Filter positions based on their exit dates.
    Args:
        positions (list): A list of Position objects.
        start (datetime, optional): The start date for filtering. Defaults to None.
        end (datetime, optional): The end date for filtering. Defaults to None.
    Returns:
        list: A list of filtered Position objects.
    """
    return [
        pos for pos in positions
        if pos.exit_date is not None
        and (start is None or pos.exit_date > start)
        and (end is None or pos.exit_date <= end)
    ]


def get_individual_trade_returns(positions: list) -> list:
    """
    Get the individual trade returns from a list of positions.
    Args:
        positions (list): A list of Position objects.
    Returns:
        list: A list of individual trade returns.
    """
    returns = [
        # Long position return
        (pos.exit_price - pos.entry_price) / pos.entry_price
        if pos.type == 'long' else
        # Short position return
        (pos.entry_price - pos.exit_price) / pos.entry_price
        for pos in positions if pos.exit_price is not None
    ]
    return returns