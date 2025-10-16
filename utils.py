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


def estandarize_pair(data: pd.DataFrame) -> pd.DataFrame:
    """
    Estandarize the stock prices for plotting purposes.
    Args:
        data (pd.DataFrame): DataFrame containing stock prices.
    Returns:
        pd.DataFrame: Estandarized stock prices.
    """
    return (data - data.mean()) / data.std()