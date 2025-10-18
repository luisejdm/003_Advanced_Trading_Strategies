import itertools
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm


def check_price_stationarity(prices: pd.Series, alpha: float = 0.05) -> bool:
    """
    Check if the price series is stationary using the Augmented Dickey-Fuller test.
    Args:
        prices (pd.Series): The price series to be tested.
        alpha (float): Significance level for the test.
    Returns:
        bool: True if the series is stationary, False otherwise.
    """
    _, p_value, _, _, _, _ = adfuller(prices)
    isstationary = p_value < alpha # True if stationary (Is expected to be False for stock prices)
    return isstationary


def get_non_stationary_stocks(data: pd.DataFrame, alpha: float = 0.05) -> list:
    """
    Identify non-stationary stocks in the given DataFrame.
    Args:
        data (pd.DataFrame): DataFrame containing stock price series.
        alpha (float): Significance level for the stationarity test.
    Returns:
        list: List of non-stationary stock names.
    """
    non_stationary_stocks = []
    for stock in data.columns:
        if not check_price_stationarity(data[stock], alpha):
            non_stationary_stocks.append(stock)
    return non_stationary_stocks


def check_pair_cointegration(data, tickers, alpha=0.05) -> list:
    """
    Check if a pair of stocks is cointegrated using the Engle-Granger two-step method.
    Args:
        data (pd.DataFrame): DataFrame containing stock price series.
        tickers (list): List of stock tickers to be tested for cointegration.
        alpha (float): Significance level for the cointegration test.
    Returns:
        list: List of tuples containing cointegrated pairs and their p-values.
    """
    coint_pairs = []

    for t1, t2 in itertools.combinations(tickers, 2):
        series1 = data[t1].dropna()
        series2 = data[t2].dropna()

        # Alinear fechas
        combined = pd.concat([series1, series2], axis=1, join="inner").dropna()

        y = combined[t1]
        x = sm.add_constant(combined[t2])

        # Regresión lineal
        model = sm.OLS(y, x).fit()
        residuals = model.resid

        # Prueba ADF sobre los residuos
        adf_stat, p_value, _, _, _, _ = adfuller(residuals)

        # Si los residuos son estacionarios, los precios están cointegrados
        if p_value < alpha:
            coint_pairs.append((t1, t2, p_value))

    return coint_pairs


def get_best_cointegrated_pair(data: pd.DataFrame, sectors: dict, alpha: float = 0.05) -> tuple[dict, tuple, float, str]:
    """
    Get cointegrated pairs of stocks grouped by their sectors.
    Args:
        data (pd.DataFrame): DataFrame containing stock price series.
        sectors (dict): Dictionary mapping sector names to lists of stock tickers.
        alpha (float): Significance level for the cointegration test.
    Returns:
        dict: Dictionary with sector names as keys and lists of cointegrated pairs as values.
        tuple: The best cointegrated pair and its p-value.
        float: The p-value of the best cointegrated pair.
        str: The sector of the best cointegrated pair.
    """
    # Get pairs that are cointegrated by sectors
    coint_results = {}
    print(f'\n{"=" * 50}\n\nSTARTING COINTEGRATION ANALYSIS\n')
    for sector_name, tickers in sectors.items():
        print(f'  Analizing {sector_name} sector...')
        pairs = check_pair_cointegration(data, tickers, alpha)
        coint_results[sector_name] = pairs
    total_pairs = sum(len(pairs) for pairs in coint_results.values())
    print(f'\nTotal cointegrated pairs found: {total_pairs}. (Confidence level: {1 - alpha:.2%})')

    # Initialize variables to track the best pair
    best_pair = None
    best_pvalue = 1
    best_sector = None

    # Get the best pairs overall
    for sector, pairs in coint_results.items():
        for p in pairs:
            if p[2] < best_pvalue:
                best_pvalue = p[2]
                best_pair = p
                best_sector = sector

    return coint_results, best_pair, best_pvalue, best_sector


def get_best_pair() -> tuple:
    """
    Get the best cointegrated pair.
    Returns:
        tuple: The best cointegrated pair of stock tickers.
    """
    return 'MSFT', 'INTU'