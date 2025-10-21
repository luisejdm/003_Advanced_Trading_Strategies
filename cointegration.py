import itertools
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
import numpy as np


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


def check_pair_correlation_and_cointegration(
    data: pd.DataFrame,
    tickers: list,
    alpha: float,
    window: int,
    correlation_threshold: float
) -> list:
    """
    Primero filtra pares con suficiente correlación rolling en rendimientos y,
    para los que pasen el filtro, aplica Engle-Granger sobre PRECIOS.
    Retorna lista de tuplas (t1, t2, p_value) sólo para pares cointegrados.

    Params:
        data: precios (columnas = tickers)
        tickers: lista de símbolos a evaluar
        alpha: nivel de significancia para ADF de residuos
        window: ventana para la correlación rolling de rendimientos
        correlation_threshold: umbral mínimo de correlación media (rolling)

    Nota:
      - Se alinea por fechas para cada par.
      - Se usa la media de la correlación rolling sobre el periodo disponible.
      - Se asegura que los residuos de OLS entren a ADF como vector 1D.
    """
    coint_pairs = []

    for t1, t2 in itertools.combinations(tickers, 2):
        # Filter by rolling correlation of returns
        pair_prices = pd.concat([data[t1], data[t2]], axis=1, join="inner").dropna()
        returns = pair_prices.pct_change().dropna()
        roll_corr = returns[t1].rolling(window).corr(returns[t2]).dropna()
        mean_roll_corr = roll_corr.mean()

        if np.abs(mean_roll_corr) < correlation_threshold:
            continue  # Skip pairs that do not meet the correlation threshold

        y = pair_prices[t1]
        x = sm.add_constant(pair_prices[t2])

        model = sm.OLS(y, x).fit()
        residuals = model.resid

        _, p_value, _, _, _, _ = adfuller(residuals)

        if p_value < alpha:
            coint_pairs.append((t1, t2, p_value, mean_roll_corr))

    return coint_pairs


def get_best_cointegrated_pair(
        data: pd.DataFrame,
        sectors: dict,
        alpha: float,
        window: int,
        correlation_threshold: float
) -> tuple[dict, tuple, float, str]:
    """
    Get cointegrated pairs of stocks grouped by their sectors.
    Args:
        data (pd.DataFrame): DataFrame containing stock price series.
        sectors (dict): Dictionary mapping sector names to lists of stock tickers.
        alpha (float): Significance level for the cointegration test.
        window (int): Window size for rolling correlation.
        correlation_threshold (float): Correlation threshold for the cointegration test.
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
        pairs = check_pair_correlation_and_cointegration(
            data, tickers, alpha, window, correlation_threshold
        )

        #Print all the found pairs in the sector
        if pairs:
            print(f'    Found {len(pairs)} cointegrated pairs:')
            for t1, t2, pvalue, corr in pairs:
                print(f'      - {t1} & {t2} | p-value: {pvalue:.6f} | Rolling Corr: {corr:.6f}')
            print()
        else:
            print(f'    No cointegrated pairs found.\n')

        # Store results
        coint_results[sector_name] = pairs
    total_pairs = sum(len(pairs) for pairs in coint_results.values())
    print(f'Total cointegrated pairs found: {total_pairs}. (Confidence level: {1 - alpha:.2%})\n')

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


def get_best_pair() -> tuple[tuple, float, str]:
    """
    Get the best cointegrated pair.
    Returns:
        tuple: The best cointegrated pair, its p-value, and its sector.
    """
    best_pair = ('INTU', 'MSFT')
    best_pvalue = 0.000077
    best_sector = 'Information Technology'
    return best_pair, best_pvalue, best_sector