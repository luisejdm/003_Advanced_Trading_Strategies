def print_best_pair(best_pair: tuple, best_pvalue: float, sector: str) -> None:
    """
    Print the best cointegrated pair and its p-value.
    Args:
        best_pair (tuple): The best cointegrated pair.
        best_pvalue (float): The p-value of the best cointegrated pair.
        sector (str): The sector of the best cointegrated pair.
    """
    print(
        f'{'=' * 50}\n' +
        f'\nBest cointegrated pair: {best_pair[0]} - {best_pair[1]} ' +
        f'\nP-value: {best_pvalue:.6f}' +
        f'\nSector: {sector}'
    )


def print_metrics(
        metrics: dict, z_threshold: float #data_set: str, n_long_trades: int, n_short_trades: int
) -> None:
    """
    Print the performance metrics.
    Args:
        metrics (dict): A dictionary containing performance metrics.
        z_threshold (float): The z-score threshold used in the backtest.
        # data_set (str): The dataset on which the metrics were evaluated.
        # n_long_trades (int): Number of long trades executed.
        # n_short_trades (int): Number of short trades executed.
    """
    print('\n' + '=' * 50)
    print(f'\nPerformance Metrics with z_threshold={z_threshold}:')
    for metric, value in metrics.items():
        print(f'  {metric}: {value:.4f}')
    # print(f'  Number of Long Trades: {n_long_trades}')
    # print(f'  Number of Short Trades: {n_short_trades}')