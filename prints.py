def print_best_pair(best_pair: tuple, best_pvalue: float, sector: str) -> None:
    """
    Print the best cointegrated pair and its p-value.
    Args:
        best_pair (tuple): The best cointegrated pair.
        best_pvalue (float): The p-value of the best cointegrated pair.
        sector (str): The sector of the best cointegrated pair.
    """
    print(
        f'{'=' * 75}\n' +
        f'\nBest cointegrated pair: {best_pair[0]} - {best_pair[1]} ' +
        f'\nP-value: {best_pvalue:.6f}' +
        f'\nSector: {sector}'
    )


def print_metrics(
        metrics: dict, z_threshold: float, data_set: str, #n_long_trades: int, n_short_trades: int
) -> None:
    """
    Print the performance metrics.
    Args:
        metrics (dict): A dictionary containing performance metrics.
        z_threshold (float): The z-score threshold used in the backtest.
        data_set (str): The dataset on which the metrics were evaluated.
    """
    print('\n' + '=' * 75)
    print(f'\nPerformance Metrics on {data_set} with optimal z_threshold:')
    for metric, value in metrics.items():
        print(f'  {metric}: {value:.4f}')


def print_summary(intial_capital: int, capitals: dict, positions: list) -> None:
    """
    Print a summary of the backtest.
    Args:
        intial_capital (int): The initial capital used in the backtest.
        capitals (dict): A dictionary containing final capitals for different datasets.
        positions (list): A list of all closed Position objects.
    """
    print('\n' + '=' * 75)
    print('\nBacktest Summary:')
    print(f'  Initial Capital: ${intial_capital:,.2f}')
    for dataset, final_capital in capitals.items():
        print(f'  Final Capital on {dataset}: ${final_capital:,.2f}')
    print(f'  Total ROI: {(capitals['Validation'] - intial_capital) / intial_capital * 100:.2f}%')
    print(f'  Total Borrowed Amount Paid: ${sum(pos.borrow_cost for pos in positions):,.2f}')
    print(f'  Total Commission Paid: ${sum(pos.commission_cost for pos in positions):,.2f}')