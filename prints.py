import pandas as pd

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

def print_zscore_optimization(metrics_df: pd.DataFrame) -> None:
    """
    Print the z-score optimization results and return the optimal z-score.
    Args:
        metrics_df (pd.DataFrame): DataFrame containing z-scores and their corresponding metrics.
    Returns:
        float: The optimal z-score.
    """
    print(f'\n{"=" * 75}\n')
    print(f'Optimization of Z-Score Threshold on Train Set\n')
    print(metrics_df.to_string(index=False))

def print_metrics(metrics: dict, period: str) -> None:
    """
    Print a clean, formatted summary of performance metrics.
    Args:
        metrics (dict): A dictionary containing performance metrics.
        period (str): The time period for which the metrics are calculated.
    """

    print(f'\n{'='*70}\n')
    print(f'Performance Metrics for {period} Period\n')

    print(f'{'Sharpe Ratio':<45}: {metrics.get('Sharpe', 0):>15.4f}')
    print(f'{'Sortino Ratio':<45}: {metrics.get('Sortino', 0):>15.4f}')
    print(f'{'Maximum Drawdown':<45}: {metrics.get('Maximum Drawdown', 0)*100:>14.2f}%')
    print(f'{'Calmar Ratio':<45}: {metrics.get('Calmar', 0):>15.4f}')

    print(f'{'Number of Long Trades':<45}: {metrics.get('Number of long trades', 0):>15}')
    print(f'{'Number of Short Trades':<45}: {metrics.get('Number of short trades', 0):>15}')

    print(f'{'Total Borrowed Amount':<45}: ${metrics.get('Total borrowed amount', 0):>14,.2f}')
    print(f'{'Total Commission Cost':<45}: ${metrics.get('Total commission cost', 0):>14,.2f}')

    print(f'{'Win Rate (Long Positions)':<45}: {metrics.get('Win rate on long positions', 0)*100:>14.2f}%')
    print(f'{'Win Rate (Short Positions)':<45}: {metrics.get('Win rate on short positions', 0)*100:>14.2f}%')
    print(f'{'Overall Win Rate':<45}: {metrics.get('General win rate', 0)*100:>14.2f}%')

    print(f'{'Average Return per Long Trade':<45}: {metrics.get('Average return per long trade', 0)*100:>14.2f}%')
    print(f'{'Average Return per Short Trade':<45}: {metrics.get('Average return per short trade', 0)*100:>14.2f}%')
    print(f'{'Average Return per Trade':<45}: {metrics.get('Average return per trade', 0)*100:>14.2f}%')
    print(f"{'Profit Factor':<45}: {metrics.get('Profit Factor', 0):>15.3f}")



def print_summary(initial_capital: int, capitals: dict, positions: list) -> None:
    """
    Print a clean, formatted summary of the backtest results.
    Args:
        initial_capital (int): The initial capital used in the backtest.
        capitals (dict): A dictionary containing final capitals for different datasets.
        positions (list): A list of all closed Position objects.
    """
    print(f'\n{'='*70}\n')
    print(f'{'Backtest Summary':}\n')

    print(f'{'Initial Capital':<45}: ${initial_capital:>15,.2f}')

    for dataset, final_capital in capitals.items():
        print(f'Final Capital ({dataset})'.ljust(45) + f': ${final_capital:>15,.2f}')

    total_roi = ((capitals['Validation'] - initial_capital) / initial_capital) * 100

    print(f'{'Total ROI':<45}: {total_roi:>15.2f}%')
    print(f'{'Total Borrow Cost Paid':<45}: ${sum(pos.borrow_cost for pos in positions):>15,.2f}')
    print(f'{'Total Commission Paid':<45}: ${sum(pos.commission_cost for pos in positions):>15,.2f}')

    print(f"\n{'='*70}\n")