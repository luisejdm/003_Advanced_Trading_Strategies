import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)

from utils import train_test_validation, standarize_pair
from cointegration import get_non_stationary_stocks, get_best_cointegrated_pair, get_best_pair
from sectors import get_sectors
from visualization import plot_cointegrated_stocks, plot_all_pairs, plot_portfolio_value, plot_estimations, plot_spread_and_signal
from prints import print_best_pair, print_metrics
from backtest import run_backtest
from config import BacktestConfig


use_best_pair = False

initial_capital = 1_000_000
commission = 0.125 / 100
borrow_rate = 0.25 / 100
invest_fraction = 0.8
z_threshold = 1
window = 252
z_close_threshold = 0.1

p = 0.001
q = 0.001
r = 100_000

def main():
    # ---- Load data and split into train, test, validation sets
    data = pd.read_csv('stocks.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    train, test, validation = train_test_validation(data, 0.6, 0.2, 0.2)

    # ---- Cointegration analysis
    if not use_best_pair:
        # Get only non-stationary stocks for cointegration analysis
        non_stationary_stocks = get_non_stationary_stocks(train, 0.01)
        data = data[non_stationary_stocks]

        # Get stock classification by sectors
        sectors = get_sectors()

        # Get the best cointegrated pair by sectors
        coint_results, best_pair, best_pvalue, best_sector = get_best_cointegrated_pair(
            train, sectors, 0.01, 50, 0.5
        )
        x, y = best_pair[0], best_pair[1]

        # Estandarize the best pair for plotting
        pair_data = data[[best_pair[0], best_pair[1]]]
        standarized_pair = standarize_pair(pair_data)

    else:
        best_pair, best_pvalue, best_sector = get_best_pair()
        x, y = best_pair[0], best_pair[1]
        pair_data = data[[best_pair[0], best_pair[1]]]
        standarized_pair = standarize_pair(pair_data)

    print_best_pair(best_pair, best_pvalue, best_sector)
    plot_cointegrated_stocks(standarized_pair)
    #plot_all_pairs(train, coint_results, estandarize_pair) # Uncomment to plot all found pairs


    # ---- Backtest configurations
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission=commission,
        borrow_rate=borrow_rate,
        invest_fraction=invest_fraction,
        z_threshold=z_threshold,
        window=window,
        z_close_threshold=z_close_threshold
    )

    # ---- Run backtest
    metrics, w_pred, porfolio_values, portfolio_results = run_backtest(
        data, config, x, y, p, q, r
    )

    # ---- Print metrics and plot results
    print_metrics(metrics, z_threshold)
    plot_estimations(data.index[window:], w_pred)
    plot_portfolio_value(data.index[window:], porfolio_values, portfolio_results['Signal'])
    plot_spread_and_signal(data.index[window:], portfolio_results['Z_Score'], portfolio_results['Signal'], z_threshold)


if __name__ == '__main__':
    main()