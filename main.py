import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)

from utils import train_test_validation, standarize_pair
from cointegration import get_non_stationary_stocks, get_best_cointegrated_pair, get_best_pair
from sectors import get_sectors
from visualization import plot_cointegrated_stocks, plot_portfolio_value, plot_estimations, plot_spread_and_signal, plot_trade_returns
from prints import print_best_pair, print_metrics, print_summary
from backtest import run_backtest
from config import BacktestConfig


use_best_pair = False

initial_capital = 1_000_000
commission = 0.125 / 100
borrow_rate = 0.25 / 100
invest_fraction = 0.8
window = 252

correlation_threshold = 0.7

z_close_threshold = 0.1 # Acceptable distance to close the position
z_threshold = np.linspace(0.1, 2.75, 20)

p = 0.0001
q = 0.0001
r = 1_000

optimize_metric = 'Sortino'

def main():
    # ---- Load data and split into train, test, validation sets
    data = pd.read_csv('stocks.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    train, test, validation = train_test_validation(data, 0.6, 0.2, 0.2)
    last_train_date = train.index[-1]
    last_test_date = test.index[-1]

    # ---- Cointegration analysis
    if not use_best_pair:
        # Get only non-stationary stocks for cointegration analysis
        non_stationary_stocks = get_non_stationary_stocks(train, 0.01)
        data = data[non_stationary_stocks]

        # Get stock classification by sectors
        sectors = get_sectors()

        # Get the best cointegrated pair by sectors
        coint_results, best_pair, best_pvalue, best_sector = get_best_cointegrated_pair(
            train, sectors, 0.01, 50, correlation_threshold
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

    # ---- Run backtet on train to get optimal z-score threshold
    metrics_list = []
    for z in z_threshold:
        config = BacktestConfig(
            initial_capital=initial_capital,
            commission=commission,
            borrow_rate=borrow_rate,
            invest_fraction=invest_fraction,
            z_threshold=z,
            window=window,
            z_close_threshold=z_close_threshold
        )
        metrics, _, _, _, _, _, _, _ = run_backtest(
            train, config, x, y, p, q, r, None, None
        )
        metrics_list.append((z, metrics[optimize_metric]))
    metrics_df = pd.DataFrame(metrics_list, columns=['Z_score', optimize_metric])
    optimal_z = metrics_df.loc[metrics_df[optimize_metric].idxmax(), 'Z_score']
    print(f'\n{'=' * 75}\n Optimal Z-Score Threshold on Train Set: {optimal_z:.4f}\n')

    # ---- Run Backtest on Test + Validation with optimal z-score
    config = BacktestConfig(
        initial_capital=initial_capital,
        commission=commission,
        borrow_rate=borrow_rate,
        invest_fraction=invest_fraction,
        z_threshold=optimal_z,
        window=window,
        z_close_threshold=z_close_threshold
    )

    # ---- Run backtest
    metrics, w_pred, porfolio_values, signal, zscore, returns, capitals, all_closed_positions = run_backtest(
        data, config, x, y, p, q, r, last_train_date, last_test_date
    )

    # ---- Print metrics and plot results
    for period, metric in metrics.items():
        print_metrics(metric, optimal_z, period)
    print_summary(initial_capital, capitals, all_closed_positions)
    plot_estimations(data.index[window:], w_pred)
    plot_portfolio_value(
        data.index[window:],
        porfolio_values,
        signal,
        last_train_date,
        last_test_date
    )
    plot_spread_and_signal(
        data.index[window:],
        zscore,
        signal,
        optimal_z,
        last_train_date,
        last_test_date
    )
    plot_trade_returns(returns)


if __name__ == '__main__':
    main()