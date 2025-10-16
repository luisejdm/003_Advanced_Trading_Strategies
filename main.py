import pandas as pd

from utils import train_test_validation, estandarize_pair
from cointegration import get_non_stationary_stocks, get_best_cointegrated_pair
from sectors import get_sectors
from visualization import plot_cointegrated_stocks
from prints import print_best_pair


def main():
    # Load and preprocess data
    data = pd.read_csv('stocks.csv')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    train, test, validation = train_test_validation(data, 0.6, 0.2, 0.2)

    # Get only non-stationary stocks for cointegration analysis
    non_stationary_stocks = get_non_stationary_stocks(train, alpha=0.05)
    data = data[non_stationary_stocks]

    # Get stock classification by sectors
    sectors = get_sectors()

    # Get the best cointegrated pair by sectors
    coint_results, best_pair, best_pvalue, best_sector = get_best_cointegrated_pair(
        train, sectors, alpha=0.05
    )

    # Estandarize the best pair for plotting
    pair_data = data[[best_pair[0], best_pair[1]]]
    estandarized_pair = estandarize_pair(pair_data)
    print_best_pair(best_pair, best_pvalue, best_sector)
    plot_cointegrated_stocks(estandarized_pair)


if __name__ == '__main__':
    main()