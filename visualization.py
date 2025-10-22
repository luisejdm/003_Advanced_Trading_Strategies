import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

plt.rcParams['figure.figsize'] = [14, 8]
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['grid.alpha'] = 0.8
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['legend.fontsize'] = 13
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.fancybox'] = True
plt.rcParams['figure.dpi'] = 100


def plot_cointegrated_stocks(data: pd.DataFrame)-> None:
    """
    Plot the price series of two cointegrated stocks.
    Args:
        data (pd.DataFrame): DataFrame containing the price series of two stocks.
    """
    plt.figure()
    plt.plot(data.index, data.iloc[:, 0], label=data.columns[0], color='cadetblue')
    plt.plot(data.index, data.iloc[:, 1], label=data.columns[1], color='#1C478B')
    plt.title('Price Series of Cointegrated Stocks')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_all_pairs(train: pd.DataFrame, coint_results: dict, estandarize_fn) -> None:
    """
    Recorre todos los pares cointegrados {sector: [(t1,t2,p)]},
    estandariza cada par con estandarize_fn y los grafica.
    """
    for sector_name, pairs in coint_results.items():
        for (t1, t2, pvalue) in pairs:
            pair_df = train[[t1, t2]].dropna()
            if pair_df.shape[0] < 20:
                continue
            std_df = estandarize_fn(pair_df)
            plt.figure()
            plt.plot(std_df.index, std_df.iloc[:, 0], label=std_df.columns[0])
            plt.plot(std_df.index, std_df.iloc[:, 1], label=std_df.columns[1])
            plt.title(f'{sector_name}: {t1} vs {t2}  |  p={pvalue:.4f}')
            plt.xlabel('Date')
            plt.ylabel('Price (standardized)')
            plt.legend()
            plt.grid(True)
            plt.show()


def plot_estimations(index: pd.Index, w_pred: list, ) -> None:
    """
    Plot the predicted spread values.
    Args:
        index (np.ndarray): Array of dates corresponding to the predictions.
        w_pred (list): List of predicted spread values.
    """
    w_df = pd.DataFrame(w_pred, columns=['alpha', 'beta'], index=index)
    # Plot alpha and beta over time with left and right different y-axes
    fig, ax1 = plt.subplots()
    ax1.set_title('Estimated α and β Over Time')

    ax1.set_xlabel('Date')
    ax1.set_ylabel('α', rotation=0, labelpad=15)
    ax1.plot(w_df.index, w_df['alpha'], color='cadetblue', label='α')
    ax1.tick_params(axis='y')
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.set_ylabel('β', rotation=0, labelpad=15)
    ax2.plot(w_df.index, w_df['beta'], color='#1C478B', label='β')
    ax2.tick_params(axis='y')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')
    plt.show()


def plot_portfolio_value(
        dates: pd.Index, portfolio_values: list, signals: np.ndarray,
        last_train_date, last_test_date
) -> None:
    """
    Plot the portfolio value over time.
    Args:
        dates (pd.Index): Index of dates.
        portfolio_values (list): List of portfolio values corresponding to the dates.
        signals (np.ndarray): Array of trading signals corresponding to the dates.
        last_train_date: Last date of the training set.
        last_test_date: Last date of the test set.
    """
    # Create Series for easier indexing
    idx = pd.Index(dates)
    pv = pd.Series(portfolio_values, index=idx)
    sig = pd.Series(signals, index=idx)

    # Separate for train, test and validation
    train = idx <= last_train_date
    test = (idx > last_train_date) & (idx <= last_test_date)
    val = idx > last_test_date

    # Subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.plot(pv.index[train], pv[train], label='Train', linewidth=1.8, color='#1C478B')
    ax1.plot(pv.index[test], pv[test], label='Test', linewidth=1.8, color='cadetblue')
    ax1.plot(pv.index[val], pv[val], label='Validation', linewidth=1.8, color='dodgerblue')

    ax1.axvline(x=last_train_date, color='k', linestyle='--', linewidth=1, alpha=0.8)
    ax1.axvline(x=last_test_date, color='k', linestyle='--', linewidth=1, alpha=0.8)

    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(sig.index[train], sig[train], label='Train', drawstyle='steps-post', color='#1C478B')
    ax2.plot(sig.index[test], sig[test], label='Test', drawstyle='steps-post', color='cadetblue')
    ax2.plot(sig.index[val], sig[val], label='Validation', drawstyle='steps-post', color='dodgerblue')

    ax2.axvline(x=last_train_date, color='k', linestyle='--', linewidth=1, alpha=0.8)
    ax2.axvline(x=last_test_date, color='k', linestyle='--', linewidth=1, alpha=0.8)

    ax2.set_title('Trading Signals Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Signal')
    ax2.set_ylim(-2, 2)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_spread_and_signal(
        dates: pd.Index, spread: np.ndarray, signals: np.ndarray, z_threshold: float,
        last_train_date=None, last_test_date=None
) -> None:
    """
    Plot the spread and trading signals over time in subplots.
    Args:
        dates (pd.Index): Index of dates.
        spread (np.ndarray): Array of spread values corresponding to the dates.
        signals (np.ndarray): Array of trading signals corresponding to the dates.
        z_threshold (np.ndarray): Array of z-threshold values for reference.
        last_train_date: Last date of the training set.
        last_test_date: Last date of the test set.
    """
    # Create Series for easier indexing
    idx = pd.Index(dates)
    spd = pd.Series(spread, index=idx)
    sig = pd.Series(signals, index=idx)

    # Separate for train, test and validation
    train = idx <= last_train_date
    test = (idx > last_train_date) & (idx <= last_test_date)
    val = idx > last_test_date

    # Subplots
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 10),
        gridspec_kw={'height_ratios': [3, 1]}
    )

    ax1.plot(spd.index[train], spd[train], label='Train', linewidth=1.5, color='#1C478B')
    ax1.plot(spd.index[test], spd[test], label='Test', linewidth=1.5, color='cadetblue')
    ax1.plot(spd.index[val], spd[val], label='Validation', linewidth=1.5, color='dodgerblue')

    ax1.axhline(y=z_threshold, color='firebrick', linestyle='--', linewidth=2.5, label='Z-Score Thresholds')
    ax1.axhline(y=-z_threshold, color='firebrick', linestyle='--', linewidth=2.5)

    if last_train_date:
        ax1.axvline(x=last_train_date, color='k', linestyle='--', linewidth=1, alpha=0.8)
    if last_test_date:
        ax1.axvline(x=last_test_date, color='k', linestyle='--', linewidth=1, alpha=0.8)

    ax1.text(dates[0], z_threshold + 0.1, f'+{z_threshold:.2f}', color='firebrick',
             fontsize=13, va='bottom', fontweight='bold')
    ax1.text(dates[0], -z_threshold - 0.1, f'-{z_threshold:.2f}', color='firebrick',
             fontsize=13, va='top', fontweight='bold')

    ax1.set_title('Spread Over Time')
    ax1.set_ylabel('Spread')
    ax1.set_ylim(spread.min() - 0.5, spread.max() + 0.5)
    ax1.legend(loc='upper right')
    ax1.grid(True)

    ax2.plot(sig.index[train], sig[train],
             label='Train', drawstyle='steps-post', linewidth=1.5, color='#1C478B')
    ax2.plot(sig.index[test], sig[test],
             label='Test', drawstyle='steps-post', linewidth=1.5, color='cadetblue')
    ax2.plot(sig.index[val], sig[val],
             label='Validation', drawstyle='steps-post', linewidth=1.5, color='dodgerblue')

    if last_train_date:
        ax2.axvline(x=last_train_date, color='k', linestyle='--', linewidth=1, alpha=0.8)
    if last_test_date:
        ax2.axvline(x=last_test_date, color='k', linestyle='--', linewidth=1, alpha=0.8)

    ax2.set_title('Trading Signals Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Signal')
    ax2.set_ylim(-2, 2)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_trade_returns(returns: list) -> None:
    """
    Plot a histogram of portfolio returns.
    Args:
        returns (list): A list of portfolio returns.
    """
    plt.figure()
    sns.histplot(returns, color='#1C478B', alpha=0.3, kde=True, bins=50, edgecolor=None,
                 label='Trade Returns')
    plt.title('Overall Trade Returns Distribution')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.axvline(x=0, color='k', linestyle='--')
    plt.grid()
    plt.show()